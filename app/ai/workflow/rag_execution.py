"""One RAG execution and grounding implementation.

Top-level RAG and Planning RAG workers run the *same* compiled graph. Two
implementations were how one path could quietly skip validation, so there is no
inline special case here and no rollout branch that turns grounding off.

The model drives retrieval. It calls the document tools, reads what came back,
and calls again until it is ready to answer — the graph does not retrieve once
up front and hand the model a fixed pack, because which documents matter is
something only the answer-in-progress knows.

Grounding runs for every result, including a retrieval that returned nothing.
Validation reports; it does not rewrite. A zero-evidence answer may ask a
bounded clarifying question, and it may not make a source-backed claim --
enforced by neutralizing any citation the server cannot resolve, not by
replacing the answer after the reader has already seen it. There is
deliberately no regeneration edge and no abstention node: a draft that can be
replaced wholesale cannot be streamed.

Evidence identity is server-owned: IDs come from a per-run allocator, and a
duplicate or ambiguous ID fails validation rather than silently selecting the
first match.

The topology is compiled once per workflow construction. Everything
authenticated -- tools, credentials, messages, the evidence allocator, and
state -- is created per invocation and reaches nodes through the invocation
scope, never through a value captured at compile time.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

from app.ai.workflow.execution_budget import (
    ExecutionBudgetAccountant,
    ExecutionBudgetLimits,
    ExecutionBudgetState,
)
from app.observability.routing import get_routing_metrics_recorder
from app.services.rag_grounding import (
    EvidencePack,
    GroundedAnswer,
    GroundedAnswerGate,
    merge_evidence_payloads,
    render_grounded_answer,
)

logger = logging.getLogger(__name__)

#: Returned in place of a refused retrieval. Phrased as a state of the world:
#: an error-shaped answer invites the model to retry the same call.
_BUDGET_REFUSAL_TEXT = (
    "Evidence gathering has ended for this execution epoch, so this retrieval "
    "was not run. Answer now from the evidence already gathered, and say "
    "plainly what remains unknown."
)

__all__ = [
    "EvidenceIdAllocator",
    "RagExecutionGraph",
    "RagExecutionRequest",
    "RagExecutionResult",
    "RagGroundingReport",
    "RagInvocationScope",
    "RagModelTurn",
    "RagToolOutcome",
    "ProductionRagRuntime",
    "merge_turn_evidence",
]

RagMode = Literal["public", "worker"]
GroundingOutcome = Literal["accepted", "accepted_with_findings", "clarification"]

_DEFAULT_MAX_TOOL_ITERATIONS = 8


class EvidenceIdAllocator:
    """Issues every evidence ID for one RAG run.

    A single allocator per run is what makes an ID mean the same record for the
    whole turn. Reissuing one would let a model cite ``E1`` and get a different
    source than the one validation checked.
    """

    def __init__(self, prefix: str = "E") -> None:
        self._prefix = prefix
        self._next = 1
        self._issued: set[str] = set()

    def allocate(self) -> str:
        evidence_id = f"{self._prefix}{self._next}"
        self._next += 1
        self._issued.add(evidence_id)
        return evidence_id

    def claim(self, evidence_id: str) -> str:
        """Reserve a specific ID, refusing one already issued this run."""
        if evidence_id in self._issued:
            raise ValueError(f"evidence id {evidence_id!r} was already issued this run")
        self._issued.add(evidence_id)
        return evidence_id

    @property
    def issued(self) -> frozenset[str]:
        return frozenset(self._issued)


def merge_turn_evidence(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[EvidencePack, int]:
    """Merge this run's evidence packs and report ID collisions.

    The collision count is surfaced rather than folded away: an ambiguous ID
    means validation cannot know which record a citation refers to.
    """
    return merge_evidence_payloads(payloads)


class RagExecutionRequest(BaseModel):
    """One RAG invocation's objective and authenticated scope."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    objective: str
    conversation_id: str | None = None
    user_id: str | None = None
    device_id: str | None = None
    persona: str | None = None
    model_request: dict[str, Any] | None = None
    history: list[Any] = Field(default_factory=list)
    allowed_tool_ids: tuple[str, ...] = ()
    hitl_policy: dict[str, Any] = Field(default_factory=dict)
    attachments: list[Any] = Field(default_factory=list)
    mode: RagMode = "public"
    dispatch_id: str | None = None
    task_id: str | None = None
    # Carried in rather than read from process memory: a Continue may be served
    # by a worker that never ran the previous epoch, and an absent budget there
    # would look exactly like a fresh turn with a full quota.
    execution_budget: dict[str, Any] | None = None
    # The key research accounting is stored under (R4). Without it a RAG turn's
    # tool calls would fall back to conversation scoping and share a bucket
    # with any other turn in flight for the same conversation.
    logical_turn_id: str | None = None


class RagModelTurn(BaseModel):
    """One RAG model turn: either tool calls or a final answer.

    A turn carrying tool calls is not an answer. Grounding it would validate a
    message the model must still pair with its own tool results.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    text: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    answer: GroundedAnswer | None = None


class RagToolOutcome(BaseModel):
    """What one round of server-executed RAG tool calls produced."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tool_messages: tuple[Any, ...] = ()
    evidence_payloads: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    images: tuple[dict[str, Any], ...] = ()


class RagGroundingReport(BaseModel):
    """What validation actually decided, and on what basis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    validated: bool
    outcome: GroundingOutcome
    regeneration_count: int = 0
    claim_count: int = 0
    cited_claim_count: int = 0
    evidence_id_count: int = 0
    ambiguous_evidence_id_count: int = 0
    tool_iterations: int = 0
    reason_codes: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RagExecutionResult(BaseModel):
    """The grounded outcome of one RAG run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    content: str
    abstained: bool
    grounding: RagGroundingReport
    evidence_ids: tuple[str, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    images: tuple[dict[str, Any], ...] = ()
    mode: RagMode = "public"
    dispatch_id: str | None = None
    task_id: str | None = None
    execution_budget: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RagInvocationScope:
    """Per-invocation collaborators for one RAG run.

    Nothing here may be captured at compile time. The allocator in particular
    is what makes an evidence ID mean one record for one run; sharing it across
    runs would let one caller's ``E1`` resolve to another's document.
    """

    request: RagExecutionRequest
    allocator: EvidenceIdAllocator = field(default_factory=EvidenceIdAllocator)
    #: Budget and provider facts from the model attempt that just ran. The tool
    #: round reads them so a provider fallback is not metered against the
    #: provider that failed.
    turn_metadata: dict[str, Any] = field(default_factory=dict)


def _empty_evidence_pack() -> EvidencePack:
    """An explicitly empty pack, for a run whose model retrieved nothing."""
    return EvidencePack(records=(), token_count=0, omitted_count=0)


def _extend_dicts(
    existing: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Append-only reducer for records several tool rounds contribute to."""
    merged = list(existing or [])
    if update is None:
        return merged
    if isinstance(update, dict):
        merged.append(update)
        return merged
    merged.extend(update)
    return merged


class RagExecutionState(TypedDict):
    """Private state of one RAG run. Never shared across users or requests."""

    request: RagExecutionRequest
    messages: Annotated[list[BaseMessage], add_messages]
    evidence_payloads: Annotated[list[dict[str, Any]], _extend_dicts]
    artifacts: Annotated[list[dict[str, Any]], _extend_dicts]
    images: Annotated[list[dict[str, Any]], _extend_dicts]
    evidence: NotRequired[EvidencePack | None]
    ambiguous_evidence_id_count: NotRequired[int]
    answer: NotRequired[GroundedAnswer | None]
    tool_iterations: NotRequired[int]
    # Last write wins, which is what a linear loop wants: each node hands the
    # next one the counters as they stand.
    execution_budget: NotRequired[ExecutionBudgetState | None]
    finalization: NotRequired[Any]
    result: NotRequired[RagExecutionResult | None]


class RagExecutionGraph:
    """The one compiled RAG topology, shared by public and worker callers.

    Compiled once at construction. ``compile_count`` is asserted by the
    cutover tests: a second compile would mean the two entry points had drifted
    back into two graphs.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        grounded_answer_gate: GroundedAnswerGate,
        settings: Any,
    ) -> None:
        self._runtime = runtime
        self._gate = grounded_answer_gate
        self._settings = settings
        self.compile_count = 0
        self._compiled = self._compile()

    # -- topology --------------------------------------------------------

    def _compile(self) -> Any:
        graph = StateGraph(RagExecutionState)

        graph.add_node("rag_model", self._rag_model)
        graph.add_node("rag_tools", self._rag_tools)
        graph.add_node("collect_rag_outputs", self._collect_outputs)
        graph.add_node("validate_grounding", self._validate)
        graph.add_node("package_rag_result", self._package)

        graph.add_edge(START, "rag_model")
        graph.add_conditional_edges(
            "rag_model",
            self._after_model,
            {"tools": "rag_tools", "validate": "validate_grounding"},
        )
        graph.add_edge("rag_tools", "collect_rag_outputs")
        graph.add_edge("collect_rag_outputs", "rag_model")
        # No regeneration edge and no abstention node: validation records what
        # it found and the same answer is packaged, so the answer can stream.
        graph.add_edge("validate_grounding", "package_rag_result")
        graph.add_edge("package_rag_result", END)

        self.compile_count += 1
        return graph.compile()

    def _accountant(self, state: RagExecutionState) -> ExecutionBudgetAccountant:
        """This run's accountant, rebuilt per node from the carried counters.

        The graph is compiled once and shared by every caller, so per-run state
        cannot live on ``self``. It lives in the graph state instead, and each
        node returns the counters it advanced.
        """
        carried = state.get("execution_budget")
        if carried is None:
            raw = state["request"].execution_budget
            if isinstance(raw, dict):
                try:
                    carried = ExecutionBudgetState.model_validate(raw)
                except Exception:
                    logger.warning("Ignored an unreadable carried RAG execution budget")
        return ExecutionBudgetAccountant(
            limits=ExecutionBudgetLimits.from_settings(self._settings), state=carried
        )

    @property
    def max_tool_iterations(self) -> int:
        return int(
            getattr(self._settings, "rag_max_tool_iterations", _DEFAULT_MAX_TOOL_ITERATIONS)
            or _DEFAULT_MAX_TOOL_ITERATIONS
        )

    # -- invocation ------------------------------------------------------

    async def ainvoke(
        self,
        request: RagExecutionRequest,
        *,
        config: dict[str, Any] | None = None,
        scope: RagInvocationScope | None = None,
    ) -> RagExecutionResult:
        """Run one RAG turn and return its validated result.

        ``scope`` is built here rather than accepted from a caller by default,
        so a fresh evidence allocator per run is the path of least resistance.
        """
        invocation = scope or RagInvocationScope(request=request)
        state = await self._compiled.ainvoke(
            {
                "request": request,
                "messages": [],
                "evidence_payloads": [],
                "artifacts": [],
                "images": [],
            },
            config=self._config_with_scope(config, invocation),
        )
        result = state.get("result")
        if not isinstance(result, RagExecutionResult):  # pragma: no cover - defensive
            raise RuntimeError("RAG execution produced no validated result")
        return result

    @staticmethod
    def _config_with_scope(
        config: dict[str, Any] | None, invocation: RagInvocationScope
    ) -> dict[str, Any]:
        """Carry the invocation scope on the run config, never in state.

        State is checkpointed; the allocator and the authenticated request are
        not things a checkpoint may hold.
        """
        merged = dict(config or {})
        configurable = dict(merged.get("configurable") or {})
        configurable["rag_invocation_scope"] = invocation
        merged["configurable"] = configurable
        return merged

    @staticmethod
    def _scope() -> RagInvocationScope | None:
        """Read the invocation scope from the live run config.

        Reading it here rather than accepting it as a node argument keeps the
        node signatures free of framework injection, which ``from __future__
        import annotations`` makes unreliable.
        """
        try:
            config = get_config()
        except RuntimeError:  # pragma: no cover - node called outside a run
            return None
        if not isinstance(config, Mapping):
            return None
        configurable = config.get("configurable")
        if not isinstance(configurable, Mapping):
            return None
        scope = configurable.get("rag_invocation_scope")
        return scope if isinstance(scope, RagInvocationScope) else None

    # -- nodes -----------------------------------------------------------

    async def _rag_model(self, state: RagExecutionState) -> dict[str, Any]:
        """One model turn. Tool calls and a final answer are different things."""
        scope = self._scope()
        accountant = self._accountant(state)
        decision = accountant.note_model_call()
        turn = await self._runtime.model_turn(
            state["request"],
            messages=list(state.get("messages") or []),
            evidence=state.get("evidence"),
            scope=scope,
            # The RAG agent already has this seam: it maps to ``disable_tools``
            # on the model binding, so the reserved answer call cannot ask for
            # another retrieval however the prompt reads.
            force_final=decision.tools_suppressed,
        )
        turn = _as_model_turn(turn)

        message = AIMessage(
            content=turn.text,
            tool_calls=[dict(call) for call in turn.tool_calls] if turn.tool_calls else [],
        )
        update: dict[str, Any] = {"messages": [message], "execution_budget": accountant.state}
        if not turn.tool_calls:
            update["answer"] = turn.answer if turn.answer is not None else GroundedAnswer()
        return update

    def _after_model(self, state: RagExecutionState) -> str:
        """Route on the model's own last message, not on a counter.

        The iteration ceiling is a backstop: exceeding it forces validation of
        whatever the model has said rather than looping forever.
        """
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return "validate"
        budget = state.get("execution_budget")
        if budget is not None and budget.forced_synthesis:
            # The answer call was already reserved and made tool-free. Anything
            # it still asked for is not going to run, so validate what it said.
            logger.info("RAG execution budget spent; validating the reserved answer")
            return "validate"
        if int(state.get("tool_iterations") or 0) >= self.max_tool_iterations:
            logger.info("RAG tool iterations exhausted; forcing grounding validation")
            return "validate"
        return "tools"

    async def _rag_tools(self, state: RagExecutionState) -> dict[str, Any]:
        """Execute the model's tool calls through the server-owned executor.

        No generic exception boundary here: ``GraphBubbleUp`` and
        ``GraphInterrupt`` must cross unchanged, or an approval pause becomes a
        tool error and the turn answers without the human.
        """
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        tool_calls = [dict(call) for call in (getattr(last, "tool_calls", None) or [])]

        accountant = self._accountant(state)
        affordable: list[dict[str, Any]] = []
        refused: list[ToolMessage] = []
        for call in tool_calls:
            if accountant.note_tool_call().allowed:
                affordable.append(call)
            else:
                # Paired, not dropped. A transcript with an unanswered tool call
                # is rejected by the provider, so refusing without answering
                # turns a budget stop into an error on the very next call.
                refused.append(
                    ToolMessage(
                        content=_BUDGET_REFUSAL_TEXT,
                        tool_call_id=str(call.get("id") or ""),
                        name=str(call.get("name") or "tool"),
                        status="success",
                    )
                )

        outcome = (
            _as_tool_outcome(
                await self._runtime.execute_tools(
                    state["request"],
                    tool_calls=affordable,
                    iteration=int(state.get("tool_iterations") or 0),
                    scope=self._scope(),
                )
            )
            if affordable
            else RagToolOutcome()
        )
        return {
            "messages": [*outcome.tool_messages, *refused],
            "evidence_payloads": [dict(payload) for payload in outcome.evidence_payloads],
            "artifacts": [dict(artifact) for artifact in outcome.artifacts],
            "images": [dict(image) for image in outcome.images],
            "execution_budget": accountant.state,
        }

    async def _collect_outputs(self, state: RagExecutionState) -> dict[str, Any]:
        """Merge every pack this run produced through the typed reducer."""
        payloads = [
            payload
            for payload in (state.get("evidence_payloads") or [])
            if isinstance(payload, dict)
        ]
        evidence, ambiguous = merge_turn_evidence(payloads)
        return {
            "evidence": evidence,
            "ambiguous_evidence_id_count": ambiguous,
            "tool_iterations": int(state.get("tool_iterations") or 0) + 1,
        }

    async def _validate(self, state: RagExecutionState) -> dict[str, Any]:
        """Validate every result. There is no path that skips this node."""
        evidence = state.get("evidence") or _empty_evidence_pack()
        answer = state.get("answer") or GroundedAnswer()

        finalization = await self._gate.finalize_answer(
            evidence=evidence,
            answer=answer,
            ambiguous_evidence_id_count=int(state.get("ambiguous_evidence_id_count") or 0),
        )
        return {"answer": finalization.answer, "finalization": finalization}

    async def _package(self, state: RagExecutionState) -> dict[str, Any]:
        request = state["request"]
        evidence = state.get("evidence") or _empty_evidence_pack()
        finalization = state.get("finalization")
        answer = state.get("answer") or GroundedAnswer()

        outcome = _outcome_for(finalization, answer, evidence)
        # Grounding is record-only, so this is the signal that tells an operator
        # whether validation is finding problems at all. `abstained` never
        # rises: nothing withholds an answer any more.
        get_routing_metrics_recorder().grounding_outcome(outcome=outcome)
        content = _render(answer, evidence)
        report = RagGroundingReport(
            validated=True,
            outcome=outcome,
            regeneration_count=0,
            claim_count=getattr(finalization, "claim_count", len(answer.claims)),
            cited_claim_count=getattr(
                finalization,
                "cited_claim_count",
                sum(bool(claim.evidence_ids) for claim in answer.claims),
            ),
            evidence_id_count=len(evidence.evidence_ids),
            ambiguous_evidence_id_count=int(state.get("ambiguous_evidence_id_count") or 0),
            tool_iterations=int(state.get("tool_iterations") or 0),
            reason_codes=tuple(
                getattr(getattr(finalization, "validation", None), "reason_codes", ())
            ),
        )
        result = RagExecutionResult(
            content=content,
            abstained=False,
            grounding=report,
            evidence_ids=_evidence_ids(evidence),
            evidence=_evidence_records(evidence),
            artifacts=tuple(
                artifact
                for artifact in (state.get("artifacts") or [])
                if isinstance(artifact, dict)
            ),
            images=tuple(image for image in (state.get("images") or []) if isinstance(image, dict)),
            mode=request.mode,
            dispatch_id=request.dispatch_id,
            task_id=request.task_id,
            execution_budget=_budget_snapshot(state.get("execution_budget")),
        )
        return {"result": result}


def _budget_snapshot(state: Any) -> dict[str, Any]:
    """The budget as the caller stores it, or an empty dict when unused."""
    if isinstance(state, ExecutionBudgetState):
        return state.model_dump(mode="json")
    return {}


def _as_model_turn(value: Any) -> RagModelTurn:
    """Accept a typed turn, or a bare answer from a runtime that has none."""
    if isinstance(value, RagModelTurn):
        return value
    if isinstance(value, GroundedAnswer):
        return RagModelTurn(text=str(value.raw_text or ""), answer=value)
    if value is None:
        return RagModelTurn()
    raise TypeError(f"RAG runtime returned an unsupported model turn: {type(value).__name__}")


def _as_tool_outcome(value: Any) -> RagToolOutcome:
    if isinstance(value, RagToolOutcome):
        return value
    if value is None:
        return RagToolOutcome()
    if isinstance(value, Mapping):
        return RagToolOutcome(**value)
    raise TypeError(f"RAG runtime returned an unsupported tool outcome: {type(value).__name__}")


def _evidence_ids(evidence: EvidencePack) -> tuple[str, ...]:
    """Evidence IDs in the order the server produced them.

    ``EvidencePack.evidence_ids`` is a frozenset, so reading it directly would
    make the reported order arbitrary between runs.
    """
    return tuple(record.evidence_id for record in evidence.records or ())


def _evidence_records(evidence: EvidencePack) -> tuple[dict[str, Any], ...]:
    """Server-produced evidence records, JSON-safe and citation-addressable."""
    payload = evidence.to_dict()
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        return ()
    return tuple(dict(record) for record in records if isinstance(record, Mapping))


def _outcome_for(
    finalization: Any, answer: GroundedAnswer, evidence: EvidencePack
) -> GroundingOutcome:
    """Name what validation decided.

    ``clarification`` is a real outcome: with no evidence retrieved and nothing
    asserted, asking the user a bounded question is the correct grounded
    response. ``accepted_with_findings`` means validation recorded something —
    a density shortfall, an unresolvable id — that the reader is protected from
    at the citation level rather than by suppressing the answer.
    """
    if not answer.claims and not evidence.records and (answer.raw_text or "").strip():
        return "clarification"
    validation = getattr(finalization, "validation", None)
    return "accepted" if getattr(validation, "valid", True) else "accepted_with_findings"


def _render(answer: GroundedAnswer, evidence: EvidencePack) -> str:
    if answer.abstained:
        return render_grounded_answer(answer, evidence, text=answer.raw_text)
    if not answer.claims:
        return str(answer.raw_text or "")
    return render_grounded_answer(answer, evidence, text=answer.raw_text)


class ProductionRagRuntime:
    """The model and tool halves of production RAG, behind the graph protocol.

    This is an adapter, not a second implementation. Retrieval, evidence
    packing, and citation identity stay in the server-owned helpers the RAG
    agent already uses; what changes is who drives the loop — the compiled
    graph rather than a hand-rolled node pair.

    Non-search tool calls run through the common execution pipeline for the
    same reason every other specialist does: the execution context, artifact
    and image records, blob offloading, and error normalization all live there.
    """

    def __init__(self, *, rag_agent: Any, agent_lookup: Any, settings: Any) -> None:
        self._rag_agent = rag_agent
        self._agent_lookup = agent_lookup
        self._settings = settings

    async def model_turn(
        self,
        request: RagExecutionRequest,
        *,
        messages: Sequence[Any],
        evidence: Any,
        scope: RagInvocationScope | None = None,
        force_final: bool = False,
    ) -> RagModelTurn:
        """One RAG model call.

        A response carrying tool calls is returned as tool calls only. Parsing
        it as an answer would ground a message the model must still pair with
        its own tool results.

        ``force_final`` reaches the agent as ``rag_force_final_response``, which
        is the flag it already maps to ``disable_tools`` on the model binding.
        Reusing it rather than adding a second switch keeps one answer to "may
        this call use tools".
        """
        from app.ai.schemas import AgentMessage, MessageRole
        from app.services.rag_grounding import parse_grounded_answer

        agent_message = AgentMessage(
            role=MessageRole.USER,
            content=request.objective,
            metadata={
                "persona": request.persona,
                "history": list(request.history),
                "original_query": request.objective,
                "rag_tool_messages": list(messages),
                "model_request": request.model_request,
                "user_id": request.user_id,
                "device_id": request.device_id,
                "rag_force_final_response": bool(force_final),
            },
            attachments=list(request.attachments),
        )
        response = await self._rag_agent.process_message(agent_message, request.conversation_id)

        tool_calls = tuple(response.message.tool_calls or ())
        if tool_calls:
            self._record_turn_metadata(scope, response)
            return RagModelTurn(text="", tool_calls=tool_calls)

        self._record_turn_metadata(scope, response)
        text = str(response.message.content or "")
        return RagModelTurn(text=text, answer=parse_grounded_answer(text))

    async def execute_tools(
        self,
        request: RagExecutionRequest,
        *,
        tool_calls: Sequence[dict[str, Any]],
        iteration: int,
        scope: RagInvocationScope | None = None,
    ) -> RagToolOutcome:
        """Execute one round of the model's tool calls.

        No generic exception boundary: an approval interrupt raised inside the
        pipeline is control flow and must reach the parent graph unchanged.
        """
        from langchain_core.messages import ToolMessage

        from app.ai.rag_tool_actions import (
            canonicalize_rag_tool_call,
            execute_rag_search_tool_call,
            fit_rag_tool_message_content,
        )
        from app.ai.tool_context import tool_execution_context
        from app.ai.tool_execution import ensure_agent_tool_map, execute_tool_calls
        from app.ai.utils import normalize_tool_call

        normalized = [canonicalize_rag_tool_call(normalize_tool_call(call)) for call in tool_calls]
        turn = _turn_metadata(scope)
        budget = _EvidenceBudget.from_metadata(turn, rag_agent=self._rag_agent)
        context: dict[str, Any] = dict(turn.get("context") or {})

        tool_messages: list[Any] = []
        evidence_payloads: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []

        non_search = [call for call in normalized if call.get("name") != "search_documents"]
        non_search_outputs: dict[str, dict[str, Any]] = {}
        if non_search:
            agent = self._agent_lookup(request)
            tool_map = (
                await ensure_agent_tool_map(
                    agent,
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                    device_id=request.device_id,
                )
                if agent is not None
                else {}
            )
            with tool_execution_context(
                request.conversation_id,
                request.user_id,
                _agent_key(agent),
                request.device_id,
                rich_response_capable=True,
                logical_turn_id=request.logical_turn_id,
            ):
                outputs, produced_artifacts, produced_images = await execute_tool_calls(
                    tool_calls=non_search,
                    tool_map=tool_map,
                    capture_images=True,
                    device_id=request.device_id,
                    agent=agent,
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                )
            for output in outputs:
                call_id = str(output.get("tool_call_id") or "")
                if call_id:
                    non_search_outputs[call_id] = output
            artifacts.extend(produced_artifacts)
            images.extend(produced_images)

        for call in normalized:
            call_id = str(call.get("id") or "")
            name = str(call.get("name") or "")

            if name != "search_documents":
                stored = non_search_outputs.get(call_id)
                content = (
                    str(stored.get("content") or "")
                    if stored is not None
                    else f"Error: Tool {name} not found"
                )
                fitted, consumed, omitted = fit_rag_tool_message_content(
                    content=content,
                    allowance=budget.allowance_for(name),
                    token_counter=budget.token_counter,
                    provider=budget.provider,
                    model=budget.model,
                    tool_call_id=call_id,
                    tool_name=name,
                )
                budget.consume(consumed)
                tool_messages.append(
                    ToolMessage(
                        content=fitted,
                        tool_call_id=call_id,
                        name=name,
                        status="error" if stored is None else "success",
                    )
                )
                if omitted:
                    logger.info("RAG tool %s output omitted for budget", name)
                continue

            search = await execute_rag_search_tool_call(
                rag_agent=self._rag_agent,
                tool_call=call,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                context=context,
                question=request.objective,
                max_agentic_images=int(getattr(self._settings, "agentic_rag_max_images", 6) or 6),
                allowance=budget.allowance_for(name),
                remaining_allowance=budget.remaining,
                evidence_token_counter=budget.token_counter,
                evidence_provider=budget.provider,
                evidence_model=budget.model,
            )
            budget.consume(search.consumed_tokens)
            artifacts.append(search.artifact)
            artifact = search.artifact if isinstance(search.artifact, dict) else {}
            pack = artifact.get("rag_evidence")
            if isinstance(pack, dict):
                evidence_payloads.append(pack)
            tool_messages.append(
                ToolMessage(content=search.public_text, tool_call_id=call_id, name=name)
            )

        images.extend(
            image for image in (context.get("agentic_images") or []) if isinstance(image, dict)
        )
        return RagToolOutcome(
            tool_messages=tuple(tool_messages),
            evidence_payloads=tuple(evidence_payloads),
            artifacts=tuple(artifacts),
            images=tuple(images),
        )

    @staticmethod
    def _record_turn_metadata(scope: RagInvocationScope | None, response: Any) -> None:
        """Carry the model attempt's budget/provider facts to the tool round.

        They belong to the attempt that just ran, not to the request, so a
        provider fallback does not leave the next round counting tokens
        against the provider that failed.
        """
        if scope is None:
            return
        metadata = getattr(response, "metadata", None)
        scope.turn_metadata = dict(metadata) if isinstance(metadata, dict) else {}


def _turn_metadata(scope: RagInvocationScope | None) -> dict[str, Any]:
    metadata = getattr(scope, "turn_metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _agent_key(agent: Any) -> str:
    return str(
        getattr(agent, "tool_state_key", None) or getattr(agent, "agent_config_key", None) or "rag"
    )


class _EvidenceBudget:
    """The evidence token allowance one RAG tool round may spend.

    Evidence packs fail closed on a missing allowance; non-pack results must
    not. "Absent" means the request budget never ran, so there is no
    authoritative remainder to enforce against, and bounding to zero would
    silently blank a tool result nobody metered.
    """

    #: Results whose model-visible text is server-generated and re-read after
    #: execution. Replacing one with a budget marker would discard feedback the
    #: model needs in order to stop retrying.
    UNBOUNDABLE = frozenset({"hand_off"})

    def __init__(
        self, *, allowance: int | None, token_counter: Any, provider: str, model: str
    ) -> None:
        self._authoritative = allowance is not None
        self.remaining = max(0, int(allowance or 0))
        self.token_counter = token_counter
        self.provider = provider
        self.model = model

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any], *, rag_agent: Any) -> _EvidenceBudget:
        from app.ai.token_counter import TokenCounter

        request_budget = metadata.get("request_budget") or {}
        allowance = (
            request_budget.get("evidence_token_allowance")
            if isinstance(request_budget, Mapping)
            else None
        )
        provider = str(metadata.get("provider") or "gemini")
        model = str(metadata.get("model") or "gemini-2.5-flash")

        counter: Any = TokenCounter()
        resolver = getattr(rag_agent, "_take_evidence_token_counter", None)
        if callable(resolver):
            try:
                counter = resolver(
                    metadata.get("evidence_tokenization"), provider=provider, model=model
                )
            except Exception:  # noqa: BLE001 - a counter is never worth failing a turn
                logger.exception("Failed to resolve ephemeral evidence counter")

        return cls(allowance=allowance, token_counter=counter, provider=provider, model=model)

    def allowance_for(self, tool_name: str) -> int | None:
        """The remainder this result may be bounded to, or None when unbounded."""
        if not self._authoritative or tool_name in self.UNBOUNDABLE:
            return None
        return self.remaining

    def consume(self, tokens: int) -> None:
        self.remaining = max(0, self.remaining - int(tokens or 0))
