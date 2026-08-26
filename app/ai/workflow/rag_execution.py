"""One RAG execution and grounding implementation.

Top-level RAG and Planning RAG workers run the *same* compiled graph. Two
implementations were how one path could quietly skip validation, so there is no
inline special case here and no rollout branch that turns grounding off.

Grounding runs for every result, including a retrieval that returned nothing. A
zero-evidence answer may ask a bounded clarifying question or abstain, but it
may never make a source-backed claim. One invalid answer earns exactly one
constrained regeneration; a second invalid answer becomes an explicit
abstention.

Evidence identity is server-owned: IDs come from a per-run allocator, and a
duplicate or ambiguous ID fails validation rather than silently selecting the
first match.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

from app.services.rag_grounding import (
    EvidencePack,
    GroundedAnswer,
    GroundedAnswerGate,
    merge_evidence_payloads,
    render_grounded_answer,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EvidenceIdAllocator",
    "RagExecutionGraphFactory",
    "RagExecutionRequest",
    "RagExecutionResult",
    "RagGroundingReport",
    "merge_turn_evidence",
]

RagMode = Literal["public", "worker"]
GroundingOutcome = Literal["accepted", "regenerated", "clarification", "abstained"]


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
    model_request: dict[str, Any] | None = None
    history: list[Any] = Field(default_factory=list)
    allowed_tool_ids: tuple[str, ...] = ()
    mode: RagMode = "public"
    task_id: str | None = None


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
    task_id: str | None = None


class RagExecutionState(TypedDict):
    """Private state of one RAG run. Never shared across users or requests."""

    request: RagExecutionRequest
    evidence: NotRequired[EvidencePack | None]
    ambiguous_evidence_id_count: NotRequired[int]
    answer: NotRequired[GroundedAnswer | None]
    regeneration_count: NotRequired[int]
    finalization: NotRequired[Any]
    result: NotRequired[RagExecutionResult | None]


class RagExecutionGraphFactory:
    """Builds one compiled RAG graph per invocation.

    Request-scoped graph state is never shared across users, so the graph is
    built per run rather than cached.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        grounded_answer_gate: GroundedAnswerGate,
        settings: Any,
        checkpointer: Any | None = None,
    ) -> None:
        self._runtime = runtime
        self._gate = grounded_answer_gate
        self._settings = settings
        self._checkpointer = checkpointer
        self.build_count = 0

    def build(self) -> Any:
        """Compile the shared RAG topology for one invocation."""
        self.build_count += 1
        graph = StateGraph(RagExecutionState)

        graph.add_node("prepare_request", self._prepare_request)
        graph.add_node("collect_evidence", self._collect_evidence)
        graph.add_node("rag_model", self._rag_model)
        graph.add_node("validate_grounding", self._validate_grounding)
        graph.add_node("package_result", self._package_result)

        graph.add_edge(START, "prepare_request")
        graph.add_edge("prepare_request", "collect_evidence")
        graph.add_edge("collect_evidence", "rag_model")
        graph.add_edge("rag_model", "validate_grounding")
        graph.add_edge("validate_grounding", "package_result")
        graph.add_edge("package_result", END)

        compiled = (
            graph.compile(checkpointer=self._checkpointer)
            if self._checkpointer
            else graph.compile()
        )
        return _RagExecutionRun(compiled)

    # -- nodes -----------------------------------------------------------

    async def _prepare_request(self, state: RagExecutionState) -> dict[str, Any]:
        return {"regeneration_count": 0}

    async def _collect_evidence(self, state: RagExecutionState) -> dict[str, Any]:
        """Retrieve, then merge this run's packs through the typed reducer."""
        payload = await self._runtime.retrieve(state["request"])
        payloads = [payload] if isinstance(payload, Mapping) and payload else []
        evidence, ambiguous = merge_turn_evidence(payloads)
        return {"evidence": evidence, "ambiguous_evidence_id_count": ambiguous}

    async def _rag_model(self, state: RagExecutionState) -> dict[str, Any]:
        answer = await self._runtime.answer(state["request"], state.get("evidence"))
        return {"answer": answer}

    async def _validate_grounding(self, state: RagExecutionState) -> dict[str, Any]:
        """Validate every result. There is no path that skips this node."""
        request = state["request"]
        evidence = state.get("evidence") or EvidencePack()
        answer = state.get("answer") or GroundedAnswer()

        finalization = await self._gate.finalize_answer(
            question=request.objective,
            evidence=evidence,
            answer=answer,
            regenerate=self._regenerator(),
            mode="enforced",
            ambiguous_evidence_id_count=int(state.get("ambiguous_evidence_id_count") or 0),
        )
        return {
            "answer": finalization.answer,
            "regeneration_count": 1 if finalization.regenerated else 0,
            "finalization": finalization,
        }

    async def _package_result(self, state: RagExecutionState) -> dict[str, Any]:
        request = state["request"]
        evidence = state.get("evidence") or EvidencePack()
        finalization = state.get("finalization")
        answer = state.get("answer") or GroundedAnswer()

        outcome = _outcome_for(finalization, answer, evidence)
        content = _render(answer, evidence)
        report = RagGroundingReport(
            validated=True,
            outcome=outcome,
            regeneration_count=int(state.get("regeneration_count") or 0),
            claim_count=getattr(finalization, "claim_count", len(answer.claims)),
            cited_claim_count=getattr(
                finalization,
                "cited_claim_count",
                sum(bool(claim.evidence_ids) for claim in answer.claims),
            ),
            evidence_id_count=len(evidence.evidence_ids),
            ambiguous_evidence_id_count=int(state.get("ambiguous_evidence_id_count") or 0),
            reason_codes=tuple(
                getattr(getattr(finalization, "validation", None), "reason_codes", ())
            ),
        )
        result = RagExecutionResult(
            content=content,
            abstained=bool(answer.abstained),
            grounding=report,
            evidence_ids=tuple(evidence.evidence_ids),
            mode=request.mode,
            task_id=request.task_id,
        )
        return {"result": result}

    # -- helpers ---------------------------------------------------------

    def _regenerator(self) -> Any | None:
        regenerate = getattr(self._runtime, "regenerate", None)
        return regenerate if callable(regenerate) else None


class _RagExecutionRun:
    """Thin adapter so callers await one result instead of a state dict."""

    def __init__(self, compiled: Any) -> None:
        self._compiled = compiled

    async def ainvoke(
        self, request: RagExecutionRequest, config: dict[str, Any] | None = None
    ) -> RagExecutionResult:
        state = await self._compiled.ainvoke({"request": request}, config=config)
        result = state.get("result")
        if not isinstance(result, RagExecutionResult):  # pragma: no cover - defensive
            raise RuntimeError("RAG execution produced no validated result")
        return result


def _outcome_for(
    finalization: Any, answer: GroundedAnswer, evidence: EvidencePack
) -> GroundingOutcome:
    """Name what validation decided.

    ``clarification`` is a real outcome, not a soft abstention: with no
    evidence retrieved and nothing asserted, asking the user a bounded question
    is the correct grounded response.
    """
    if answer.abstained:
        return "abstained"
    if not answer.claims and not evidence.records and (answer.raw_text or "").strip():
        return "clarification"
    return "regenerated" if getattr(finalization, "regenerated", False) else "accepted"


def _render(answer: GroundedAnswer, evidence: EvidencePack) -> str:
    if answer.abstained:
        return render_grounded_answer(answer, evidence, text=answer.raw_text)
    if not answer.claims:
        return str(answer.raw_text or "")
    return render_grounded_answer(answer, evidence, text=answer.raw_text)
