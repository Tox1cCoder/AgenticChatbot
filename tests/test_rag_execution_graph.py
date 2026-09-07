"""One shared RAG execution graph, with grounding that is never optional.

Top-level RAG and Planning RAG workers must be the *same* compiled graph: two
implementations were how one path could quietly skip validation. The model
drives retrieval — it calls the document tools, reads the results, and calls
again until it is ready to answer — so evidence arrives through the tool loop
rather than from a single retrieval the graph performs up front.

Grounding runs for every result, including a retrieval that found nothing. An
answer with no evidence may ask for clarification, but it may not claim a
source. Validation records findings and never regenerates or abstains: a draft
that can be replaced wholesale cannot be streamed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from app.ai.workflow.rag_execution import (
    EvidenceIdAllocator,
    RagExecutionGraph,
    RagExecutionRequest,
    RagExecutionResult,
    RagModelTurn,
    RagToolOutcome,
    merge_turn_evidence,
)
from app.services.rag_grounding import GroundedAnswer, GroundedAnswerGate, GroundedClaim


def _gate(min_coverage: float = 0.5) -> GroundedAnswerGate:
    return GroundedAnswerGate(min_coverage=min_coverage)


def _evidence_payload(*evidence_ids: str) -> dict:
    return {
        "records": [
            {
                "evidence_id": evidence_id,
                "document_id": "11111111-1111-1111-1111-111111111111",
                "filename": "manual.pdf",
                "text": f"content for {evidence_id}",
            }
            for evidence_id in evidence_ids
        ]
    }


def _answer(*evidence_ids: str, text: str = "The answer is 42.") -> GroundedAnswer:
    return GroundedAnswer(
        claims=[GroundedClaim(text=text, evidence_ids=list(evidence_ids))],
        raw_text=text,
    )


def _search_call(call_id: str = "call-1") -> dict:
    return {
        "name": "search_documents",
        "id": call_id,
        "args": {"action": "search_chunks", "query": "what does the manual say?"},
    }


def _request(**overrides) -> RagExecutionRequest:
    payload = {
        "objective": "what does the manual say?",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "model_request": None,
        "history": [],
        "mode": "public",
    }
    payload.update(overrides)
    return RagExecutionRequest(**payload)


class ScriptedRagRuntime:
    """Deterministic stand-in for the model/tool half of the RAG graph.

    ``turns`` is the script the model follows. A turn carrying tool calls sends
    the graph through ``rag_tools``; a turn carrying an answer sends it to
    validation.
    """

    def __init__(self, *, turns, evidence=None, tool_error=None, artifacts=(), images=()):
        self._turns = list(turns)
        self.evidence = evidence or {}
        self._tool_error = tool_error
        self._artifacts = tuple(artifacts)
        self._images = tuple(images)
        self.model_calls = 0
        self.tool_calls_executed: list[dict] = []
        self.seen_evidence_ids: list[tuple[str, ...]] = []
        self.seen_scopes: list[object] = []
        self.regeneration_calls = 0
        self.force_final_flags: list[bool] = []

    async def model_turn(self, request, *, messages, evidence, scope=None, force_final=False):
        self.model_calls += 1
        self.seen_scopes.append(scope)
        self.force_final_flags.append(bool(force_final))
        records = getattr(evidence, "records", ()) or ()
        self.seen_evidence_ids.append(tuple(record.evidence_id for record in records))
        # Index by this run's own turn count, not a cumulative counter, so the
        # script replays identically for every invocation of a shared graph.
        turn = sum(1 for message in messages if getattr(message, "type", None) == "ai")
        return self._turns[min(turn, len(self._turns) - 1)]

    async def execute_tools(self, request, *, tool_calls, iteration, scope=None):
        if self._tool_error is not None:
            raise self._tool_error
        self.tool_calls_executed.extend(tool_calls)
        return RagToolOutcome(
            tool_messages=tuple(
                ToolMessage(content="retrieved", tool_call_id=str(call.get("id") or ""))
                for call in tool_calls
            ),
            evidence_payloads=(self.evidence,) if self.evidence else (),
            artifacts=self._artifacts,
            images=self._images,
        )


def _graph(runtime, gate=None, **overrides) -> RagExecutionGraph:
    payload = {
        "runtime": runtime,
        "grounded_answer_gate": gate or _gate(),
        "settings": SimpleNamespace(
            rag_evidence_max_tokens=0,
            enable_citation_verification=True,
            rag_max_tool_iterations=8,
        ),
    }
    payload.update(overrides)
    return RagExecutionGraph(**payload)


def _retrieve_then_answer(*evidence_ids: str, text: str = "The answer is 42.") -> list:
    """The ordinary shape: one retrieval round, then a grounded answer."""
    return [
        RagModelTurn(tool_calls=(_search_call(),)),
        RagModelTurn(text=text, answer=_answer(*evidence_ids, text=text)),
    ]


# ----------------------------------------------------------------------
# one compiled graph
# ----------------------------------------------------------------------


def test_the_topology_is_compiled_exactly_once():
    graph = _graph(ScriptedRagRuntime(turns=_retrieve_then_answer("E1")))
    assert graph.compile_count == 1


async def test_repeated_invocations_do_not_recompile():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    graph = _graph(runtime)

    await graph.ainvoke(_request(mode="public"))
    await graph.ainvoke(_request(mode="worker"))

    assert graph.compile_count == 1


async def test_top_level_and_worker_share_one_graph_object():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    graph = _graph(runtime)

    public = await graph.ainvoke(_request(mode="public"))
    worker = await graph.ainvoke(_request(mode="worker", dispatch_id="d1", task_id="t1"))

    assert isinstance(public, RagExecutionResult)
    assert isinstance(worker, RagExecutionResult)
    assert worker.dispatch_id == "d1"
    assert worker.task_id == "t1"


async def test_worker_and_public_enforce_the_same_evidence_budget():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    graph = _graph(runtime)

    public = await graph.ainvoke(_request(mode="public"))
    worker = await graph.ainvoke(_request(mode="worker"))

    assert public.grounding.validated is worker.grounding.validated is True
    assert public.evidence_ids == worker.evidence_ids == ("E1",)


async def test_each_invocation_gets_its_own_evidence_allocator():
    """A shared allocator would let one caller's ``E1`` resolve to another's."""
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    graph = _graph(runtime)

    await graph.ainvoke(_request())
    await graph.ainvoke(_request())

    allocators = {id(scope.allocator) for scope in runtime.seen_scopes if scope is not None}
    assert len(allocators) == 2


# ----------------------------------------------------------------------
# the model drives retrieval
# ----------------------------------------------------------------------


async def test_the_model_calls_the_document_tools_itself():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    result = await _graph(runtime).ainvoke(_request())

    assert [call["name"] for call in runtime.tool_calls_executed] == ["search_documents"]
    assert runtime.model_calls == 2
    assert result.evidence_ids == ("E1",)


async def test_the_second_model_turn_sees_what_the_first_retrieved():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    await _graph(runtime).ainvoke(_request())

    assert runtime.seen_evidence_ids == [(), ("E1",)]


async def test_a_turn_with_tool_calls_is_never_grounded_as_an_answer():
    """Grounding a tool-calling turn would validate a message with no answer."""
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    result = await _graph(runtime).ainvoke(_request())

    assert result.grounding.tool_iterations == 1
    assert result.content


async def test_an_answer_without_any_tool_call_still_reaches_validation():
    runtime = ScriptedRagRuntime(turns=[RagModelTurn(text="No documents.", answer=_answer())])
    result = await _graph(runtime).ainvoke(_request())

    assert runtime.tool_calls_executed == []
    assert result.grounding.validated is True
    assert result.grounding.tool_iterations == 0


async def test_the_tool_loop_has_a_backstop_ceiling():
    """An always-retrieving model is forced to validation rather than looping."""
    runtime = ScriptedRagRuntime(
        turns=[RagModelTurn(tool_calls=(_search_call(),))], evidence=_evidence_payload("E1")
    )
    graph = _graph(
        runtime,
        settings=SimpleNamespace(
            rag_evidence_max_tokens=0,
            enable_citation_verification=True,
            rag_max_tool_iterations=2,
        ),
    )

    result = await graph.ainvoke(_request())

    assert result.grounding.tool_iterations == 2
    assert result.grounding.validated is True


async def test_control_flow_exceptions_cross_the_tool_node_unchanged():
    """An approval pause is not a tool error, so nothing normalizes it."""
    runtime = ScriptedRagRuntime(
        turns=[RagModelTurn(tool_calls=(_search_call(),))],
        tool_error=GraphBubbleUp("paused for approval"),
    )
    with pytest.raises(GraphBubbleUp):
        await _graph(runtime).ainvoke(_request())


# ----------------------------------------------------------------------
# grounding is mandatory
# ----------------------------------------------------------------------


async def test_a_grounded_answer_is_accepted():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    result = await _graph(runtime).ainvoke(_request())

    assert result.abstained is False
    assert result.grounding.outcome == "accepted"
    assert result.grounding.validated is True
    assert result.grounding.regeneration_count == 0


async def test_invalid_citations_are_recorded_without_a_second_generation():
    """An unresolvable id is a finding, not grounds for regenerating the turn.

    The reader is protected where the citation renders, so there is no second
    model call and no draft to discard.
    """
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E99"), evidence=_evidence_payload("E1")
    )
    result = await _graph(runtime).ainvoke(_request())

    assert result.abstained is False
    assert result.grounding.regeneration_count == 0
    assert runtime.regeneration_calls == 0
    assert result.grounding.outcome == "accepted_with_findings"
    assert "unknown_evidence_id" in result.grounding.reason_codes


async def test_unknown_evidence_id_never_reaches_the_public_result():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E404"), evidence=_evidence_payload("E1")
    )
    result = await _graph(runtime).ainvoke(_request())

    assert "E404" not in result.content
    assert "E404" not in result.evidence_ids


async def test_no_evidence_still_runs_grounding_and_cannot_claim_sources():
    """With nothing retrieved, every citation is unresolvable — so none render.

    The invariant is unchanged; only its enforcement moved. It is no longer
    "replace the answer", it is "a citation to nothing is not a citation".
    """
    runtime = ScriptedRagRuntime(turns=_retrieve_then_answer("E1"), evidence={})
    result = await _graph(runtime).ainvoke(_request())

    assert result.grounding.validated is True
    assert result.evidence_ids == ()
    assert "[E1]" not in result.content
    assert "E1" not in result.content


async def test_a_zero_evidence_clarification_is_allowed_through():
    clarification = GroundedAnswer(claims=[], raw_text="Which document should I look in?")
    runtime = ScriptedRagRuntime(
        turns=[RagModelTurn(text=clarification.raw_text, answer=clarification)], evidence={}
    )
    result = await _graph(runtime).ainvoke(_request())

    assert result.abstained is False
    assert result.grounding.outcome == "clarification"
    assert result.grounding.validated is True
    assert "Which document" in result.content


async def test_grounding_never_reports_a_shadow_outcome():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    result = await _graph(runtime).ainvoke(_request())

    assert result.grounding.outcome in {"accepted", "accepted_with_findings", "clarification"}
    assert "shadow" not in result.grounding.outcome


def test_no_rollout_branch_or_regeneration_node_remains():
    import pathlib

    source = pathlib.Path("app/ai/workflow/rag_execution.py").read_text(encoding="utf-8")
    assert "rag_grounded_answer_gate_enabled" not in source
    assert '"shadow"' not in source
    assert "regenerate_grounded" not in source
    assert 'add_node("regenerate' not in source


def test_the_required_nodes_and_only_those_are_registered():
    graph = _graph(ScriptedRagRuntime(turns=_retrieve_then_answer("E1")))
    nodes = set(graph._compiled.get_graph().nodes) - {"__start__", "__end__"}

    assert nodes == {
        "rag_model",
        "rag_tools",
        "collect_rag_outputs",
        "validate_grounding",
        "package_rag_result",
    }


# ----------------------------------------------------------------------
# server-owned evidence identity
# ----------------------------------------------------------------------


def test_evidence_ids_come_from_one_per_run_allocator():
    allocator = EvidenceIdAllocator()
    assert [allocator.allocate() for _ in range(3)] == ["E1", "E2", "E3"]

    fresh = EvidenceIdAllocator()
    assert fresh.allocate() == "E1"


def test_allocator_refuses_to_reissue_an_id():
    allocator = EvidenceIdAllocator()
    allocator.allocate()
    with pytest.raises(ValueError):
        allocator.claim("E1")


def test_duplicate_evidence_ids_are_dropped_not_silently_first_wins():
    """An ambiguous id cannot be validated, so neither record survives.

    Keeping the first would make a citation mean whichever record happened to
    merge first — exactly the silent wrong-source failure grounding exists to
    prevent.
    """
    merged, ambiguous = merge_turn_evidence(
        [
            {
                "records": [
                    {
                        "evidence_id": "E1",
                        "document_id": "11111111-1111-1111-1111-111111111111",
                        "filename": "a.pdf",
                        "text": "first",
                    }
                ]
            },
            {
                "records": [
                    {
                        "evidence_id": "E1",
                        "document_id": "22222222-2222-2222-2222-222222222222",
                        "filename": "b.pdf",
                        "text": "second",
                    }
                ]
            },
        ]
    )
    assert ambiguous >= 1
    assert "E1" not in merged.evidence_ids


async def test_ambiguous_evidence_fails_validation_rather_than_picking_one():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"),
        evidence={
            "records": [
                {
                    "evidence_id": "E1",
                    "document_id": "11111111-1111-1111-1111-111111111111",
                    "filename": "a.pdf",
                    "text": "first",
                },
                {
                    "evidence_id": "E1",
                    "document_id": "22222222-2222-2222-2222-222222222222",
                    "filename": "b.pdf",
                    "text": "second",
                },
            ]
        },
    )
    result = await _graph(runtime).ainvoke(_request())
    assert result.grounding.ambiguous_evidence_id_count >= 1


# ----------------------------------------------------------------------
# result shape
# ----------------------------------------------------------------------


async def test_result_carries_server_owned_provenance():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"),
        evidence=_evidence_payload("E1"),
        artifacts=({"tool_call_id": "call-1", "kind": "rag_search"},),
        images=({"image_id": "img-1"},),
    )
    result = await _graph(runtime).ainvoke(_request())

    assert result.evidence_ids == ("E1",)
    assert result.grounding.claim_count == 1
    assert result.grounding.cited_claim_count == 1
    assert result.artifacts == ({"tool_call_id": "call-1", "kind": "rag_search"},)
    assert result.images == ({"image_id": "img-1"},)
    assert [record["evidence_id"] for record in result.evidence] == ["E1"]


async def test_worker_mode_result_is_private():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    result = await _graph(runtime).ainvoke(_request(mode="worker"))

    assert result.mode == "worker"
    assert "public_messages" not in RagExecutionResult.model_fields


# ----------------------------------------------------------------------
# execution budget
# ----------------------------------------------------------------------


def _budget_settings(**overrides) -> SimpleNamespace:
    values = {
        "rag_evidence_max_tokens": 0,
        "enable_citation_verification": True,
        "rag_max_tool_iterations": 8,
        "generation_soft_tool_calls_per_epoch": 1,
        "generation_hard_tool_calls_per_epoch": 4,
        "generation_soft_model_calls_per_epoch": 6,
        "generation_hard_model_calls_per_epoch": 8,
        "generation_total_epochs_per_turn": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _two_retrievals_then_answer() -> list:
    """The model asks twice; the budget only pays for one."""
    return [
        RagModelTurn(tool_calls=(_search_call("call-1"),)),
        RagModelTurn(tool_calls=(_search_call("call-2"),)),
        RagModelTurn(text="partial", answer=_answer("E1", text="partial")),
    ]


async def test_the_rag_loop_refuses_a_tool_call_past_the_soft_budget():
    """R1. RAG bypasses create_agent, so no middleware could do this."""
    runtime = ScriptedRagRuntime(
        turns=_two_retrievals_then_answer(), evidence=_evidence_payload("E1")
    )
    graph = _graph(runtime, settings=_budget_settings())

    await graph.ainvoke(_request())

    assert [call["id"] for call in runtime.tool_calls_executed] == ["call-1"]


async def test_the_refused_rag_call_is_still_answered():
    """An unanswered tool call is a provider error on the next request."""
    runtime = ScriptedRagRuntime(
        turns=_two_retrievals_then_answer(), evidence=_evidence_payload("E1")
    )
    graph = _graph(runtime, settings=_budget_settings())

    result = await graph.ainvoke(_request())

    assert isinstance(result, RagExecutionResult)
    assert result.execution_budget["exhausted_by"] == "tool_calls"


async def test_the_rag_answer_call_is_told_that_gathering_has_ended():
    """Suppression rides the agent's existing rag_force_final_response seam."""
    runtime = ScriptedRagRuntime(
        turns=_two_retrievals_then_answer(), evidence=_evidence_payload("E1")
    )
    graph = _graph(runtime, settings=_budget_settings())

    await graph.ainvoke(_request())

    assert runtime.force_final_flags[-1] is True
    assert runtime.force_final_flags[0] is False


async def test_a_rag_run_within_budget_is_never_forced():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    graph = _graph(runtime, settings=_budget_settings(
            generation_soft_tool_calls_per_epoch=4,
            generation_hard_tool_calls_per_epoch=6,
        ))

    result = await graph.ainvoke(_request())

    assert result.execution_budget["exhausted_by"] is None
    assert result.execution_budget["forced_synthesis"] is False
    assert runtime.force_final_flags == [False, False]


async def test_the_rag_budget_counts_what_the_run_actually_spent():
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    graph = _graph(runtime, settings=_budget_settings(
            generation_soft_tool_calls_per_epoch=4,
            generation_hard_tool_calls_per_epoch=6,
        ))

    result = await graph.ainvoke(_request())

    assert result.execution_budget["tool_calls"] == 1
    assert result.execution_budget["model_calls"] == 2


async def test_a_carried_budget_resumes_rather_than_restarting():
    """R4. A Continue served here must not be handed a fresh quota."""
    runtime = ScriptedRagRuntime(
        turns=_retrieve_then_answer("E1"), evidence=_evidence_payload("E1")
    )
    graph = _graph(runtime, settings=_budget_settings(
            generation_soft_tool_calls_per_epoch=4,
            generation_hard_tool_calls_per_epoch=6,
        ))

    result = await graph.ainvoke(
        _request(
            execution_budget={
                "execution_epoch": 1,
                "turn_tool_calls": 5,
                "turn_model_calls": 4,
                "epochs_used": 2,
            }
        )
    )

    assert result.execution_budget["execution_epoch"] == 1
    assert result.execution_budget["turn_tool_calls"] == 6
    assert result.execution_budget["turn_model_calls"] == 6
