"""One shared RAG execution graph, with grounding that is never optional.

Top-level RAG and Planning RAG workers must be the *same* graph: two
implementations were how one path could quietly skip validation. Grounding runs
for every result, including a retrieval that found nothing — an answer with no
evidence may ask for clarification, but it may not claim a source.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.workflow.rag_execution import (
    EvidenceIdAllocator,
    RagExecutionGraphFactory,
    RagExecutionRequest,
    RagExecutionResult,
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
    """Deterministic stand-in for the model/tool half of the RAG graph."""

    def __init__(self, *, answers, evidence=None, regenerations=None):
        self._answers = list(answers)
        self._regenerations = list(regenerations or [])
        self.evidence = evidence or {}
        self.final_answer_calls = 0
        self.regeneration_calls = 0

    async def answer(self, request, evidence):
        self.final_answer_calls += 1
        return self._answers[min(self.final_answer_calls, len(self._answers)) - 1]

    async def regenerate(self, *, reason_codes):
        self.regeneration_calls += 1
        if not self._regenerations:
            return None
        return self._regenerations[
            min(self.regeneration_calls, len(self._regenerations)) - 1
        ]

    async def retrieve(self, request):
        return self.evidence


def _factory(runtime, gate=None, **overrides):
    payload = {
        "runtime": runtime,
        "grounded_answer_gate": gate or _gate(),
        "settings": SimpleNamespace(
            rag_evidence_max_tokens=0,
            enable_citation_verification=True,
        ),
    }
    payload.update(overrides)
    return RagExecutionGraphFactory(**payload)


# ----------------------------------------------------------------------
# one shared factory
# ----------------------------------------------------------------------


async def test_top_level_and_worker_use_the_same_factory():
    runtime = ScriptedRagRuntime(
        answers=[_answer("E1")], evidence=_evidence_payload("E1")
    )
    factory = _factory(runtime)

    public = await factory.build().ainvoke(_request(mode="public"))
    worker = await factory.build().ainvoke(_request(mode="worker"))

    assert isinstance(public, RagExecutionResult)
    assert isinstance(worker, RagExecutionResult)
    assert factory.build_count == 2


async def test_worker_and_public_enforce_the_same_evidence_budget():
    runtime = ScriptedRagRuntime(
        answers=[_answer("E1")], evidence=_evidence_payload("E1")
    )
    factory = _factory(runtime)

    public = await factory.build().ainvoke(_request(mode="public"))
    worker = await factory.build().ainvoke(_request(mode="worker"))

    assert public.grounding.validated is worker.grounding.validated is True
    assert public.evidence_ids == worker.evidence_ids == ("E1",)


# ----------------------------------------------------------------------
# grounding is mandatory
# ----------------------------------------------------------------------


async def test_a_grounded_answer_is_accepted():
    runtime = ScriptedRagRuntime(
        answers=[_answer("E1")], evidence=_evidence_payload("E1")
    )
    result = await _factory(runtime).build().ainvoke(_request())

    assert result.abstained is False
    assert result.grounding.outcome == "accepted"
    assert result.grounding.validated is True
    assert result.grounding.regeneration_count == 0


async def test_invalid_citations_regenerate_once_then_abstain():
    runtime = ScriptedRagRuntime(
        answers=[_answer("E99")],
        regenerations=[_answer("E98")],
        evidence=_evidence_payload("E1"),
    )
    result = await _factory(runtime).build().ainvoke(_request())

    assert result.abstained is True
    assert result.grounding.regeneration_count == 1
    assert runtime.regeneration_calls == 1


async def test_one_regeneration_can_rescue_the_answer():
    runtime = ScriptedRagRuntime(
        answers=[_answer("E99")],
        regenerations=[_answer("E1")],
        evidence=_evidence_payload("E1"),
    )
    result = await _factory(runtime).build().ainvoke(_request())

    assert result.abstained is False
    assert result.grounding.outcome == "regenerated"
    assert result.grounding.regeneration_count == 1


async def test_unknown_evidence_id_never_reaches_the_public_result():
    runtime = ScriptedRagRuntime(
        answers=[_answer("E404")], evidence=_evidence_payload("E1")
    )
    result = await _factory(runtime).build().ainvoke(_request())

    assert "E404" not in result.content
    assert "E404" not in result.evidence_ids


async def test_no_evidence_still_runs_grounding_and_cannot_claim_sources():
    runtime = ScriptedRagRuntime(answers=[_answer("E1")], evidence={})
    result = await _factory(runtime).build().ainvoke(_request())

    assert result.grounding.validated is True
    assert result.grounding.outcome in {"abstained", "clarification"}
    assert result.evidence_ids == ()


async def test_a_zero_evidence_clarification_is_allowed_through():
    clarification = GroundedAnswer(
        claims=[], raw_text="Which document should I look in?"
    )
    runtime = ScriptedRagRuntime(answers=[clarification], evidence={})
    result = await _factory(runtime).build().ainvoke(_request())

    assert result.abstained is False
    assert result.grounding.outcome == "clarification"
    assert result.grounding.validated is True
    assert "Which document" in result.content


async def test_grounding_never_reports_a_shadow_outcome():
    runtime = ScriptedRagRuntime(
        answers=[_answer("E1")], evidence=_evidence_payload("E1")
    )
    result = await _factory(runtime).build().ainvoke(_request())

    assert result.grounding.outcome in {
        "accepted",
        "regenerated",
        "clarification",
        "abstained",
    }
    assert "shadow" not in result.grounding.outcome


def test_no_shadow_only_rollout_branch_remains():
    import pathlib

    source = pathlib.Path("app/ai/workflow/rag_execution.py").read_text(encoding="utf-8")
    assert "rag_grounded_answer_gate_enabled" not in source
    assert '"shadow"' not in source


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
        answers=[_answer("E1")],
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
    result = await _factory(runtime).build().ainvoke(_request())
    assert result.grounding.ambiguous_evidence_id_count >= 1


# ----------------------------------------------------------------------
# result shape
# ----------------------------------------------------------------------


async def test_result_carries_server_owned_provenance():
    runtime = ScriptedRagRuntime(
        answers=[_answer("E1")], evidence=_evidence_payload("E1")
    )
    result = await _factory(runtime).build().ainvoke(_request())

    assert result.evidence_ids == ("E1",)
    assert result.grounding.claim_count == 1
    assert result.grounding.cited_claim_count == 1


async def test_worker_mode_result_is_private():
    runtime = ScriptedRagRuntime(
        answers=[_answer("E1")], evidence=_evidence_payload("E1")
    )
    result = await _factory(runtime).build().ainvoke(_request(mode="worker"))

    assert result.mode == "worker"
    assert "public_messages" not in RagExecutionResult.model_fields
