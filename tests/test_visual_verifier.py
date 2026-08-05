from __future__ import annotations

import asyncio

import pytest

from app.ai.visual_verifier import (
    SubmittedCandidate,
    VisualCandidateDecision,
    VisualVerificationResult,
    admit_candidates,
    verify_candidates,
)
from app.services.thumbnail_batch import FetchedThumbnail
from app.services.web_image_service import FetchedWebImage


def _submitted(candidate_id: str, title: str = "t") -> SubmittedCandidate:
    return SubmittedCandidate(
        candidate_id=candidate_id,
        thumbnail=FetchedThumbnail(
            url=f"https://cdn.example/{candidate_id}.jpg",
            image=FetchedWebImage(
                content=b"bytes", media_type="image/jpeg", width=995, height=565
            ),
        ),
        title=title,
        description="d",
    )


def _decision(candidate_id: str, **overrides) -> VisualCandidateDecision:
    payload = {
        "candidate_id": candidate_id,
        "depicts_requested_subject": True,
        "materially_supports_answer": True,
        "confidence": 0.95,
        "content_kind": "photo",
    }
    payload.update(overrides)
    return VisualCandidateDecision(**payload)


def _admit(decisions, submitted, **overrides):
    kwargs = {"threshold": 0.85, "max_items": 2, "requested_kinds": frozenset()}
    kwargs.update(overrides)
    return admit_candidates(VisualVerificationResult(decisions=decisions), submitted, **kwargs)


def test_admits_a_confident_relevant_photo():
    submitted = [_submitted("c1")]

    assert _admit([_decision("c1")], submitted) == submitted


def test_rejects_low_confidence_even_at_provider_rank_one():
    submitted = [_submitted("c1"), _submitted("c2")]
    decisions = [_decision("c1", confidence=0.5), _decision("c2", confidence=0.9)]

    assert [item.candidate_id for item in _admit(decisions, submitted)] == ["c2"]


def test_rejects_a_relevant_image_that_does_not_support_the_answer():
    submitted = [_submitted("c1")]

    assert _admit([_decision("c1", materially_supports_answer=False)], submitted) == []


def test_rejects_a_portrait_unless_the_user_asked_for_one():
    submitted = [_submitted("c1")]
    decisions = [_decision("c1", content_kind="portrait")]

    assert _admit(decisions, submitted) == []
    assert len(_admit(decisions, submitted, requested_kinds=frozenset({"portrait"}))) == 1


def test_an_all_uncertain_batch_admits_nothing():
    submitted = [_submitted("c1"), _submitted("c2")]
    decisions = [_decision("c1", confidence=0.4), _decision("c2", confidence=0.6)]

    assert _admit(decisions, submitted) == []


def test_hallucinated_and_duplicate_ids_fail_closed():
    submitted = [_submitted("c1")]

    assert _admit([_decision("ghost")], submitted) == []
    assert _admit([_decision("c1"), _decision("c1")], submitted) == []


def test_out_of_range_confidence_rejects_only_that_candidate():
    submitted = [_submitted("c1"), _submitted("c2")]
    decisions = [_decision("c1", confidence=1.9), _decision("c2")]

    assert [item.candidate_id for item in _admit(decisions, submitted)] == ["c2"]


def test_a_response_level_failure_rejects_the_whole_batch():
    submitted = [_submitted("c1"), _submitted("c2")]

    assert admit_candidates(
        None, submitted, threshold=0.85, max_items=2, requested_kinds=frozenset()
    ) == []


def test_provider_order_is_preserved_and_capped():
    submitted = [_submitted("c1"), _submitted("c2"), _submitted("c3")]
    decisions = [_decision("c3"), _decision("c1"), _decision("c2")]

    assert [item.candidate_id for item in _admit(decisions, submitted)] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_verifier_timeout_returns_none():
    class _SlowModel:
        async def ainvoke(self, _messages):
            await asyncio.sleep(1.0)
            return VisualVerificationResult(decisions=[])

    result = await verify_candidates(
        [_submitted("c1")],
        user_request="cho t thong tin ve t1",
        image_query="T1 League of Legends team photo",
        factual_query="T1 roster 2026",
        result_titles=["LoL: T1 completed 2026 LCK roster"],
        model=_SlowModel(),
        timeout=0.05,
    )

    assert result is None


@pytest.mark.asyncio
async def test_verifier_provider_error_returns_none():
    class _BrokenModel:
        async def ainvoke(self, _messages):
            raise RuntimeError("provider refused")

    result = await verify_candidates(
        [_submitted("c1")],
        user_request="q",
        image_query="i",
        factual_query="f",
        result_titles=[],
        model=_BrokenModel(),
        timeout=1.0,
    )

    assert result is None


@pytest.mark.asyncio
async def test_verifier_sends_one_message_with_every_thumbnail():
    captured: list = []

    class _Recorder:
        async def ainvoke(self, messages):
            captured.append(messages)
            return VisualVerificationResult(decisions=[_decision("c1")])

    await verify_candidates(
        [_submitted("c1"), _submitted("c2")],
        user_request="q",
        image_query="i",
        factual_query="f",
        result_titles=["title"],
        model=_Recorder(),
        timeout=1.0,
    )

    assert len(captured) == 1
    blocks = captured[0][0].content
    assert sum(1 for block in blocks if block.get("type") == "image_url") == 2


def test_configured_timeouts_fit_inside_the_image_deadline():
    from app.core.config import settings

    brave_timeout = float(settings.brave_image_search_timeout_seconds)
    thumbnail_timeout = float(settings.image_verification_thumbnail_timeout_seconds)
    deadline = float(settings.image_verification_deadline_seconds)

    assert brave_timeout + thumbnail_timeout < deadline, (
        "image search plus one thumbnail must leave room for the verifier call"
    )
