from __future__ import annotations

import asyncio
import json
import logging
import warnings
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

import pytest
from prometheus_client import CollectorRegistry

from app.ai.visual_verifier import (
    SubmittedCandidate,
    VisualCandidateDecision,
    VisualVerificationResult,
    admit_candidates,
    build_verifier_model,
    verify_candidates,
)
from app.observability.model_usage import ModelUsageMetrics
from app.repositories.model_usage import RecordEventCommand, RecordResult
from app.services.thumbnail_batch import FetchedThumbnail
from app.services.web_image_service import FetchedWebImage
from app.usage.recorder import ModelUsageRecorder


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


def test_rejects_an_image_that_does_not_depict_the_requested_subject():
    submitted = [_submitted("c1")]

    assert _admit([_decision("c1", depicts_requested_subject=False)], submitted) == []


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
async def test_a_batch_level_parse_failure_rejects_the_whole_response():
    class _MalformedModel:
        async def ainvoke(self, _messages):
            # Missing every required field but candidate_id: the strict
            # structured-output model can't coerce this into a decision, and
            # that failure is not isolated to one record — it sinks the batch.
            return {"decisions": [{"candidate_id": "c1"}]}

    result = await verify_candidates(
        [_submitted("c1")],
        user_request="q",
        image_query="i",
        factual_query="f",
        result_titles=[],
        model=_MalformedModel(),
        timeout=1.0,
    )

    assert result is None
    assert (
        admit_candidates(
            result,
            [_submitted("c1")],
            threshold=0.85,
            max_items=2,
            requested_kinds=frozenset(),
        )
        == []
    )


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


class _FakeUsageRepo:
    def __init__(self) -> None:
        self.commands: list[RecordEventCommand] = []

    def record_event(self, command: RecordEventCommand) -> RecordResult:
        self.commands.append(command)
        return RecordResult(inserted=True, event_id=uuid4())


def _test_recorder(repo: _FakeUsageRepo) -> ModelUsageRecorder:
    return ModelUsageRecorder(
        repository=repo,
        enqueue_failed_write=lambda payload: None,
        metrics=ModelUsageMetrics(registry=CollectorRegistry()),
    )


@pytest.mark.asyncio
async def test_verify_candidates_records_one_image_verification_attempt():
    """A recorder must see exactly one attempt, with real tokens, no verdict content.

    This is the only test standing between the verifier's real, billed Gemini
    call and an unnoticed silent-cost regression: if the recording call were
    ever deleted from ``verify_candidates``, ``repo.commands`` would stay
    empty and this test would fail. The fake model returns the
    ``{"raw": ..., "parsed": ..., "parsing_error": ...}`` shape that
    ``with_structured_output(..., include_raw=True)`` actually produces, with
    a real ``usage_metadata`` envelope on ``raw`` -- so the test also fails if
    usage capture ever regresses back to a permanent ``source="unavailable"``
    with zero tokens, which is the whole reason this call is worth recording.
    """

    # A UUID is all hex digits and hyphens, so a candidate id containing a
    # letter outside a-f (like "z") can never collide with the recorded
    # operation_id/attempt fields the way a hex-like id such as "c1" could.
    candidate_id = "candidate-zebra"
    parsed_result = VisualVerificationResult(
        decisions=[_decision(candidate_id, confidence=0.97, content_kind="portrait")]
    )

    class _StructuredOutputModel:
        async def ainvoke(self, _messages):
            return {
                "raw": SimpleNamespace(
                    usage_metadata={"input_tokens": 812, "output_tokens": 47, "total_tokens": 859}
                ),
                "parsed": parsed_result,
                "parsing_error": None,
            }

    repo = _FakeUsageRepo()

    result = await verify_candidates(
        [_submitted(candidate_id)],
        user_request="q",
        image_query="i",
        factual_query="f",
        result_titles=[],
        model=_StructuredOutputModel(),
        timeout=1.0,
        recorder=_test_recorder(repo),
    )

    assert result == parsed_result
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.status == "success"
    assert command.provider == "gemini"
    assert command.context.operation == "image_verification"
    assert command.usage.source == "provider_reported"
    assert command.usage.input_tokens == 812
    assert command.usage.output_tokens == 47
    assert command.usage.total_tokens == 859

    serialized = json.dumps(asdict(command), default=str)
    assert candidate_id not in serialized
    assert "0.97" not in serialized
    assert "portrait" not in serialized
    assert "confidence" not in serialized
    assert "content_kind" not in serialized


@pytest.mark.asyncio
async def test_verify_candidates_tolerates_a_bare_parsed_response_when_recording():
    """Every pre-existing test injects a bare parsed result, not the include_raw dict.

    Recording must not assume the wrapped shape: a bare response still records
    one attempt (with ``source="unavailable"``, since it carries no usage
    envelope at all) rather than raising and losing the verification.
    """

    class _BareModel:
        async def ainvoke(self, _messages):
            return VisualVerificationResult(decisions=[_decision("c1")])

    repo = _FakeUsageRepo()

    result = await verify_candidates(
        [_submitted("c1")],
        user_request="q",
        image_query="i",
        factual_query="f",
        result_titles=[],
        model=_BareModel(),
        timeout=1.0,
        recorder=_test_recorder(repo),
    )

    assert result is not None
    assert len(repo.commands) == 1
    assert repo.commands[0].usage.source == "unavailable"


def _settings_with(**overrides):
    from app.core.config import Settings

    return Settings(
        _env_file=None,
        secret_key="test-secret",
        environment="development",
        **overrides,
    )


def test_shipped_defaults_leave_headroom_inside_the_image_deadline():
    """Guard the shipped code defaults, not whatever a live `.env` happens to set.

    A test that reads the live ``settings`` singleton only proves that
    whatever is loaded right now satisfies the arithmetic — it would pass in
    a clean CI environment while a real deployment silently violated the
    deadline. `Field.default` is env-independent: it is what ships.
    """
    from app.core.config import Settings

    fields = Settings.model_fields
    brave_default = float(fields["brave_image_search_timeout_seconds"].default)
    thumbnail_default = float(fields["image_verification_thumbnail_timeout_seconds"].default)
    deadline_default = float(fields["image_verification_deadline_seconds"].default)

    assert brave_default + thumbnail_default < deadline_default, (
        "shipped image search plus thumbnail timeout defaults must leave verifier headroom"
    )


def test_warns_when_the_image_verification_budget_has_no_headroom(caplog):
    with caplog.at_level(logging.WARNING, logger="app.core.config"):
        _settings_with(
            brave_image_search_timeout_seconds=3.0,
            image_verification_thumbnail_timeout_seconds=1.5,
            image_verification_deadline_seconds=4.0,
        )

    assert "headroom" in caplog.text


def test_does_not_warn_when_the_image_verification_budget_has_headroom(caplog):
    with caplog.at_level(logging.WARNING, logger="app.core.config"):
        _settings_with(
            brave_image_search_timeout_seconds=2.0,
            image_verification_thumbnail_timeout_seconds=1.5,
            image_verification_deadline_seconds=4.0,
        )

    assert "headroom" not in caplog.text


def test_build_verifier_model_maps_media_resolution_to_a_canonical_value(monkeypatch):
    from google.genai.types import MediaResolution

    from app.ai import visual_verifier

    monkeypatch.setattr(visual_verifier.settings, "gemini_api_key", "dummy-test-key")
    monkeypatch.setattr(
        visual_verifier.settings, "image_verification_model", "gemini-3-flash-preview"
    )
    monkeypatch.setattr(visual_verifier.settings, "image_verification_media_resolution", "low")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = build_verifier_model()

    assert not any(issubclass(item.category, UserWarning) for item in caught)
    assert model is not None
    # include_raw=True (added for usage capture) makes `.first` a RunnableParallel
    # with a "raw" step wrapping the bound chat model, not the chat model itself.
    resolved = model.first.steps__["raw"].media_resolution
    assert resolved in set(MediaResolution), (
        "media_resolution must be a canonical enum member, not a synthetic one "
        "the SDK invents for an unrecognized string"
    )
