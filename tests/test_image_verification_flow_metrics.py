"""Flow-level proof that each branch of ``discover_and_verify_images`` records
exactly one terminal outcome.

The unit tests in ``test_visual_verification_metrics.py`` exercise
``RichImageMetrics`` directly, and prove the labels are bounded. They cannot
catch a future refactor that reorders an early return or silently drops one
of the ``_outcome(...)`` calls from the flow itself -- which would defeat the
whole point of this instrumentation, since the flow's job is to make a silent
collapse to text-only visible. These tests patch ``rich_image_metrics`` in
``app.ai.image_verification_flow`` and assert the exact stage/outcome calls
recorded for each of the six branches, including that exactly one outcome is
recorded per call (a double-record would pass a test that only checked which
label appeared).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock, call

import pytest

from app.ai import image_verification_flow
from app.ai.image_verification_flow import discover_and_verify_images
from app.ai.visual_verifier import VisualCandidateDecision, VisualVerificationResult
from app.services.web_image_service import FetchedWebImage

_TEAM_PAYLOAD = json.dumps(
    {
        "query": "T1 team photo",
        "provider": "brave_image_search",
        "images": [
            {
                "url": "https://cdn.example/team.jpg",
                "provider": "brave_image_search",
                "mime_type": "image/jpeg",
                "title": "T1 roster",
                "description": "T1 roster",
                "width": 995,
                "height": 565,
                "source_url": "https://sheepesports.example/t1",
            }
        ],
        "total_results": 1,
    }
)
_EMPTY_PAYLOAD = json.dumps(
    {"query": "nothing", "provider": "brave_image_search", "images": [], "total_results": 0}
)


class _BraveTool:
    def __init__(self, payload: str):
        self.payload = payload

    async def ainvoke(self, args: dict) -> str:
        return self.payload


class _WorkingImageService:
    async def fetch_url(self, url: str, *, provider: str = "other") -> FetchedWebImage:
        return FetchedWebImage(content=b"bytes", media_type="image/jpeg", width=995, height=565)


class _FailingImageService:
    async def fetch_url(self, url: str, *, provider: str = "other") -> FetchedWebImage:
        raise RuntimeError("network down")


class _RaisingBraveTool:
    async def ainvoke(self, args: dict) -> str:
        raise RuntimeError("brave down")


class _RaisingVerifier:
    async def ainvoke(self, messages):
        raise RuntimeError("model unavailable")


class _SlowVerifier:
    async def ainvoke(self, messages):
        await asyncio.sleep(1.0)
        return VisualVerificationResult(decisions=[])


class _MalformedVerifier:
    async def ainvoke(self, messages):
        return {"decisions": [{"candidate_id": "c0"}]}


class _RejectAllVerifier:
    async def ainvoke(self, messages):
        return VisualVerificationResult(decisions=[])


class _ApproveVerifier:
    async def ainvoke(self, messages):
        return VisualVerificationResult(
            decisions=[
                VisualCandidateDecision(
                    candidate_id="c0",
                    depicts_requested_subject=True,
                    materially_supports_answer=True,
                    confidence=0.9,
                    content_kind="photo",
                )
            ]
        )


@pytest.fixture
def metrics(monkeypatch):
    """A ``rich_image_metrics`` double that records every call it receives."""
    fake = type(
        "Metrics",
        (),
        {"record_verification": Mock(), "record_verification_outcome": Mock()},
    )()
    monkeypatch.setattr(image_verification_flow, "rich_image_metrics", fake)
    return fake


def _outcome_labels(fake) -> list[str]:
    return [kwargs["outcome"] for _, kwargs in fake.record_verification_outcome.call_args_list]


async def _discover(**overrides):
    kwargs = {
        "brave_tool": _BraveTool(_TEAM_PAYLOAD),
        "web_image_service": _WorkingImageService(),
        "verifier_model": _ApproveVerifier(),
        "user_request": "T1 roster",
        "image_query": "T1 team photo",
        "factual_query": "T1 roster",
    }
    kwargs.update(overrides)
    return await discover_and_verify_images(**kwargs)


@pytest.mark.asyncio
async def test_missing_dependency_records_unavailable_once(metrics):
    result = await _discover(brave_tool=None, verifier_model=None)

    assert result == []
    metrics.record_verification.assert_not_called()
    metrics.record_verification_outcome.assert_called_once()
    assert _outcome_labels(metrics) == ["unavailable"]


@pytest.mark.asyncio
async def test_empty_discovery_records_no_match_once(metrics):
    result = await _discover(brave_tool=_BraveTool(_EMPTY_PAYLOAD))

    assert result == []
    metrics.record_verification.assert_called_once_with(stage="discovered", count=0)
    metrics.record_verification_outcome.assert_called_once()
    assert _outcome_labels(metrics) == ["no_match"]


@pytest.mark.asyncio
async def test_a_failing_image_search_records_search_failure_once(metrics):
    result = await _discover(brave_tool=_RaisingBraveTool())

    assert result == []
    metrics.record_verification.assert_not_called()
    metrics.record_verification_outcome.assert_called_once()
    assert _outcome_labels(metrics) == ["search_failure"]


@pytest.mark.asyncio
async def test_all_thumbnails_failing_records_fetch_failure_once(metrics):
    result = await _discover(web_image_service=_FailingImageService())

    assert result == []
    assert metrics.record_verification.call_args_list == [
        call(stage="discovered", count=1),
        call(stage="fetched", count=0),
        call(stage="submitted", count=0),
    ]
    metrics.record_verification_outcome.assert_called_once()
    assert _outcome_labels(metrics) == ["fetch_failure"]


@pytest.mark.asyncio
async def test_a_verifier_timeout_records_verifier_timeout_once(metrics, monkeypatch):
    monkeypatch.setattr(
        image_verification_flow.settings,
        "image_verification_timeout_seconds",
        0.01,
        raising=False,
    )

    result = await _discover(verifier_model=_SlowVerifier())

    assert result == []
    metrics.record_verification_outcome.assert_called_once()
    assert _outcome_labels(metrics) == ["verifier_timeout"]


@pytest.mark.asyncio
async def test_a_raising_verifier_records_verifier_failure_once(metrics):
    result = await _discover(verifier_model=_RaisingVerifier())

    assert result == []
    assert metrics.record_verification.call_args_list == [
        call(stage="discovered", count=1),
        call(stage="fetched", count=1),
        call(stage="submitted", count=1),
    ]
    metrics.record_verification_outcome.assert_called_once()
    assert _outcome_labels(metrics) == ["verifier_failure"]


@pytest.mark.asyncio
async def test_unparsable_verifier_output_records_malformed_once(metrics):
    result = await _discover(verifier_model=_MalformedVerifier())

    assert result == []
    metrics.record_verification_outcome.assert_called_once()
    assert _outcome_labels(metrics) == ["malformed"]


@pytest.mark.asyncio
async def test_zero_approvals_records_no_match_once(metrics):
    result = await _discover(verifier_model=_RejectAllVerifier())

    assert result == []
    assert metrics.record_verification.call_args_list == [
        call(stage="discovered", count=1),
        call(stage="fetched", count=1),
        call(stage="submitted", count=1),
        call(stage="approved", count=0),
    ]
    metrics.record_verification_outcome.assert_called_once()
    assert _outcome_labels(metrics) == ["no_match"]


@pytest.mark.asyncio
async def test_approved_candidate_records_approved_once(metrics):
    result = await _discover()

    assert len(result) == 1
    assert metrics.record_verification.call_args_list == [
        call(stage="discovered", count=1),
        call(stage="fetched", count=1),
        call(stage="submitted", count=1),
        call(stage="approved", count=1),
    ]
    metrics.record_verification_outcome.assert_called_once()
    assert _outcome_labels(metrics) == ["approved"]
