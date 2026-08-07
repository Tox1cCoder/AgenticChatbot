"""Verified bytes stored at registration are served without a second fetch.

Before this, an image could pass every check — public-address assertion, byte
cap, decoded MIME, and the visual verifier — be placed in the answer, and then
die at render time on a fetch nobody was watching.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.services.web_image_service import (
    FetchedWebImage,
    WebImageService,
)


class _ExplodingResolver:
    def __call__(self, hostname):  # pragma: no cover - must never run
        raise AssertionError("a cached image must not touch the network")


class _RecordingRepository:
    def __init__(self):
        self.created: list[dict] = []

    async def acreate(self, payload: dict):
        self.created.append(dict(payload))
        return payload


def _service(repository=None, resolver=None) -> WebImageService:
    return WebImageService(
        repository=repository if repository is not None else _RecordingRepository(),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_redirects=1,
        max_bytes=4096,
        max_pixels=10_000_000,
        resolver=resolver or _ExplodingResolver(),
    )


@pytest.mark.asyncio
async def test_cached_bytes_are_served_without_a_fetch():
    record = {
        "upstream_url": "https://cdn.example/a.jpg",
        "provider": "brave",
        "content": b"cached-bytes",
        "expected_mime": "image/jpeg",
        "cached_width": 995,
        "cached_height": 565,
    }

    fetched = await _service().fetch(record)

    assert fetched == FetchedWebImage(
        content=b"cached-bytes", media_type="image/jpeg", width=995, height=565
    )


@pytest.mark.asyncio
async def test_a_record_object_serves_its_cache_too():
    class _Record:
        upstream_url = "https://cdn.example/a.jpg"
        provider = "brave"
        expected_mime = "image/webp"
        content = b"cached"
        cached_width = 800
        cached_height = 600

    fetched = await _service().fetch(_Record())

    assert fetched.media_type == "image/webp"
    assert fetched.width == 800


@pytest.mark.asyncio
async def test_a_disallowed_cached_mime_falls_back_to_fetching():
    """A cache entry must not become a way past the MIME allowlist."""

    record = {
        "upstream_url": "https://cdn.example/a.svg",
        "provider": "brave",
        "content": b"<svg/>",
        "expected_mime": "image/svg+xml",
        "cached_width": 10,
        "cached_height": 10,
    }

    with pytest.raises(AssertionError, match="must not touch the network"):
        await _service().fetch(record)


@pytest.mark.asyncio
async def test_incomplete_cache_metadata_falls_back_to_fetching():
    record = {
        "upstream_url": "https://cdn.example/a.jpg",
        "provider": "brave",
        "content": b"cached",
        "expected_mime": "image/jpeg",
        "cached_width": 995,
        "cached_height": None,
    }

    with pytest.raises(AssertionError, match="must not touch the network"):
        await _service().fetch(record)


@pytest.mark.asyncio
async def test_register_persists_supplied_bytes():
    repository = _RecordingRepository()

    await _service(repository).register(
        conversation_id=uuid4(),
        user_id=uuid4(),
        upstream_url="https://cdn.example/a.jpg",
        expected_mime="image/jpeg",
        provider="brave",
        cached=FetchedWebImage(
            content=b"verified", media_type="image/jpeg", width=995, height=565
        ),
    )

    stored = repository.created[0]
    assert stored["content"] == b"verified"
    assert stored["cached_width"] == 995
    assert stored["cached_height"] == 565


@pytest.mark.asyncio
async def test_register_without_bytes_stores_none_and_still_works():
    """The fallback path: verification was skipped or its hand-off expired."""

    repository = _RecordingRepository()

    await _service(repository).register(
        conversation_id=uuid4(),
        user_id=uuid4(),
        upstream_url="https://cdn.example/a.jpg",
        expected_mime="image/jpeg",
        provider="brave",
    )

    stored = repository.created[0]
    assert stored["content"] is None
    assert stored["cached_width"] is None


@pytest.mark.asyncio
async def test_verification_hands_its_bytes_to_registration(monkeypatch):
    """End to end: the bytes the verifier validated are the bytes persisted.

    The two halves live in different layers and run at different times, so a
    test that only covers one of them proves nothing about the round trip.
    """
    import json

    from app.ai.image_verification_flow import discover_and_verify_images
    from app.ai.tool_context import clear_tool_context, tool_execution_context
    from app.ai.visual_verifier import VisualCandidateDecision, VisualVerificationResult
    from app.services.verified_image_bytes import (
        forget_conversation_bytes,
        take_verified_bytes,
    )

    conversation_id = "55555555-5555-5555-5555-555555555555"
    url = "https://cdn.example/team.jpg"
    brave_payload = json.dumps(
        {
            "query": "T1 team photo",
            "provider": "brave_image_search",
            "images": [
                {
                    "url": url,
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

    class _Brave:
        async def ainvoke(self, args):
            return brave_payload

    class _Service:
        async def fetch_url(self, target, *, provider="other"):
            return FetchedWebImage(
                content=b"verified-team-bytes",
                media_type="image/jpeg",
                width=995,
                height=565,
            )

    class _Verifier:
        async def ainvoke(self, messages):
            return VisualVerificationResult(
                decisions=[
                    VisualCandidateDecision(
                        candidate_id="c0",
                        depicts_requested_subject=True,
                        materially_supports_answer=True,
                        confidence=0.95,
                        content_kind="photo",
                    )
                ]
            )

    clear_tool_context()
    forget_conversation_bytes(conversation_id)
    monkeypatch.setattr(
        "app.ai.image_verification_flow.settings.image_verification_confidence_threshold",
        0.85,
        raising=False,
    )
    try:
        with tool_execution_context(
            conversation_id=conversation_id, user_id="u1", agent_key="search"
        ):
            approved = await discover_and_verify_images(
                brave_tool=_Brave(),
                web_image_service=_Service(),
                verifier_model=_Verifier(),
                user_request="T1 roster 2026",
                image_query="T1 team photo",
                factual_query="T1 roster 2026",
            )

        assert len(approved) == 1

        held = take_verified_bytes(conversation_id, url)
        assert held is not None, "approved bytes must survive the tool call"
        assert held.content == b"verified-team-bytes"

        repository = _RecordingRepository()
        await _service(repository).register(
            conversation_id=uuid4(),
            user_id=uuid4(),
            upstream_url=url,
            expected_mime="image/jpeg",
            provider="brave",
            cached=held,
        )
        assert repository.created[0]["content"] == b"verified-team-bytes"
    finally:
        clear_tool_context()
        forget_conversation_bytes(conversation_id)
