"""Already-owned bytes stored at registration are served without a fetch."""

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
    """Registration remains metadata-only when the caller owns no bytes."""

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
