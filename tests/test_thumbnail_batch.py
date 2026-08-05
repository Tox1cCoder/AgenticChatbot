from __future__ import annotations

import asyncio

import pytest

from app.services.thumbnail_batch import fetch_thumbnails
from app.services.web_image_service import (
    FetchedWebImage,
    WebImageRejected,
    WebImageUpstreamFailure,
)


class _FakeService:
    def __init__(self, behavior: dict[str, object], delay: float = 0.0):
        self.behavior = behavior
        self.delay = delay
        self.calls: list[str] = []
        self.cancelled: list[str] = []

    async def fetch_url(self, url: str, *, provider: str = "other") -> FetchedWebImage:
        self.calls.append(url)
        if self.delay:
            try:
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                self.cancelled.append(url)
                raise
        outcome = self.behavior[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _image(width: int = 800) -> FetchedWebImage:
    return FetchedWebImage(content=b"bytes", media_type="image/jpeg", width=width, height=600)


@pytest.mark.asyncio
async def test_one_failure_does_not_discard_the_batch():
    urls = ["https://a.example/1.jpg", "https://b.example/2.jpg", "https://c.example/3.jpg"]
    service = _FakeService(
        {
            urls[0]: _image(800),
            urls[1]: WebImageRejected("private_address"),
            urls[2]: _image(900),
        }
    )

    fetched = await fetch_thumbnails(
        service, urls, provider="brave", per_item_timeout=1.0, batch_deadline=2.0
    )

    assert [item is None for item in fetched] == [False, True, False]
    assert fetched[0].url == urls[0]
    assert fetched[2].image.width == 900


@pytest.mark.asyncio
async def test_upstream_failure_is_isolated_too():
    urls = ["https://a.example/1.jpg"]
    service = _FakeService({urls[0]: WebImageUpstreamFailure("timeout")})

    assert await fetch_thumbnails(
        service, urls, provider="brave", per_item_timeout=1.0, batch_deadline=1.0
    ) == [None]


@pytest.mark.asyncio
async def test_downloads_run_concurrently_within_the_batch_deadline():
    urls = [f"https://a.example/{index}.jpg" for index in range(6)]
    service = _FakeService({url: _image() for url in urls}, delay=0.2)

    started = asyncio.get_running_loop().time()
    fetched = await fetch_thumbnails(
        service, urls, provider="brave", per_item_timeout=1.0, batch_deadline=2.0
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert all(item is not None for item in fetched)
    assert elapsed < 0.6, "six 0.2s fetches must overlap, not serialize"


@pytest.mark.asyncio
async def test_batch_deadline_fires_when_it_is_the_tighter_bound():
    """The outer deadline must bound the batch even when per-item timeouts cannot."""
    urls = ["https://a.example/1.jpg", "https://b.example/2.jpg"]
    service = _FakeService({url: _image() for url in urls}, delay=2.0)

    started = asyncio.get_running_loop().time()
    fetched = await fetch_thumbnails(
        service, urls, provider="brave", per_item_timeout=10.0, batch_deadline=0.2
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert fetched == [None, None]
    assert elapsed < 1.0, "the batch deadline, not the per-item timeout, must bound this"

    await asyncio.sleep(0)  # let the already-scheduled cancellation land in the fakes
    assert sorted(service.cancelled) == sorted(urls), "abandoned fetches must be cancelled"


@pytest.mark.asyncio
async def test_empty_url_list_makes_no_calls():
    service = _FakeService({})

    assert await fetch_thumbnails(
        service, [], provider="brave", per_item_timeout=1.0, batch_deadline=1.0
    ) == []
    assert service.calls == []
