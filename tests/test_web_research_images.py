from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.research_budget import ResearchBudget
from app.ai.web_research.contracts import (
    ProviderImageCandidate,
    ProviderSource,
    ResearchRequest,
    ResearchScope,
)
from app.ai.web_research.providers import ProviderResolver
from app.ai.web_research.service import WebResearchService
from app.services.web_image_service import FetchedWebImage


class TextProvider:
    name = "text"
    health_key = "text:key"

    async def search(self, _request, *, query_index: int):
        return (
            ProviderSource(
                provider="text",
                url="https://source.test/article",
                rank=1,
                query_index=query_index,
            ),
        )


class ImageProvider:
    name = "images"
    health_key = "images:key"

    def __init__(self, urls: tuple[str, ...]) -> None:
        self.urls = urls

    async def search(self, _request):
        return tuple(
            ProviderImageCandidate(
                provider="images",
                image_url=url,
                source_url="https://source.test/article",
                rank=index,
            )
            for index, url in enumerate(self.urls, start=1)
        )


class ImageService:
    max_bytes = 10_000

    def __init__(self, fetched: dict[str, FetchedWebImage]) -> None:
        self.fetched = fetched
        self.records: dict[object, SimpleNamespace] = {}
        self.selected: list[object] = []
        self.released: list[object] = []
        self.suspended: list[object] = []

    async def fetch_url(self, url: str, *, provider: str):
        return self.fetched[url]

    async def register(self, **kwargs):
        record = SimpleNamespace(id=uuid4(), **kwargs)
        self.records[record.id] = record
        return record

    async def mark_selected(self, ids, **_scope):
        self.selected.extend(ids)

    async def release_references(self, ids, **_scope):
        self.released.extend(ids)

    async def suspend_references(self, ids, **_scope):
        self.suspended.extend(ids)


def _image(content: bytes) -> FetchedWebImage:
    return FetchedWebImage(content=content, media_type="image/png", width=640, height=360)


def _scope() -> ResearchScope:
    return ResearchScope(
        conversation_id=str(uuid4()),
        user_id=str(uuid4()),
        logical_turn_id=str(uuid4()),
    )


async def _session(contents: tuple[bytes, ...], *, max_model_bytes: int = 100_000):
    urls = tuple(f"https://images.test/{index}.png" for index in range(len(contents)))
    image_service = ImageService(dict(zip(urls, map(_image, contents), strict=True)))
    service = WebResearchService(
        resolver=ProviderResolver(text=(TextProvider(),), images=(ImageProvider(urls),)),
        image_service=image_service,
        max_download_bytes=100_000,
        max_model_bytes=max_model_bytes,
        now=lambda: datetime(2026, 9, 15, tzinfo=timezone.utc),
    )
    session = service.new_session(_scope(), ResearchBudget(), mode="quick")
    bundle = await session.search(
        ResearchRequest(
            query="release image",
            objective="find release image",
            mode="quick",
            visual_intent="comparison",
            image_query="release interface",
        )
    )
    return session, bundle, image_service


@pytest.mark.asyncio
async def test_validated_bytes_become_candidates_and_duplicate_digests_collapse() -> None:
    session, bundle, _service = await _session((b"same", b"same", b"different"))

    assert [image.candidate_id for image in bundle.images] == ["I1", "I2"]
    assert session.model_evidence_blocks(supports_vision=True)[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.asyncio
async def test_finish_preserves_authored_order_and_releases_unselected() -> None:
    session, _bundle, service = await _session((b"one", b"two", b"three"))

    result = await session.finish(("I2", "I1", "I2"))

    assert result.selected_candidate_ids == ("I2", "I1")
    assert service.selected == [
        session.prepared_images["I2"].reference_id,
        session.prepared_images["I1"].reference_id,
    ]
    assert service.released == [session.prepared_images["I3"].reference_id]


@pytest.mark.asyncio
async def test_text_only_model_gets_no_candidate_ids_or_pixels() -> None:
    session, _bundle, _service = await _session((b"one", b"two"))

    assert session.model_evidence_blocks(supports_vision=False) == []
    assert "answer_model_not_vision_capable" in session.reason_codes


@pytest.mark.asyncio
async def test_aggregate_model_byte_limit_rejects_later_candidates() -> None:
    _session_value, bundle, _service = await _session((b"1234", b"5678"), max_model_bytes=5)

    assert [image.candidate_id for image in bundle.images] == ["I1"]
    assert bundle.omitted_image_count == 1


@pytest.mark.asyncio
async def test_interrupt_suspends_but_later_abort_releases() -> None:
    session, _bundle, service = await _session((b"one", b"two"))
    reference_ids = [prepared.reference_id for prepared in session.prepared_images.values()]

    await session.suspend()
    await session.abort()

    assert service.suspended == reference_ids
    assert service.released == reference_ids
