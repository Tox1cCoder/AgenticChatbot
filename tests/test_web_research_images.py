from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.research_budget import ResearchBudget
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.web_research.contracts import (
    ProviderImageCandidate,
    ProviderSource,
    ResearchRequest,
    ResearchScope,
)
from app.ai.web_research.grounding import GroundingParser
from app.ai.web_research.providers import ProviderResolver
from app.ai.web_research.service import WebResearchService
from app.core.config import settings
from app.core.response_constants import build_bot_metadata
from app.core.rich_response import ImageRichItem, validate_public_rich_item
from app.services.web_image_service import FetchedWebImage, WebImageRejected


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

    async def fetch_url(self, url: str, *, provider: str, max_bytes: int | None = None):
        image = self.fetched[url]
        if max_bytes is not None and len(image.content) > max_bytes:
            raise WebImageRejected("size")
        return image

    async def register(self, **kwargs):
        record = SimpleNamespace(id=uuid4(), **kwargs)
        self.records[record.id] = record
        return record

    async def mark_selected(self, ids, **_scope):
        self.selected.extend(ids)

    async def release_references(self, ids, **_scope):
        self.released.extend(ids)


def _image(content: bytes) -> FetchedWebImage:
    return FetchedWebImage(content=content, media_type="image/png", width=640, height=360)


def _scope() -> ResearchScope:
    return ResearchScope(
        conversation_id=str(uuid4()),
        user_id=str(uuid4()),
        logical_turn_id=str(uuid4()),
    )


async def _session(
    contents: tuple[bytes, ...],
    *,
    max_model_bytes: int = 100_000,
    max_download_bytes: int = 100_000,
):
    urls = tuple(f"https://images.test/{index}.png" for index in range(len(contents)))
    image_service = ImageService(dict(zip(urls, map(_image, contents), strict=True)))
    service = WebResearchService(
        resolver=ProviderResolver(text=(TextProvider(),), images=(ImageProvider(urls),)),
        image_service=image_service,
        max_download_bytes=max_download_bytes,
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
async def test_prepared_web_image_satisfies_public_rich_contract() -> None:
    session, _bundle, _service = await _session((b"one",))
    item = session.prepared_images["I1"].rich_item

    validated = validate_public_rich_item(item)

    assert isinstance(validated, ImageRichItem)
    assert validated.alt_text
    assert str(validated.payload.source_url) == "https://source.test/article"
    assert validated.payload.url.startswith("/web-images/")


@pytest.mark.asyncio
async def test_selected_web_image_survives_grounding_and_metadata_finalization(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    session, _bundle, _service = await _session((b"one",))
    resolution = GroundingParser(session).resolve(
        "Current view [[source:S1]].\n\n[[image:I1]]"
    )
    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=resolution.text),
        metadata={
            "_rich_item_candidates": list(resolution.rich_items),
            "_inline_rich_response_v1": True,
        },
    )

    metadata = build_bot_metadata(response)

    assert "<!--rich:image:web:" in response.message.content
    assert len(metadata["rich_items"]) == 1
    item = metadata["rich_items"][0]
    assert item["payload"]["url"].startswith("/web-images/")
    assert item["payload"]["source_url"] == "https://source.test/article"
    assert item["alt_text"]
    assert metadata["rich_reference_warnings"] == []


@pytest.mark.asyncio
async def test_finish_preserves_authored_order_and_releases_unselected() -> None:
    session, _bundle, service = await _session((b"one", b"two", b"three"))

    result = await session.finish(("I2", "I1", "I2"))

    assert result.selected_candidate_ids == ("I2", "I1")
    assert service.selected == []
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
async def test_aggregate_download_limit_bounds_accepted_bytes() -> None:
    session, bundle, _service = await _session(
        (b"1234",) * 8,
        max_download_bytes=10,
    )

    assert session._downloaded_image_bytes <= 10
    assert sum(image.byte_size for image in bundle.images) <= 10


@pytest.mark.asyncio
async def test_abort_releases_all_pending_references() -> None:
    session, _bundle, service = await _session((b"one", b"two"))
    reference_ids = [prepared.reference_id for prepared in session.prepared_images.values()]

    await session.abort()

    assert service.released == reference_ids
