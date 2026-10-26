"""Media lifetime of composed source figures.

Creating a figure, placing its marker, and persisting the message are
separate states. A figure pins the media it was composed from so a later
search cannot evict them; finishing the turn keeps exactly the media of
figures the answer placed, and releases everything else once.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.research_budget import ResearchBudget
from app.ai.source_figure_tools import (
    COMPOSE_SOURCE_FIGURE_TOOL_NAME,
    create_compose_source_figure_tool,
)
from app.ai.tool_context import tool_execution_context
from app.ai.tool_execution import execute_tool_calls
from app.ai.web_research.contracts import (
    ProviderImageCandidate,
    ProviderSource,
    ResearchRequest,
    ResearchScope,
)
from app.ai.web_research.providers import ProviderResolver
from app.ai.web_research.service import WebResearchService
from app.core.source_figures import SourceFigureError
from app.services.web_image_service import FetchedWebImage


class _TextProvider:
    name = "text"
    health_key = "text:key"

    async def search(self, _request, *, query_index: int):
        return (
            ProviderSource(
                provider="text", url="https://source.test/article", rank=1, query_index=query_index
            ),
        )


class _BatchedImageProvider:
    """Each search returns the next batch, so a later search can outrank earlier ones."""

    name = "images"
    health_key = "images:key"

    def __init__(self, batches: list[list[tuple[str, int]]]) -> None:
        self.batches = batches

    async def search(self, _request):
        batch = self.batches.pop(0)
        return tuple(
            ProviderImageCandidate(
                provider="images",
                image_url=url,
                source_url="https://source.test/article",
                width=width,
                height=width // 2,
                rank=index,
            )
            for index, (url, width) in enumerate(batch, start=1)
        )


class _ImageService:
    max_bytes = 10_000

    def __init__(self) -> None:
        self.released: list[object] = []
        self.selected: list[object] = []

    async def fetch_url(self, url: str, *, provider: str, max_bytes: int | None = None):
        return FetchedWebImage(
            content=url.encode(), media_type="image/png", width=640, height=360
        )

    async def register(self, **kwargs):
        return SimpleNamespace(id=uuid4(), **kwargs)

    async def mark_selected(self, ids, **_scope):
        self.selected.extend(ids)

    async def release_references(self, ids, **_scope):
        self.released.extend(ids)


def _urls(prefix: str, count: int, width: int) -> list[tuple[str, int]]:
    return [(f"https://images.test/{prefix}{n}.png", width) for n in range(count)]


async def _search(session, query: str = "graphics settings") -> None:
    await session.search(
        ResearchRequest(
            query=query,
            objective="compare shadow settings",
            mode="quick",
            visual_intent="comparison",
            image_query="shadow quality comparison",
        )
    )


async def _session(batches: list[list[tuple[str, int]]]):
    image_service = _ImageService()
    service = WebResearchService(
        resolver=ProviderResolver(
            text=(_TextProvider(),), images=(_BatchedImageProvider(batches),)
        ),
        image_service=image_service,
        now=lambda: datetime(2026, 10, 26, tzinfo=UTC),
    )
    session = service.new_session(
        ResearchScope(
            conversation_id=str(uuid4()), user_id=str(uuid4()), logical_turn_id="turn-1"
        ),
        ResearchBudget(),
        mode="quick",
    )
    await _search(session)
    return session, image_service


def _ref(session, candidate_id: str):
    return session.known_image(candidate_id).reference_id


async def _compose(session, cells: list[dict], **extra) -> dict:
    with tool_execution_context(
        conversation_id=session.scope.conversation_id,
        logical_turn_id="turn-1",
        web_research_session=session,
    ):
        _outputs, artifacts, _images = await execute_tool_calls(
            tool_calls=[
                {
                    "id": f"call-{uuid4()}",
                    "name": COMPOSE_SOURCE_FIGURE_TOOL_NAME,
                    "args": {"cells": cells, **extra},
                }
            ],
            tool_map={COMPOSE_SOURCE_FIGURE_TOOL_NAME: create_compose_source_figure_tool()},
        )
    return {"output": json.loads(_outputs[0]["content"]), "artifact": artifacts[0]}


def _cells(*ids: str) -> list[dict]:
    return [
        {"image_id": image_id, "label": f"View {image_id}", "alt_text": f"Screenshot {image_id}"}
        for image_id in ids
    ]


def _figure(result: dict) -> dict:
    [candidate] = result["artifact"]["_rich_item_candidates"]
    return candidate


# ---------------------------------------------------------------------------
# Inspection before composition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_composition_requires_images_shown_to_a_vision_model():
    session, _service = await _session([_urls("a", 2, 640)])

    result = await _compose(session, _cells("I1", "I2"))

    assert result["output"]["status"] == "error"
    assert "inspect" in result["output"]["message"]
    assert "_rich_item_candidates" not in result["artifact"]


@pytest.mark.asyncio
async def test_composition_after_falling_back_to_a_non_vision_model_is_refused():
    session, _service = await _session([_urls("a", 2, 640)])
    session.model_evidence_blocks(supports_vision=True)
    session.model_evidence_blocks(supports_vision=False)

    result = await _compose(session, _cells("I1", "I2"))

    assert result["output"]["status"] == "error"
    assert "cannot see images" in result["output"]["message"]


@pytest.mark.asyncio
async def test_unknown_candidate_is_refused():
    session, _service = await _session([_urls("a", 2, 640)])
    session.model_evidence_blocks(supports_vision=True)

    result = await _compose(session, _cells("I1", "I9"))

    assert result["output"]["status"] == "error"
    assert "I9" in result["output"]["message"]


@pytest.mark.asyncio
async def test_composed_figure_cells_carry_their_own_sources_and_protected_media():
    session, _service = await _session([_urls("a", 2, 640)])
    session.model_evidence_blocks(supports_vision=True)

    result = await _compose(session, _cells("I1", "I2"), caption="Shadows on and off.")

    assert result["output"]["status"] == "ok"
    figure = _figure(result)
    assert result["output"]["marker"] == f"<!--rich:{figure['id']}-->"
    for cell in figure["payload"]["items"]:
        assert cell["url"].startswith("/web-images/")
        assert cell["source_url"] == "https://source.test/article"
        assert "images.test" not in json.dumps(figure)
    assert figure["_internal"]["candidate_ids"] == ["I1", "I2"]
    assert "fallback_markdown" not in json.dumps(result["output"])


# ---------------------------------------------------------------------------
# Retention and release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_later_search_cannot_release_a_figure_dependency():
    session, service = await _session([_urls("small", 2, 320), _urls("big", 4, 1280)])
    session.model_evidence_blocks(supports_vision=True)
    figure = _figure(await _compose(session, _cells("I1", "I2")))
    pinned = {_ref(session, "I1"), _ref(session, "I2")}

    await _search(session, "shadow quality screenshots")

    assert "I1" not in session.prepared_images  # evicted from the vision window
    assert not pinned & set(service.released)

    await session.finish((), selected_figure_ids=(figure["id"],))
    assert not pinned & set(service.released)


@pytest.mark.asyncio
async def test_a_figure_selected_by_its_marker_alone_keeps_its_media():
    session, service = await _session([_urls("a", 3, 640)])
    session.model_evidence_blocks(supports_vision=True)
    figure = _figure(await _compose(session, _cells("I1", "I2")))

    closeout = await session.finish((), selected_figure_ids=(figure["id"],))

    assert set(closeout.selected_candidate_ids) == {"I1", "I2"}
    assert service.released == [_ref(session, "I3")]


@pytest.mark.asyncio
async def test_an_unused_figure_releases_its_media():
    session, service = await _session([_urls("a", 2, 640)])
    session.model_evidence_blocks(supports_vision=True)
    await _compose(session, _cells("I1", "I2"))

    await session.finish(())

    assert set(service.released) == {_ref(session, "I1"), _ref(session, "I2")}


@pytest.mark.asyncio
async def test_a_reference_shared_by_a_used_and_an_unused_figure_is_kept_once():
    session, service = await _session([_urls("a", 3, 640)])
    session.model_evidence_blocks(supports_vision=True)
    used = _figure(await _compose(session, _cells("I1", "I2")))
    await _compose(session, _cells("I1", "I3"))

    await session.finish((), selected_figure_ids=(used["id"],))

    assert service.released == [_ref(session, "I3")]
    assert len(service.released) == len(set(service.released))


@pytest.mark.asyncio
async def test_direct_selection_and_figures_union():
    session, service = await _session([_urls("a", 3, 640)])
    session.model_evidence_blocks(supports_vision=True)
    figure = _figure(await _compose(session, _cells("I2")))

    closeout = await session.finish(("I1",), selected_figure_ids=(figure["id"],))

    assert set(closeout.selected_candidate_ids) == {"I1", "I2"}
    assert service.released == [_ref(session, "I3")]


@pytest.mark.asyncio
async def test_an_invented_figure_id_selects_nothing():
    session, service = await _session([_urls("a", 2, 640)])
    session.model_evidence_blocks(supports_vision=True)
    await _compose(session, _cells("I1", "I2"))

    closeout = await session.finish((), selected_figure_ids=("figure:invented",))

    assert closeout.selected_candidate_ids == ()
    assert len(service.released) == 2


@pytest.mark.asyncio
async def test_finish_leaves_selected_media_pending_for_persistence_to_mark():
    session, service = await _session([_urls("a", 2, 640)])
    session.model_evidence_blocks(supports_vision=True)
    figure = _figure(await _compose(session, _cells("I1", "I2")))

    await session.finish((), selected_figure_ids=(figure["id"],))

    # A failed persistence never calls mark_selected; the pending TTL reclaims them.
    assert service.selected == []
    assert service.released == []


@pytest.mark.asyncio
async def test_cancellation_releases_pinned_dependencies_too():
    session, service = await _session([_urls("small", 2, 320), _urls("big", 4, 1280)])
    session.model_evidence_blocks(supports_vision=True)
    await _compose(session, _cells("I1", "I2"))
    pinned = {_ref(session, "I1"), _ref(session, "I2")}
    await _search(session, "shadow quality screenshots")

    await session.abort()

    assert pinned <= set(service.released)
    assert len(service.released) == len(set(service.released))


@pytest.mark.asyncio
async def test_a_retired_candidate_id_is_never_rebound():
    session, _service = await _session([_urls("small", 2, 320), _urls("big", 4, 1280)])
    session.model_evidence_blocks(supports_vision=True)
    await _compose(session, _cells("I1"))

    await _search(session, "shadow quality screenshots")

    assert "I1" not in session.prepared_images
    assert all(candidate_id not in {"I1", "I2"} for candidate_id in session.prepared_images)


def test_composition_errors_are_value_errors_for_the_tool():
    assert issubclass(SourceFigureError, ValueError)


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def _bound(model_config_key: str = "chat", *, worker: bool = False, figures: bool = True):
    from app.ai.workflow.specialists import with_answer_authoring_tools

    request = SimpleNamespace(
        state={"context": {"inline_rich_response_v1": True, "source_figures_v1": figures}},
        extras={"worker": True} if worker else {},
    )
    tools = with_answer_authoring_tools(
        [], SimpleNamespace(model_config_key=model_config_key), request
    )
    return [tool.name for tool in tools]


@pytest.fixture
def _figures_rollout(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "source_figures_enabled", True)
    monkeypatch.setattr(settings, "answer_visuals_enabled", False)


@pytest.mark.usefixtures("_figures_rollout")
@pytest.mark.parametrize("key", ["chat", "search"])
def test_public_web_answers_bind_figure_composition(key):
    assert _bound(key) == [COMPOSE_SOURCE_FIGURE_TOOL_NAME]


@pytest.mark.usefixtures("_figures_rollout")
def test_private_planning_workers_never_compose_figures():
    assert _bound(worker=True) == []


@pytest.mark.usefixtures("_figures_rollout")
def test_a_client_without_figure_support_is_not_offered_composition():
    assert _bound(figures=False) == []


@pytest.mark.usefixtures("_figures_rollout")
@pytest.mark.parametrize("key", ["rag", "canvas", "custom"])
def test_agents_without_web_images_never_compose(key):
    assert _bound(key) == []


def test_figure_rollout_gate_defaults_off():
    from app.core.config import Settings

    assert Settings.model_fields["source_figures_enabled"].default is False
