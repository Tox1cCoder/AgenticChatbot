"""Composed source figures: labelled cells, captions, selection, degradation.

A figure reuses the ``image`` / ``image_group`` family. These tests pin that
its labels and caption survive every path that used to rebuild an image
payload from scratch, and that a comparison never outlives its cells.
"""

from __future__ import annotations

import copy
from uuid import uuid4

import pytest

from app.core.answer_visuals import ResolvedEvidence
from app.core.config import settings
from app.core.response_constants import build_bot_metadata
from app.core.rich_image_selection import ImageSelectionPolicy, select_rich_item_candidates
from app.core.rich_response import validate_public_rich_item
from app.core.source_figures import FigureCell, build_source_figure, degrade_source_figure
from app.schemas.workflow import WorkflowResponse, WorkflowResponseMessage

_POLICY = ImageSelectionPolicy(
    max_items=4, min_width_px=1, min_height_px=1, min_aspect_ratio=0.1, max_aspect_ratio=10
)


def _cell(n: int, *, label: str | None = None) -> FigureCell:
    return FigureCell(
        url=f"/web-images/{uuid4()}",
        mime_type="image/png",
        source_url=f"https://site{n}.example/page",
        source_title=f"Site {n}",
        width=640,
        height=360,
        label=label or f"Setting {n}",
        alt_text=f"Screenshot of setting {n}",
    )


def _figure(cells: int = 2, **kwargs) -> dict:
    return build_source_figure(
        [_cell(n) for n in range(1, cells + 1)],
        caption=kwargs.get("caption", "Low shadows remove the soft edge under the car."),
        caption_evidence=kwargs.get("caption_evidence", ()),
        turn_key=kwargs.get("turn_key", "turn-1"),
    )


def _public(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "_internal"}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_comparison_is_an_image_group_with_labelled_cells_and_a_caption():
    item = _figure()

    validated = validate_public_rich_item(_public(item))
    assert validated.type == "image_group"
    payload = item["payload"]
    assert [cell["label"] for cell in payload["items"]] == ["Setting 1", "Setting 2"]
    assert payload["items"][0]["alt_text"] == "Screenshot of setting 1"
    assert payload["caption"] == "Low shadows remove the soft edge under the car."
    assert item["display_policy"] == "inline_only"
    assert item["id"].startswith("figure:")


def test_single_cell_is_an_image_with_its_own_label_and_caption():
    item = _figure(cells=1, caption="The settings menu in version 2.")

    validated = validate_public_rich_item(_public(item))
    assert validated.type == "image"
    assert item["payload"]["label"] == "Setting 1"
    assert item["payload"]["caption"] == "The settings menu in version 2."
    assert item["alt_text"] == "Screenshot of setting 1"


def test_fallback_names_each_cell_with_its_source():
    fallback = _figure()["payload"]["fallback_markdown"]

    assert fallback == (
        "**Figure:** Low shadows remove the soft edge under the car.\n"
        "\n"
        "- Setting 1 — Screenshot of setting 1 ([site1.example](https://site1.example/page))\n"
        "- Setting 2 — Screenshot of setting 2 ([site2.example](https://site2.example/page))"
    )


def test_caption_evidence_is_resolved_into_public_links():
    item = _figure(
        caption_evidence=(ResolvedEvidence(title="Guide", url="https://guide.example/a"),)
    )

    assert item["payload"]["caption_evidence"] == [
        {"title": "Guide", "url": "https://guide.example/a", "label": "guide.example"}
    ]
    assert "[guide.example](https://guide.example/a)" in item["payload"]["fallback_markdown"]


def test_figure_payload_names_only_protected_media():
    item = _figure()

    for cell in item["payload"]["items"]:
        assert cell["url"].startswith("/web-images/")


@pytest.mark.parametrize("count", [0, 5])
def test_figure_needs_one_to_four_cells(count):
    with pytest.raises(ValueError):
        build_source_figure(
            [_cell(n) for n in range(count)], caption=None, caption_evidence=(), turn_key="t"
        )


def test_duplicate_cells_are_rejected():
    cell = _cell(1)
    with pytest.raises(ValueError, match="once"):
        build_source_figure([cell, cell], caption=None, caption_evidence=(), turn_key="t")


def test_same_composition_reuses_its_id_and_a_revision_gets_a_new_one():
    cells = [_cell(1), _cell(2)]
    first = build_source_figure(cells, caption="a", caption_evidence=(), turn_key="t")
    again = build_source_figure(cells, caption="a", caption_evidence=(), turn_key="t")
    revised = build_source_figure(cells, caption="b", caption_evidence=(), turn_key="t")

    assert first["id"] == again["id"]
    assert first["id"] != revised["id"]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_labelled_cells_survive_selection():
    [selected] = select_rich_item_candidates([_figure()], policy=_POLICY)

    assert selected["type"] == "image_group"
    assert [cell["label"] for cell in selected["payload"]["items"]] == ["Setting 1", "Setting 2"]
    assert selected["payload"]["caption"]


def test_singleton_figure_keeps_its_own_metadata_through_selection():
    figure = _figure(cells=1, caption="One view.")

    [selected] = select_rich_item_candidates([figure], policy=_POLICY)

    assert selected["payload"]["label"] == "Setting 1"
    assert selected["payload"]["caption"] == "One view."


def test_selected_group_survives_a_competing_singleton_for_the_same_media():
    figure = _figure()
    singleton = {
        "id": "image:tool-content:c1:0",
        "type": "image",
        "source": "tool_image",
        "display_policy": "inline_only",
        "alt_text": "same picture",
        "payload": {"url": figure["payload"]["items"][0]["url"], "mime_type": "image/png"},
    }

    selected = select_rich_item_candidates([singleton, figure], policy=_POLICY)

    group = next(item for item in selected if item["id"] == figure["id"])
    assert len(group["payload"]["items"]) == 2


def test_unselected_group_stays_private(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)

    metadata = build_bot_metadata(
        WorkflowResponse(
            message=WorkflowResponseMessage(content="No figure placed."),
            metadata={"_rich_item_candidates": [_figure()]},
        )
    )

    assert metadata["rich_items"] == []


def test_placed_group_is_persisted_without_its_private_record(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    figure = _figure()

    metadata = build_bot_metadata(
        WorkflowResponse(
            message=WorkflowResponseMessage(content=f"<!--rich:{figure['id']}-->"),
            metadata={"_rich_item_candidates": [figure]},
        )
    )

    [persisted] = metadata["rich_items"]
    assert persisted["id"] == figure["id"]
    assert "_internal" not in persisted


# ---------------------------------------------------------------------------
# Degradation when a cell disappears
# ---------------------------------------------------------------------------


def test_losing_a_cell_removes_the_comparison_caption():
    figure = _public(_figure())
    surviving = [figure["payload"]["items"][1]]

    degraded = degrade_source_figure(figure, surviving)

    payload = degraded["payload"]
    assert [cell["label"] for cell in payload["items"]] == ["Setting 2"]
    assert "caption" not in payload
    assert payload["comparison_unavailable"] is True
    assert "Low shadows" not in payload["fallback_markdown"]
    assert "comparison is unavailable" in payload["fallback_markdown"]
    validate_public_rich_item(degraded)


def test_degrading_never_mutates_its_input():
    figure = _public(_figure())
    before = copy.deepcopy(figure)

    degrade_source_figure(figure, [figure["payload"]["items"][0]])

    assert figure == before


def test_losing_every_cell_drops_the_figure():
    assert degrade_source_figure(_public(_figure()), []) is None


def test_a_complete_figure_is_unchanged():
    figure = _public(_figure())

    assert degrade_source_figure(figure, figure["payload"]["items"]) == figure


def test_fork_drops_an_unavailable_cell_and_the_comparison_caption():
    from app.services.conversation_fork_metadata import drop_unavailable_media, media_references

    figure = _public(_figure())
    metadata = {"rich_items_version": 1, "rich_items": [figure], "rich_reference_warnings": []}
    references = sorted(media_references(metadata))
    content = f"<!--rich:{figure['id']}-->"

    result = drop_unavailable_media(content, metadata, available={references[0]})

    [kept] = result["rich_items"]
    assert len(kept["payload"]["items"]) == 1
    assert "caption" not in kept["payload"]
    assert kept["payload"]["comparison_unavailable"] is True


def test_selection_that_leaves_one_cell_keeps_its_label_and_marks_the_loss():
    figure = _figure()
    figure["payload"]["items"][0]["url"] = "/not-a-protected-route/x"

    [selected] = select_rich_item_candidates([figure], policy=_POLICY)

    assert selected["type"] == "image"
    assert selected["payload"]["label"] == "Setting 2"
    assert selected["payload"]["comparison_unavailable"] is True
    assert "caption" not in selected["payload"]
    validate_public_rich_item(_public(selected))


# ---------------------------------------------------------------------------
# Capability projection
# ---------------------------------------------------------------------------


def _message(figure: dict) -> dict:
    return {
        "id": "m-1",
        "role": "assistant",
        "content": f"Compare:\n\n<!--rich:{figure['id']}-->\n\nDone.",
        "metadata": {
            "rich_items_version": 1,
            "rich_items": [figure],
            "rich_reference_warnings": [],
        },
    }


def _project(message: dict, *, rich: bool, figures: bool) -> dict:
    from app.services.event_streaming.ai_sdk_projection import (
        project_ai_sdk_message_for_capability,
    )

    return project_ai_sdk_message_for_capability(
        message, inline_rich_response_v1=rich, source_figures_v1=figures
    )


def test_figure_capable_client_gets_the_canonical_figure():
    message = _message(_public(_figure()))

    assert _project(message, rich=True, figures=True) == message


def test_base_rich_client_gets_text_then_a_legacy_group():
    figure = _public(_figure())
    projected = _project(_message(figure), rich=True, figures=False)

    marker = f"<!--rich:{figure['id']}-->"
    assert projected["content"] == (
        f"Compare:\n\n{figure['payload']['fallback_markdown']}\n\n{marker}\n\nDone."
    )
    [legacy] = projected["metadata"]["rich_items"]
    assert legacy["id"] == figure["id"]
    assert set(legacy["payload"]) == {"items"}
    for cell in legacy["payload"]["items"]:
        assert "label" not in cell
        assert "alt_text" not in cell
    validate_public_rich_item(legacy)


def test_non_rich_client_gets_text_only_and_the_media_as_file_parts():
    from app.services.event_streaming.ai_sdk_projection import visible_image_file_parts

    figure = _public(_figure())
    message = _message(figure)
    projected = _project(message, rich=False, figures=False)

    assert figure["payload"]["fallback_markdown"] in projected["content"]
    assert "<!--rich:" not in projected["content"]
    parts = visible_image_file_parts(message, inline_rich_response_v1=False)
    assert [part["url"] for part in parts] == [
        cell["url"] for cell in figure["payload"]["items"]
    ]


@pytest.fixture
def _figures_rollout(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "source_figures_enabled", True)


@pytest.mark.usefixtures("_figures_rollout")
def test_effective_figure_capability_requires_base_and_rollout(monkeypatch):
    from app.services.event_streaming.ai_sdk_projection import effective_source_figures

    assert effective_source_figures(base=True, requested=True) is True
    assert effective_source_figures(base=False, requested=True) is False
    monkeypatch.setattr(settings, "source_figures_enabled", False)
    assert effective_source_figures(base=True, requested=True) is False


@pytest.mark.usefixtures("_figures_rollout")
@pytest.mark.parametrize("requested", [True, False])
def test_figure_capability_reaches_generation(requested):
    from app.ai.graph import MultiAgentWorkflow
    from app.ai.schemas import WorkflowExecutionRequest as AIRequest
    from app.schemas.message import MessageCreate
    from app.schemas.workflow import WorkflowExecutionRequest
    from app.services.ai_service import AIService

    create = MessageCreate.model_validate(
        {"conversationId": str(uuid4()), "content": "hi", "sourceFiguresV1": requested}
    )
    assert create.source_figures_v1 is requested
    ai_request = AIService._to_ai_request(
        WorkflowExecutionRequest(message="hi", source_figures_v1=requested)
    )
    assert ai_request.source_figures_v1 is requested
    state = MultiAgentWorkflow.__new__(MultiAgentWorkflow)._build_initial_state_from_request(
        AIRequest(message="hi", source_figures_v1=requested)
    )
    assert state["context"]["source_figures_v1"] is requested


def test_message_service_forwards_figure_capability():
    import inspect

    from app.services.message_service import MessageService

    source = inspect.getsource(MessageService._build_user_message_workflow_request)
    assert 'getattr(message_create_data, "source_figures_v1", False)' in source


@pytest.mark.usefixtures("_figures_rollout")
@pytest.mark.parametrize("key", ["sourceFiguresV1", "source_figures_v1"])
def test_every_request_schema_accepts_the_figure_capability(key):
    from app.api.ai_sdk import AISDKChatRequest
    from app.schemas.message import ContinueGenerationRequest, InterruptResumeRequest

    chat = AISDKChatRequest.model_validate({"messages": [], key: True})
    resume = InterruptResumeRequest.model_validate(
        {
            "threadId": "t",
            "interruptId": "i",
            "conversationId": str(uuid4()),
            "decisions": [{"type": "approve"}],
            key: True,
        }
    )
    resumed_continue = ContinueGenerationRequest.model_validate(
        {
            "conversationId": str(uuid4()),
            "generationId": str(uuid4()),
            "continuationId": str(uuid4()),
            "idempotencyKey": "continue-test-1",
            "expectedVersion": 1,
            key: True,
        }
    )
    assert chat.source_figures_v1 and resume.source_figures_v1
    assert resumed_continue.source_figures_v1


@pytest.mark.usefixtures("_figures_rollout")
@pytest.mark.parametrize("figures", [True, False])
def test_history_projects_figures_by_capability(figures):
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from app.api.ai_sdk import get_conversation_messages_ai_sdk
    from app.repositories.utils.pagination import PaginationMeta
    from app.schemas.pagination import MessagePaginationParams

    figure = _public(_figure())
    stored = _message(figure)
    row = SimpleNamespace(
        id=uuid4(),
        sender=2,
        content=stored["content"],
        created_at=datetime.now(UTC),
        message_metadata=stored["metadata"],
    )
    service = MagicMock()
    service.get_conversation_messages.return_value = SimpleNamespace(
        items=[row], meta=PaginationMeta.calculate(1, 100, 1)
    )

    response = get_conversation_messages_ai_sdk(
        uuid4(),
        service,
        uuid4(),
        MessagePaginationParams(),
        inline_rich_response_v1=True,
        source_figures_v1=figures,
    )

    message = response.data.messages[0].model_dump(by_alias=True)
    [item] = message["metadata"]["rich_items"]
    has_labels = "label" in item["payload"]["items"][0]
    assert has_labels is figures
    assert (figure["payload"]["fallback_markdown"] in message["content"]) is not figures


@pytest.mark.asyncio
@pytest.mark.usefixtures("_figures_rollout")
@pytest.mark.parametrize("figures", [True, False])
async def test_stream_completion_projects_figures_by_capability(figures):
    import json

    from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
    from app.services.event_streaming.events import make_event

    figure = _public(_figure())
    stored = _message(figure)

    async def source():
        yield make_event("message_delta", sequence=1, data={"text": stored["content"]})
        yield make_event("complete", sequence=2, data={"message": stored})

    state = AISDKV6StreamState(
        "m-1", "t-1", "r-1", inline_rich_response_v1=True, source_figures_v1=figures
    )
    adapter = AISDKV6StreamAdapter(source, state, heartbeat_interval_seconds=60)
    body = "".join([frame async for frame in adapter.iter_sse()])
    frames = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    [final] = [f for f in frames if f["type"] == "data-assistant-message"]
    [item] = final["data"]["message"]["metadata"]["rich_items"]
    assert ("caption" in item["payload"]) is figures


def test_figure_projection_never_mutates_the_stored_message():
    message = _message(_public(_figure()))
    before = copy.deepcopy(message)

    _project(message, rich=True, figures=False)
    _project(message, rich=False, figures=False)

    assert message == before
