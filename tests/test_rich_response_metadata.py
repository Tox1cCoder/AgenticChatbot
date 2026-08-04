"""Tests for `build_bot_metadata()` finalization with rich items.

See response_format.md Task 2.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.core.response_constants import build_bot_metadata
from app.schemas.workflow import WorkflowResponse, WorkflowResponseMessage


@pytest.fixture(autouse=True)
def _enable_inline_rich_response_for_contract_tests(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)


def test_build_bot_metadata_persists_only_inline_selected_images():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="See this.\n\n<!--rich:image:tool:c1:0-->"),
        metadata={
            "_rich_item_candidates": [
                {
                    "id": "image:tool:c1:0",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Selected image",
                    "payload": {
                        "url": "https://img.test/selected.png",
                        "mime_type": "image/png",
                    },
                },
                {
                    "id": "image:tool:c1:1",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Hidden image",
                    "payload": {
                        "url": "https://img.test/hidden.png",
                        "mime_type": "image/png",
                    },
                },
            ]
        },
    )
    metadata = build_bot_metadata(response)
    image_ids = [item["id"] for item in metadata["rich_items"] if item["type"] == "image"]
    assert image_ids == ["image:tool:c1:0"]
    assert all("hidden.png" not in str(item) for item in metadata["rich_items"])
    assert "hidden.png" not in str(metadata.get("images", []))
    assert "_rich_item_candidates" not in metadata
    assert metadata["rich_items_version"] == 1


def test_build_bot_metadata_strips_original_image_url_from_public_provenance():
    candidate = {
        "id": "image:tool:c1:0",
        "type": "image",
        "display_policy": "inline_only",
        "alt_text": "Selected image",
        "payload": {
            "url": "https://img.test/selected.png",
            "mime_type": "image/png",
        },
        "provenance": {
            "provider": "tavily",
            "original_image_url": "https://img.test/original.png",
        },
    }
    response = WorkflowResponse(
        message=WorkflowResponseMessage(
            content="See this.\n\n<!--rich:image:tool:c1:0-->"
        ),
        metadata={"_rich_item_candidates": [candidate]},
    )

    metadata = build_bot_metadata(response)

    assert "original_image_url" not in metadata["rich_items"][0]["provenance"]
    assert candidate["provenance"]["original_image_url"].endswith("original.png")


def test_build_bot_metadata_persists_embedded_selected_images():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(
            content=(
                "* **Review:** <!--rich:image:tool:c1:0--> Fastest drive.\n\n"
                "* **Review:** <!--rich:image:tool:c1:1--> Better value."
            )
        ),
        metadata={
            "_rich_item_candidates": [
                {
                    "id": "image:tool:c1:0",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Fastest drive",
                    "payload": {
                        "url": "https://img.test/first.png",
                        "mime_type": "image/png",
                    },
                },
                {
                    "id": "image:tool:c1:1",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Better value",
                    "payload": {
                        "url": "https://img.test/second.png",
                        "mime_type": "image/png",
                    },
                },
            ]
        },
    )

    metadata = build_bot_metadata(response)

    image_ids = [item["id"] for item in metadata["rich_items"] if item["type"] == "image"]
    assert image_ids == ["image:tool:c1:0", "image:tool:c1:1"]
    assert metadata["rich_reference_warnings"] == []


def test_unreferenced_widget_is_kept_for_appended_compatibility():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="The comparison widget is available below."),
        metadata={"_inline_rich_response_v1": True},
        tool_artifacts=[
            {
                "tool_call_id": "widget-call",
                "tool": "widget_create",
                "args": {},
                "output": (
                    '{"widget_id":"w-1","session_id":"conv-1","widget_type":"chart",'
                    '"status":"active","version":1}'
                ),
                "status": "success",
            }
        ],
    )
    metadata = build_bot_metadata(response)
    widget = next(item for item in metadata["rich_items"] if item["type"] == "live_widget")
    assert widget["display_policy"] == "inline_or_append"
    # Legacy live_widgets and tool_artifacts must remain intact for the widget
    # restoration code paths.
    assert metadata["live_widgets"][0]["widget_id"] == "w-1"
    assert metadata["tool_artifacts"][0]["tool_call_id"] == "widget-call"


def test_non_capable_widget_response_keeps_legacy_metadata_only():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="The widget is available below."),
        tool_artifacts=[
            {
                "tool_call_id": "widget-call",
                "tool": "widget_create",
                "output": (
                    '{"widget_id":"w-legacy","session_id":"conv-1","widget_type":"chart",'
                    '"status":"active","version":1}'
                ),
                "status": "success",
            }
        ],
    )

    metadata = build_bot_metadata(response)

    assert metadata["live_widgets"][0]["widget_id"] == "w-legacy"
    assert "rich_items_version" not in metadata


def test_disabled_rollout_keeps_widget_on_legacy_metadata_only(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", False)
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="The widget is available below."),
        tool_artifacts=[
            {
                "tool_call_id": "widget-call",
                "tool": "widget_create",
                "output": (
                    '{"widget_id":"w-legacy","session_id":"conv-1","widget_type":"chart",'
                    '"status":"active","version":1}'
                ),
                "status": "success",
            }
        ],
    )

    metadata = build_bot_metadata(response)

    assert metadata["live_widgets"][0]["widget_id"] == "w-legacy"
    assert "rich_items_version" not in metadata
    assert "rich_items" not in metadata


def test_referenced_widget_marker_is_not_duplicated_in_append():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="Here is the widget:\n\n<!--rich:widget:w-2-->"),
        tool_artifacts=[
            {
                "tool_call_id": "widget-call",
                "tool": "widget_create",
                "args": {},
                "output": (
                    '{"widget_id":"w-2","session_id":"conv-1","widget_type":"chart",'
                    '"status":"active","version":1}'
                ),
                "status": "success",
            }
        ],
    )
    metadata = build_bot_metadata(response)
    widget_items = [item for item in metadata["rich_items"] if item["type"] == "live_widget"]
    assert len(widget_items) == 1
    assert widget_items[0]["id"] == "widget:w-2"


def test_capable_canvas_response_promotes_canvas_rich_item():
    """CanvasAgent output must reach AI SDK clients through ``rich_items`` —
    the wire projection scrubs the legacy ``canvas_artifact`` field, so
    without this promotion the artifact would be invisible to them."""

    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="Here is your page."),
        metadata={
            "_inline_rich_response_v1": True,
            "canvas_artifact": {
                "content": "<!doctype html><title>Hello</title><h1>Hello</h1>",
                "language": "html",
                "title": "Hello",
            },
        },
    )

    metadata = build_bot_metadata(response)

    canvas = next(item for item in metadata["rich_items"] if item["type"] == "canvas_artifact")
    assert canvas["id"] == "canvas:main"
    assert canvas["display_policy"] == "inline_or_append"
    assert canvas["payload"]["content"].startswith("<!doctype html>")
    assert canvas["payload"]["language"] == "html"
    assert canvas["payload"]["title"] == "Hello"
    assert metadata["rich_items_version"] == 1
    # Legacy canvas_artifact stays persisted for the Streamlit path.
    assert metadata["canvas_artifact"]["title"] == "Hello"


def test_canvas_revision_metadata_is_additive_in_rich_payload():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="Updated your page."),
        metadata={
            "_inline_rich_response_v1": True,
            "canvas_artifact": {
                "artifact_id": "canvas:main",
                "revision": 2,
                "operation": "update",
                "content": "<!doctype html><title>Updated</title>",
                "language": "html",
                "title": "Updated",
            },
        },
    )

    metadata = build_bot_metadata(response)

    canvas = next(item for item in metadata["rich_items"] if item["type"] == "canvas_artifact")
    assert canvas["id"] == "canvas:main"
    assert canvas["payload"]["revision"] == 2
    assert canvas["payload"]["operation"] == "update"


def test_capable_canvas_response_defaults_missing_title_and_language():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="Here is your page."),
        metadata={
            "_inline_rich_response_v1": True,
            "canvas_artifact": {"content": "<svg></svg>"},
        },
    )

    metadata = build_bot_metadata(response)

    canvas = next(item for item in metadata["rich_items"] if item["type"] == "canvas_artifact")
    assert canvas["payload"]["title"] == "Canvas"
    assert canvas["payload"]["language"] == "html"


def test_non_capable_canvas_response_keeps_legacy_metadata_only():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="Here is your page."),
        metadata={
            "canvas_artifact": {
                "content": "<!doctype html><title>Hello</title>",
                "language": "html",
                "title": "Hello",
            }
        },
    )

    metadata = build_bot_metadata(response)

    assert metadata["canvas_artifact"]["title"] == "Hello"
    assert "rich_items_version" not in metadata


def test_canvas_artifact_without_content_is_not_promoted():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="Nothing to render."),
        metadata={
            "_inline_rich_response_v1": True,
            "canvas_artifact": {"language": "html", "title": "Empty"},
        },
    )

    metadata = build_bot_metadata(response)

    assert "rich_items_version" not in metadata
    assert "rich_items" not in metadata


def test_unknown_marker_produces_warning():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="<!--rich:image:does-not-exist-->"),
        metadata={"_rich_item_candidates": []},
    )
    metadata = build_bot_metadata(response)
    warnings = metadata.get("rich_reference_warnings", [])
    assert any(
        w.get("code") == "unknown_rich_item" and w.get("id") == "image:does-not-exist"
        for w in warnings
    )


def test_legacy_response_without_candidates_or_markers_keeps_existing_images_field():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="Legacy answer"),
        metadata={"images": [{"url": "https://img.test/legacy.png", "mime": "image/png"}]},
    )
    metadata = build_bot_metadata(response)
    # Legacy messages without v1 rich_items_version must keep their gallery
    # behavior. The version is only set when v1 finalization actually runs.
    assert metadata.get("rich_items_version") != 1 or metadata.get("images")


def test_v1_message_strips_unselected_image_candidates_from_public_images():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="No relevant images."),
        metadata={
            "_rich_item_candidates": [
                {
                    "id": "image:tool:c1:0",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Hidden",
                    "payload": {"url": "https://img.test/hidden.png", "mime_type": "image/png"},
                }
            ],
            # Even if some upstream populated metadata["images"] with all
            # candidates, v1 finalization must not leak them.
            "images": [{"url": "https://img.test/hidden.png", "mime": "image/png"}],
        },
    )
    metadata = build_bot_metadata(response)
    assert metadata.get("rich_items_version") == 1
    assert metadata.get("rich_items") == []
    images = metadata.get("images", [])
    assert all("hidden.png" not in str(image) for image in images)


def _group_candidate() -> dict:
    return {
        "id": "imagegroup:tool:c1",
        "type": "image_group",
        "source": "image_search",
        "display_policy": "inline_only",
        "alt_text": "Images of a red panda",
        "payload": {
            "items": [
                {"url": "https://img.test/hidden-a.jpg", "mime_type": "image/jpeg"},
                {"url": "https://img.test/hidden-b.jpg", "mime_type": "image/jpeg"},
            ]
        },
    }


def test_unreferenced_image_group_candidate_is_never_persisted_or_flattened():
    """An ``image_group`` is an image candidate and obeys selected-only
    persistence. This is the ``rich_auto_place_enabled=False`` + no-marker
    shape: nothing placed the group, so nothing may persist it, register
    ``/web-images`` rows for its cells, or flatten it into AI SDK file parts.
    """
    from app.services.event_streaming.ai_sdk_projection import (
        selected_image_file_parts_from_rich_items,
    )

    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="No marker in this answer."),
        metadata={"_rich_item_candidates": [_group_candidate()]},
    )

    metadata = build_bot_metadata(response)

    assert metadata["rich_items"] == []
    assert "hidden-a.jpg" not in str(metadata)
    assert selected_image_file_parts_from_rich_items(metadata) == []


def test_selected_image_group_is_not_duplicated_in_legacy_gallery():
    """Typed v1 images render through rich_items, never a second gallery."""
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="Look:\n\n<!--rich:imagegroup:tool:c1-->"),
        metadata={
            "_rich_item_candidates": [_group_candidate()],
            "images": [
                {
                    "rich_item_id": "imagegroup:tool:c1",
                    "url": "https://img.test/hidden-a.jpg",
                    "mime": "image/jpeg",
                },
                {
                    "id": "generated:local-1",
                    "url": "/chat-images/local-1",
                    "mime": "image/png",
                },
            ],
        },
    )

    metadata = build_bot_metadata(response)

    assert [item["id"] for item in metadata["rich_items"]] == ["imagegroup:tool:c1"]
    assert metadata["images"] == [
        {
            "id": "generated:local-1",
            "url": "/chat-images/local-1",
            "mime": "image/png",
        }
    ]


def test_v1_finalization_rejects_selected_image_with_invalid_url_scheme():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="<!--rich:image:tool:c1:0-->"),
        metadata={
            "_rich_item_candidates": [
                {
                    "id": "image:tool:c1:0",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Unsafe",
                    "payload": {
                        "url": "javascript:alert(1)",
                        "mime_type": "image/png",
                    },
                }
            ]
        },
    )

    metadata = build_bot_metadata(response)

    assert metadata["rich_items"] == []
    assert any(
        warning.get("code") == "invalid_rich_item" and warning.get("id") == "image:tool:c1:0"
        for warning in metadata["rich_reference_warnings"]
    )


def test_v1_finalization_rejects_selected_image_over_decoded_byte_limit(monkeypatch):
    monkeypatch.setattr(settings, "rich_item_selected_image_max_bytes", 3)
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="<!--rich:image:document:1-->"),
        metadata={
            "_rich_item_candidates": [
                {
                    "id": "image:document:1",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Oversized",
                    "payload": {
                        "data": "QUJDRA==",
                        "mime_type": "image/png",
                    },
                }
            ]
        },
    )

    metadata = build_bot_metadata(response)

    assert metadata["rich_items"] == []
    assert any(
        warning.get("code") == "invalid_rich_item" and warning.get("id") == "image:document:1"
        for warning in metadata["rich_reference_warnings"]
    )
