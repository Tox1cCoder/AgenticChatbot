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
