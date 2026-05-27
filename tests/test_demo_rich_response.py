"""Tests for the pure rich-response view model used by the Streamlit renderer
(response_format.md Task 6).
"""

from __future__ import annotations

from app.ui.rich_response import build_rich_response_view


metadata_with_selected_image = {
    "rich_items_version": 1,
    "rich_items": [
        {
            "id": "image:document:1",
            "type": "image",
            "display_policy": "inline_only",
            "alt_text": "Selected document figure",
            "payload": {"url": "https://img.test/doc.png", "mime_type": "image/png"},
        }
    ],
}
metadata_with_image_and_widget = {
    "rich_items_version": 1,
    "rich_items": [
        metadata_with_selected_image["rich_items"][0],
        {
            "id": "widget:w-1",
            "type": "live_widget",
            "display_policy": "inline_or_append",
            "payload": {
                "widget_id": "w-1",
                "session_id": "conv-1",
                "widget_type": "chart",
                "status": "active",
                "version": 1,
                "connection_endpoint": "/widgets/w-1/connection",
            },
        },
    ],
}
legacy_metadata_with_image = {
    "images": [{"url": "https://img.test/legacy.png", "mime": "image/png"}]
}


def test_build_view_interleaves_markdown_and_selected_image():
    body = "Intro\n\n<!--rich:image:document:1-->\n\n*Figure 1.*\n\nConclusion"
    view = build_rich_response_view(body, metadata_with_selected_image)
    assert [segment.kind for segment in view.segments] == ["markdown", "rich", "markdown"]
    assert view.segments[1].item["id"] == "image:document:1"
    assert view.append_items == []


def test_build_view_hides_unreferenced_image_and_appends_unreferenced_widget():
    view = build_rich_response_view("Answer", metadata_with_image_and_widget)
    # Unreferenced image is hidden (inline_only policy).
    assert all(item["type"] != "image" for item in view.append_items)
    # Unreferenced widget appends.
    assert [item["type"] for item in view.append_items] == ["live_widget"]


def test_legacy_message_without_version_keeps_existing_gallery_fallback():
    view = build_rich_response_view("Historic answer", legacy_metadata_with_image)
    assert view.use_legacy_image_gallery is True


def test_v1_message_disables_legacy_image_gallery():
    view = build_rich_response_view("Answer", metadata_with_selected_image)
    assert view.use_legacy_image_gallery is False


def test_unknown_marker_becomes_unavailable_segment():
    body = "Before\n\n<!--rich:image:nope-->\n\nAfter"
    view = build_rich_response_view(body, metadata_with_selected_image)
    kinds = [segment.kind for segment in view.segments]
    assert "unavailable" in kinds
    unavailable = next(segment for segment in view.segments if segment.kind == "unavailable")
    assert unavailable.item_id == "image:nope"


def test_referenced_widget_renders_inline_not_appended():
    body = "Body\n\n<!--rich:widget:w-1-->"
    view = build_rich_response_view(body, metadata_with_image_and_widget)
    inline_widget = next(
        segment for segment in view.segments
        if segment.kind == "rich" and segment.item["type"] == "live_widget"
    )
    assert inline_widget.item["id"] == "widget:w-1"
    # Should not also appear in the append section.
    assert all(item["id"] != "widget:w-1" for item in view.append_items)


def test_view_skips_markers_inside_code_fences():
    body = "Code:\n\n```\n<!--rich:image:document:1-->\n```\n\nEnd"
    view = build_rich_response_view(body, metadata_with_selected_image)
    rich_segments = [segment for segment in view.segments if segment.kind == "rich"]
    assert rich_segments == []


def test_empty_metadata_returns_single_markdown_segment():
    view = build_rich_response_view("Just text", None)
    assert [segment.kind for segment in view.segments] == ["markdown"]
    assert view.segments[0].text == "Just text"
    assert view.use_legacy_image_gallery is False
    assert view.append_items == []


# ---------------------------------------------------------------------------
# Live stream state
# ---------------------------------------------------------------------------


def test_stream_state_resolves_complete_marker_after_upsert():
    from app.ui.rich_response import RichStreamState

    state = RichStreamState()
    state.append_text("Intro\n\n<!--rich:widget:w-1-->\n\nMore")
    view = state.build_view()
    # Marker exists but registry is empty → unavailable segment.
    assert any(s.kind == "unavailable" and s.item_id == "widget:w-1" for s in view.segments)
    state.apply_rich_items_upsert(
        [
            {
                "id": "widget:w-1",
                "type": "live_widget",
                "display_policy": "inline_or_append",
                "payload": {
                    "widget_id": "w-1",
                    "session_id": "conv-1",
                    "widget_type": "chart",
                    "status": "active",
                    "version": 1,
                    "connection_endpoint": "/widgets/w-1/connection",
                },
            }
        ]
    )
    view = state.build_view()
    assert any(s.kind == "rich" and s.item["id"] == "widget:w-1" for s in view.segments)


def test_stream_state_holds_partial_marker_as_markdown_until_complete():
    from app.ui.rich_response import RichStreamState

    state = RichStreamState()
    state.append_text("Intro\n\n<!--rich:widget:w")
    view = state.build_view()
    # Incomplete marker stays as markdown text; no exceptions.
    assert all(s.kind == "markdown" for s in view.segments)
    state.append_text("-1-->\n\nMore")
    state.apply_rich_items_upsert(
        [
            {
                "id": "widget:w-1",
                "type": "live_widget",
                "display_policy": "inline_or_append",
                "payload": {
                    "widget_id": "w-1",
                    "session_id": "conv-1",
                    "widget_type": "chart",
                    "status": "active",
                    "version": 1,
                    "connection_endpoint": "/widgets/w-1/connection",
                },
            }
        ]
    )
    view = state.build_view()
    rich_segments = [s for s in view.segments if s.kind == "rich"]
    assert rich_segments and rich_segments[0].item["id"] == "widget:w-1"


def test_stream_state_finalized_replaces_transient_registry():
    from app.ui.rich_response import RichStreamState

    state = RichStreamState()
    state.apply_rich_items_upsert(
        [
            {
                "id": "widget:transient",
                "type": "live_widget",
                "display_policy": "inline_or_append",
                "payload": {
                    "widget_id": "transient",
                    "session_id": "conv-1",
                    "widget_type": "chart",
                    "status": "active",
                    "version": 1,
                    "connection_endpoint": "/widgets/transient/connection",
                },
            }
        ]
    )
    state.replace_with_finalized(
        [
            {
                "id": "widget:final",
                "type": "live_widget",
                "display_policy": "inline_or_append",
                "payload": {
                    "widget_id": "final",
                    "session_id": "conv-1",
                    "widget_type": "chart",
                    "status": "active",
                    "version": 1,
                    "connection_endpoint": "/widgets/final/connection",
                },
            }
        ]
    )
    assert "widget:transient" not in state.items_by_id
    assert "widget:final" in state.items_by_id


def test_stream_state_does_not_track_unselected_image_until_finalize():
    """Image candidates are never streamed via transient upserts; only final
    metadata may surface them. RichStreamState therefore does not accept
    image-type records through upsert."""
    from app.ui.rich_response import RichStreamState

    state = RichStreamState()
    # Caller code in demo.py is expected to filter image records before
    # calling apply_rich_items_upsert(). The state container itself does not
    # enforce that today, but build_view() will still hide unreferenced
    # inline_only images.
    state.append_text("Answer with no image references")
    state.replace_with_finalized(
        [
            {
                "id": "image:tool:c1:0",
                "type": "image",
                "display_policy": "inline_only",
                "alt_text": "Hidden",
                "payload": {"url": "https://img.test/hidden.png", "mime_type": "image/png"},
            }
        ]
    )
    view = state.build_view()
    # No marker → image is not in segments and not appended (inline_only).
    assert all(
        s.kind != "rich" or (s.item or {}).get("type") != "image" for s in view.segments
    )
    assert view.append_items == []
