"""Tests for the inline rich-response contract: marker parser, payload validation,
display-policy filters, and bounded prompt inventory serialization.

See response_format.md Task 1.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.rich_response import (
    ImageRichItem,
    LiveWidgetRichItem,
    RichDisplayPolicy,
    RichItemType,
    build_rich_item_inventory_block,
    parse_inline_rich_references,
    remove_inline_rich_reference,
    select_append_fallback_items,
    select_transient_upsert_items,
    validate_public_rich_item,
    validate_rich_references,
)


def test_inline_rich_response_rollout_is_enabled_by_default_kill_switch(monkeypatch):
    """Inline rich-response is enabled by default; the flag remains as a kill switch."""
    from app.core.config import Settings

    monkeypatch.delenv("INLINE_RICH_RESPONSE_ENABLED", raising=False)

    assert Settings(_env_file=None).inline_rich_response_enabled is True


# ---------------------------------------------------------------------------
# Marker parser
# ---------------------------------------------------------------------------


def test_marker_is_recognized_only_as_standalone_markdown_block():
    body = "Before\n\n<!--rich:image:tool:call-1:0-->\n\nAfter\n`<!--rich:image:nope-->`"
    assert parse_inline_rich_references(body) == ["image:tool:call-1:0"]


def test_marker_allows_up_to_three_leading_spaces_and_trailing_whitespace():
    body = "Intro\n\n   <!--rich:image:doc:1-->   \n\nMid"
    assert parse_inline_rich_references(body) == ["image:doc:1"]


def test_marker_inside_fenced_code_block_is_ignored():
    body = "```\n<!--rich:image:tool:call-1:0-->\n```\n\n<!--rich:image:tool:call-2:0-->"
    assert parse_inline_rich_references(body) == ["image:tool:call-2:0"]


def test_marker_inside_tilde_fenced_code_block_is_ignored():
    body = "~~~\n<!--rich:image:hidden-->\n~~~\n\n<!--rich:image:visible-->"
    assert parse_inline_rich_references(body) == ["image:visible"]


def test_marker_inside_longer_closing_fence_remains_ignored():
    body = "````\n<!--rich:image:hidden-->\n````\n\n<!--rich:image:visible-->"
    assert parse_inline_rich_references(body) == ["image:visible"]


def test_marker_indented_four_spaces_is_treated_as_code_not_marker():
    body = "Intro\n\n    <!--rich:image:hidden-->\n\nEnd"
    assert parse_inline_rich_references(body) == []


def test_marker_handles_crlf_line_endings():
    body = "Intro\r\n\r\n<!--rich:image:doc:1-->\r\n\r\nEnd"
    assert parse_inline_rich_references(body) == ["image:doc:1"]


def test_marker_rejects_invalid_characters():
    # Spaces and other unsupported characters are not valid id chars.
    body = "<!--rich:bad id-->"
    assert parse_inline_rich_references(body) == []


def test_marker_rejects_oversized_ids():
    long_id = "x" * 129
    body = f"<!--rich:{long_id}-->"
    assert parse_inline_rich_references(body) == []


def test_marker_duplicate_ids_preserve_order():
    body = "<!--rich:widget:1-->\n\nMid\n\n<!--rich:widget:1-->"
    assert parse_inline_rich_references(body) == ["widget:1", "widget:1"]


def test_remove_one_rich_reference_preserves_other_markers_and_code_examples():
    body = (
        "Before\n\n<!--rich:image:drop-->\n\n"
        "```\n<!--rich:image:drop-->\n```\n\n"
        "<!--rich:image:keep-->\n\nAfter"
    )

    cleaned = remove_inline_rich_reference(body, "image:drop")

    assert parse_inline_rich_references(cleaned) == ["image:keep"]
    assert "```\n<!--rich:image:drop-->\n```" in cleaned
    assert "\n\n\n" not in cleaned


def test_embedded_markers_are_recognized_outside_inline_code():
    body = (
        "* **Review:** <!--rich:image:tool:c1:0--> Strong option.\n"
        "See `<!--rich:image:tool:hidden:0-->` for the marker grammar.\n"
        "* **Review:** <!--rich:image:tool:c1:1--> Better value."
    )
    assert parse_inline_rich_references(body) == [
        "image:tool:c1:0",
        "image:tool:c1:1",
    ]


# ---------------------------------------------------------------------------
# Display-policy filters
# ---------------------------------------------------------------------------


def _make_image(id_: str, url: str = "https://img.test/a.png") -> ImageRichItem:
    return ImageRichItem(
        id=id_,
        type=RichItemType.image,
        display_policy=RichDisplayPolicy.inline_only,
        alt_text="Example image",
        payload={"url": url, "mime_type": "image/png"},
    )


def test_image_payload_accepts_display_metadata_and_protected_web_reference():
    image = ImageRichItem(
        id="image:web:1",
        type=RichItemType.image,
        display_policy=RichDisplayPolicy.inline_only,
        alt_text="A red panda in a tree",
        payload={
            "url": "/web-images/55d170b5-b0f0-44fc-9155-af8af484513d",
            "mime_type": "image/jpeg",
            "source_url": "https://publisher.example/story",
            "width": 640,
            "height": 360,
            "caption": "Publisher-supplied figure caption",
        },
    )

    assert image.payload.width == 640
    assert image.payload.height == 360
    assert image.payload.caption == "Publisher-supplied figure caption"


def test_image_payload_accepts_existing_api_protected_route_forms():
    for url in (
        "/chat-images/55d170b5-b0f0-44fc-9155-af8af484513d",
        "/api/chat-images/55d170b5-b0f0-44fc-9155-af8af484513d",
        "/api/web-images/55d170b5-b0f0-44fc-9155-af8af484513d",
    ):
        assert _make_image("image:web:1", url=url).payload.url == url


def test_image_payload_rejects_unknown_relative_url():
    with pytest.raises(ValidationError, match="protected image url"):
        _make_image("image:web:1", url="/proxy?url=https://internal.example")


def _make_widget(id_: str = "widget:w-1") -> LiveWidgetRichItem:
    return LiveWidgetRichItem(
        id=id_,
        type=RichItemType.live_widget,
        display_policy=RichDisplayPolicy.inline_or_append,
        payload={
            "widget_id": "w-1",
            "session_id": "conv-1",
            "status": "active",
            "version": 1,
            "connection_endpoint": "/widgets/w-1/connection",
        },
    )


def test_unreferenced_images_are_never_append_fallbacks():
    image = _make_image("image:tool:call-1:0")
    assert select_append_fallback_items([image], referenced_ids=set()) == []


def test_unreferenced_widget_is_an_append_fallback():
    widget = _make_widget()
    assert select_append_fallback_items([widget], referenced_ids=set()) == [widget]


def test_referenced_items_are_never_append_fallbacks():
    widget = _make_widget()
    assert select_append_fallback_items([widget], referenced_ids={widget.id}) == []


def test_transient_upserts_exclude_image_items_entirely():
    image = _make_image("image:generated:m-1:0")
    assert select_transient_upsert_items([image]) == []


def test_transient_upserts_include_safe_widget_records():
    widget = _make_widget()
    [item] = select_transient_upsert_items([widget])
    assert item["id"] == widget.id
    assert item["type"] == "live_widget"
    assert item["provenance"] == {}
    assert item["payload"]["widget_id"] == "w-1"


def test_unselected_image_data_is_not_serialized_for_streaming():
    image = ImageRichItem(
        id="image:generated:m-1:0",
        type=RichItemType.image,
        display_policy=RichDisplayPolicy.inline_only,
        alt_text="Generated image",
        payload={"data": "QUJDRA==", "mime_type": "image/png"},
    )
    assert select_transient_upsert_items([image]) == []


def test_transient_upserts_exclude_nested_inline_binary_data():
    tool_render = {
        "id": "tool:call-1",
        "type": "tool_render",
        "source": "tool",
        "display_policy": "inline_or_append",
        "payload": {
            "render": {
                "version": 1,
                "type": "image",
                "content": [
                    {
                        "type": "image",
                        "data": "QUJDRA==",
                        "mimeType": "image/png",
                    }
                ],
            }
        },
        "provenance": {"tool_call_id": "call-1", "tool": "custom_tool"},
    }

    assert select_transient_upsert_items([tool_render]) == []


def test_transient_upserts_omit_null_keys_and_default_provenance():
    widget = {
        "id": "widget:w-1",
        "type": "live_widget",
        "source": "widget_tool",
        "display_policy": "inline_or_append",
        "title": None,
        "payload": {
            "widget_id": "w-1",
            "session_id": "conv-1",
            "status": "active",
            "version": 1,
            "connection_endpoint": "/widgets/w-1/connection",
        },
    }

    [item] = select_transient_upsert_items([widget])

    assert "title" not in item
    assert item["provenance"] == {}
    assert item["source"] == "widget_tool"


def test_image_record_rejects_append_display_policy():
    with pytest.raises(ValidationError):
        ImageRichItem(
            id="image:tool:call-1:0",
            type=RichItemType.image,
            display_policy=RichDisplayPolicy.inline_or_append,
            alt_text="Example image",
            payload={"url": "https://img.test/a.png", "mime_type": "image/png"},
        )


# ---------------------------------------------------------------------------
# Validation warnings
# ---------------------------------------------------------------------------


def test_validate_rich_references_flags_unknown_ids():
    body = "Intro\n\n<!--rich:image:missing-->\n\nEnd"
    warnings = validate_rich_references(body, items=[])
    assert any(w["code"] == "unknown_rich_item" and w["id"] == "image:missing" for w in warnings)


def test_validate_rich_references_returns_no_warnings_for_resolved_ids():
    image = _make_image("image:tool:c1:0")
    body = "Intro\n\n<!--rich:image:tool:c1:0-->\n\nEnd"
    assert validate_rich_references(body, items=[image]) == []


# ---------------------------------------------------------------------------
# Inventory budget
# ---------------------------------------------------------------------------


def test_inventory_omits_payload_data_and_respects_budget():
    image = ImageRichItem(
        id="image:document:1",
        type=RichItemType.image,
        display_policy=RichDisplayPolicy.inline_only,
        alt_text="Document image",
        title="A" * 300,
        payload={"data": "QUJDRA==", "mime_type": "image/png"},
    )
    # Single image item: image_max_items is generous so it never interferes
    # with the char-budget behavior this test actually exercises.
    block = build_rich_item_inventory_block(
        [image], max_items=1, max_chars=220, summary_chars=30, image_max_items=10
    )
    assert "QUJDRA==" not in block
    assert len(block) <= 220


def test_inventory_truncates_summary_to_summary_chars():
    image = ImageRichItem(
        id="image:document:2",
        type=RichItemType.image,
        display_policy=RichDisplayPolicy.inline_only,
        alt_text="Document image",
        payload={
            "url": "https://img.test/b.png",
            "mime_type": "image/png",
            "description": "Z" * 400,
        },
    )
    # Single image item: image_max_items is generous so it never interferes
    # with the summary-truncation behavior this test actually exercises.
    block = build_rich_item_inventory_block(
        [image], max_items=1, max_chars=2400, summary_chars=20, image_max_items=10
    )
    # The 'Z' repeat should not appear in full because of summary truncation.
    assert "Z" * 21 not in block


def test_inventory_prefers_non_image_items_when_trimming():
    widget = _make_widget()
    images = [_make_image(f"image:tool:c1:{i}") for i in range(20)]
    # image_max_items is set to the full candidate count so the image cap
    # itself never trims the list; this test exercises max_items ordering
    # (non-image survives ahead of image candidates), not the image cap.
    block = build_rich_item_inventory_block(
        [*images, widget],
        max_items=1,
        max_chars=2400,
        summary_chars=80,
        image_max_items=len(images),
    )
    assert widget.id in block


def test_inventory_block_is_empty_when_no_items():
    assert (
        build_rich_item_inventory_block(
            [], max_items=12, max_chars=2400, summary_chars=180, image_max_items=2
        )
        == ""
    )


# ---------------------------------------------------------------------------
# Schema-level validation
# ---------------------------------------------------------------------------


def test_image_payload_requires_exactly_one_source():
    with pytest.raises(ValidationError):
        ImageRichItem(
            id="image:invalid",
            type=RichItemType.image,
            display_policy=RichDisplayPolicy.inline_only,
            alt_text="Bad",
            payload={"mime_type": "image/png"},
        )


def test_image_payload_rejects_both_url_and_data():
    with pytest.raises(ValidationError):
        ImageRichItem(
            id="image:invalid",
            type=RichItemType.image,
            display_policy=RichDisplayPolicy.inline_only,
            alt_text="Bad",
            payload={"url": "https://img.test/x.png", "data": "QUJDRA==", "mime_type": "image/png"},
        )


def test_image_payload_rejects_unknown_mime_category():
    with pytest.raises(ValidationError):
        ImageRichItem(
            id="image:invalid",
            type=RichItemType.image,
            display_policy=RichDisplayPolicy.inline_only,
            alt_text="Bad",
            payload={"url": "https://img.test/x.exe", "mime_type": "application/octet-stream"},
        )


def test_image_payload_rejects_invalid_url_scheme():
    with pytest.raises(ValidationError):
        ImageRichItem(
            id="image:invalid",
            type=RichItemType.image,
            display_policy=RichDisplayPolicy.inline_only,
            alt_text="Bad",
            payload={"url": "ftp://img.test/x.png", "mime_type": "image/png"},
        )


def test_shared_mime_and_url_validation_helper_is_reused_by_both_models():
    """ImagePayload._has_exactly_one_source and ImageGroupItem._validate_cell
    duplicated the mime-type check and the two URL validation calls
    line-for-line. Both must delegate to one shared helper instead."""
    from app.core.rich_response import _validate_image_mime_and_urls

    # Valid input never raises.
    _validate_image_mime_and_urls("image/jpeg", "https://img.test/a.jpg", "https://img.test/src")

    with pytest.raises(ValueError, match="unsupported image mime_type"):
        _validate_image_mime_and_urls("image/svg+xml", "https://img.test/a.svg", None)

    with pytest.raises(ValueError, match="unsupported url scheme"):
        _validate_image_mime_and_urls("image/png", "ftp://img.test/a.png", None)

    with pytest.raises(ValueError, match="unsupported url scheme"):
        _validate_image_mime_and_urls("image/png", "https://img.test/a.png", "ftp://img.test/src")


# ---------------------------------------------------------------------------
# image_group schema
# ---------------------------------------------------------------------------


def _group(cells):
    return {
        "id": "imagegroup:tool:call_1",
        "type": "image_group",
        "source": "image_search",
        "display_policy": "inline_only",
        "alt_text": "Photos of a red panda",
        "payload": {"items": cells},
        "provenance": {"provider": "brave_image_search", "query": "red panda photo"},
    }


def _cell(url="https://e.com/a.jpg"):
    return {"url": url, "mime_type": "image/jpeg"}


def test_image_group_validates_with_two_cells():
    item = validate_public_rich_item(_group([_cell(), _cell("https://e.com/b.jpg")]))
    assert item.type.value == "image_group"
    assert len(item.payload.items) == 2


def test_image_group_rejects_fewer_than_two_cells():
    with pytest.raises(ValidationError):
        validate_public_rich_item(_group([_cell()]))


def test_image_group_rejects_unsupported_mime():
    with pytest.raises(ValidationError):
        validate_public_rich_item(
            _group([{"url": "https://e.com/a.svg", "mime_type": "image/svg+xml"}, _cell()])
        )


def test_image_group_accepts_protected_relative_cell_url():
    item = validate_public_rich_item(
        _group([{"url": "/web-images/abc", "mime_type": "image/jpeg"}, _cell()])
    )
    assert item.payload.items[0].url == "/web-images/abc"


def test_image_group_rejects_unknown_payload_field():
    bad = _group([_cell(), _cell("https://e.com/b.jpg")])
    bad["payload"]["carousel"] = True
    with pytest.raises(ValidationError):
        validate_public_rich_item(bad)
