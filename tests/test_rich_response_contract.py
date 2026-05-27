"""Tests for the inline rich-response contract: marker parser, payload validation,
display-policy filters, and bounded prompt inventory serialization.

See response_format.md Task 1.
"""

from __future__ import annotations

import pytest

from app.core.rich_response import (
    ImageRichItem,
    LiveWidgetRichItem,
    RichDisplayPolicy,
    RichItemType,
    build_rich_item_inventory_block,
    parse_inline_rich_references,
    select_append_fallback_items,
    select_transient_upsert_items,
    validate_rich_references,
)


def test_inline_rich_response_rollout_is_disabled_by_default(monkeypatch):
    from app.core.config import Settings

    monkeypatch.delenv("INLINE_RICH_RESPONSE_ENABLED", raising=False)

    assert Settings(_env_file=None).inline_rich_response_enabled is False


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


def _make_widget(id_: str = "widget:w-1") -> LiveWidgetRichItem:
    return LiveWidgetRichItem(
        id=id_,
        type=RichItemType.live_widget,
        display_policy=RichDisplayPolicy.inline_or_append,
        payload={
            "widget_id": "w-1",
            "session_id": "conv-1",
            "widget_type": "chart",
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
    assert (
        select_append_fallback_items([widget], referenced_ids={widget.id}) == []
    )


def test_transient_upserts_exclude_image_items_entirely():
    image = _make_image("image:generated:m-1:0")
    assert select_transient_upsert_items([image]) == []


def test_transient_upserts_include_safe_widget_records():
    widget = _make_widget()
    assert select_transient_upsert_items([widget]) == [widget]


def test_unselected_image_data_is_not_serialized_for_streaming():
    image = ImageRichItem(
        id="image:generated:m-1:0",
        type=RichItemType.image,
        display_policy=RichDisplayPolicy.inline_only,
        alt_text="Generated image",
        payload={"data": "QUJDRA==", "mime_type": "image/png"},
    )
    assert select_transient_upsert_items([image]) == []


def test_image_record_rejects_append_display_policy():
    with pytest.raises(Exception):
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
    block = build_rich_item_inventory_block(
        [image], max_items=1, max_chars=220, summary_chars=30
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
    block = build_rich_item_inventory_block(
        [image], max_items=1, max_chars=2400, summary_chars=20
    )
    # The 'Z' repeat should not appear in full because of summary truncation.
    assert "Z" * 21 not in block


def test_inventory_prefers_non_image_items_when_trimming():
    widget = _make_widget()
    images = [_make_image(f"image:tool:c1:{i}") for i in range(20)]
    block = build_rich_item_inventory_block(
        [*images, widget], max_items=1, max_chars=2400, summary_chars=80
    )
    assert widget.id in block


def test_inventory_block_is_empty_when_no_items():
    assert build_rich_item_inventory_block([], max_items=12, max_chars=2400, summary_chars=180) == ""


# ---------------------------------------------------------------------------
# Schema-level validation
# ---------------------------------------------------------------------------


def test_image_payload_requires_exactly_one_source():
    with pytest.raises(Exception):
        ImageRichItem(
            id="image:invalid",
            type=RichItemType.image,
            display_policy=RichDisplayPolicy.inline_only,
            alt_text="Bad",
            payload={"mime_type": "image/png"},
        )


def test_image_payload_rejects_both_url_and_data():
    with pytest.raises(Exception):
        ImageRichItem(
            id="image:invalid",
            type=RichItemType.image,
            display_policy=RichDisplayPolicy.inline_only,
            alt_text="Bad",
            payload={"url": "https://img.test/x.png", "data": "QUJDRA==", "mime_type": "image/png"},
        )


def test_image_payload_rejects_unknown_mime_category():
    with pytest.raises(Exception):
        ImageRichItem(
            id="image:invalid",
            type=RichItemType.image,
            display_policy=RichDisplayPolicy.inline_only,
            alt_text="Bad",
            payload={"url": "https://img.test/x.exe", "mime_type": "application/octet-stream"},
        )


def test_image_payload_rejects_invalid_url_scheme():
    with pytest.raises(Exception):
        ImageRichItem(
            id="image:invalid",
            type=RichItemType.image,
            display_policy=RichDisplayPolicy.inline_only,
            alt_text="Bad",
            payload={"url": "ftp://img.test/x.png", "mime_type": "image/png"},
        )
