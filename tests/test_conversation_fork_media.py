"""Media references in forked history are kept only when the owner can still read them.

The fork repository asks the database which chat images and *selected* web
images the source owner still owns, then this pure policy drops every other
reference. A dropped inline image keeps its marker: the renderer already shows
an unknown marker as an unavailable block, so the surrounding text survives.
"""

from __future__ import annotations

import copy
import uuid

from app.services.conversation_fork_metadata import (
    CHAT_IMAGE,
    WEB_IMAGE,
    drop_unavailable_media,
    media_references,
)

CHAT_OK = uuid.UUID("0d4e7a52-8b6f-4c1e-9f0a-5b2c3d4e5f60")
CHAT_GONE = uuid.UUID("1d4e7a52-8b6f-4c1e-9f0a-5b2c3d4e5f61")
WEB_OK = uuid.UUID("6c1f0b5e-3d6a-4bb3-9a43-0b1f6c2d9e11")
WEB_PENDING = uuid.UUID("7c1f0b5e-3d6a-4bb3-9a43-0b1f6c2d9e12")


def _image_item(item_id: str, url: str) -> dict:
    return {
        "id": item_id,
        "type": "image",
        "display": "inline_or_append",
        "payload": {"url": url, "mime_type": "image/png"},
    }


def _group_item(item_id: str, *urls: str) -> dict:
    return {
        "id": item_id,
        "type": "image_group",
        "display": "inline_or_append",
        "payload": {"items": [{"url": url, "mime_type": "image/png"} for url in urls]},
    }


def _metadata() -> dict:
    return {
        "images": [
            {"image_id": str(CHAT_OK), "url": f"/chat-images/{CHAT_OK}", "mime": "image/png"},
            {"image_id": str(CHAT_GONE), "url": f"/chat-images/{CHAT_GONE}"},
            {"url": "https://example.com/remote.png"},
        ],
        "attachments": [
            {"image_id": str(CHAT_GONE), "url": f"/api/chat-images/{CHAT_GONE}"},
            {"name": "inline", "data": "aGVsbG8=", "mime": "image/png"},
        ],
        "rich_items": [
            _image_item("img-ok", f"/web-images/{WEB_OK}"),
            _image_item("img-pending", f"/web-images/{WEB_PENDING}?v=1"),
            _group_item("grp", f"/web-images/{WEB_OK}", f"/web-images/{WEB_PENDING}"),
            _group_item("grp-gone", f"/web-images/{WEB_PENDING}", f"/chat-images/{CHAT_GONE}"),
            {"id": "cite-1", "type": "citation", "payload": {"url": "/web-images/not-a-uuid"}},
        ],
        "rich_items_version": 1,
    }


def test_references_are_collected_from_every_image_shape():
    references = media_references(_metadata())

    assert references == {
        (CHAT_IMAGE, CHAT_OK),
        (CHAT_IMAGE, CHAT_GONE),
        (WEB_IMAGE, WEB_OK),
        (WEB_IMAGE, WEB_PENDING),
    }


def test_references_ignore_remote_urls_malformed_ids_and_non_dict_metadata():
    assert media_references(None) == set()
    assert media_references({"images": "nope", "rich_items": [None, 3]}) == set()
    assert media_references({"images": [{"url": "/chat-images/xyz"}]}) == set()


def test_unavailable_references_are_dropped_and_available_ones_kept():
    content = "Look <!--rich:img-ok--> and <!--rich:img-pending--> and <!--rich:grp-gone-->"
    available = {(CHAT_IMAGE, CHAT_OK), (WEB_IMAGE, WEB_OK)}

    result = drop_unavailable_media(content, _metadata(), available=available)

    assert [entry.get("image_id") for entry in result["images"]] == [str(CHAT_OK), None]
    assert result["attachments"] == [{"name": "inline", "data": "aGVsbG8=", "mime": "image/png"}]
    kept_ids = [item["id"] for item in result["rich_items"]]
    assert kept_ids == ["img-ok", "grp", "cite-1"]
    group = next(item for item in result["rich_items"] if item["id"] == "grp")
    assert [cell["url"] for cell in group["payload"]["items"]] == [f"/web-images/{WEB_OK}"]
    # Markers of dropped items stay; the renderer shows them as unavailable.
    assert {warning["id"] for warning in result["rich_reference_warnings"]} == {
        "img-pending",
        "grp-gone",
    }


def test_dropping_media_never_mutates_the_input():
    metadata = _metadata()
    before = copy.deepcopy(metadata)

    drop_unavailable_media("text", metadata, available=set())

    assert metadata == before


def test_metadata_without_unavailable_media_is_returned_unchanged():
    metadata = {"images": [{"url": f"/chat-images/{CHAT_OK}"}], "model": "m"}

    result = drop_unavailable_media("text", metadata, available={(CHAT_IMAGE, CHAT_OK)})

    assert result == metadata
    assert "rich_reference_warnings" not in result


def test_empty_media_lists_are_removed_rather_than_left_empty():
    metadata = {
        "images": [{"image_id": str(CHAT_GONE)}],
        "rich_items": [_image_item("img", f"/chat-images/{CHAT_GONE}")],
    }

    result = drop_unavailable_media("<!--rich:img-->", metadata, available=set())

    assert "images" not in result
    assert result["rich_items"] == []
    assert result["rich_reference_warnings"] == [{"code": "unknown_rich_item", "id": "img"}]
