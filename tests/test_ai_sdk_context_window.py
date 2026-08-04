"""Regression test for AI SDK message history context-window contract.

This test pins the contract that ``GET .../messages`` (handled by
``app.api.ai_sdk.get_conversation_messages_ai_sdk``) exposes the persisted
``context_window`` metadata on assistant messages under ``metadata`` — the
canonical AI SDK ``UIMessage`` field. The legacy ``messageMetadata`` mirror
was removed in the 2026-07-02 response-format cleanup and must stay absent.

Scope note — Live streaming context-window updates (i.e. emitting
context-window deltas mid-SSE on the chat stream endpoint) are explicitly
OUT OF SCOPE for this feature and are deferred as a future extension. This
contract only covers the post-completion message history endpoint.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.params import Depends

from app.api.ai_sdk import get_conversation_messages_ai_sdk
from app.repositories.utils.pagination import PaginationMeta
from app.schemas.pagination import MessagePaginationParams
from app.services.event_streaming.ai_sdk_projection import (
    selected_image_file_parts_from_rich_items,
)


def _build_assistant_message_with_context_window() -> SimpleNamespace:
    """Construct a fake persisted assistant message whose ``message_metadata``
    carries a fully-populated ``context_window`` block (the same shape
    produced by ``BaseAgent._merge_context_window_usage`` on completion)."""

    return SimpleNamespace(
        id=uuid4(),
        sender=2,  # assistant
        content="hello from the model",
        created_at=datetime.now(UTC),
        message_metadata={
            "provider": "openai",
            "model": "gpt-4o",
            "context_window": {
                "provider": "openai",
                "model": "gpt-4o",
                "context_window_tokens": 128000,
                "max_input_tokens": 128000,
                "max_output_tokens": 16384,
                "source": "registry",
                "known": True,
                "used_tokens": 12000,
                "used_token_source": "actual_input",
                "usage_ratio": 12000 / 128000,
                "display_state": "ok",
            },
        },
    )


def _build_assistant_message_with_rich_items() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        sender=2,
        content="Intro\n\n<!--rich:widget:w-1-->\n\nDone",
        created_at=datetime.now(UTC),
        message_metadata={
            "rich_items_version": 1,
            "rich_items": [
                {
                    "id": "widget:w-1",
                    "type": "live_widget",
                    "display_policy": "inline_or_append",
                    "payload": {
                        "widget_id": "w-1",
                        "session_id": "conversation-1",
                        "widget_type": "table",
                        "status": "active",
                        "version": 1,
                        "connection_endpoint": "/widgets/w-1/connection",
                    },
                }
            ],
            "rich_reference_warnings": [],
        },
    )


def _build_assistant_message_with_selected_image() -> SimpleNamespace:
    image_url = "/web-images/11111111-1111-4111-8111-111111111111"
    return SimpleNamespace(
        id=uuid4(),
        sender=2,
        content="Intro\n\n<!--rich:image:tool:c1:0-->\n\nDone",
        created_at=datetime.now(UTC),
        message_metadata={
            "rich_items_version": 1,
            "rich_items": [
                {
                    "id": "image:tool:c1:0",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Selected",
                    "payload": {"url": image_url, "mime_type": "image/jpeg"},
                    "provenance": {
                        "provider": "tavily",
                        "original_image_url": "https://img.test/original.jpg",
                    },
                }
            ],
            "rich_reference_warnings": [],
        },
    )


def _build_paginated_result(
    items: list[SimpleNamespace],
    *,
    total: int | None = None,
    page: int = 1,
    limit: int = 100,
) -> SimpleNamespace:
    return SimpleNamespace(
        items=items,
        meta=PaginationMeta.calculate(total or len(items), limit, page),
    )


def _build_message_service(
    items: list[SimpleNamespace],
    *,
    total: int | None = None,
    page: int = 1,
    limit: int = 100,
) -> MagicMock:
    service = MagicMock()
    service.get_conversation_messages.return_value = _build_paginated_result(
        items,
        total=total,
        page=page,
        limit=limit,
    )
    return service


def _history_payload(message: SimpleNamespace, *, capable: bool) -> dict:
    response = asyncio.run(
        get_conversation_messages_ai_sdk(
            uuid4(),
            _build_message_service([message]),
            uuid4(),
            MessagePaginationParams(),
            inline_rich_response_v1=capable,
        )
    )
    return response.data.messages[0].model_dump(by_alias=True)


def test_ai_sdk_messages_route_signature_exposes_pagination_dependency():
    signature = inspect.signature(get_conversation_messages_ai_sdk)

    pagination_param = signature.parameters["pagination"]

    assert pagination_param.annotation is MessagePaginationParams
    assert isinstance(pagination_param.default, Depends)


def test_ai_sdk_messages_expose_context_window_on_metadata():
    """``context_window`` persisted on assistant messages must round-trip
    through the AI SDK message history endpoint under ``metadata``.

    This is the contract the Next.js client relies on to render the
    context-window indicator from the existing history view. Filtering the
    metadata dict or dropping the ``context_window`` sub-key would break the
    indicator silently — hence this regression test.
    """

    assistant_msg = _build_assistant_message_with_context_window()
    message_service = _build_message_service([assistant_msg])

    conversation_id = uuid4()
    current_user_id = uuid4()

    response = asyncio.run(
        get_conversation_messages_ai_sdk(
            conversation_id,
            message_service,
            current_user_id,
            MessagePaginationParams(),
        )
    )

    message_service.get_conversation_messages.assert_called_once()

    assert response.success is True
    assert response.data is not None
    assert len(response.data.messages) == 1
    assert response.data.meta.total == 1
    assert response.data.meta.current_page == 1

    ui_message = response.data.messages[0]
    assert ui_message.role == "assistant"

    # ``model_dump(by_alias=True)`` materializes the camelCase wire shape the
    # Next.js client actually consumes.
    payload = ui_message.model_dump(by_alias=True)

    assert "metadata" in payload, (
        "AI SDK response must expose `metadata` (canonical UIMessage field)"
    )
    assert "messageMetadata" not in payload, (
        "the legacy `messageMetadata` mirror must not come back"
    )

    cw = payload["metadata"]["context_window"]
    assert cw == assistant_msg.message_metadata["context_window"]

    # Spot-check critical sub-fields explicitly so a future refactor that
    # subtly mangles values (rather than dropping the key entirely) still
    # trips this test.
    assert cw["known"] is True
    assert cw["context_window_tokens"] == 128000
    assert cw["used_tokens"] == 12000
    assert cw["display_state"] == "ok"


def test_ai_sdk_messages_forwards_pagination_to_message_service():
    assistant_msg = _build_assistant_message_with_context_window()
    message_service = _build_message_service([assistant_msg], total=9, page=2, limit=1)

    conversation_id = uuid4()
    current_user_id = uuid4()
    pagination = MessagePaginationParams(
        page=2,
        limit=1,
        orderBy="updatedAt",
        orderDirection="desc",
    )

    response = asyncio.run(
        get_conversation_messages_ai_sdk(
            conversation_id,
            message_service,
            current_user_id,
            pagination,
        )
    )

    message_service.get_conversation_messages.assert_called_once_with(
        conversation_id,
        current_user_id,
        page=2,
        limit=1,
        order_by="updated_at",
        order_direction="desc",
        include_feedback=False,
    )
    assert response.data.meta.total == 9
    assert response.data.meta.current_page == 2
    assert response.data.meta.per_page == 1


def test_ai_sdk_messages_preserve_unknown_metadata_fields():
    """``metadata`` must pass through persisted fields the endpoint does not
    know about (only the legacy renderer fields are scrubbed). A curated
    allowlist would break the context-window indicator's access to fields it
    doesn't know about yet.
    """

    assistant_msg = _build_assistant_message_with_context_window()
    message_service = _build_message_service([assistant_msg])

    response = asyncio.run(
        get_conversation_messages_ai_sdk(
            uuid4(),
            message_service,
            uuid4(),
            MessagePaginationParams(),
        )
    )

    ui_message = response.data.messages[0]
    payload = ui_message.model_dump(by_alias=True)

    assert payload["metadata"] == assistant_msg.message_metadata


def test_ai_sdk_messages_scrub_legacy_renderer_fields():
    """Legacy renderer fields must not reach AI SDK clients; images surface
    as ``file`` parts instead, and a leading ``text`` part is guaranteed."""

    now = datetime.now(UTC)
    assistant_msg = SimpleNamespace(
        id=uuid4(),
        sender=2,
        content="Answer with an image.",
        created_at=now,
        message_metadata={
            "provider": "openai",
            "images": [{"url": "https://img.test/a.png", "mime": "image/png"}],
            "has_images": True,
            "images_count": 1,
            "agentic_images_count": 1,
            "live_widgets": [{"widget_id": "w-1"}],
            "canvas_artifact": {"content": "<html></html>", "language": "html"},
            "pending_tool_calls": [],
            "_rich_item_candidates": [],
            "conversation_id": "conversation-1",
            "has_tool_calls": True,
            "context_messages": 5,
        },
    )
    user_msg = SimpleNamespace(
        id=uuid4(),
        sender=1,
        content="show me",
        created_at=now,
        message_metadata=None,
    )
    message_service = _build_message_service([user_msg, assistant_msg])

    response = asyncio.run(
        get_conversation_messages_ai_sdk(
            uuid4(),
            message_service,
            uuid4(),
            MessagePaginationParams(),
        )
    )

    user_payload = response.data.messages[0].model_dump(by_alias=True)
    assistant_payload = response.data.messages[1].model_dump(by_alias=True)

    assert assistant_payload["metadata"] == {"provider": "openai"}
    assert assistant_payload["parts"][0] == {
        "type": "text",
        "text": "Answer with an image.",
    }
    assert {
        "type": "file",
        "url": "https://img.test/a.png",
        "mediaType": "image/png",
    } in assistant_payload["parts"]
    assert user_payload["parts"] == [{"type": "text", "text": "show me"}]


def test_ai_sdk_messages_hide_v1_image_candidates_without_capability():
    """A v1 message must never fall back to legacy ``images`` for file parts,
    even for non-capable clients where the capability projection has already
    stripped ``rich_items_version`` (regression: hidden candidates leaked as
    file parts through exactly that ordering)."""

    assistant_msg = _build_assistant_message_with_rich_items()
    assistant_msg.message_metadata = {
        **assistant_msg.message_metadata,
        "images": [{"url": "https://img.test/hidden-candidate.png", "mime": "image/png"}],
    }
    message_service = _build_message_service([assistant_msg])

    response = asyncio.run(
        get_conversation_messages_ai_sdk(
            uuid4(),
            message_service,
            uuid4(),
            MessagePaginationParams(),
        )
    )

    payload = response.data.messages[0].model_dump(by_alias=True)

    assert all(part["type"] != "file" for part in payload["parts"])
    assert "images" not in payload["metadata"]


def test_ai_sdk_messages_strip_rich_v1_fields_without_capability():
    assistant_msg = _build_assistant_message_with_rich_items()
    message_service = _build_message_service([assistant_msg])

    response = asyncio.run(
        get_conversation_messages_ai_sdk(
            uuid4(),
            message_service,
            uuid4(),
            MessagePaginationParams(),
        )
    )

    payload = response.data.messages[0].model_dump(by_alias=True)

    assert "<!--rich:" not in payload["content"]
    assert "rich_items" not in payload["metadata"]
    assert "rich_items_version" not in payload["metadata"]
    assert "rich_reference_warnings" not in payload["metadata"]


def test_ai_sdk_messages_preserve_rich_v1_fields_with_capability():
    assistant_msg = _build_assistant_message_with_rich_items()
    message_service = _build_message_service([assistant_msg])

    response = asyncio.run(
        get_conversation_messages_ai_sdk(
            uuid4(),
            message_service,
            uuid4(),
            MessagePaginationParams(),
            inline_rich_response_v1=True,
        )
    )

    payload = response.data.messages[0].model_dump(by_alias=True)

    assert "<!--rich:widget:w-1-->" in payload["content"]
    assert payload["metadata"]["rich_items_version"] == 1
    assert payload["metadata"]["rich_items"][0]["id"] == "widget:w-1"


def test_ai_sdk_rich_history_uses_image_rich_item_without_file_part():
    payload = _history_payload(_build_assistant_message_with_selected_image(), capable=True)

    assert payload["metadata"]["rich_items"][0]["type"] == "image"
    assert "original_image_url" not in payload["metadata"]["rich_items"][0]["provenance"]
    assert all(part["type"] != "file" for part in payload["parts"])


def test_ai_sdk_non_rich_history_projects_selected_image_once():
    payload = _history_payload(_build_assistant_message_with_selected_image(), capable=False)

    assert "rich_items" not in payload["metadata"]
    assert "<!--rich:" not in payload["content"]
    files = [part for part in payload["parts"] if part["type"] == "file"]
    assert len(files) == 1
    assert files[0]["url"].startswith("/web-images/")


def test_group_cells_become_file_parts_in_order():
    metadata = {
        "rich_items_version": 1,
        "rich_items": [
            {
                "id": "imagegroup:tool:c1",
                "type": "image_group",
                "payload": {
                    "items": [
                        {"url": "/web-images/1", "mime_type": "image/jpeg"},
                        {"url": "/web-images/2", "mime_type": "image/png"},
                    ]
                },
            }
        ],
    }
    parts = selected_image_file_parts_from_rich_items(metadata)
    assert [p["url"] for p in parts] == ["/web-images/1", "/web-images/2"]
    assert [p["mediaType"] for p in parts] == ["image/jpeg", "image/png"]


def _v1_selected_image_message():
    return {
        "content": "Intro\n\n<!--rich:image:tool:c1:0-->",
        "metadata": {
            "rich_items_version": 1,
            "rich_items": [
                {
                    "id": "image:tool:c1:0",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Selected",
                    "payload": {
                        "url": "/web-images/11111111-1111-4111-8111-111111111111",
                        "mime_type": "image/jpeg",
                    },
                }
            ],
        },
    }


def test_rich_capable_v1_message_has_no_image_file_parts():
    from app.services.event_streaming.ai_sdk_projection import visible_image_file_parts

    assert (
        visible_image_file_parts(
            _v1_selected_image_message(),
            is_v1=True,
            inline_rich_response_v1=True,
        )
        == []
    )


def test_non_rich_v1_message_keeps_selected_image_file_parts():
    from app.services.event_streaming.ai_sdk_projection import visible_image_file_parts

    assert visible_image_file_parts(
        _v1_selected_image_message(),
        is_v1=True,
        inline_rich_response_v1=False,
    ) == [
        {
            "url": "/web-images/11111111-1111-4111-8111-111111111111",
            "mediaType": "image/jpeg",
        }
    ]


def test_explicit_empty_image_parts_prevents_rederivation():
    from app.services.event_streaming.ai_sdk_projection import attach_image_parts_to_message

    projected = attach_image_parts_to_message(
        _v1_selected_image_message(),
        image_parts=[],
        is_v1=True,
    )

    assert all(part.get("type") != "file" for part in projected.get("parts") or [])


def test_group_and_image_duplicates_are_deduplicated():
    metadata = {
        "rich_items_version": 1,
        "rich_items": [
            {
                "id": "image:tool:c1:0",
                "type": "image",
                "payload": {"url": "/web-images/1", "mime_type": "image/jpeg"},
            },
            {
                "id": "imagegroup:tool:c2",
                "type": "image_group",
                "payload": {"items": [{"url": "/web-images/1", "mime_type": "image/jpeg"}]},
            },
        ],
    }
    assert len(selected_image_file_parts_from_rich_items(metadata)) == 1


def test_unknown_rich_type_is_skipped_not_rendered():
    metadata = {
        "rich_items_version": 1,
        "rich_items": [{"id": "future:1", "type": "future_thing", "payload": {"x": 1}}],
    }
    assert selected_image_file_parts_from_rich_items(metadata) == []


def test_pre_v1_message_still_yields_image_file_parts():
    from app.services.event_streaming.ai_sdk_projection import visible_image_file_parts

    message = {
        "content": "old answer",
        "metadata": {"images": [{"url": "/chat-images/9", "mime_type": "image/png"}]},
    }
    assert visible_image_file_parts(message) != []
