"""Regression test for AI SDK message history context-window contract.

This test pins the contract that ``GET .../messages`` (handled by
``app.api.ai_sdk.get_conversation_messages_ai_sdk``) exposes the persisted
``context_window`` metadata on assistant messages under BOTH:

- ``messageMetadata`` (the canonical AI SDK ``UIMessage`` field, used by
  ``useChat()`` for client-side metadata access), and
- ``metadata``       (a mirrored generic key kept for UI compatibility with
  components that read a flat ``metadata`` blob).

The mirroring is performed by ``get_conversation_messages_ai_sdk`` at the
point where ``msg.message_metadata`` is copied onto the response payload.
Breaking either key would silently strip context-window data from the
frontend's history view, so we lock both down here.

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


def test_ai_sdk_messages_route_signature_exposes_pagination_dependency():
    signature = inspect.signature(get_conversation_messages_ai_sdk)

    pagination_param = signature.parameters["pagination"]

    assert pagination_param.annotation is MessagePaginationParams
    assert isinstance(pagination_param.default, Depends)


def test_ai_sdk_messages_expose_context_window_on_both_metadata_keys():
    """``context_window`` persisted on assistant messages must round-trip
    through the AI SDK message history endpoint under both
    ``messageMetadata`` and ``metadata``.

    This is the contract the Next.js client relies on to render the
    context-window indicator from the existing history view. Removing the
    mirror, filtering the metadata dict, or dropping the ``context_window``
    sub-key would break the indicator silently — hence this regression test.
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

    # ``messageMetadata`` is exposed via the alias on ``AISDKUIMessage``.
    # ``model_dump(by_alias=True)`` materializes the camelCase wire shape the
    # Next.js client actually consumes.
    payload = ui_message.model_dump(by_alias=True)

    assert "messageMetadata" in payload, (
        "AI SDK response must expose `messageMetadata` (canonical UIMessage field)"
    )
    assert "metadata" in payload, (
        "AI SDK response must also mirror metadata to `metadata` for UI compatibility"
    )

    expected_cw = assistant_msg.message_metadata["context_window"]

    assert payload["messageMetadata"]["context_window"] == expected_cw
    assert payload["metadata"]["context_window"] == expected_cw

    # Spot-check critical sub-fields explicitly so a future refactor that
    # subtly mangles values (rather than dropping the key entirely) still
    # trips this test.
    cw_via_message_metadata = payload["messageMetadata"]["context_window"]
    assert cw_via_message_metadata["known"] is True
    assert cw_via_message_metadata["context_window_tokens"] == 128000
    assert cw_via_message_metadata["used_tokens"] == 12000
    assert cw_via_message_metadata["display_state"] == "ok"


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
    assert response.data.total == 9
    assert response.data.meta.total == 9
    assert response.data.meta.current_page == 2
    assert response.data.meta.per_page == 1


def test_ai_sdk_messages_metadata_keys_share_identity_for_assistant_messages():
    """The two metadata keys must reference the same persisted dict — i.e.
    ``metadata`` is a true mirror, not a filtered/stripped subset. Future
    refactors that only forward a curated allowlist would break the
    context-window indicator's access to fields it doesn't know about yet.
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

    assert payload["messageMetadata"] == payload["metadata"], (
        "`metadata` must be a full mirror of `messageMetadata`, not a subset"
    )
    # And both must equal the originally persisted metadata dict (the
    # endpoint must not strip or rewrite fields).
    assert payload["messageMetadata"] == assistant_msg.message_metadata


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
    assert "rich_items" not in payload["messageMetadata"]
    assert "rich_items_version" not in payload["messageMetadata"]
    assert "rich_reference_warnings" not in payload["messageMetadata"]
    assert payload["metadata"] == payload["messageMetadata"]


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
    assert payload["messageMetadata"]["rich_items_version"] == 1
    assert payload["messageMetadata"]["rich_items"][0]["id"] == "widget:w-1"
    assert payload["metadata"] == payload["messageMetadata"]
