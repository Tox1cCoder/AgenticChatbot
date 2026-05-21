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
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from app.api.ai_sdk import get_conversation_messages_ai_sdk


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


def _build_paginated_result(items: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(items=items, meta=SimpleNamespace(total=len(items)))


def _build_message_service(items: list[SimpleNamespace]) -> MagicMock:
    service = MagicMock()
    service.get_conversation_messages.return_value = _build_paginated_result(items)
    return service


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
            conversation_id, message_service, current_user_id
        )
    )

    message_service.get_conversation_messages.assert_called_once()

    assert response.success is True
    assert response.data is not None
    assert len(response.data.messages) == 1

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


def test_ai_sdk_messages_metadata_keys_share_identity_for_assistant_messages():
    """The two metadata keys must reference the same persisted dict — i.e.
    ``metadata`` is a true mirror, not a filtered/stripped subset. Future
    refactors that only forward a curated allowlist would break the
    context-window indicator's access to fields it doesn't know about yet.
    """

    assistant_msg = _build_assistant_message_with_context_window()
    message_service = _build_message_service([assistant_msg])

    response = asyncio.run(
        get_conversation_messages_ai_sdk(uuid4(), message_service, uuid4())
    )

    ui_message = response.data.messages[0]
    payload = ui_message.model_dump(by_alias=True)

    assert payload["messageMetadata"] == payload["metadata"], (
        "`metadata` must be a full mirror of `messageMetadata`, not a subset"
    )
    # And both must equal the originally persisted metadata dict (the
    # endpoint must not strip or rewrite fields).
    assert payload["messageMetadata"] == assistant_msg.message_metadata
