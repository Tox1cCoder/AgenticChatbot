from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.api.ai_sdk import AISDKChatRequest, chat_ui_message_stream
from app.services.event_streaming.events import make_event


@pytest.mark.asyncio
async def test_chat_endpoint_returns_assistant_ui_stream():
    conversation_id = uuid4()
    user_id = uuid4()

    class FakeMessageService:
        def create_message_stream(self, *_args, **_kwargs):
            async def source():
                yield make_event("message_delta", sequence=1, data={"text": "hello"})
                yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

            return source()

    response = await chat_ui_message_stream(
        conversation_id=conversation_id,
        payload=AISDKChatRequest(
            messages=[{"role": "user", "parts": [{"type": "text", "text": "hello"}]}]
        ),
        message_service=FakeMessageService(),
        current_user_id=user_id,
    )

    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
    body = "".join([chunk async for chunk in response.body_iterator])
    payloads = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]

    assert payloads[0]["type"] == "start"
    assert any(payload.get("type") == "text-delta" for payload in payloads)
    assert payloads[-1]["type"] == "finish"
    assert body.rstrip().endswith("data: [DONE]")
