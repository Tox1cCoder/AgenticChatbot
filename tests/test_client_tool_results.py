"""What a client tool hands the model, and the size contract with the sidecar.

The sidecar replies with MCP content blocks. Serializing them to a JSON string
before rendering made the model read ``[{"type": "text", ...}]`` with escaped
newlines, and an image's base64 as text. The size budgets travel with every
request so the sidecar caps a result before it crosses the bridge.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.ai.client_runtime_tools import _CLIENT_TOOL_CACHE, get_client_runtime_tools
from app.ai.tool_context import tool_execution_context
from app.ai.tool_result_rendering import normalize_tool_result_for_rendering
from app.core.config import settings
from app.schemas.runtime_protocol import RUNTIME_MAX_MESSAGE_BYTES
from app.services import client_runtime_store as runtime_store_module
from app.services.client_device_service import ClientDeviceService
from app.services.client_runtime_store import (
    DeviceSessionRecord,
    InMemoryClientRuntimeStore,
    reset_client_runtime_store,
)


@pytest.fixture
async def connected():
    reset_client_runtime_store()
    _CLIENT_TOOL_CACHE.clear()
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store
    user_id, device_id = uuid4(), uuid4()
    await store.put_session(
        DeviceSessionRecord(
            device_id=device_id,
            session_id="session-1",
            user_id=user_id,
            tool_catalog={
                "tools": [
                    {
                        "name": "read_file",
                        "origin": "mcp",
                        "server_name": "desktop-commander",
                        "qualified_id": "desktop-commander::read_file",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ]
            },
            tool_catalog_version=1,
        )
    )
    try:
        yield store, user_id, device_id
    finally:
        reset_client_runtime_store()
        _CLIENT_TOOL_CACHE.clear()


async def _call_tool(user_id, device_id):
    tool = get_client_runtime_tools(user_id=str(user_id), device_id=str(device_id))[0]
    with tool_execution_context(
        conversation_id="conversation-1",
        user_id=str(user_id),
        agent_key="chat",
        device_id=str(device_id),
    ):
        return tool, await tool.coroutine()


def _sidecar_replies(monkeypatch, result) -> None:
    async def dispatch(**_kwargs):
        return {"success": True, "result": result}

    monkeypatch.setattr(ClientDeviceService, "dispatch_tool_call", dispatch)


async def test_model_reads_the_tools_text_not_its_json_envelope(connected, monkeypatch):
    _, user_id, device_id = connected
    _sidecar_replies(monkeypatch, [{"type": "text", "text": 'line 1\nline "2"', "id": "lc_1"}])

    tool, result = await _call_tool(user_id, device_id)

    model_text = normalize_tool_result_for_rendering(result, tool_name=tool.name).model_content
    assert model_text == 'line 1\nline "2"'


async def test_model_does_not_read_image_data_as_text(connected, monkeypatch):
    _, user_id, device_id = connected
    image = {"type": "image", "data": "A" * 40_000, "mimeType": "image/png"}
    _sidecar_replies(monkeypatch, [image])

    tool, result = await _call_tool(user_id, device_id)

    model_text = normalize_tool_result_for_rendering(result, tool_name=tool.name).model_content
    assert "AAAA" not in model_text


async def test_result_over_the_limit_from_an_older_sidecar_is_capped(connected, monkeypatch):
    _, user_id, device_id = connected
    monkeypatch.setattr(settings, "client_runtime_max_tool_result_size_bytes", 2_000)
    _sidecar_replies(monkeypatch, [{"type": "text", "text": "x" * 50_000}])

    _, result = await _call_tool(user_id, device_id)

    assert len(json.dumps(result)) <= 2_000


async def _dispatch(user_id, device_id, arguments):
    return await ClientDeviceService.dispatch_tool_call(
        user_id=str(user_id),
        device_id=str(device_id),
        tool_name="read_file",
        qualified_tool_id="desktop-commander::read_file",
        arguments=arguments,
        execution_timeout_seconds=5,
        response_timeout_seconds=6,
        bound_session_id="session-1",
        bound_catalog_version=1,
    )


async def test_each_request_carries_the_result_budgets(connected, monkeypatch):
    store, user_id, device_id = connected
    monkeypatch.setattr(settings, "client_runtime_max_tool_result_size_bytes", 123_456)
    monkeypatch.setattr(settings, "client_runtime_max_tool_result_media_bytes", 654_321)
    sent = []

    async def capture(_session, request, _timeout):
        sent.append(request)
        return {"success": True, "result": "ok"}

    monkeypatch.setattr(store, "dispatch_request", capture)

    await _dispatch(user_id, device_id, {"path": "notes.txt"})

    assert sent[0].max_result_text_bytes == 123_456
    assert sent[0].max_result_media_bytes == 654_321


async def test_arguments_too_large_for_the_bridge_are_refused_before_dispatch(
    connected, monkeypatch
):
    store, user_id, device_id = connected
    sent = []

    async def capture(_session, request, _timeout):
        sent.append(request)
        return {"success": True, "result": "ok"}

    monkeypatch.setattr(store, "dispatch_request", capture)

    with pytest.raises(ValueError, match="too large"):
        await _dispatch(user_id, device_id, {"content": "x" * RUNTIME_MAX_MESSAGE_BYTES})

    assert sent == []
