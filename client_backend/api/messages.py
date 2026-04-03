"""
Message and streaming proxy endpoints for the local client backend.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse

from client_backend.api.common import add_device_context, raise_server_error
from client_backend.core.auth import require_local_session
from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.security import LocalSessionPayload
from client_backend.services.runtime_bridge import get_runtime_bridge
from client_backend.services.server_api import get_server_client
from client_backend.services.upstream_auth import get_upstream_auth_service

router = APIRouter(tags=["messages"])
ai_sdk_router = APIRouter(tags=["messages"])
logger = get_logger(__name__)


def _build_sse_response(
    event_source: AsyncIterator[dict[str, Any]],
    *,
    ai_sdk: bool = False,
) -> StreamingResponse:
    async def event_generator():
        try:
            async for event in event_source:
                if "raw" in event and len(event) == 1:
                    yield f"data: {event['raw']}\n\n"
                else:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            if ai_sdk:
                yield f"data: {json.dumps({'type': 'error', 'errorText': str(exc)})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        finally:
            if ai_sdk:
                yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    if ai_sdk:
        headers["x-vercel-ai-ui-message-stream"] = "v1"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


async def _ensure_runtime_bridge_for_message_flow() -> None:
    """
    Best-effort runtime reconnection for message/HITL flows.

    The UI can keep using a restored access token after the client backend
    restarts, so message routes must re-establish the device runtime before
    forwarding chat or resume requests when possible.
    """
    auth_service = get_upstream_auth_service()
    if not auth_service.is_authenticated():
        return

    bridge = get_runtime_bridge()
    if bridge.is_connected():
        return

    try:
        started = await bridge.start(wait_for_connection=False)
        if not started:
            return

        await bridge.start(
            wait_for_connection=True,
            timeout_seconds=min(5, client_settings.server_api_timeout_seconds),
        )
    except Exception as exc:
        logger.warning("Failed to pre-connect runtime bridge for message flow: %s", exc)


async def _build_upstream_json_response(
    *,
    method: str,
    path: str,
    **kwargs: Any,
) -> Response:
    response = await get_server_client().request_response(method, path, **kwargs)
    content_type = response.headers.get("content-type", "").lower()
    if content_type.startswith("application/json"):
        return JSONResponse(status_code=response.status_code, content=response.json())

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
    )


@router.post("/messages")
async def create_message(
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    """Create a non-streaming message."""
    try:
        await _ensure_runtime_bridge_for_message_flow()
        normalized_payload = add_device_context(payload)
        return await _build_upstream_json_response(
            method="POST",
            path="/messages/",
            json=normalized_payload,
        )
    except Exception as exc:
        raise_server_error(exc)


@router.post("/messages/stream")
async def create_message_stream(
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Proxy the server's internal message SSE stream."""
    await _ensure_runtime_bridge_for_message_flow()
    normalized_payload = add_device_context(payload)
    return _build_sse_response(get_server_client().stream_internal_message(normalized_payload))


@router.post("/messages/resume-interrupt")
async def resume_interrupt(
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Resume an interrupted internal message stream."""
    await _ensure_runtime_bridge_for_message_flow()
    normalized_payload = add_device_context(payload)
    return _build_sse_response(get_server_client().resume_interrupt(normalized_payload))


@router.post("/messages/stop")
async def stop_generation(
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Proxy a stop-generation request."""
    try:
        normalized_payload = add_device_context(payload)
        return await get_server_client().post("/messages/stop", json=normalized_payload)
    except Exception as exc:
        raise_server_error(exc)


@router.get("/messages/{message_id}")
async def get_message(
    message_id: str,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Fetch a specific message."""
    try:
        return await get_server_client().get_message(message_id)
    except Exception as exc:
        raise_server_error(exc)


@router.get("/messages")
async def list_messages(
    page: int = 1,
    limit: int = 50,
    include: list[str] = Query(default=[]),  # noqa: B008
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """List user messages."""
    try:
        return await get_server_client().get(
            "/messages/",
            params={"page": page, "limit": limit, "include": include},
        )
    except Exception as exc:
        raise_server_error(exc)


@ai_sdk_router.post("/api/chat/{conversation_id}")
async def ai_sdk_chat(
    conversation_id: str,
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Proxy the AI SDK UI message stream."""
    await _ensure_runtime_bridge_for_message_flow()
    normalized_payload = add_device_context(payload)
    return _build_sse_response(
        get_server_client().stream_ai_sdk_chat(conversation_id, normalized_payload),
        ai_sdk=True,
    )


@ai_sdk_router.post("/ai/chat/{conversation_id}", include_in_schema=False)
async def ai_sdk_chat_alias(
    conversation_id: str,
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Alias of the AI SDK UI message stream endpoint."""
    await _ensure_runtime_bridge_for_message_flow()
    normalized_payload = add_device_context(payload)
    return _build_sse_response(
        get_server_client().stream_ai_sdk_chat(conversation_id, normalized_payload),
        ai_sdk=True,
    )


@ai_sdk_router.post("/ai/resume-interrupt")
async def ai_sdk_resume_interrupt(
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Resume an interrupted AI SDK stream."""
    await _ensure_runtime_bridge_for_message_flow()
    normalized_payload = add_device_context(payload)
    return _build_sse_response(
        get_server_client().resume_ai_sdk_interrupt(normalized_payload),
        ai_sdk=True,
    )
