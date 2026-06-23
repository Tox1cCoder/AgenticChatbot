"""
Widget HTTP and WebSocket endpoints.

POST /widgets/{widget_id}/connection  — mint a short-lived widget WS token
WS   /widgets/{widget_id}/connect     — real-time widget state stream
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import jwt as pyjwt
from fastapi import APIRouter, Body, Depends, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import Text, cast, select

from app.core.auth import get_current_user_id
from app.models.message import Message
from app.services.widget_contract import (
    SUPPORTED_WIDGET_TYPE,
    resolve_widget_action_message,
    validate_html_widget_state,
)
from app.services.widget_runtime import (
    get_widget_connection_manager,
    get_widget_store,
    get_widget_token_service,
)


class WidgetActionRequest(BaseModel):
    input_values: dict[str, Any] | None = Field(default=None)
    state_patch: dict[str, Any] | None = Field(default=None)


WidgetActionRequest.model_rebuild()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/widgets", tags=["widgets"])
WIDGET_POLL_INTERVAL_SECONDS = 0.5


def _get_container():
    from app.core.container import get_container

    return get_container()


def _parse_conversation_uuid(session_id: str | None) -> UUID | None:
    try:
        return UUID(str(session_id))
    except (TypeError, ValueError):
        return None


def _user_can_access_widget_session(user_id: UUID, session_id: str) -> bool:
    """Revalidate widget ownership against the backing conversation."""
    conversation_id = _parse_conversation_uuid(session_id)
    if conversation_id is None:
        logger.warning("Widget session_id is not a valid conversation UUID: %s", session_id)
        return False

    try:
        repository = _get_container().conversation_repository()
        return bool(repository.user_owns_conversation(user_id, conversation_id))
    except Exception:
        logger.warning(
            "Failed to validate widget conversation access for user %s / session %s",
            user_id,
            session_id,
            exc_info=True,
        )
        return False


def _message_metadata_contains_widget(metadata: Any, widget_id: str) -> bool:
    if not isinstance(metadata, dict):
        return False

    live_widgets = metadata.get("live_widgets")
    if isinstance(live_widgets, list):
        for widget in live_widgets:
            if isinstance(widget, dict) and str(widget.get("widget_id") or "") == widget_id:
                return True

    tool_artifacts = metadata.get("tool_artifacts")
    if not isinstance(tool_artifacts, list):
        return False

    for artifact in tool_artifacts:
        if not isinstance(artifact, dict):
            continue

        for key in ("output", "tool_output", "result"):
            value = artifact.get(key)
            if isinstance(value, dict) and str(value.get("widget_id") or "") == widget_id:
                return True
            if isinstance(value, str) and widget_id in value:
                return True

    return False


def _iter_widget_messages_for_user(user_id: UUID, widget_id: str) -> list[Message]:
    repository = _get_container().message_repository()
    session_factory = getattr(repository, "session_factory", None)
    if session_factory is None:
        return []

    with session_factory() as session:
        statement = (
            select(Message)
            .join(Message.conversation)
            .where(
                Message.conversation.has(owner_id=user_id),
                Message.message_metadata.is_not(None),
                cast(Message.message_metadata, Text).ilike(f"%{widget_id}%"),
            )
            .order_by(Message.created_at.desc())
            .limit(25)
        )
        return list(session.execute(statement).scalars().all())


def _recover_widget_session_id_from_messages(
    user_id: UUID,
    widget_id: str,
) -> str | None:
    """Recover a widget's conversation ID from persisted assistant message metadata."""
    try:
        for message in _iter_widget_messages_for_user(user_id, widget_id):
            if _message_metadata_contains_widget(
                getattr(message, "message_metadata", None),
                widget_id,
            ):
                return str(message.conversation_id)
    except Exception:
        logger.warning(
            "Failed to recover widget conversation from message metadata for widget %s",
            widget_id,
            exc_info=True,
        )

    return None


def _parse_json_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _extract_widget_snapshot_from_metadata(
    *,
    metadata: Any,
    widget_id: str,
    conversation_id: str,
) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None

    live_widget = None
    for candidate in metadata.get("live_widgets") or []:
        if isinstance(candidate, dict) and str(candidate.get("widget_id") or "") == widget_id:
            live_widget = candidate
            break

    latest_state: dict[str, Any] | None = None
    widget_type = str((live_widget or {}).get("widget_type") or "").strip()
    title = (live_widget or {}).get("title")
    status = str((live_widget or {}).get("status") or "active").strip().lower() or "active"
    version = int((live_widget or {}).get("version") or 1)

    for artifact in metadata.get("tool_artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("error") or str(artifact.get("status") or "").strip().lower() in {
            "error",
            "failed",
            "rejected",
        }:
            continue

        tool_name = artifact.get("tool") or artifact.get("tool_name")
        if tool_name not in {"widget_create", "widget_update", "widget_close"}:
            continue

        output_payload = None
        for key in ("output", "tool_output", "result"):
            output_payload = _parse_json_dict(artifact.get(key))
            if output_payload:
                break
        if not output_payload or str(output_payload.get("widget_id") or "") != widget_id:
            continue

        args = artifact.get("args")
        if not isinstance(args, dict):
            args = {}

        widget_type = str(
            output_payload.get("widget_type") or args.get("widget_type") or widget_type
        ).strip()
        title = output_payload.get("title") or args.get("title") or title
        status = str(output_payload.get("status") or status or "active").strip().lower() or "active"
        version = int(output_payload.get("version") or version or 1)

        if tool_name == "widget_create":
            initial_state = _parse_json_dict(args.get("initial_state"))
            if initial_state is not None and latest_state is None:
                latest_state = initial_state
        elif tool_name == "widget_update":
            updated_state = _parse_json_dict(args.get("state"))
            if updated_state is not None:
                latest_state = updated_state

    if not latest_state or not widget_type:
        return None

    return {
        "widget_id": widget_id,
        "session_id": conversation_id,
        "widget_type": widget_type,
        "title": title,
        "state": latest_state,
        "status": status,
        "version": version,
    }


async def _restore_widget_record_from_messages(
    user_id: UUID,
    widget_id: str,
) -> Any | None:
    """Rehydrate an expired widget from persisted assistant message metadata."""
    try:
        messages = _iter_widget_messages_for_user(user_id, widget_id)
        store = get_widget_store()
        for message in messages:
            snapshot = _extract_widget_snapshot_from_metadata(
                metadata=getattr(message, "message_metadata", None),
                widget_id=widget_id,
                conversation_id=str(message.conversation_id),
            )
            if not snapshot:
                continue

            restored = await store.restore(
                widget_id=snapshot["widget_id"],
                session_id=snapshot["session_id"],
                widget_type=snapshot["widget_type"],
                state=snapshot["state"],
                title=snapshot.get("title"),
                status=snapshot.get("status", "active"),
                version=int(snapshot.get("version") or 1),
            )
            logger.info(
                "Restored expired widget %s for conversation %s from persisted metadata",
                widget_id,
                snapshot["session_id"],
            )
            return restored
    except Exception:
        logger.warning(
            "Failed to restore expired widget %s from persisted metadata",
            widget_id,
            exc_info=True,
        )

    return None


def _resolve_widget_session_id(
    *,
    user_id: UUID,
    widget_id: str,
    stored_session_id: str,
) -> str | None:
    """Resolve the effective conversation ID for a widget.

    This covers both the normal case (widget store already has a valid
    conversation UUID) and legacy/broken widgets created with placeholders like
    ``current_session``.
    """
    stored_conversation_id = _parse_conversation_uuid(stored_session_id)
    if stored_conversation_id and _user_can_access_widget_session(
        user_id, str(stored_conversation_id)
    ):
        return str(stored_conversation_id)

    recovered_session_id = _recover_widget_session_id_from_messages(user_id, widget_id)
    if recovered_session_id and _user_can_access_widget_session(user_id, recovered_session_id):
        logger.info(
            "Recovered widget %s session_id from %r to conversation %s",
            widget_id,
            stored_session_id,
            recovered_session_id,
        )
        return recovered_session_id

    if stored_conversation_id is None:
        logger.warning("Widget session_id is not a valid conversation UUID: %s", stored_session_id)

    return None


def _widget_record_signature(record: Any | None) -> tuple[int, str, float]:
    if record is None:
        return (0, "missing", 0.0)
    return (
        int(getattr(record, "version", 0) or 0),
        str(getattr(getattr(record, "status", None), "value", getattr(record, "status", ""))),
        float(getattr(record, "updated_at", 0.0) or 0.0),
    )


def _html_patch_contract_error(record: Any, patch: dict[str, Any]) -> str | None:
    """Return an error string if shallow-merging ``patch`` into an HTML widget's
    state would break the minimal HTML contract; otherwise ``None``.

    Legacy non-HTML widgets are out of scope — they are a read/restore-only
    compatibility concern, so no contract is enforced on their patches.
    """
    if getattr(record, "widget_type", "") != SUPPORTED_WIDGET_TYPE:
        return None
    base_state = record.state if isinstance(getattr(record, "state", None), dict) else {}
    try:
        validate_html_widget_state({**base_state, **patch})
    except ValueError as exc:
        return str(exc)
    return None


def _widget_event_payload(event_type: str, record: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "widget_id": record.widget_id,
        "widget_type": record.widget_type,
        "title": record.title,
        "state": record.state,
        "status": record.status.value,
        "version": record.version,
    }


async def _send_widget_event(
    websocket: WebSocket,
    payload: dict[str, Any],
    send_lock: asyncio.Lock,
) -> None:
    async with send_lock:
        await websocket.send_text(json.dumps(payload))


async def _ping_loop(websocket: WebSocket, send_lock: asyncio.Lock) -> None:
    """Send periodic pings to keep the connection alive."""
    try:
        while True:
            await asyncio.sleep(30)
            await _send_widget_event(
                websocket,
                {
                    "type": "ping",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                send_lock,
            )
    except (asyncio.CancelledError, Exception):
        pass


async def _watch_widget_updates(
    *,
    websocket: WebSocket,
    widget_id: str,
    store: Any,
    last_seen: dict[str, tuple[int, str, float]],
    send_lock: asyncio.Lock,
) -> None:
    """Poll the canonical widget store so out-of-process MCP updates reach clients."""
    try:
        while True:
            await asyncio.sleep(WIDGET_POLL_INTERVAL_SECONDS)
            record = await store.get(widget_id)
            if record is None:
                await _send_widget_event(
                    websocket,
                    {"type": "error", "message": "Widget is no longer available."},
                    send_lock,
                )
                with contextlib.suppress(Exception):
                    await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                return

            signature = _widget_record_signature(record)
            if signature == last_seen["value"]:
                continue

            event_type = "widget_close" if record.status.value == "closed" else "widget_update"
            await _send_widget_event(
                websocket,
                _widget_event_payload(event_type, record),
                send_lock,
            )
            last_seen["value"] = signature
            if event_type == "widget_close":
                with contextlib.suppress(Exception):
                    await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                return
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Widget watch loop error for %s", widget_id, exc_info=True)


# ---------------------------------------------------------------------------
# POST /widgets/{widget_id}/connection
# ---------------------------------------------------------------------------
@router.post("/{widget_id}/connection")
async def widget_connection(
    widget_id: str,
    user_id: UUID = Depends(get_current_user_id),  # noqa: B008
) -> JSONResponse:
    """Mint a short-lived widget connection token.

    The frontend calls this with normal bearer auth, then uses the returned
    ``ws_url`` and ``token`` to open the widget WebSocket.
    """
    store = get_widget_store()
    record = await store.get(widget_id)
    if record is None:
        record = await _restore_widget_record_from_messages(user_id, widget_id)
        if record is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": f"Widget {widget_id} not found"},
            )
    session_id = _resolve_widget_session_id(
        user_id=user_id,
        widget_id=widget_id,
        stored_session_id=record.session_id,
    )
    if not session_id:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "Access denied to this widget"},
        )

    token_service = get_widget_token_service()
    token, expires_at = token_service.mint(
        widget_id=widget_id,
        session_id=session_id,
        user_id=str(user_id),
    )

    return JSONResponse(
        content={
            "widget_id": record.widget_id,
            "session_id": session_id,
            "widget_type": record.widget_type,
            "title": record.title,
            "status": record.status.value,
            "version": record.version,
            "ws_url": f"/widgets/{widget_id}/connect?session_id={session_id}&token={token}",
            "token": token,
            "expires_at": expires_at.isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# POST /widgets/{widget_id}/actions/{action_key}
# ---------------------------------------------------------------------------
@router.post("/{widget_id}/actions/{action_key}")
async def widget_action(
    widget_id: str,
    action_key: str,
    payload: WidgetActionRequest | None = Body(default=None),  # noqa: B008
    user_id: UUID = Depends(get_current_user_id),  # noqa: B008
) -> JSONResponse:
    """Resolve a widget action into a chat message.

    Applies an optional ``state_patch`` to the widget, then renders the action's
    ``message_template`` against the resulting state. The endpoint does **not**
    invoke the assistant — frontends submit the returned ``content`` through their
    normal stream path.
    """
    request_body = payload or WidgetActionRequest()
    store = get_widget_store()
    record = await store.get(widget_id)
    if record is None:
        record = await _restore_widget_record_from_messages(user_id, widget_id)
        if record is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": f"Widget {widget_id} not found"},
            )

    session_id = _resolve_widget_session_id(
        user_id=user_id,
        widget_id=widget_id,
        stored_session_id=record.session_id,
    )
    if not session_id:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "Access denied to this widget"},
        )

    if request_body.state_patch:
        contract_error = _html_patch_contract_error(record, request_body.state_patch)
        if contract_error:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": contract_error},
            )
        try:
            record = await store.patch(widget_id, request_body.state_patch)
        except (KeyError, ValueError) as exc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": str(exc)},
            )

    try:
        content = resolve_widget_action_message(
            record.state,
            action_key,
            input_values=request_body.input_values or {},
        )
    except KeyError as exc:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": str(exc).strip("'\"")},
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(exc)},
        )

    last_action_entry = {
        "action_key": action_key,
        "content": content,
        "input_values": request_body.input_values or {},
        "control_values": record.state.get("control_values")
        if isinstance(record.state.get("control_values"), dict)
        else {},
        "at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await store.patch(widget_id, {"last_action": last_action_entry})
    except (KeyError, ValueError):
        logger.debug("Failed to record last_action for widget %s", widget_id, exc_info=True)

    return JSONResponse(
        content={
            "widget_id": widget_id,
            "session_id": session_id,
            "action_key": action_key,
            "content": content,
        }
    )


# ---------------------------------------------------------------------------
# WS /widgets/{widget_id}/connect
# ---------------------------------------------------------------------------
@router.websocket("/{widget_id}/connect")
async def widget_connect(
    websocket: WebSocket,
    widget_id: str,
    session_id: str = Query(...),  # noqa: B008
    token: str = Query(...),  # noqa: B008
) -> None:
    """Real-time widget state stream.

    Events server → client:
      widget_state_sync  — full state snapshot on connect
      widget_update      — state changed (by agent or user patch)
      widget_close       — widget was closed
      ping               — keepalive

    Events client → server:
      user_state_patch   — shallow-merge patch from UI interaction
      pong               — keepalive reply
    """
    # Verify token
    token_service = get_widget_token_service()
    try:
        claims = token_service.verify(token)
    except pyjwt.InvalidTokenError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if claims.get("wid") != widget_id or claims.get("sid") != session_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    store = get_widget_store()
    record = await store.get(widget_id)
    if record is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    manager = get_widget_connection_manager()
    await manager.connect(widget_id, websocket)
    send_lock = asyncio.Lock()
    last_seen = {"value": _widget_record_signature(record)}

    # Send initial state sync
    try:
        await _send_widget_event(
            websocket,
            _widget_event_payload("widget_state_sync", record),
            send_lock,
        )
    except Exception:
        await manager.disconnect(widget_id, websocket)
        return

    # Start ping task
    ping_task = asyncio.create_task(_ping_loop(websocket, send_lock))
    watch_task = asyncio.create_task(
        _watch_widget_updates(
            websocket=websocket,
            widget_id=widget_id,
            store=store,
            last_seen=last_seen,
            send_lock=send_lock,
        )
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "pong":
                continue

            if msg_type == "user_state_patch":
                patch_data = msg.get("patch")
                if not isinstance(patch_data, dict):
                    continue
                if record.widget_type == SUPPORTED_WIDGET_TYPE:
                    current = await store.get(widget_id)
                    contract_error = _html_patch_contract_error(
                        current if current is not None else record, patch_data
                    )
                    if contract_error:
                        await _send_widget_event(
                            websocket,
                            {"type": "error", "message": contract_error},
                            send_lock,
                        )
                        continue
                try:
                    updated = await store.patch(widget_id, patch_data)
                    last_seen["value"] = _widget_record_signature(updated)
                    await _send_widget_event(
                        websocket,
                        _widget_event_payload("widget_update", updated),
                        send_lock,
                    )
                except (KeyError, ValueError) as e:
                    await _send_widget_event(
                        websocket,
                        {"type": "error", "message": str(e)},
                        send_lock,
                    )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("Widget WS error for %s", widget_id, exc_info=True)
    finally:
        ping_task.cancel()
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ping_task
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task
        await manager.disconnect(widget_id, websocket)
