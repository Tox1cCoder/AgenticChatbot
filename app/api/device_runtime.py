"""
WebSocket endpoint and runtime gateway for device connections.

Handles the persistent WebSocket connection between server and client devices
for tool dispatch and runtime events.
"""

import asyncio
import contextlib
import json
import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.config import settings
from app.database.session import get_db
from app.models.user import User
from app.schemas.runtime_protocol import (
    RUNTIME_MESSAGE_ACK,
    RUNTIME_MESSAGE_ERROR,
    RUNTIME_MESSAGE_HEARTBEAT,
    RUNTIME_MESSAGE_TOOL_REQUEST,
    RUNTIME_MESSAGE_TOOL_RESULT,
    RuntimeAckMessage,
    RuntimeErrorContext,
    RuntimeErrorMessage,
    ToolDispatchResult,
)
from app.services.client_device_service import ClientDeviceService, DeviceSession
from app.services.client_runtime_store import get_client_runtime_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/device-runtime", tags=["device-runtime"])


class WebSocketMessage:
    """Base message structure for WebSocket communication."""

    TYPE_TOOL_REQUEST = RUNTIME_MESSAGE_TOOL_REQUEST
    TYPE_TOOL_RESULT = RUNTIME_MESSAGE_TOOL_RESULT
    TYPE_HEARTBEAT = RUNTIME_MESSAGE_HEARTBEAT
    TYPE_ERROR = RUNTIME_MESSAGE_ERROR
    TYPE_ACK = RUNTIME_MESSAGE_ACK

    @staticmethod
    def heartbeat() -> dict:
        """Create a heartbeat message."""
        return {"type": RUNTIME_MESSAGE_HEARTBEAT}

    @staticmethod
    def error(message: str, code: str | None = None) -> dict:
        """Create an error message."""
        return RuntimeErrorMessage(message=message, code=code).model_dump(mode="json")

    @staticmethod
    def ack(message_id: str | None = None) -> dict:
        """Create an acknowledgment message."""
        return RuntimeAckMessage(message_id=message_id).model_dump(mode="json")


class DeviceRuntimeGateway:
    """
    Gateway for managing device runtime WebSocket connections.

    Handles bidirectional communication for tool dispatch and results.
    """

    def __init__(
        self,
        websocket: WebSocket,
        device_id: UUID,
        session: DeviceSession,
        service: ClientDeviceService,
    ):
        self.websocket = websocket
        self.device_id = device_id
        self.session = session
        self.service = service
        self._running = False
        self._send_lock = asyncio.Lock()
        self._dispatch_task: asyncio.Task | None = None

    async def send_message(self, message: dict) -> None:
        """
        Send a message to the device.

        Args:
            message: The message dictionary to send.
        """
        async with self._send_lock:
            try:
                await self.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to device {self.device_id}: {e}")
                raise

    async def receive_message(self) -> dict | None:
        """
        Receive a message from the device.

        Returns:
            The message dictionary, or None on error.
        """
        try:
            data = await self.websocket.receive_text()
            return json.loads(data)
        except WebSocketDisconnect:
            logger.info(f"Device {self.device_id} disconnected")
            return None
        except Exception as e:
            logger.error(f"Error receiving message from device {self.device_id}: {e}")
            return None

    async def handle_connection(self) -> None:
        """
        Main connection handler loop.

        Processes incoming messages and manages the connection lifecycle.
        """
        self._running = True
        logger.info(f"Device runtime connection established: {self.device_id}")

        # Send initial ack
        await self.send_message(WebSocketMessage.ack())
        self._dispatch_task = asyncio.create_task(self._dispatch_requests_loop())

        try:
            while self._running:
                message = await self.receive_message()
                if message is None:
                    break

                msg_type = message.get("type")

                if msg_type == WebSocketMessage.TYPE_HEARTBEAT:
                    await self._handle_heartbeat(message)
                elif msg_type == WebSocketMessage.TYPE_TOOL_RESULT:
                    await self._handle_tool_result(message)
                elif msg_type == WebSocketMessage.TYPE_ERROR:
                    await self._handle_error(message)
                else:
                    logger.warning(f"Unknown message type from device {self.device_id}: {msg_type}")

        except Exception as e:
            logger.error(f"Error in device runtime connection {self.device_id}: {e}")
        finally:
            if self._dispatch_task is not None:
                self._dispatch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._dispatch_task
                self._dispatch_task = None
            await self._cleanup()

    async def _handle_heartbeat(self, message: dict) -> None:
        """Handle heartbeat message from device."""
        await self.service.update_heartbeat(self.device_id)
        await self.send_message(WebSocketMessage.ack())

    async def _handle_tool_result(self, message: dict) -> None:
        """
        Handle tool result message from device.

        Args:
            message: The tool result message.
        """
        try:
            payload = ToolDispatchResult.model_validate(message)
        except Exception as exc:
            logger.warning(
                "Invalid tool result payload from device %s: %s",
                self.device_id,
                exc,
            )
            return

        request_id = payload.request_id
        if not request_id:
            logger.warning(f"Tool result missing request_id from device {self.device_id}")
            return

        await get_client_runtime_store().publish_result(payload)

        await self.send_message(WebSocketMessage.ack(request_id))

    async def _handle_error(self, message: dict) -> None:
        """Handle error message from device."""
        logger.error(f"Device {self.device_id} error: {message.get('message')}")

    async def _cleanup(self) -> None:
        """Cleanup when connection closes."""
        self._running = False
        await self.service.end_session(
            self.device_id,
            reason="Client runtime disconnected before completing the request.",
        )
        logger.info(f"Device runtime connection closed: {self.device_id}")

    async def _dispatch_requests_loop(self) -> None:
        """Bridge queued runtime requests from the shared store to this WebSocket."""
        store = get_client_runtime_store()

        while self._running:
            try:
                request = await store.get_next_request(self.device_id, timeout_seconds=1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Failed pulling queued runtime request for device %s: %s",
                    self.device_id,
                    exc,
                )
                await asyncio.sleep(1)
                continue

            if request is None:
                continue

            try:
                await self.send_message(request.model_dump(mode="json"))
            except Exception as exc:
                logger.error(
                    "Failed forwarding queued runtime request %s to device %s: %s",
                    request.request_id,
                    self.device_id,
                    exc,
                )
                await store.publish_result(
                    ToolDispatchResult(
                        request_id=request.request_id,
                        success=False,
                        error=str(exc),
                        error_context=RuntimeErrorContext(
                            message=str(exc),
                            code=exc.__class__.__name__,
                            detail={
                                "device_id": str(self.device_id),
                                "tool_name": request.tool_name,
                                "qualified_tool_id": request.qualified_tool_id,
                            },
                        ),
                        execution_time_ms=0,
                    )
                )

    async def dispatch_tool_call(
        self,
        request_id: str,
        tool_name: str,
        qualified_tool_id: str,
        arguments: dict[str, Any],
        timeout_seconds: int = 30,
    ) -> dict:
        """
        Dispatch a tool call to the device and wait for the result.

        Args:
            request_id: Unique request identifier.
            tool_name: The tool name.
            qualified_tool_id: Fully qualified tool ID.
            arguments: Tool arguments.
            timeout_seconds: Timeout for the call.

        Returns:
            The tool result message.

        Raises:
            TimeoutError: If the tool call times out.
            RuntimeError: If the tool call fails.
        """
        return await ClientDeviceService.dispatch_tool_call(
            user_id=str(self.session.user_id),
            device_id=str(self.session.device_id),
            tool_name=tool_name,
            qualified_tool_id=qualified_tool_id,
            arguments=arguments,
            timeout_seconds=timeout_seconds,
            bound_session_id=self.session.session_id,
        )


@router.websocket("/{device_id}/connect")
async def device_runtime_connect(
    device_id: str,
    session_id: str,
    websocket: WebSocket,
    db: Session = Depends(get_db),
) -> None:
    """
    WebSocket endpoint for device runtime connection.

    Args:
        device_id: The device UUID.
        session_id: The session ID from registration.
        websocket: The WebSocket connection.
    """
    if not settings.enable_client_runtime_bridge:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    device_uuid = UUID(device_id)
    service = ClientDeviceService(db)

    # Verify device exists
    device = service.repository.get_by_id(device_uuid)
    if not device:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    try:
        # Start session
        session = await service.start_session(
            device_id=device_uuid,
            session_id=session_id,
        )

        # Create gateway
        gateway = DeviceRuntimeGateway(
            websocket=websocket,
            device_id=device_uuid,
            session=session,
            service=service,
        )

        # Handle connection
        await gateway.handle_connection()

    except ValueError as e:
        logger.warning(f"Rejected device runtime connection for {device_uuid}: {e}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    except Exception as e:
        logger.error(f"Device runtime connection error: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)


@router.get("/connected-devices")
async def list_connected_devices(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    List currently connected devices for the user.

    Returns device IDs and connection status.
    """
    service = ClientDeviceService(db)
    sessions = service.get_active_sessions_for_user(user.id)

    return {
        "connected_devices": [
            {
                "device_id": str(session.device_id),
                "session_id": session.session_id,
                "connected_at": session.connected_at.isoformat(),
                "last_heartbeat": session.last_heartbeat.isoformat(),
                "is_alive": session.is_alive(),
            }
            for session in sessions
        ],
        "total": len(sessions),
    }
