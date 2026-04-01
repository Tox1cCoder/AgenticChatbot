"""
WebSocket endpoint and runtime gateway for device connections.

Handles the persistent WebSocket connection between server and client devices
for tool dispatch and runtime events.
"""

import asyncio
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
    RuntimeErrorMessage,
    ToolDispatchRequest,
    ToolDispatchResult,
)
from app.services.client_device_service import ClientDeviceService, DeviceSession

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

        # Store result in session for retrieval by the waiting request
        if request_id in self.session.pending_tool_requests:
            result_future = self.session.pending_tool_requests[request_id]
            result_future.set_result(payload.model_dump(mode="json"))
            logger.debug(f"Tool result received for request {request_id}")
        else:
            logger.warning(
                f"Unexpected tool result for request {request_id} from device {self.device_id}"
            )

        await self.send_message(WebSocketMessage.ack(request_id))

    async def _handle_error(self, message: dict) -> None:
        """Handle error message from device."""
        logger.error(f"Device {self.device_id} error: {message.get('message')}")

    async def _cleanup(self) -> None:
        """Cleanup when connection closes."""
        self._running = False
        await self.service.end_session(self.device_id)
        logger.info(f"Device runtime connection closed: {self.device_id}")

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
        # Create a future for this request
        result_future = asyncio.Future()
        self.session.pending_tool_requests[request_id] = result_future

        try:
            # Send tool request
            message = ToolDispatchRequest(
                request_id=request_id,
                tool_name=tool_name,
                qualified_tool_id=qualified_tool_id,
                arguments=arguments,
                timeout_seconds=timeout_seconds,
            )
            await self.send_message(message.model_dump(mode="json"))

            # Wait for result with timeout
            result = await asyncio.wait_for(result_future, timeout=timeout_seconds)
            return result

        except asyncio.TimeoutError:
            logger.error(f"Tool call timeout for request {request_id} on device {self.device_id}")
            raise TimeoutError(f"Tool call timed out after {timeout_seconds}s")

        finally:
            # Cleanup pending request
            self.session.pending_tool_requests.pop(request_id, None)


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
            websocket=websocket,
        )

        # Create gateway
        gateway = DeviceRuntimeGateway(
            websocket=websocket,
            device_id=device_uuid,
            session=session,
            service=service,
        )

        # Store gateway in session for tool dispatch access
        session.websocket = gateway

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
