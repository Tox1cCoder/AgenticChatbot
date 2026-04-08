"""
Service for managing client device registration and sessions.

This service handles:
- Device registration and identification
- Heartbeat tracking
- Device status management
- Tool and skill catalog caching
"""

import asyncio
import logging
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.client_device import ClientDevice, DeviceStatus
from app.repositories.client_device import ClientDeviceRepository
from app.schemas.runtime_protocol import ToolDispatchRequest
from app.services.client_runtime_store import (
    DeviceSessionRecord,
    get_client_runtime_store,
)

DeviceSession = DeviceSessionRecord
logger = logging.getLogger(__name__)


class ClientDeviceService:
    """Service for managing client devices."""

    def __init__(self, session: Session):
        self.session = session
        self.repository = ClientDeviceRepository(session)

    @classmethod
    async def issue_runtime_session_id(cls, device_id: UUID, session_id: str) -> None:
        """Register a one-time session token that authorizes a runtime connect."""
        await get_client_runtime_store().issue_runtime_session_id(device_id, session_id)

    @classmethod
    async def consume_runtime_session_id(cls, device_id: UUID, session_id: str) -> bool:
        """Validate and consume a pending runtime session token."""
        return await get_client_runtime_store().consume_runtime_session_id(device_id, session_id)

    @classmethod
    def lookup_active_session(cls, device_id: UUID) -> DeviceSession | None:
        """Class-level access to an active device session."""
        return get_client_runtime_store().get_session(device_id)

    @classmethod
    async def dispatch_tool_call(
        cls,
        *,
        user_id: str,
        device_id: str,
        tool_name: str,
        qualified_tool_id: str,
        arguments: dict[str, Any],
        timeout_seconds: int,
        bound_session_id: str | None = None,
        bound_catalog_version: int | None = None,
        tool_instance_id: str | None = None,
    ) -> dict[str, Any]:
        """Dispatch a tool call to the correct sidecar session."""
        session = cls.lookup_active_session(UUID(str(device_id)))
        if session is None or str(session.user_id) != str(user_id):
            raise RuntimeError("Client device is not connected for this user.")

        if bound_session_id and session.session_id != bound_session_id:
            raise RuntimeError(
                "Client device session changed after tool binding. Retry from the active device."
            )

        if (
            bound_catalog_version is not None
            and session.tool_catalog_version != bound_catalog_version
        ):
            raise RuntimeError(
                "Client device tool catalog changed after tool binding. Retry from the active device."
            )

        # Server-side catalog validation (defense-in-depth; sidecar also validates)
        if session.tool_catalog:
            tools = session.tool_catalog.get("tools", [])
            catalog_by_qid = {
                str(entry.get("qualified_id")): entry
                for entry in tools
                if entry.get("qualified_id")
            }
            catalog_entry = catalog_by_qid.get(qualified_tool_id)
            if catalog_entry is None:
                raise RuntimeError(
                    f"Tool {qualified_tool_id!r} is not in the active session catalog. "
                    "The catalog may have changed; re-sync and retry."
                )
            current_instance_id = str(catalog_entry.get("tool_instance_id") or "")
            if tool_instance_id and current_instance_id and tool_instance_id != current_instance_id:
                raise RuntimeError(
                    "Client device capability changed after tool binding. Retry from the active device."
                )

        request = ToolDispatchRequest(
            request_id=str(uuid4()),
            tool_name=tool_name,
            qualified_tool_id=qualified_tool_id,
            arguments=arguments,
            timeout_seconds=timeout_seconds,
            tool_instance_id=tool_instance_id,
            expected_session_id=session.session_id,
            expected_catalog_version=(
                bound_catalog_version
                if bound_catalog_version is not None
                else session.tool_catalog_version
            ),
        )
        return await get_client_runtime_store().dispatch_request(session, request, timeout_seconds)

    async def register_or_update_device(
        self,
        user_id: UUID,
        device_identifier: str,
        display_name: str,
        platform: str,
        app_version: str,
        runtime_version: str,
        capabilities: dict | None = None,
    ) -> ClientDevice:
        """
        Register a new device or update an existing one.

        Args:
            user_id: The user ID.
            device_identifier: Stable device identifier.
            display_name: Device display name.
            platform: Platform string.
            app_version: App version.
            runtime_version: Runtime version.
            capabilities: Optional capabilities metadata.

        Returns:
            The ClientDevice instance.
        """
        # Check if device exists
        existing = self.repository.get_by_user_and_identifier(user_id, device_identifier)

        if existing:
            # Update metadata
            device = self.repository.update_metadata(
                existing.id,
                display_name=display_name,
                app_version=app_version,
                runtime_version=runtime_version,
                capabilities_json=capabilities,
            )
        else:
            # Create new device
            device = self.repository.create(
                user_id=user_id,
                device_identifier=device_identifier,
                display_name=display_name,
                platform=platform,
                app_version=app_version,
                runtime_version=runtime_version,
                capabilities_json=capabilities,
            )

        return device

    async def start_session(
        self,
        device_id: UUID,
        session_id: str,
    ) -> DeviceSession:
        """
        Start a new device runtime session.

        Args:
            device_id: The device ID.
            session_id: Unique session identifier.
        Returns:
            The DeviceSession instance.
        """
        if not await self.consume_runtime_session_id(device_id, session_id):
            raise ValueError("Invalid or expired runtime session ID")

        # Mark device as online
        device = self.repository.update_status(device_id, DeviceStatus.ONLINE)
        if not device:
            raise ValueError(f"Device {device_id} not found")

        # Create session
        session = DeviceSession(
            device_id=device_id,
            session_id=session_id,
            user_id=device.user_id,
        )
        await get_client_runtime_store().put_session(session)

        return session

    async def end_session(self, device_id: UUID, *, reason: str | None = None) -> bool:
        """
        End a device runtime session.

        Args:
            device_id: The device ID.

        Returns:
            True if session was ended, False if not found.
        """
        store = get_client_runtime_store()
        session = store.get_session(device_id)
        if session is None:
            return False

        await store.fail_pending_requests(
            device_id,
            reason or "Client runtime disconnected before completing the request.",
        )
        await store.delete_session(device_id)

        # Mark device as offline
        self.repository.update_status(device_id, DeviceStatus.OFFLINE)

        return True

    async def update_heartbeat(self, device_id: UUID) -> bool:
        """
        Update the heartbeat for a device session.

        Args:
            device_id: The device ID.

        Returns:
            True if heartbeat updated, False if session not found.
        """
        session = await get_client_runtime_store().update_heartbeat(device_id)
        if not session:
            return False

        # Also update last_seen_at in database
        self.repository.update_status(device_id, DeviceStatus.ONLINE)

        return True

    def get_active_session(self, device_id: UUID) -> DeviceSession | None:
        """Get an active session by device ID."""
        return get_client_runtime_store().get_session(device_id)

    def get_active_sessions_for_user(self, user_id: UUID) -> list[DeviceSession]:
        """Get all active sessions for a user."""
        return get_client_runtime_store().list_sessions_for_user(user_id)

    def get_device_tool_catalog(
        self,
        *,
        user_id: UUID,
        device_id: UUID,
    ) -> dict[str, Any] | None:
        """Get the current tool catalog for an active device owned by the user."""
        session = self.get_active_session(device_id)
        if not session or session.user_id != user_id:
            return None
        return session.tool_catalog

    def get_device_skill_catalog(
        self,
        *,
        user_id: UUID,
        device_id: UUID,
    ) -> dict[str, Any] | None:
        """Get the current skill catalog for an active device owned by the user."""
        session = self.get_active_session(device_id)
        if not session or session.user_id != user_id:
            return None
        return session.skill_catalog

    async def update_tool_catalog(
        self,
        device_id: UUID,
        catalog: dict,
    ) -> bool:
        """
        Update the tool catalog for a device session.

        Args:
            device_id: The device ID.
            catalog: The tool catalog dictionary.

        Returns:
            True if updated, False if session not found.
        """
        session = await get_client_runtime_store().update_tool_catalog(device_id, catalog)
        return session is not None

    async def update_skill_catalog(
        self,
        device_id: UUID,
        catalog: dict,
    ) -> bool:
        """
        Update the skill catalog for a device session.

        Args:
            device_id: The device ID.
            catalog: The skill catalog dictionary.

        Returns:
            True if updated, False if session not found.
        """
        session = await get_client_runtime_store().update_skill_catalog(device_id, catalog)
        return session is not None

    async def cleanup_stale_sessions(self) -> int:
        """
        Clean up stale sessions that haven't sent heartbeats.

        Returns:
            Number of sessions cleaned up.
        """
        try:
            store_stale_count = await get_client_runtime_store().cleanup_stale_sessions()
        except Exception as exc:
            logger.warning(
                "Client runtime store stale-session cleanup failed; continuing with DB cleanup only: %s",
                exc,
            )
            store_stale_count = 0

        # Also mark stale devices as offline in database
        db_stale_count = self.repository.mark_stale_devices_offline(
            settings.client_runtime_heartbeat_interval_seconds * 2
        )

        return max(store_stale_count, db_stale_count)

    async def list_user_devices(
        self,
        user_id: UUID,
        include_offline: bool = True,
    ) -> list[ClientDevice]:
        """List all devices for a user."""
        self.repository.mark_stale_devices_offline(
            settings.client_runtime_heartbeat_interval_seconds * 2
        )
        return self.repository.list_by_user(user_id, include_offline)


# Background task for periodic session cleanup
async def periodic_session_cleanup_task(session_factory: callable, interval_seconds: int = 60):
    """
    Background task to periodically clean up stale sessions.

    Args:
        session_factory: Callable returning a new SQLAlchemy session.
        interval_seconds: Cleanup interval in seconds.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            db_session = session_factory()
            try:
                service = ClientDeviceService(db_session)
                count = await service.cleanup_stale_sessions()
            finally:
                db_session.close()
            if count > 0:
                logger.info("Cleaned up %d stale device sessions", count)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("Error in session cleanup task: %s", e)
