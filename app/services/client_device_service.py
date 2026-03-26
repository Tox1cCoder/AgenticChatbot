"""
Service for managing client device registration and sessions.

This service handles:
- Device registration and identification
- Heartbeat tracking
- Device status management
- Tool and skill catalog caching
"""

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.client_device import ClientDevice, DeviceStatus
from app.repositories.client_device import ClientDeviceRepository


class DeviceSession:
    """
    Represents an active device runtime session.

    Tracks WebSocket connection state and catalog information.
    """

    def __init__(
        self,
        device_id: UUID,
        session_id: str,
        user_id: UUID,
        websocket=None,
    ):
        self.device_id = device_id
        self.session_id = session_id
        self.user_id = user_id
        self.websocket = websocket
        self.connected_at = datetime.now(timezone.utc)
        self.last_heartbeat = self.connected_at
        self.tool_catalog: dict = {}
        self.skill_catalog: dict = {}
        self.pending_tool_requests: dict = {}
        self.tool_catalog_version: int = 0
        self.skill_catalog_version: int = 0
        self.tool_catalog_updated_at: datetime | None = None
        self.skill_catalog_updated_at: datetime | None = None

    def is_alive(self) -> bool:
        """Check if the session is still consider alive based on heartbeat."""
        timeout = settings.client_runtime_heartbeat_interval_seconds * 2
        elapsed = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
        return elapsed < timeout

    def update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = datetime.now(timezone.utc)

    def update_tool_catalog(self, catalog: dict) -> None:
        """Store a new device-local tool catalog and bump its generation."""
        self.tool_catalog = catalog
        self.tool_catalog_version += 1
        self.tool_catalog_updated_at = datetime.now(timezone.utc)

    def update_skill_catalog(self, catalog: dict) -> None:
        """Store a new device-local skill catalog and bump its generation."""
        self.skill_catalog = catalog
        self.skill_catalog_version += 1
        self.skill_catalog_updated_at = datetime.now(timezone.utc)

    def get_tool_cache_key(self) -> tuple[str, str, str, int]:
        """Return a stable cache key for server-side client-tool bindings."""
        return (
            str(self.user_id),
            str(self.device_id),
            self.session_id,
            self.tool_catalog_version,
        )


class ClientDeviceService:
    """Service for managing client devices."""

    _active_sessions: dict[UUID, DeviceSession] = {}
    _issued_session_ids: dict[UUID, str] = {}
    _registry_lock = threading.RLock()

    def __init__(self, session: Session):
        self.session = session
        self.repository = ClientDeviceRepository(session)

    @classmethod
    def issue_runtime_session_id(cls, device_id: UUID, session_id: str) -> None:
        """Register a one-time session token that authorizes a runtime connect."""
        with cls._registry_lock:
            cls._issued_session_ids[device_id] = session_id

    @classmethod
    def consume_runtime_session_id(cls, device_id: UUID, session_id: str) -> bool:
        """Validate and consume a pending runtime session token."""
        with cls._registry_lock:
            expected = cls._issued_session_ids.get(device_id)
            if expected != session_id:
                return False
            cls._issued_session_ids.pop(device_id, None)
            return True

    @classmethod
    def lookup_active_session(cls, device_id: UUID) -> DeviceSession | None:
        """Class-level access to an active device session."""
        with cls._registry_lock:
            return cls._active_sessions.get(device_id)

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
        websocket=None,
    ) -> DeviceSession:
        """
        Start a new device runtime session.

        Args:
            device_id: The device ID.
            session_id: Unique session identifier.
            websocket: Optional WebSocket connection.

        Returns:
            The DeviceSession instance.
        """
        if not self.consume_runtime_session_id(device_id, session_id):
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
            websocket=websocket,
        )

        with self._registry_lock:
            self._active_sessions[device_id] = session

        return session

    async def end_session(self, device_id: UUID) -> bool:
        """
        End a device runtime session.

        Args:
            device_id: The device ID.

        Returns:
            True if session was ended, False if not found.
        """
        with self._registry_lock:
            if device_id not in self._active_sessions:
                return False

            # Remove from active sessions
            del self._active_sessions[device_id]

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
        session = self.get_active_session(device_id)
        if not session:
            return False

        session.update_heartbeat()

        # Also update last_seen_at in database
        self.repository.update_status(device_id, DeviceStatus.ONLINE)

        return True

    def get_active_session(self, device_id: UUID) -> DeviceSession | None:
        """Get an active session by device ID."""
        with self._registry_lock:
            return self._active_sessions.get(device_id)

    def get_active_sessions_for_user(self, user_id: UUID) -> list[DeviceSession]:
        """Get all active sessions for a user."""
        with self._registry_lock:
            return [
                session for session in self._active_sessions.values() if session.user_id == user_id
            ]

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
        session = self.get_active_session(device_id)
        if not session:
            return False

        session.update_tool_catalog(catalog)
        return True

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
        session = self.get_active_session(device_id)
        if not session:
            return False

        session.update_skill_catalog(catalog)
        return True

    async def cleanup_stale_sessions(self) -> int:
        """
        Clean up stale sessions that haven't sent heartbeats.

        Returns:
            Number of sessions cleaned up.
        """
        stale_devices = []

        with self._registry_lock:
            active_sessions = list(self._active_sessions.items())

        for device_id, session in active_sessions:
            if not session.is_alive():
                stale_devices.append(device_id)

        for device_id in stale_devices:
            await self.end_session(device_id)

        # Also mark stale devices as offline in database
        self.repository.mark_stale_devices_offline(
            settings.client_runtime_heartbeat_interval_seconds * 2
        )

        return len(stale_devices)

    async def list_user_devices(
        self,
        user_id: UUID,
        include_offline: bool = True,
    ) -> list[ClientDevice]:
        """List all devices for a user."""
        return self.repository.list_by_user(user_id, include_offline)


# Background task for periodic session cleanup
async def periodic_session_cleanup_task(service: ClientDeviceService, interval_seconds: int = 60):
    """
    Background task to periodically clean up stale sessions.

    Args:
        service: The ClientDeviceService instance.
        interval_seconds: Cleanup interval in seconds.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            count = await service.cleanup_stale_sessions()
            if count > 0:
                print(f"Cleaned up {count} stale device sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in session cleanup task: {e}")
