"""
Repository for managing client device records.

Provides CRUD operations for the client_devices table.
"""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.client_device import ClientDevice, DeviceStatus


class ClientDeviceRepository:
    """Repository for ClientDevice model."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        user_id: UUID,
        device_identifier: str,
        display_name: str,
        platform: str,
        app_version: str,
        runtime_version: str,
        capabilities_json: dict | None = None,
    ) -> ClientDevice:
        """
        Create a new client device record.

        Args:
            user_id: The user who owns this device.
            device_identifier: Unique stable identifier for the device.
            display_name: Human-readable device name.
            platform: Platform string (e.g., "windows", "macos", "linux").
            app_version: Application version.
            runtime_version: Runtime version.
            capabilities_json: Optional device capabilities metadata.

        Returns:
            The created ClientDevice instance.
        """
        device = ClientDevice(
            user_id=user_id,
            device_identifier=device_identifier,
            display_name=display_name,
            platform=platform,
            app_version=app_version,
            runtime_version=runtime_version,
            status=DeviceStatus.OFFLINE,
            capabilities_json=capabilities_json or {},
        )

        self.session.add(device)
        self.session.commit()
        self.session.refresh(device)

        return device

    def get_by_id(self, device_id: UUID) -> ClientDevice | None:
        """Get a device by its ID."""
        result = self.session.execute(select(ClientDevice).where(ClientDevice.id == device_id))
        return result.scalar_one_or_none()

    def get_by_user_and_identifier(
        self,
        user_id: UUID,
        device_identifier: str,
    ) -> ClientDevice | None:
        """
        Get a device by user ID and device identifier.

        This is the natural key for device lookup.

        Args:
            user_id: The user ID.
            device_identifier: The device's stable identifier.

        Returns:
            The ClientDevice if found, None otherwise.
        """
        result = self.session.execute(
            select(ClientDevice).where(
                ClientDevice.user_id == user_id,
                ClientDevice.device_identifier == device_identifier,
            )
        )
        return result.scalar_one_or_none()

    def list_by_user(
        self,
        user_id: UUID,
        include_offline: bool = True,
    ) -> list[ClientDevice]:
        """
        List all devices for a user.

        Args:
            user_id: The user ID.
            include_offline: If False, only return online devices.

        Returns:
            List of ClientDevice instances.
        """
        query = select(ClientDevice).where(ClientDevice.user_id == user_id)

        if not include_offline:
            query = query.where(ClientDevice.status == DeviceStatus.ONLINE)

        result = self.session.execute(query.order_by(ClientDevice.last_seen_at.desc()))
        return list(result.scalars().all())

    def update_status(
        self,
        device_id: UUID,
        status: DeviceStatus,
    ) -> ClientDevice | None:
        """
        Update a device's status and last_seen_at timestamp.

        Args:
            device_id: The device ID.
            status: The new status.

        Returns:
            The updated device, or None if not found.
        """
        device = self.get_by_id(device_id)
        if not device:
            return None

        device.status = status
        device.last_seen_at = datetime.now(timezone.utc)

        self.session.commit()
        self.session.refresh(device)

        return device

    def update_metadata(
        self,
        device_id: UUID,
        display_name: str | None = None,
        app_version: str | None = None,
        runtime_version: str | None = None,
        capabilities_json: dict | None = None,
    ) -> ClientDevice | None:
        """
        Update device metadata fields.

        Args:
            device_id: The device ID.
            display_name: Optional new display name.
            app_version: Optional new app version.
            runtime_version: Optional new runtime version.
            capabilities_json: Optional new capabilities.

        Returns:
            The updated device, or None if not found.
        """
        device = self.get_by_id(device_id)
        if not device:
            return None

        if display_name is not None:
            device.display_name = display_name
        if app_version is not None:
            device.app_version = app_version
        if runtime_version is not None:
            device.runtime_version = runtime_version
        if capabilities_json is not None:
            device.capabilities_json = capabilities_json

        self.session.commit()
        self.session.refresh(device)

        return device

    def delete(self, device_id: UUID) -> bool:
        """
        Delete a device record.

        Args:
            device_id: The device ID.

        Returns:
            True if deleted, False if not found.
        """
        device = self.get_by_id(device_id)
        if not device:
            return False

        self.session.delete(device)
        self.session.commit()

        return True

    def mark_stale_devices_offline(
        self,
        timeout_seconds: int = 120,
    ) -> int:
        """
        Mark devices as offline if they haven't been seen recently.

        Args:
            timeout_seconds: Timeout in seconds for considering a device offline.

        Returns:
            Number of devices marked offline.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - timeout_seconds

        result = self.session.execute(
            select(ClientDevice).where(
                ClientDevice.status == DeviceStatus.ONLINE,
                ClientDevice.last_seen_at < datetime.fromtimestamp(cutoff, tz=timezone.utc),
            )
        )

        devices = list(result.scalars().all())
        count = 0

        for device in devices:
            device.status = DeviceStatus.OFFLINE
            count += 1

        if count > 0:
            self.session.commit()

        return count
