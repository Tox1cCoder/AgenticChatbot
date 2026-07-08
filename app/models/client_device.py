"""Model for tracking client devices registered by users."""

import enum
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class DeviceStatus(str, enum.Enum):
    """Status of a client device."""

    ONLINE = "online"
    OFFLINE = "offline"
    IDLE = "idle"


class DevicePlatform(str, enum.Enum):
    """Platform type for client devices."""

    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    OTHER = "other"


class ClientDevice(Base):
    """
    Represents a registered client device for a user.

    A device is uniquely identified by the combination of user_id and device_identifier.
    The device_identifier is a stable per-installation/device value, while device sessions
    are ephemeral per connected runtime session.
    """

    __tablename__ = "client_devices"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "device_identifier",
            name="uq_client_devices_user_device_id",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    # Ownership
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    # Device identity
    device_identifier = Column(String(255), nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    platform = Column(String(64), nullable=False, default=DevicePlatform.WINDOWS.value)
    app_version = Column(String(64), nullable=True)
    runtime_version = Column(String(64), nullable=True)

    # Runtime status
    last_seen_at = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)
    status = Column(
        SQLEnum(
            DeviceStatus,
            name="device_status",
            native_enum=False,
            length=32,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=DeviceStatus.OFFLINE,
        index=True,
    )

    # Device capabilities (sanitized tool/skill catalogs, etc.)
    capabilities_json = Column(JSONB, nullable=False, default=dict)

    # Relationships
    user = relationship("User", backref="client_devices")

    def __repr__(self) -> str:
        return (
            f"<ClientDevice(id={self.id}, display_name='{self.display_name}', "
            f"status={self.status}, user_id={self.user_id})>"
        )
