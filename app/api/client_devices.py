"""
API endpoints for client device management.

Handles device registration, heartbeats, and catalog syncing.
"""

import secrets
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.config import settings
from app.database.session import get_db
from app.models.user import User
from app.services.client_device_service import ClientDeviceService

router = APIRouter(prefix="/client-devices", tags=["client-devices"])


class DeviceRegistrationRequest(BaseModel):
    """Request to register a new device."""

    device_identifier: str = Field(..., description="Stable unique identifier for the device")
    display_name: str = Field(..., description="Human-readable device name")
    platform: str = Field(..., description="Platform (windows, macos, linux, etc.)")
    app_version: str = Field(..., description="Application version")
    runtime_version: str = Field(..., description="Runtime version")
    capabilities: dict = Field(
        default_factory=dict, description="Optional device capabilities metadata"
    )


class DeviceRegistrationResponse(BaseModel):
    """Response for device registration."""

    device_id: str
    session_id: str
    status: str
    message: str


class DeviceHeartbeatRequest(BaseModel):
    """Request to send a heartbeat."""

    device_id: UUID


class DeviceHeartbeatResponse(BaseModel):
    """Response for heartbeat."""

    status: str
    last_heartbeat: str


class ToolCatalogUpdateRequest(BaseModel):
    """Request to update the tool catalog."""

    device_id: UUID
    catalog: dict = Field(..., description="Tool catalog with schema and metadata")


class SkillCatalogUpdateRequest(BaseModel):
    """Request to update the skill catalog."""

    device_id: UUID
    catalog: dict = Field(..., description="Skill catalog with enabled skills")


class DeviceInfoResponse(BaseModel):
    """Response for device info."""

    id: str
    device_identifier: str
    display_name: str
    platform: str
    app_version: str
    runtime_version: str
    status: str
    last_seen_at: str | None
    created_at: str


class ToolCatalogUpdateResponse(BaseModel):
    """Response for a device tool-catalog sync."""

    status: str = Field(..., description='Always "updated" on success')
    tool_count: int


class SkillCatalogUpdateResponse(BaseModel):
    """Response for a device skill-catalog sync."""

    status: str = Field(..., description='Always "updated" on success')
    skill_count: int


def get_device_service(db: Session = Depends(get_db)) -> ClientDeviceService:
    """Dependency to get the device service."""
    return ClientDeviceService(db)


@router.post("/register", response_model=DeviceRegistrationResponse)
async def register_device(
    request: DeviceRegistrationRequest,
    user: User = Depends(get_current_user),
    service: ClientDeviceService = Depends(get_device_service),
) -> DeviceRegistrationResponse:
    """
    Register or update a client device.

    Creates a new device record or updates an existing one, and returns
    a session ID for establishing a runtime connection.
    """
    if not settings.enable_client_runtime_bridge:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Client runtime bridge is not enabled on this server",
        )

    # Register the device
    device = await service.register_or_update_device(
        user_id=user.id,
        device_identifier=request.device_identifier,
        display_name=request.display_name,
        platform=request.platform,
        app_version=request.app_version,
        runtime_version=request.runtime_version,
        capabilities=request.capabilities,
    )

    # Generate a session ID
    session_id = secrets.token_urlsafe(32)
    await service.issue_runtime_session_id(device.id, session_id)

    return DeviceRegistrationResponse(
        device_id=str(device.id),
        session_id=session_id,
        status="registered",
        message=f"Device {request.display_name} registered successfully",
    )


@router.post("/heartbeat", response_model=DeviceHeartbeatResponse)
async def send_heartbeat(
    request: DeviceHeartbeatRequest,
    user: User = Depends(get_current_user),
    service: ClientDeviceService = Depends(get_device_service),
) -> DeviceHeartbeatResponse:
    """
    Send a heartbeat to keep the device session alive.

    Devices should send heartbeats at the configured interval to maintain
    their online status.
    """
    device = service.repository.get_by_id(request.device_id)
    if not device or device.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found",
        )

    session = service.get_active_session(request.device_id)
    if not session or session.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device session not found or not active",
        )

    # Update heartbeat
    success = await service.update_heartbeat(request.device_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device session not found or not active",
        )

    return DeviceHeartbeatResponse(
        status="ok",
        last_heartbeat=service.get_active_session(request.device_id).last_heartbeat.isoformat(),
    )


@router.get("/me", response_model=list[DeviceInfoResponse])
async def list_my_devices(
    user: User = Depends(get_current_user),
    service: ClientDeviceService = Depends(get_device_service),
) -> list[DeviceInfoResponse]:
    """
    List all devices registered for the current user.
    """
    devices = await service.list_user_devices(user.id, include_offline=True)

    return [
        DeviceInfoResponse(
            id=str(device.id),
            device_identifier=device.device_identifier,
            display_name=device.display_name,
            platform=device.platform,
            app_version=device.app_version,
            runtime_version=device.runtime_version,
            status=device.status.value,
            last_seen_at=device.last_seen_at.isoformat() if device.last_seen_at else None,
            created_at=device.created_at.isoformat(),
        )
        for device in devices
    ]


@router.put("/{device_id}/tool-catalog", response_model=ToolCatalogUpdateResponse)
async def update_tool_catalog(
    device_id: str,
    request: ToolCatalogUpdateRequest,
    user: User = Depends(get_current_user),
    service: ClientDeviceService = Depends(get_device_service),
) -> ToolCatalogUpdateResponse:
    """
    Update the tool catalog for a device.

    The client sends a sanitized catalog of available tools (native and MCP).
    """
    device_uuid = UUID(device_id)
    if request.device_id != device_uuid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request device_id does not match path device_id",
        )

    # Verify device ownership
    session = service.get_active_session(device_uuid)
    if not session or session.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device not found or access denied",
        )

    # Update catalog
    success = await service.update_tool_catalog(device_uuid, request.catalog)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device session not found",
        )

    return ToolCatalogUpdateResponse(
        status="updated",
        tool_count=len(request.catalog.get("tools", [])),
    )


@router.put("/{device_id}/skill-catalog", response_model=SkillCatalogUpdateResponse)
async def update_skill_catalog(
    device_id: str,
    request: SkillCatalogUpdateRequest,
    user: User = Depends(get_current_user),
    service: ClientDeviceService = Depends(get_device_service),
) -> SkillCatalogUpdateResponse:
    """
    Update the skill catalog for a device.

    The client sends a catalog of available local skills.
    """
    device_uuid = UUID(device_id)
    if request.device_id != device_uuid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request device_id does not match path device_id",
        )

    # Verify device ownership
    session = service.get_active_session(device_uuid)
    if not session or session.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device not found or access denied",
        )

    # Update catalog
    success = await service.update_skill_catalog(device_uuid, request.catalog)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device session not found",
        )

    return SkillCatalogUpdateResponse(
        status="updated",
        skill_count=len(request.catalog.get("skills", [])),
    )


@router.get("/{device_id}")
async def get_device_info(
    device_id: str,
    user: User = Depends(get_current_user),
    service: ClientDeviceService = Depends(get_device_service),
) -> DeviceInfoResponse:
    """
    Get information about a specific device.
    """
    device_uuid = UUID(device_id)
    device = service.repository.get_by_id(device_uuid)

    if not device or device.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found",
        )

    return DeviceInfoResponse(
        id=str(device.id),
        device_identifier=device.device_identifier,
        display_name=device.display_name,
        platform=device.platform,
        app_version=device.app_version,
        runtime_version=device.runtime_version,
        status=device.status.value,
        last_seen_at=device.last_seen_at.isoformat() if device.last_seen_at else None,
        created_at=device.created_at.isoformat(),
    )
