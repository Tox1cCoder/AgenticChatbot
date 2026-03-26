"""
Health and status API endpoints for the client backend.
"""

import time

from fastapi import APIRouter

from client_backend import __version__
from client_backend.core.config import client_settings
from client_backend.schemas.runtime import (
    DeviceInfo,
    HealthCheckResponse,
    HealthStatus,
    RuntimeState,
    RuntimeStatus,
)
from client_backend.services.runtime_bridge import get_runtime_bridge

router = APIRouter(tags=["health"])

# Track startup time
_startup_time = time.time()


def get_runtime_state() -> RuntimeState:
    """Get the current runtime state."""
    return get_runtime_bridge().get_runtime_state()


def set_runtime_state(state: RuntimeState) -> None:
    """Compatibility shim kept for older callers."""
    bridge = get_runtime_bridge()
    bridge._state = state


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for monitoring.

    Returns basic health status and component checks.
    """
    runtime_state = get_runtime_state()
    device_info = get_runtime_bridge().get_device_info()

    checks = {
        "config_loaded": True,
        "profile_accessible": _check_profile_accessible(),
        "server_reachable": runtime_state.status == RuntimeStatus.CONNECTED,
    }

    # Determine overall status
    if all(checks.values()):
        status = HealthStatus.HEALTHY
    elif checks["config_loaded"] and checks["profile_accessible"]:
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.UNHEALTHY

    return HealthCheckResponse(
        status=status,
        version=__version__,
        uptime_seconds=time.time() - _startup_time,
        server_connected=runtime_state.status == RuntimeStatus.CONNECTED,
        device_id=device_info.device_id,
        device_identifier=device_info.device_identifier,
        checks=checks,
    )


@router.get("/health/ready")
async def readiness_check() -> dict:
    """
    Kubernetes-style readiness probe.

    Returns 200 if the service is ready to accept traffic.
    """
    return {"ready": True}


@router.get("/health/live")
async def liveness_check() -> dict:
    """
    Kubernetes-style liveness probe.

    Returns 200 if the service is alive.
    """
    return {"alive": True}


@router.get("/status", response_model=RuntimeState)
async def get_status() -> RuntimeState:
    """
    Get the current runtime status.

    Returns detailed information about the runtime connection state.
    """
    return get_runtime_state()


@router.get("/device", response_model=DeviceInfo)
async def get_device_info() -> DeviceInfo:
    """
    Get information about the local device.
    """
    return get_runtime_bridge().get_device_info()


def _check_profile_accessible() -> bool:
    """Check if the profile directory is accessible."""
    from pathlib import Path

    try:
        profile_path = Path(client_settings.profile_root)
        return profile_path.exists() and profile_path.is_dir()
    except Exception:
        return False
