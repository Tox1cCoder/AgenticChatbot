"""
Runtime management endpoints for the local client backend.
"""

from fastapi import APIRouter, Depends

from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.services.runtime_bridge import get_runtime_bridge

router = APIRouter(prefix="/runtime", tags=["runtime"])


@router.get("/status")
async def get_runtime_status(
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict:
    """Return the current runtime bridge state."""
    return get_runtime_bridge().get_runtime_state().model_dump(mode="json")


@router.post("/connect")
async def connect_runtime(
    wait_for_connection: bool = True,
    timeout_seconds: int | None = None,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict:
    """Start or reconnect the runtime bridge."""
    connected = await get_runtime_bridge().start(
        wait_for_connection=wait_for_connection,
        timeout_seconds=timeout_seconds,
    )
    return {
        "started": connected if wait_for_connection else True,
        "waited_for_connection": wait_for_connection,
        "runtime_state": get_runtime_bridge().get_runtime_state().model_dump(mode="json"),
    }


@router.post("/disconnect")
async def disconnect_runtime(
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict:
    """Stop the runtime bridge."""
    await get_runtime_bridge().stop()
    return {"status": "stopped"}


@router.post("/refresh-catalogs")
async def refresh_catalogs(
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict:
    """Refresh synced tool and skill catalogs."""
    await get_runtime_bridge().refresh_catalogs()
    return {
        "status": "refreshed",
        "runtime_state": get_runtime_bridge().get_runtime_state().model_dump(mode="json"),
    }
