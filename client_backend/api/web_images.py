"""Protected selected web-image media proxy for the local sidecar."""

from uuid import UUID

from fastapi import APIRouter, Depends, Response

from client_backend.api.common import proxy_media_request
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload

router = APIRouter(prefix="/web-images", tags=["web-images"])


@router.get("/{image_id}")
async def read_web_image(
    image_id: UUID,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    """Proxy one authenticated, user-owned web-image reference upstream."""
    return await proxy_media_request(upstream_path=f"/web-images/{image_id}")
