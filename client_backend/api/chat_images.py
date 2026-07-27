"""Protected chat-image media proxy for the local sidecar.

The canonical server owns the per-user ``GET /chat-images/{id}`` read; the
desktop/Streamlit origin is this sidecar (port 8100), which historically had no
such route, so a streamed protected reference (``/chat-images/{id}``) 404'd
when fetched locally. This route proxies the credentialed upstream read: it
requires a valid local session, attaches the upstream auth, and streams the
bytes back without buffering the whole payload.

Registered at both ``/chat-images`` and ``/api/chat-images`` by
``client_backend.main`` (every compatibility router is mounted twice), so both
origins a streamed reference might use resolve.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Response

from client_backend.api.common import proxy_media_request
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload

router = APIRouter(prefix="/chat-images", tags=["chat-images"])


@router.get("/{image_id}")
async def read_chat_image(
    image_id: UUID,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    """Proxy the protected per-user image read to the canonical server.

    ``image_id`` is validated as a UUID by FastAPI, so the upstream path can
    never be steered anywhere but a canonical chat-image read.
    """
    return await proxy_media_request(upstream_path=f"/chat-images/{image_id}")
