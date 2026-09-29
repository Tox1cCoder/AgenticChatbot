"""Per-user read endpoint for externalized chat images."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.auth import get_current_user_id
from app.repositories.chat_image import ChatImageRepository
from app.services.chat_image_service import ChatImageStorageService

router = APIRouter(prefix="/chat-images", tags=["chat-images"])


def _container():
    """The process-wide container, not a fresh ``Container()``.

    A fresh declarative container rebuilds its ``Database`` singleton, so a
    per-request one opened a new engine and connection pool on every image
    read. Imported lazily for the reason given in ``app.api.tool_result_blobs``.
    """
    from app.core.container import get_container

    return get_container()


def _get_repository() -> ChatImageRepository:
    return _container().chat_image_repository()


def _get_service() -> ChatImageStorageService:
    return _container().chat_image_service()


@router.get("/{image_id}")
async def read_chat_image(
    image_id: UUID,
    current_user_id: UUID = Depends(get_current_user_id),
    repository: ChatImageRepository = Depends(_get_repository),
    service: ChatImageStorageService = Depends(_get_service),
) -> Response:
    record = repository.get_for_user(image_id, current_user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    # The content type is whatever ``image/*`` the uploader declared, which
    # includes ``image/svg+xml``. Opened directly, an SVG would run script on
    # the API origin; these headers keep it an inert image.
    return Response(
        content=service.read_bytes(record),
        media_type=record.content_type,
        headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'",
        },
    )
