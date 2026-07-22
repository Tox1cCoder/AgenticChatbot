"""Per-user read endpoint for externalized chat images."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.auth import get_current_user_id
from app.core.container import Container
from app.repositories.chat_image import ChatImageRepository
from app.services.chat_image_service import ChatImageStorageService

router = APIRouter(prefix="/chat-images", tags=["chat-images"])


def _get_repository() -> ChatImageRepository:
    return Container().chat_image_repository()


def _get_service() -> ChatImageStorageService:
    return Container().chat_image_service()


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
    return Response(content=service.read_bytes(record), media_type=record.content_type)
