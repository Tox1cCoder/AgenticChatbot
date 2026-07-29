"""Authenticated delivery endpoint for selected third-party rich images."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.auth import get_current_user_id
from app.core.container import Container
from app.repositories.web_image_reference import WebImageReferenceRepository
from app.services.web_image_service import (
    WebImageRejected,
    WebImageService,
    WebImageUpstreamFailure,
)

router = APIRouter(prefix="/web-images", tags=["web-images"])


def _get_repository() -> WebImageReferenceRepository:
    return Container().web_image_reference_repository()


def _get_service() -> WebImageService:
    return Container().web_image_service()


@router.get("/{image_id}")
async def read_web_image(
    image_id: UUID,
    current_user_id: UUID = Depends(get_current_user_id),
    repository: WebImageReferenceRepository = Depends(_get_repository),
    service: WebImageService = Depends(_get_service),
) -> Response:
    record = await repository.aget_for_user(image_id, current_user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    try:
        image = await service.fetch(record)
    except WebImageUpstreamFailure as exc:
        code = (
            status.HTTP_504_GATEWAY_TIMEOUT
            if exc.reason == "timeout"
            else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(status_code=code, detail="Visual unavailable") from exc
    except WebImageRejected as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Visual unavailable",
        ) from exc
    return Response(
        content=image.content,
        media_type=image.media_type,
        headers={
            "Cache-Control": "private, max-age=300",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'",
        },
    )
