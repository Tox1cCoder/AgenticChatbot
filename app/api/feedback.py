from typing import List, Optional
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.core.auth import get_current_user_id
from app.services.feedback_service import FeedbackService
from app.schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackRead

router = APIRouter(prefix="/messages", tags=["feedback"])


@router.post(
    "/{message_id}/feedback",
    response_model=FeedbackRead,
    status_code=status.HTTP_201_CREATED,
)
@inject
async def create_feedback(
    message_id: UUID,
    feedback_data: FeedbackCreate,
    feedback_service: Annotated[
        FeedbackService, Depends(Provide[Container.feedback_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
) -> FeedbackRead:
    """Create new feedback for a message or update existing feedback"""
    # Set the message_id from the URL path
    feedback_data.message_id = message_id
    return feedback_service.create_feedback(feedback_data, user_id)


@router.get("/{message_id}/feedback/{feedback_id}", response_model=FeedbackRead)
@inject
async def get_feedback(
    message_id: UUID,
    feedback_id: UUID,
    feedback_service: Annotated[
        FeedbackService, Depends(Provide[Container.feedback_service])
    ],
) -> FeedbackRead:
    """Get specific feedback for a message"""
    return feedback_service.get_feedback_by_id(feedback_id)


@router.get("/{message_id}/feedback", response_model=List[FeedbackRead])
@inject
async def get_message_feedback(
    message_id: UUID,
    feedback_service: Annotated[
        FeedbackService, Depends(Provide[Container.feedback_service])
    ],
    skip: int = 0,
    limit: int = 100,
) -> List[FeedbackRead]:
    """Get all feedback for a message"""
    return feedback_service.get_feedback_by_message(message_id, skip=skip, limit=limit)


@router.get("/user/{user_id}", response_model=List[FeedbackRead])
@inject
async def get_user_feedback(
    user_id: UUID,
    feedback_service: Annotated[
        FeedbackService, Depends(Provide[Container.feedback_service])
    ],
    authenticated_user_id: UUID = Depends(get_current_user_id),
    skip: int = 0,
    limit: int = 100,
) -> List[FeedbackRead]:
    """Get all feedback by authenticated user (user_id must match authenticated user)"""
    # Validate user can only access their own feedback
    if user_id != authenticated_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: can only access your own feedback",
        )
    return feedback_service.get_feedback_by_user(user_id, skip=skip, limit=limit)


@router.get("/{message_id}/feedback/user/{user_id}", response_model=FeedbackRead)
@inject
async def get_user_feedback_for_message(
    message_id: UUID,
    user_id: UUID,
    feedback_service: Annotated[
        FeedbackService, Depends(Provide[Container.feedback_service])
    ],
    authenticated_user_id: UUID = Depends(get_current_user_id),
) -> FeedbackRead:
    """Get authenticated user's feedback for a message (user_id must match authenticated user)"""
    # Validate user can only access their own feedback
    if user_id != authenticated_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: can only access your own feedback",
        )
    feedback = feedback_service.get_user_feedback_for_message(message_id, user_id)
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found"
        )
    return feedback


@router.get("/{message_id}/feedback/stats")
@inject
async def get_message_rating_stats(
    message_id: UUID,
    feedback_service: Annotated[
        FeedbackService, Depends(Provide[Container.feedback_service])
    ],
) -> dict:
    """Get rating statistics for a message"""
    return feedback_service.get_message_rating_stats(message_id)


@router.put("/{message_id}/feedback/{feedback_id}", response_model=FeedbackRead)
@inject
async def update_feedback(
    message_id: UUID,
    feedback_id: UUID,
    feedback_data: FeedbackUpdate,
    feedback_service: Annotated[
        FeedbackService, Depends(Provide[Container.feedback_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
) -> FeedbackRead:
    """Update feedback (requires user ownership)"""
    return feedback_service.update_feedback(feedback_id, user_id, feedback_data)
