from typing import List
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, status

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.core.auth import get_current_user_id
from app.interfaces.feedback_service_interface import IFeedbackService
from app.schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackRead
from app.schemas.responses import ApiResponse

router = APIRouter(prefix="/messages", tags=["feedback"])


@router.post(
    "/{message_id}/feedback",
    response_model=ApiResponse[FeedbackRead],
    status_code=status.HTTP_201_CREATED,
)
@inject
async def create_feedback(
    message_id: UUID,
    feedback_data: FeedbackCreate,
    feedback_service: Annotated[
        IFeedbackService, Depends(Provide[Container.feedback_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
) -> ApiResponse[FeedbackRead]:
    """Create new feedback for a message or update existing feedback"""
    feedback_data.message_id = message_id
    result = feedback_service.create_feedback(feedback_data, user_id)
    return ApiResponse(data=result, message="Feedback created successfully")


@router.get("/{message_id}/feedback/stats")
@inject
async def get_message_rating_stats(
    message_id: UUID,
    feedback_service: Annotated[
        IFeedbackService, Depends(Provide[Container.feedback_service])
    ],
) -> dict:
    """Get rating statistics for a message"""
    return feedback_service.get_message_rating_stats(message_id)


@router.get(
    "/{message_id}/feedback/{feedback_id}", response_model=ApiResponse[FeedbackRead]
)
@inject
async def get_feedback(
    message_id: UUID,
    feedback_id: UUID,
    feedback_service: Annotated[
        IFeedbackService, Depends(Provide[Container.feedback_service])
    ],
) -> ApiResponse[FeedbackRead]:
    """Get specific feedback for a message"""
    result = feedback_service.get_by_id(feedback_id)
    return ApiResponse(data=result, message="Feedback retrieved successfully")


@router.get("/{message_id}/feedback", response_model=ApiResponse[List[FeedbackRead]])
@inject
async def get_message_feedback(
    message_id: UUID,
    feedback_service: Annotated[
        IFeedbackService, Depends(Provide[Container.feedback_service])
    ],
    skip: int = 0,
    limit: int = 100,
) -> ApiResponse[List[FeedbackRead]]:
    """Get all feedback for a message"""
    result = feedback_service.get_by_message(message_id, skip=skip, limit=limit)
    return ApiResponse(
        data=result, message="Message feedback retrieved successfully"
    )


@router.get("/user/{user_id}", response_model=ApiResponse[List[FeedbackRead]])
@inject
async def get_user_feedback(
    user_id: UUID,
    feedback_service: Annotated[
        IFeedbackService, Depends(Provide[Container.feedback_service])
    ],
    authenticated_user_id: UUID = Depends(get_current_user_id),
    skip: int = 0,
    limit: int = 100,
) -> ApiResponse[List[FeedbackRead]]:
    """Get all feedback by authenticated user (user_id must match authenticated user)"""
    result = feedback_service.get_feedback_by_user(user_id, skip=skip, limit=limit)
    return ApiResponse(data=result, message="User feedback retrieved successfully")


@router.get(
    "/{message_id}/feedback/user/{user_id}", response_model=ApiResponse[FeedbackRead]
)
@inject
async def get_user_feedback_for_message(
    message_id: UUID,
    user_id: UUID,
    feedback_service: Annotated[
        IFeedbackService, Depends(Provide[Container.feedback_service])
    ],
    authenticated_user_id: UUID = Depends(get_current_user_id),
) -> ApiResponse[FeedbackRead]:
    """Get authenticated user's feedback for a message (user_id must match authenticated user)"""
    feedback = feedback_service.get_user_feedback_for_message(message_id, user_id)
    return ApiResponse(
        data=feedback, message="User feedback retrieved successfully"
    )


@router.put(
    "/{message_id}/feedback/{feedback_id}", response_model=ApiResponse[FeedbackRead]
)
@inject
async def update_feedback(
    message_id: UUID,
    feedback_id: UUID,
    feedback_data: FeedbackUpdate,
    feedback_service: Annotated[
        IFeedbackService, Depends(Provide[Container.feedback_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
) -> ApiResponse[FeedbackRead]:
    """Update feedback (requires user ownership)"""
    result = feedback_service.update_feedback(feedback_id, user_id, feedback_data)
    return ApiResponse(data=result, message="Feedback updated successfully")