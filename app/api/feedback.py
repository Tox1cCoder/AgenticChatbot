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

router = APIRouter(prefix="/messages", tags=["feedbacks"])


@router.post(
    "/{message_id}/feedbacks",
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
    """Create a new feedback for a message or update existing feedback"""
    feedback_data.message_id = message_id
    result = feedback_service.create_feedback(feedback_data, user_id)
    return ApiResponse(
        success=True, message="Feedback created successfully", data=result
    )


@router.get("/{message_id}/feedbacks/stats", response_model=ApiResponse[dict])
@inject
async def get_message_rating_stats(
    message_id: UUID,
    feedback_service: Annotated[
        IFeedbackService, Depends(Provide[Container.feedback_service])
    ],
) -> ApiResponse[dict]:
    """Get rating statistics for a message"""
    result = feedback_service.get_message_rating_stats(message_id)
    return ApiResponse(
        success=True,
        message="Rating statistics retrieved successfully",
        data=result,
    )


@router.get(
    "/{message_id}/feedbacks/user/{user_id}", response_model=ApiResponse[FeedbackRead]
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
        success=True,
        message="User feedback retrieved successfully",
        data=feedback,
    )


@router.get("/{message_id}/feedbacks", response_model=ApiResponse[List[FeedbackRead]])
@inject
async def get_message_feedbacks(
    message_id: UUID,
    feedback_service: Annotated[
        IFeedbackService, Depends(Provide[Container.feedback_service])
    ],
) -> ApiResponse[List[FeedbackRead]]:
    """Get all feedbacks for a message"""
    result = feedback_service.get_feedbacks_by_message_id(message_id)
    return ApiResponse(
        success=True,
        message="Message feedbacks retrieved successfully",
        data=result,
    )


@router.put(
    "/{message_id}/feedbacks/{feedback_id}", response_model=ApiResponse[FeedbackRead]
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
    return ApiResponse(
        success=True, code="ok", message="Feedback updated successfully", data=result
    )
