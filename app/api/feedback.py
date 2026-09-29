from uuid import UUID

from fastapi import APIRouter, status

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.feedback_service_interface import IFeedbackService
from app.interfaces.message_service_interface import IMessageService
from app.schemas.feedback import FeedbackCreate, FeedbackRead, FeedbackUpdate
from app.schemas.responses import ApiResponse

router = APIRouter(prefix="/messages", tags=["feedbacks"])


def _require_message_access(
    message_service: IMessageService, message_id: UUID, user_id: UUID
) -> None:
    """Refuse a message the caller does not own (404 missing, 403 someone else's).

    The feedback service only checks that the message exists, so without this
    any caller could read another user's rating and comment, or write a rating
    onto their message; the stats and list routes did not even authenticate.
    """
    message_service.get_by_id(message_id, user_id)


@router.post(
    "/{message_id}/feedbacks",
    response_model=ApiResponse[FeedbackRead],
    status_code=status.HTTP_201_CREATED,
)
@AppAutoInjector.auto_inject()
async def create_feedback(
    message_id: UUID,
    feedback_data: FeedbackCreate,
    feedback_service: IFeedbackService,
    message_service: IMessageService,
    user_id: UUID,
) -> ApiResponse[FeedbackRead]:
    """Create a new feedback for a message or update existing feedback"""
    _require_message_access(message_service, message_id, user_id)
    feedback_data.message_id = message_id
    result = feedback_service.create_feedback(feedback_data, user_id)
    return ApiResponse(success=True, message="Feedback created successfully", data=result)


@router.get("/{message_id}/feedbacks/stats", response_model=ApiResponse[dict])
@AppAutoInjector.auto_inject()
async def get_message_rating_stats(
    message_id: UUID,
    feedback_service: IFeedbackService,
    message_service: IMessageService,
    user_id: UUID,
) -> ApiResponse[dict]:
    """Get rating statistics for a message"""
    _require_message_access(message_service, message_id, user_id)
    result = feedback_service.get_message_rating_stats(message_id)
    return ApiResponse(
        success=True,
        message="Rating statistics retrieved successfully",
        data=result,
    )


@router.get("/{message_id}/feedbacks/user", response_model=ApiResponse[FeedbackRead])
@AppAutoInjector.auto_inject()
async def get_user_feedback_for_message(
    message_id: UUID,
    user_id: UUID,
    feedback_service: IFeedbackService,
    message_service: IMessageService,
) -> ApiResponse[FeedbackRead]:
    """Get authenticated user's feedback for a message"""
    _require_message_access(message_service, message_id, user_id)
    feedback = feedback_service.get_user_feedback_for_message(message_id, user_id)
    return ApiResponse(
        success=True,
        message="User feedback retrieved successfully",
        data=feedback,
    )


@router.get("/{message_id}/feedbacks", response_model=ApiResponse[list[FeedbackRead]])
@AppAutoInjector.auto_inject()
async def get_message_feedbacks(
    message_id: UUID,
    feedback_service: IFeedbackService,
    message_service: IMessageService,
    user_id: UUID,
) -> ApiResponse[list[FeedbackRead]]:
    """Get all feedbacks for a message"""
    _require_message_access(message_service, message_id, user_id)
    result = feedback_service.get_by_message(message_id)
    feedback_list = [result] if result else []
    return ApiResponse(
        success=True,
        message="Message feedbacks retrieved successfully",
        data=feedback_list,
    )


@router.put("/{message_id}/feedbacks/{feedback_id}", response_model=ApiResponse[FeedbackRead])
@AppAutoInjector.auto_inject()
async def update_feedback(
    message_id: UUID,
    feedback_id: UUID,
    feedback_data: FeedbackUpdate,
    feedback_service: IFeedbackService,
    user_id: UUID,
) -> ApiResponse[FeedbackRead]:
    """Update feedback (requires user ownership)"""
    result = feedback_service.update_feedback(feedback_id, user_id, feedback_data)
    return ApiResponse(
        success=True, code="ok", message="Feedback updated successfully", data=result
    )
