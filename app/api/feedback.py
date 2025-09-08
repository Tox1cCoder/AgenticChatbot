from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.feedback import FeedbackService
from app.schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackRead

router = APIRouter(prefix="/feedback", tags=["feedback"])


def get_feedback_service(db: Session = Depends(get_db)) -> FeedbackService:
    """Dependency to get FeedbackService instance"""
    return FeedbackService(db)


@router.post(
    "/user/{user_id}", response_model=FeedbackRead, status_code=status.HTTP_201_CREATED
)
async def create_feedback(
    user_id: UUID,
    feedback_data: FeedbackCreate,
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackRead:
    """Create new feedback for a message"""
    return feedback_service.create_feedback(feedback_data, user_id)


@router.get("/{feedback_id}", response_model=FeedbackRead)
async def get_feedback(
    feedback_id: UUID, feedback_service: FeedbackService = Depends(get_feedback_service)
) -> FeedbackRead:
    """Get feedback by ID"""
    return feedback_service.get_feedback_by_id(feedback_id)


@router.get("/message/{message_id}", response_model=List[FeedbackRead])
async def get_message_feedback(
    message_id: UUID,
    skip: int = 0,
    limit: int = 100,
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> List[FeedbackRead]:
    """Get all feedback for a message"""
    return feedback_service.get_feedback_by_message(message_id, skip=skip, limit=limit)


@router.get("/user/{user_id}", response_model=List[FeedbackRead])
async def get_user_feedback(
    user_id: UUID,
    skip: int = 0,
    limit: int = 100,
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> List[FeedbackRead]:
    """Get all feedback by a user"""
    return feedback_service.get_feedback_by_user(user_id, skip=skip, limit=limit)


@router.get("/message/{message_id}/user/{user_id}", response_model=FeedbackRead)
async def get_user_feedback_for_message(
    message_id: UUID,
    user_id: UUID,
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackRead:
    """Get specific user's feedback for a message"""
    feedback = feedback_service.get_user_feedback_for_message(message_id, user_id)
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found"
        )
    return feedback


@router.get("/message/{message_id}/stats")
async def get_message_rating_stats(
    message_id: UUID, feedback_service: FeedbackService = Depends(get_feedback_service)
) -> dict:
    """Get rating statistics for a message"""
    return feedback_service.get_message_rating_stats(message_id)


@router.put("/{feedback_id}", response_model=FeedbackRead)
async def update_feedback(
    feedback_id: UUID,
    user_id: UUID,
    feedback_data: FeedbackUpdate,
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackRead:
    """Update feedback (requires user ownership)"""
    return feedback_service.update_feedback(feedback_id, user_id, feedback_data)


@router.delete("/{feedback_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_feedback(
    feedback_id: UUID,
    user_id: UUID,
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> None:
    """Delete feedback (requires user ownership)"""
    feedback_service.delete_feedback(feedback_id, user_id)
