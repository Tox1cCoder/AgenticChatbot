from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.core.container import get_container, DIContainer
from app.services.conversation_service import ConversationService
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationRead,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


def get_conversation_service(db: Session = Depends(get_db)) -> ConversationService:
    """Dependency to get ConversationService instance"""
    container = get_container()
    container.set_session(db)
    return container.get("conversation_service")


@router.post(
    "/",
    response_model=ConversationRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    conversation_data: ConversationCreate,
    user_id: UUID,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationRead:
    """Create a new conversation for a user"""
    return conversation_service.create_conversation(conversation_data, user_id)


@router.get("/{conversation_id}", response_model=ConversationRead)
async def get_conversation(
    conversation_id: UUID,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationRead:
    """Get conversation by ID"""
    return conversation_service.get_conversation_by_id(conversation_id)


@router.get("/", response_model=List[ConversationRead])
async def get_conversations(
    user_id: UUID,
    page: int = 1,
    limit: int = 100,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> List[ConversationRead]:
    """Get all conversations for a user"""
    skip = (page - 1) * limit
    return conversation_service.get_user_conversations(user_id, skip=skip, limit=limit)


@router.put("/{conversation_id}", response_model=ConversationRead)
async def update_conversation(
    conversation_id: UUID,
    user_id: UUID,
    conversation_data: ConversationUpdate,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationRead:
    """Update conversation (requires user ownership)"""
    return conversation_service.update_conversation(
        conversation_id, user_id, conversation_data
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    user_id: UUID,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> dict:
    """Delete conversation (requires user ownership)"""
    success = conversation_service.delete_conversation(conversation_id, user_id)
    if success:
        return {"message": "Conversation deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
