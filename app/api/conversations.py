from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.conversation import ConversationService
from app.schemas.conversation import ConversationCreate, ConversationUpdate, ConversationRead

router = APIRouter(prefix="/conversations", tags=["conversations"])


def get_conversation_service(db: Session = Depends(get_db)) -> ConversationService:
    """Dependency to get ConversationService instance"""
    return ConversationService(db)


@router.post("/", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation_data: ConversationCreate,
    conversation_service: ConversationService = Depends(get_conversation_service)
) -> ConversationRead:
    """Create a new conversation"""
    return conversation_service.create_conversation(conversation_data)


@router.get("/{conversation_id}", response_model=ConversationRead)
async def get_conversation(
    conversation_id: int,
    conversation_service: ConversationService = Depends(get_conversation_service)
) -> ConversationRead:
    """Get conversation by ID"""
    return conversation_service.get_conversation_by_id(conversation_id)


@router.get("/user/{user_id}", response_model=List[ConversationRead])
async def get_user_conversations(
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    conversation_service: ConversationService = Depends(get_conversation_service)
) -> List[ConversationRead]:
    """Get all conversations for a user"""
    return conversation_service.get_user_conversations(user_id, skip=skip, limit=limit)


@router.get("/{conversation_id}/with-messages", response_model=ConversationRead)
async def get_conversation_with_messages(
    conversation_id: int,
    user_id: int,
    conversation_service: ConversationService = Depends(get_conversation_service)
) -> ConversationRead:
    """Get conversation with messages (requires user ownership)"""
    return conversation_service.get_conversation_with_messages(conversation_id, user_id)


@router.put("/{conversation_id}", response_model=ConversationRead)
async def update_conversation(
    conversation_id: int,
    user_id: int,
    conversation_data: ConversationUpdate,
    conversation_service: ConversationService = Depends(get_conversation_service)
) -> ConversationRead:
    """Update conversation (requires user ownership)"""
    return conversation_service.update_conversation(conversation_id, user_id, conversation_data)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    user_id: int,
    conversation_service: ConversationService = Depends(get_conversation_service)
) -> None:
    """Delete conversation (requires user ownership)"""
    conversation_service.delete_conversation(conversation_id, user_id)
