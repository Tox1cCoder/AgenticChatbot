from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.container import get_container, DIContainer
from app.services.message_service import MessageService
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead

router = APIRouter(prefix="/messages", tags=["messages"])


def get_message_service(db: Session = Depends(get_db)) -> MessageService:
    """Dependency to get MessageService instance with proper DI"""
    container = get_container()
    container.set_session(db)
    return container.get("message_service")


@router.post("/", response_model=MessageRead, status_code=status.HTTP_201_CREATED)
async def create_message(
    message_data: MessageCreate,
    message_service: MessageService = Depends(get_message_service),
) -> MessageRead:
    """Create a new message (will auto-generate bot response if role is 'user')"""
    return message_service.create_message(message_data)


@router.get("/{message_id}", response_model=MessageRead)
async def get_message(
    message_id: UUID, message_service: MessageService = Depends(get_message_service)
) -> MessageRead:
    """Get message by ID"""
    return message_service.get_message_by_id(message_id)


@router.get("/conversation/{conversation_id}", response_model=List[MessageRead])
async def get_conversation_messages(
    conversation_id: UUID,
    user_id: UUID,
    skip: int = 0,
    limit: int = 100,
    message_service: MessageService = Depends(get_message_service),
) -> List[MessageRead]:
    """Get messages for a conversation (requires user ownership)"""
    return message_service.get_conversation_messages(
        conversation_id, user_id, skip=skip, limit=limit
    )


@router.get("/conversation/{conversation_id}/thread", response_model=List[MessageRead])
async def get_conversation_thread(
    conversation_id: UUID,
    user_id: UUID,
    message_service: MessageService = Depends(get_message_service),
) -> List[MessageRead]:
    """Get conversation thread ordered by timestamp (requires user ownership)"""
    return message_service.get_conversation_thread(conversation_id, user_id)
