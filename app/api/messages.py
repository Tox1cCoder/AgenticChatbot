from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.message import MessageService
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead

router = APIRouter(prefix="/messages", tags=["messages"])


def get_message_service(db: Session = Depends(get_db)) -> MessageService:
    """Dependency to get MessageService instance"""
    return MessageService(db)


@router.post("/", response_model=MessageRead, status_code=status.HTTP_201_CREATED)
async def create_message(
    message_data: MessageCreate,
    message_service: MessageService = Depends(get_message_service)
) -> MessageRead:
    """Create a new message (will auto-generate bot response if role is 'user')"""
    return message_service.create_message(message_data)


@router.get("/{message_id}", response_model=MessageRead)
async def get_message(
    message_id: int,
    message_service: MessageService = Depends(get_message_service)
) -> MessageRead:
    """Get message by ID"""
    return message_service.get_message_by_id(message_id)


@router.get("/conversation/{conversation_id}", response_model=List[MessageRead])
async def get_conversation_messages(
    conversation_id: int,
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    message_service: MessageService = Depends(get_message_service)
) -> List[MessageRead]:
    """Get messages for a conversation (requires user ownership)"""
    return message_service.get_conversation_messages(conversation_id, user_id, skip=skip, limit=limit)


@router.get("/conversation/{conversation_id}/history", response_model=List[MessageRead])
async def get_conversation_history(
    conversation_id: int,
    user_id: int,
    limit: int = 50,
    message_service: MessageService = Depends(get_message_service)
) -> List[MessageRead]:
    """Get recent conversation history (requires user ownership)"""
    return message_service.get_conversation_history(conversation_id, user_id, limit=limit)


@router.get("/user/{user_id}", response_model=List[MessageRead])
async def get_user_messages(
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    message_service: MessageService = Depends(get_message_service)
) -> List[MessageRead]:
    """Get messages by user"""
    return message_service.get_user_messages(user_id, skip=skip, limit=limit)


@router.put("/{message_id}", response_model=MessageRead)
async def update_message(
    message_id: int,
    user_id: int,
    message_data: MessageUpdate,
    message_service: MessageService = Depends(get_message_service)
) -> MessageRead:
    """Update message (requires user ownership)"""
    return message_service.update_message(message_id, user_id, message_data)


@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    message_id: int,
    user_id: int,
    message_service: MessageService = Depends(get_message_service)
) -> None:
    """Delete message (requires user ownership)"""
    message_service.delete_message(message_id, user_id)
