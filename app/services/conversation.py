from typing import List
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.repositories.conversation import ConversationRepository
from app.repositories.user import UserRepository
from app.schemas.conversation import ConversationCreate, ConversationUpdate, ConversationRead


class ConversationService:
    """Service layer for Conversation operations"""
    
    def __init__(self, db: Session):
        self.repository = ConversationRepository(db)
        self.user_repository = UserRepository(db)
    
    def create_conversation(self, conversation_data: ConversationCreate) -> ConversationRead:
        """Create a new conversation with validation"""
        # Validate user exists
        if not self.user_repository.exists(conversation_data.user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        conversation = self.repository.create(conversation_data)
        return ConversationRead.model_validate(conversation)
    
    def get_conversation_by_id(self, conversation_id: int) -> ConversationRead:
        """Get conversation by ID"""
        conversation = self.repository.get_by_id(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        return ConversationRead.model_validate(conversation)
    
    def get_user_conversations(self, user_id: int, skip: int = 0, limit: int = 100) -> List[ConversationRead]:
        """Get all conversations for a user"""
        # Validate user exists
        if not self.user_repository.exists(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        conversations = self.repository.get_by_user_id(user_id, skip=skip, limit=limit)
        return [ConversationRead.model_validate(conv) for conv in conversations]
    
    def get_conversation_with_messages(self, conversation_id: int, user_id: int) -> ConversationRead:
        """Get conversation with messages, ensuring user owns it"""
        if not self.repository.user_owns_conversation(user_id, conversation_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation"
            )
        
        conversation = self.repository.get_with_messages(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        return ConversationRead.model_validate(conversation)
    
    def update_conversation(self, conversation_id: int, user_id: int, conversation_data: ConversationUpdate) -> ConversationRead:
        """Update conversation with ownership validation"""
        if not self.repository.user_owns_conversation(user_id, conversation_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation"
            )
        
        conversation = self.repository.get_by_id(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        updated_conversation = self.repository.update(conversation, conversation_data)
        return ConversationRead.model_validate(updated_conversation)
    
    def delete_conversation(self, conversation_id: int, user_id: int) -> bool:
        """Delete conversation with ownership validation"""
        if not self.repository.user_owns_conversation(user_id, conversation_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation"
            )
        
        return self.repository.delete(conversation_id)
