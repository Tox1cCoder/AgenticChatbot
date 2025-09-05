from typing import List, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.user import UserCreate, UserUpdate, UserRead


class UserService:
    """Service layer for User operations"""
    
    def __init__(self, db: Session):
        self.repository = UserRepository(db)
    
    def create_user(self, user_data: UserCreate) -> UserRead:
        """Create a new user with validation"""
        # Check if email already exists
        if self.repository.email_exists(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Check if username already exists
        if self.repository.username_exists(user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        user = self.repository.create(user_data)
        return UserRead.model_validate(user)
    
    def get_user_by_id(self, user_id: int) -> UserRead:
        """Get user by ID"""
        user = self.repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return UserRead.model_validate(user)
    
    def get_user_by_email(self, email: str) -> Optional[UserRead]:
        """Get user by email"""
        user = self.repository.get_by_email(email)
        return UserRead.model_validate(user) if user else None
    
    def get_user_by_username(self, username: str) -> Optional[UserRead]:
        """Get user by username"""
        user = self.repository.get_by_username(username)
        return UserRead.model_validate(user) if user else None
    
    def get_all_users(self, skip: int = 0, limit: int = 100) -> List[UserRead]:
        """Get all users with pagination"""
        users = self.repository.get_all(skip=skip, limit=limit)
        return [UserRead.model_validate(user) for user in users]
    
    def update_user(self, user_id: int, user_data: UserUpdate) -> UserRead:
        """Update user with validation"""
        user = self.repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Validate email uniqueness if updating email
        if user_data.email and self.repository.email_exists(user_data.email, exclude_id=user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Validate username uniqueness if updating username
        if user_data.username and self.repository.username_exists(user_data.username, exclude_id=user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        updated_user = self.repository.update(user, user_data)
        return UserRead.model_validate(updated_user)
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user"""
        if not self.repository.exists(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return self.repository.delete(user_id)
