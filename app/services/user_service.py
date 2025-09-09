from typing import List, Optional
from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.repositories.user import UserRepository
from app.schemas.user import UserCreate, UserUpdate, UserRead
from app.factories.user_factory import UserFactory
from app.services.validation_service import UserValidationService


class UserService:
    """Service layer for User operations"""

    def __init__(self, db: Session):
        self.repository = UserRepository(db)
        self.validation_service = UserValidationService(db)

    def create_user(self, user_create_data: UserCreate) -> UserRead:
        """Create a new user with validation"""
        # Validate email and username availability
        is_valid, validation_errors = (
            self.validation_service.validate_email_and_username_availability(
                user_create_data.email, user_create_data.username
            )
        )

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=", ".join(validation_errors),
            )

        # Create user entity using factory
        user_entity = UserFactory.create_from_schema(user_create_data)

        # Save to repository
        created_user = self.repository.create(user_entity)
        return UserRead.model_validate(created_user)

    def get_user_by_id(self, user_id: UUID) -> UserRead:
        """Get user by ID"""
        user_entity = self.repository.get_by_id(user_id)
        if not user_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )
        return UserRead.model_validate(user_entity)

    def get_user_by_email(self, email: str) -> Optional[UserRead]:
        """Get user by email"""
        user_entity = self.repository.get_by_email(email)
        return UserRead.model_validate(user_entity) if user_entity else None

    def get_user_by_username(self, username: str) -> Optional[UserRead]:
        """Get user by username"""
        user_entity = self.repository.get_by_username(username)
        return UserRead.model_validate(user_entity) if user_entity else None

    def get_all_users(self, skip: int = 0, limit: int = 100) -> List[UserRead]:
        """Get all users with pagination"""
        user_entities = self.repository.get_all(skip=skip, limit=limit)
        return [UserRead.model_validate(user_entity) for user_entity in user_entities]

    def update_user(self, user_id: UUID, user_update_data: UserUpdate) -> UserRead:
        """Update user with validation"""
        user_entity = self.repository.get_by_id(user_id)
        if not user_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Validate email and username if being updated
        if user_update_data.email or user_update_data.username:
            email_to_check = user_update_data.email or user_entity.email
            username_to_check = user_update_data.username or user_entity.username

            is_valid, validation_errors = (
                self.validation_service.validate_email_and_username_availability(
                    email_to_check, username_to_check, exclude_user_id=user_id
                )
            )

            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=", ".join(validation_errors),
                )

        updated_user = self.repository.update(user_entity, user_update_data)
        return UserRead.model_validate(updated_user)

    def delete_user(self, user_id: UUID) -> bool:
        """Delete user"""
        if not self.validation_service.validate_user_exists(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        return self.repository.delete(user_id)
