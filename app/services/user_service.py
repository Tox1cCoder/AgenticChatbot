from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
from uuid import UUID
from fastapi import HTTPException, status

from app.repositories.user import UserRepository
from app.schemas.user import UserCreate, UserUpdate, UserRead, UserInDB
from app.factories.user_factory import UserFactory
from app.services.validation_service import UserValidationService
from app.utils.user_validation import UserValidationUtils
from app.interfaces.user_service_interface import IUserService


class UserService(IUserService):
    """Service layer for User operations"""

    def __init__(
        self,
        user_repository: UserRepository,
        user_validation_utils: UserValidationUtils,
    ):
        """
        Initialize UserService with injected dependencies.

        Args:
            user_repository: Injected user repository
            user_validation_utils: Injected validation utils
        """
        self.repository = user_repository
        self.validation_utils = user_validation_utils

    def create_user(self, user_create_data: UserCreate) -> UserRead:
        """Create a new user with validation"""
        # Validate email and username availability
        is_valid, validation_errors = (
            self.validation_utils.validate_email_and_username_availability(
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
        return UserRead.dto(created_user)

    def get_by_id(self, user_id: UUID) -> UserRead:
        """Get user by ID"""
        user_entity = self.repository.get_by_id(user_id)
        if not user_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )
        return UserRead.dto(user_entity)

    def get_by_email(self, email: str) -> Optional[UserRead]:
        """Get user by email"""
        user_entity = self.repository.get_by_email(email)
        return UserRead.dto(user_entity) if user_entity else None

    def get_by_email_with_password(self, email: str) -> Optional[UserInDB]:
        """Get user by email with password hash for authentication"""
        user_entity = self.repository.get_by_email(email)
        return UserInDB.dto(user_entity) if user_entity else None

    def get_by_username(self, username: str) -> Optional[UserRead]:
        """Get user by username"""
        user_entity = self.repository.get_by_username(username)
        return UserRead.dto(user_entity) if user_entity else None

    def get_all(self, skip: int = 0, limit: int = 100) -> List[UserRead]:
        """Get all users with pagination"""
        user_entities = self.repository.get_all(skip=skip, limit=limit)
        return [UserRead.dto(user_entity) for user_entity in user_entities]

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
                self.validation_utils.validate_email_and_username_availability(
                    email_to_check, username_to_check, exclude_user_id=user_id
                )
            )

            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=", ".join(validation_errors),
                )

        updated_user = self.repository.update(user_entity, user_update_data)
        return UserRead.dto(updated_user)
