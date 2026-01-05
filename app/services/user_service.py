from __future__ import annotations
from typing import List, Optional
from uuid import UUID

from app.repositories.user import UserRepository
from app.schemas.user import UserCreate, UserUpdate, UserRead, UserInDB
from app.factories.user_factory import UserFactory
from app.utils.validation.user_validation import UserValidationUtils
from app.interfaces.user_service_interface import IUserService


class UserService(IUserService):
    """Service layer for User operations"""

    def __init__(
        self,
        user_repository: UserRepository,
        user_validation_utils: UserValidationUtils,
    ):
        self.repository = user_repository
        self.validation_utils = user_validation_utils

    def create_user(self, user_create_data: UserCreate) -> UserRead:
        self.validation_utils.validate_email_and_username_availability(
            user_create_data.email, user_create_data.username
        )
        user_entity = UserFactory.create_from_schema(user_create_data)
        created_user = self.repository.create(user_entity)
        return UserRead.model_validate(created_user)

    def get_by_id(self, user_id: UUID) -> UserRead:
        self.validation_utils.validate_user_exists(user_id)
        user_entity = self.repository.get_by_id(user_id)
        return UserRead.model_validate(user_entity)

    def get_by_email(self, email: str) -> Optional[UserRead]:
        user_entity = self.repository.get_by_email(email)
        return UserRead.model_validate(user_entity) if user_entity else None

    def get_by_email_with_password(self, email: str) -> Optional[UserInDB]:
        user_entity = self.repository.get_by_email(email)
        return UserInDB.model_validate(user_entity) if user_entity else None

    def get_by_username(self, username: str) -> Optional[UserRead]:
        user_entity = self.repository.get_by_username(username)
        return UserRead.model_validate(user_entity) if user_entity else None

    def get_all(self, skip: int = 0, limit: int = 100) -> List[UserRead]:
        user_entities = self.repository.get_all(skip=skip, limit=limit)
        return [UserRead.model_validate(user_entity) for user_entity in user_entities]

    def update_user(self, user_id: UUID, user_update_data: UserUpdate) -> UserRead:
        self.validation_utils.validate_user_exists(user_id)
        user_entity = self.repository.get_by_id(user_id)

        if user_update_data.email or user_update_data.username:
            email_to_check = user_update_data.email or user_entity.email
            username_to_check = user_update_data.username or user_entity.username

            self.validation_utils.validate_email_and_username_availability(
                email_to_check, username_to_check, exclude_user_id=user_id
            )

        updated_user = self.repository.update(user_entity.id, user_update_data)
        return UserRead.model_validate(updated_user)
