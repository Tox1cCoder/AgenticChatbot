from __future__ import annotations

from uuid import UUID

from app.factories.user_factory import UserFactory
from app.interfaces.user_service_interface import IUserService
from app.repositories.user import UserRepository
from app.schemas.user import UserCreate, UserInDB, UserRead
from app.utils.validation.user_validation import UserValidationUtils


class UserService(IUserService):
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

    def get_by_email_with_password(self, email: str) -> UserInDB | None:
        user_entity = self.repository.get_by_email(email)
        return UserInDB.model_validate(user_entity) if user_entity else None
