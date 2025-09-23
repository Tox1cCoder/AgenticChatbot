from typing import Optional
from uuid import UUID
from contextlib import AbstractContextManager
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.user import User
from app.repositories.command_strategy import DefaultCommandStrategy
from app.repositories.query_strategy import DefaultQueryStrategy
from app.schemas.user import UserCreate, UserUpdate


class UserCRUDStrategy(
    DefaultCommandStrategy[User, UserCreate, UserUpdate], DefaultQueryStrategy[User]
):
    """Custom CRUD strategy for User operations"""

    def __init__(self, model: type[User]):
        DefaultCommandStrategy.__init__(self, model)
        DefaultQueryStrategy.__init__(self, model)

    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        """Get user by email address"""
        stmt = select(User).where(User.email == email)
        return db.execute(stmt).scalar_one_or_none()

    def get_by_username(self, db: Session, username: str) -> Optional[User]:
        """Get user by username"""
        stmt = select(User).where(User.username == username)
        return db.execute(stmt).scalar_one_or_none()

    def email_exists(
        self, db: Session, email: str, exclude_id: Optional[UUID] = None
    ) -> bool:
        """Check if email already exists"""
        stmt = select(User.id).where(User.email == email)
        if exclude_id:
            stmt = stmt.where(User.id != exclude_id)
        return db.execute(stmt).scalar() is not None

    def username_exists(
        self, db: Session, username: str, exclude_id: Optional[UUID] = None
    ) -> bool:
        """Check if username already exists"""
        stmt = select(User.id).where(User.username == username)
        if exclude_id:
            stmt = stmt.where(User.id != exclude_id)
        return db.execute(stmt).scalar() is not None


class UserRepository:
    """Repository for User model using strategy pattern"""

    def __init__(self, session_factory: callable):
        """Initialize repository with session factory for dependency injection."""
        self.session_factory = session_factory
        self._crud_strategy = UserCRUDStrategy(User)

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_email(session, email)

    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_username(session, username)

    def email_exists(self, email: str, exclude_id: Optional[UUID] = None) -> bool:
        """Check if email already exists"""
        with self.session_factory() as session:
            return self._crud_strategy.email_exists(session, email, exclude_id)

    def username_exists(self, username: str, exclude_id: Optional[UUID] = None) -> bool:
        """Check if username already exists"""
        with self.session_factory() as session:
            return self._crud_strategy.username_exists(session, username, exclude_id)

    def create(self, input_schema: UserCreate) -> User:
        """Create a new user"""
        with self.session_factory() as session:
            return self._crud_strategy.create(session, input_schema)

    def get_by_id(self, id: UUID) -> Optional[User]:
        """Get user by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    def get_all(self, skip: int = 0, limit: int = 100) -> list[User]:
        """Get all users with pagination"""
        with self.session_factory() as session:
            return self._crud_strategy.get_all(session, skip, limit)

    def update(self, id: UUID, input_schema: UserUpdate) -> Optional[User]:
        """Update user by ID"""
        with self.session_factory() as session:
            db_obj = self._crud_strategy.get_by_id(session, id)
            if db_obj is None:
                return None
            return self._crud_strategy.update(session, db_obj, input_schema)

    def delete(self, id: UUID) -> bool:
        """Delete user by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.delete(session, id)

    def exists(self, id: UUID) -> bool:
        """Check if user exists"""
        with self.session_factory() as session:
            return self._crud_strategy.exists(session, id)
