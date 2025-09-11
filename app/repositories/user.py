from typing import Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.user import User
from app.repositories.strategy import Repository, DefaultCRUDStrategy
from app.schemas.user import UserCreate, UserUpdate


class UserCRUDStrategy(DefaultCRUDStrategy[User, UserCreate, UserUpdate]):
    """Custom CRUD strategy for User operations"""

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


class UserRepository(Repository[User, UserCreate, UserUpdate]):
    """Repository for User model using strategy pattern"""

    def __init__(self, db: Session):
        strategy = UserCRUDStrategy(User)
        super().__init__(db, strategy)

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        return self._crud_strategy.get_by_email(self.db, email)

    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self._crud_strategy.get_by_username(self.db, username)

    def email_exists(self, email: str, exclude_id: Optional[UUID] = None) -> bool:
        """Check if email already exists"""
        return self._crud_strategy.email_exists(self.db, email, exclude_id)

    def username_exists(self, username: str, exclude_id: Optional[UUID] = None) -> bool:
        """Check if username already exists"""
        return self._crud_strategy.username_exists(self.db, username, exclude_id)
