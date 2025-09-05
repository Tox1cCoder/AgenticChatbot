from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.user import User
from app.repositories.base import BaseRepository
from app.schemas.user import UserCreate, UserUpdate


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """Repository for User model with custom methods"""
    
    def __init__(self, db: Session):
        super().__init__(User, db)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        stmt = select(User).where(User.email == email)
        return self.db.execute(stmt).scalar_one_or_none()
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        stmt = select(User).where(User.username == username)
        return self.db.execute(stmt).scalar_one_or_none()
    
    def email_exists(self, email: str, exclude_id: Optional[int] = None) -> bool:
        """Check if email already exists"""
        stmt = select(User.id).where(User.email == email)
        if exclude_id:
            stmt = stmt.where(User.id != exclude_id)
        return self.db.execute(stmt).scalar() is not None
    
    def username_exists(self, username: str, exclude_id: Optional[int] = None) -> bool:
        """Check if username already exists"""
        stmt = select(User.id).where(User.username == username)
        if exclude_id:
            stmt = stmt.where(User.id != exclude_id)
        return self.db.execute(stmt).scalar() is not None
