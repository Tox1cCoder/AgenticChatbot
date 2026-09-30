from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.security import token_version
from app.core.security.token_version import TokenState
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

    def get_by_email(self, db: Session, email: str) -> User | None:
        """Get user by email address"""
        statement = select(User).where(User.email == email)
        return db.execute(statement).scalar_one_or_none()

    def get_by_username(self, db: Session, username: str) -> User | None:
        """Get user by username"""
        statement = select(User).where(User.username == username)
        return db.execute(statement).scalar_one_or_none()

    def email_exists(self, db: Session, email: str, exclude_id: UUID | None = None) -> bool:
        """Check if email already exists"""
        statement = select(User.id).where(User.email == email)
        if exclude_id:
            statement = statement.where(User.id != exclude_id)
        return db.execute(statement).scalar() is not None

    def username_exists(self, db: Session, username: str, exclude_id: UUID | None = None) -> bool:
        """Check if username already exists"""
        statement = select(User.id).where(User.username == username)
        if exclude_id:
            statement = statement.where(User.id != exclude_id)
        return db.execute(statement).scalar() is not None


class UserRepository:
    """Repository for User model using strategy pattern"""

    def __init__(self, session_factory: callable):
        """Initialize repository with session factory for dependency injection."""
        self.session_factory = session_factory
        self._crud_strategy = UserCRUDStrategy(User)

    def get_by_email(self, email: str) -> User | None:
        """Get user by email address"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_email(session, email)

    def get_by_username(self, username: str) -> User | None:
        """Get user by username"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_username(session, username)

    def email_exists(self, email: str, exclude_id: UUID | None = None) -> bool:
        """Check if email already exists"""
        with self.session_factory() as session:
            return self._crud_strategy.email_exists(session, email, exclude_id)

    def username_exists(self, username: str, exclude_id: UUID | None = None) -> bool:
        """Check if username already exists"""
        with self.session_factory() as session:
            return self._crud_strategy.username_exists(session, username, exclude_id)

    def create(self, input_schema: UserCreate) -> User:
        """Create a new user"""
        with self.session_factory() as session:
            return self._crud_strategy.create(session, input_schema)

    def get_by_id(self, id: UUID) -> User | None:
        """Get user by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    def get_all(self, skip: int = 0, limit: int = 100) -> list[User]:
        """Live users in creation order, ``skip`` rows in, at most ``limit``.

        ``skip`` is a row offset. It used to be passed straight through as a
        page number, so ``skip=0`` produced a negative OFFSET (a PostgreSQL
        error) and ``skip=n`` skipped ``(n - 1) * limit`` rows.
        """
        statement = (
            select(User)
            .where(User.deleted_at.is_(None))
            .order_by(User.created_at.asc(), User.id.asc())
            .offset(max(0, skip))
            .limit(limit)
        )
        with self.session_factory() as session:
            return list(session.execute(statement).scalars().all())

    def update(self, id: UUID, input_schema: UserUpdate) -> User | None:
        """Update user by ID"""
        with self.session_factory() as session:
            db_obj = self._crud_strategy.get_by_id(session, id)
            if db_obj is None:
                return None
            return self._crud_strategy.update(session, db_obj, input_schema)

    def delete(self, id: UUID) -> bool:
        """Soft-delete the user and revoke every token issued to them."""
        with self.session_factory() as session:
            result = session.execute(
                update(User)
                .where(User.id == id, User.deleted_at.is_(None))
                .values(deleted_at=func.now(), token_version=User.token_version + 1)
            )
            session.commit()
        token_version.token_state_cache.invalidate(id)
        return result.rowcount > 0

    def get_token_state(self, id: UUID) -> TokenState | None:
        """The user's token version and soft-delete state, None if no such user."""
        statement = select(User.token_version, User.deleted_at).where(User.id == id)
        with self.session_factory() as session:
            row = session.execute(statement).one_or_none()
        if row is None:
            return None
        return TokenState(version=row.token_version, deleted=row.deleted_at is not None)

    def exists(self, id: UUID) -> bool:
        """Check if user exists"""
        with self.session_factory() as session:
            return self._crud_strategy.exists(session, id)
