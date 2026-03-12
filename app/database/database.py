"""Database module for dependency-injector integration."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, orm
from sqlalchemy.orm import Session

from app.database.base import Base

logger = logging.getLogger(__name__)


class Database:
    """Database utility class."""

    def __init__(self, db_url: str) -> None:
        self._engine = create_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        self._session_factory = orm.scoped_session(
            orm.sessionmaker(
                autoflush=False,
                bind=self._engine,
            ),
        )

    def create_database(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """
        Provide a database session as a context manager.

        Yields:
            Session: SQLAlchemy database session
        """
        session: Session = self._session_factory()
        try:
            yield session
        except Exception:
            logger.exception("Session rollback because of exception")
            session.rollback()
            raise
        finally:
            session.close()
