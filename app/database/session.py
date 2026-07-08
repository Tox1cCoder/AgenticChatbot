from collections.abc import Generator, Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# Create SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.api_debug,
)

# Create SessionLocal class
SessionLocal = sessionmaker(autoflush=False, expire_on_commit=False, bind=engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session scope that commits on success.

    Use for callers that own the transaction boundary. Do not use inside
    repositories that already commit internally; pass ``SessionLocal`` there.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_engine():
    """
    Get the SQLAlchemy engine instance.

    Returns:
        Engine: SQLAlchemy engine
    """
    return engine


def get_session_factory():
    """
    Get the SQLAlchemy session factory used for short-lived repositories.

    Returns:
        sessionmaker: Callable that creates new Session instances
    """
    return SessionLocal
