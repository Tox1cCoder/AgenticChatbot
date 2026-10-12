from collections.abc import Generator

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

# The conversation turn lock holds one connection for the whole of every
# in-flight turn. Drawn from the pool above, a burst of turns took all of its
# connections and stalled every sync repository call in the process, so locks
# get their own pool. A checkout waits no longer than a lock does.
turn_lock_engine = create_engine(
    settings.database_url,
    pool_size=settings.conversation_turn_lock_pool_size,
    max_overflow=0,
    pool_timeout=settings.conversation_turn_lock_timeout_seconds,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.api_debug,
)
TurnLockSessionLocal = sessionmaker(autoflush=False, expire_on_commit=False, bind=turn_lock_engine)


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
