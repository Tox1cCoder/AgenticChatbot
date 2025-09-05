# Import all models to register them with SQLAlchemy metadata
from app.models.base import Base
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message

# Make Base available for Alembic
__all__ = ["Base"]
