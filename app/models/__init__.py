# Import all models to ensure they are registered with SQLAlchemy
from app.models.base import Base, BaseModel
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message

__all__ = ["Base", "BaseModel", "User", "Conversation", "Message"]
