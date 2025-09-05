# Import all schemas for easy access
from app.schemas.user import UserBase, UserCreate, UserUpdate, UserRead, UserInDB
from app.schemas.conversation import ConversationBase, ConversationCreate, ConversationUpdate, ConversationRead, ConversationInDB
from app.schemas.message import MessageBase, MessageCreate, MessageUpdate, MessageRead, MessageInDB

__all__ = [
    # User schemas
    "UserBase", "UserCreate", "UserUpdate", "UserRead", "UserInDB",
    # Conversation schemas
    "ConversationBase", "ConversationCreate", "ConversationUpdate", "ConversationRead", "ConversationInDB",
    # Message schemas
    "MessageBase", "MessageCreate", "MessageUpdate", "MessageRead", "MessageInDB",
]
