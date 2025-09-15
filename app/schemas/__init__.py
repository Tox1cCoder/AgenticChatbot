# Import all schemas for easy access
from app.schemas.user import (
    UserCreate,
    UserUpdate,
    UserRead,
    UserInDB,
)
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationRead,
    ConversationInDB,
)
from app.schemas.message import (
    MessageCreate,
    MessageUpdate,
    MessageRead,
    MessageInDB,
)
from app.schemas.feedback import (
    FeedbackCreate,
    FeedbackUpdate,
    FeedbackRead,
    FeedbackInDB,
)

__all__ = [
    # User schemas
    "UserCreate",
    "UserUpdate",
    "UserRead",
    "UserInDB",
    # Conversation schemas
    "ConversationCreate",
    "ConversationUpdate",
    "ConversationRead",
    "ConversationInDB",
    # Message schemas
    "MessageCreate",
    "MessageUpdate",
    "MessageRead",
    "MessageInDB",
    # Feedback schemas
    "FeedbackCreate",
    "FeedbackUpdate",
    "FeedbackRead",
    "FeedbackInDB",
]
