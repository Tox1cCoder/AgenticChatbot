import enum
from sqlalchemy import Enum


class MessageRole(enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


# SQLAlchemy enum type
MessageRoleType = Enum(MessageRole, name="message_role")
