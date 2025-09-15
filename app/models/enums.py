import enum
from sqlalchemy import Integer


class MessageRole(enum.IntEnum):
    user = 1
    assistant = 2
    # system = 3


# SQLAlchemy type
MessageRoleType = Integer
