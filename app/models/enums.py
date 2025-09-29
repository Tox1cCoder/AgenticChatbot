import enum
from sqlalchemy import Integer


class MessageRole(enum.IntEnum):
    user = 1
    assistant = 2


class DocumentStatus(enum.IntEnum):
    processing = 1
    ready = 2
    failed = 3


# SQLAlchemy types
MessageRoleType = Integer
DocumentStatusType = Integer
