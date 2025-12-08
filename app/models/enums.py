import enum
from sqlalchemy import Integer, Enum


class MessageRole(enum.IntEnum):
    user = 1
    assistant = 2


class DocumentStatus(enum.IntEnum):
    processing = 1
    ready = 2
    failed = 3


class TaskStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    skipped = "skipped"


# SQLAlchemy types
MessageRoleType = Integer
DocumentStatusType = Integer
TaskStatusType = Enum(
    "pending",
    "in_progress",
    "completed",
    "skipped",
    name="task_status",
    create_type=False,
)
