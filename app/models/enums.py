import enum

from sqlalchemy import Enum, Integer, String


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


class PlanLifecycle(str, enum.Enum):
    """Explicit lifecycle state for a conversation's task plan."""

    draft = "draft"  # Tasks created/modified, not yet approved for execution
    ready = "ready"  # User confirmed; ready to execute
    executing = "executing"  # Execution in progress
    paused = "paused"  # Execution paused (clarification, approval, budget, error)
    completed = "completed"  # All tasks complete


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
# Plain String for plan lifecycle (avoids creating a new PG enum type mid-migration)
PlanLifecycleType = String(20)
