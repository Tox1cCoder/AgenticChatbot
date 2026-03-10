# Import all models to register them with SQLAlchemy metadata
from app.models.enums import MessageRole, MessageRoleType
from app.models.enums import TaskStatus, TaskStatusType
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.feedback import Feedback
from app.models.document import Document
from app.models.document_image import DocumentImage
from app.models.task_plan import TaskPlan
from app.models.agent_model_config import AgentModelConfig
from app.models.tool_approval import ToolApproval
from app.models.hitl_interrupt import HITLInterrupt

# Import shared Base for Alembic
from app.models.base import Base

# Make Base available for Alembic
__all__ = ["Base"]
