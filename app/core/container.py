"""
Dependency Injection Container.
"""

from dependency_injector import containers, providers

from app.core.config import settings
from app.database.database import Database
from app.repositories.user import UserRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository
from app.repositories.feedback import FeedbackRepository
from app.repositories.document import DocumentRepository
from app.repositories.task_plan import TaskPlanRepository

from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.services.conversation_service import ConversationService
from app.services.message_service import MessageService
from app.services.feedback_service import FeedbackService
from app.services.ai_service import AIService
from app.services.document_service import DocumentService
from app.services.document_processing_service import DocumentProcessingService
from app.services.mcp_service import MCPService
from app.services.jwt_service import JwtService
from app.services.task_plan_service import TaskPlanService

from app.ai.checkpoint import CheckpointManager
from app.ai.mcp_integration import MCPManager
from app.ai.agents.planning_agent import PlanningAgent
from app.repositories.document_image import DocumentImageRepository

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.workers.celery_app import celery_app

from app.utils.validation.user_validation import UserValidationUtils
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.feedback_validation import FeedbackValidationUtils
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.document_validation import DocumentValidationUtils
from app.utils.validation.task_plan_validation import TaskPlanValidationUtils

from app.core.dependency_injection import AppAutoInjector, AppContainerInjector


from app.interfaces import (
    IUserService,
    IConversationService,
    IMessageService,
    IFeedbackService,
    IAuthService,
    IDocumentService,
)
from app.interfaces.task_plan_service_interface import ITaskPlanService


class Container(containers.DeclarativeContainer):
    """Application dependency injection container."""

    wiring_config = containers.WiringConfiguration(
        modules=[
            "app.api.auth",
            "app.api.users",
            "app.api.conversations",
            "app.api.messages",
            "app.api.feedback",
            "app.api.documents",
            "app.api.mcp",
            "app.api.task_plans",
        ]
    )

    # Configuration
    config = providers.Configuration()

    # Database
    db = providers.Singleton(
        Database,
        db_url=settings.database_url,
    )

    # Qdrant client
    qdrant_client = providers.Singleton(
        QdrantClient,
        url=settings.qdrant_url,
    )

    # Embedding model
    embedding_model = providers.Singleton(
        SentenceTransformer,
        "google/embeddinggemma-300m",
    )

    # JWT Service
    jwt_service = providers.Factory(
        JwtService,
    )

    # Checkpoint manager
    checkpoint_manager = providers.Singleton(
        CheckpointManager,
        db_url=settings.database_url,
        settings=providers.Object(settings),
    )

    # MCP Manager
    mcp_manager = providers.Singleton(
        MCPManager,
    )

    # Repositories - use session factory from database
    user_repository = providers.Factory(
        UserRepository,
        session_factory=db.provided.session,
    )

    conversation_repository = providers.Factory(
        ConversationRepository,
        session_factory=db.provided.session,
    )

    message_repository = providers.Factory(
        MessageRepository,
        session_factory=db.provided.session,
    )

    feedback_repository = providers.Factory(
        FeedbackRepository,
        session_factory=db.provided.session,
    )

    document_repository = providers.Factory(
        DocumentRepository,
        session_factory=db.provided.session,
    )

    document_image_repository = providers.Factory(
        DocumentImageRepository,
        session_factory=db.provided.session,
    )

    task_plan_repository = providers.Factory(
        TaskPlanRepository,
        session_factory=db.provided.session,
    )

    # Validation utils
    user_validation_utils = providers.Factory(
        UserValidationUtils,
        session_factory=db.provided.session,
    )
    conversation_validation_utils = providers.Factory(
        ConversationValidationUtils,
        session_factory=db.provided.session,
    )
    message_validation_utils = providers.Factory(
        MessageValidationUtils,
        session_factory=db.provided.session,
    )
    feedback_validation_utils = providers.Factory(
        FeedbackValidationUtils,
        session_factory=db.provided.session,
    )
    document_validation_utils = providers.Factory(
        DocumentValidationUtils,
        session_factory=db.provided.session,
    )
    task_plan_validation_utils = providers.Factory(
        TaskPlanValidationUtils,
        session_factory=db.provided.session,
    )

    # Planning agent
    planning_agent = providers.Factory(
        PlanningAgent,
    )

    # Business services
    user_service: providers.Provider[IUserService] = providers.Factory(
        UserService,
        user_repository=user_repository,
        user_validation_utils=user_validation_utils,
    )

    conversation_service: providers.Provider[IConversationService] = providers.Factory(
        ConversationService,
        conversation_repository=conversation_repository,
        user_validation_utils=user_validation_utils,
        conversation_validation_utils=conversation_validation_utils,
    )

    # AI service with conditional checkpoint injection
    def _create_ai_service():
        """Factory function to create AIService with conditional checkpointer."""
        qdrant = container.qdrant_client()
        embeddings = container.embedding_model()

        # Conditionally get checkpointer based on settings
        checkpointer = None
        if settings.enable_langgraph_checkpoints:
            checkpoint_mgr = container.checkpoint_manager()
            # Get the checkpointer (now synchronous)
            checkpointer = checkpoint_mgr.get_checkpointer()

        return AIService(
            qdrant_client=qdrant,
            embedding_model=embeddings,
            conversation_repository=container.conversation_repository(),
            document_repository=container.document_repository(),
            checkpointer=checkpointer,
        )

    ai_service = providers.Factory(_create_ai_service)

    task_plan_service: providers.Provider[ITaskPlanService] = providers.Factory(
        TaskPlanService,
        task_plan_repository=task_plan_repository,
        conversation_validation_utils=conversation_validation_utils,
        task_plan_validation_utils=task_plan_validation_utils,
        planning_agent=planning_agent,
        conversation_repository=conversation_repository,
    )

    message_service: providers.Provider[IMessageService] = providers.Factory(
        MessageService,
        message_repository=message_repository,
        conversation_validation_utils=conversation_validation_utils,
        message_validation_utils=message_validation_utils,
        ai_service=ai_service,
        task_plan_service=task_plan_service,
    )

    feedback_service: providers.Provider[IFeedbackService] = providers.Factory(
        FeedbackService,
        feedback_repository=feedback_repository,
        message_repository=message_repository,
        user_repository=user_repository,
        user_validation_utils=user_validation_utils,
        message_validation_utils=message_validation_utils,
        feedback_validation_utils=feedback_validation_utils,
    )

    auth_service: providers.Provider[IAuthService] = providers.Factory(
        AuthService,
        user_service=user_service,
        jwt_service=jwt_service,
    )

    document_processing_service = providers.Factory(
        DocumentProcessingService,
        settings=providers.Object(settings),
        celery_app=providers.Object(celery_app),
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        document_image_repository=document_image_repository,
    )

    document_service: providers.Provider[IDocumentService] = providers.Factory(
        DocumentService,
        document_repository=document_repository,
        document_processing_service=document_processing_service,
        document_validation_utils=document_validation_utils,
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
    )

    mcp_service = providers.Factory(
        MCPService,
        mcp_manager=mcp_manager,
    )


# Initialize auto-injection wiring map before container instantiation
def setup_auto_injection(container_ref: Container | type[Container] | None = None):
    """Setup auto-injection wiring maps."""

    target = container_ref or Container
    AppAutoInjector.setup_wiring_map(target)
    AppContainerInjector.setup_wiring_map(target)


setup_auto_injection(Container)


# Create global container instance
container = Container()


def get_container() -> Container:
    """Get the global container instance."""
    return container
