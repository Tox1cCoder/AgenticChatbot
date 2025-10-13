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

from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.services.conversation_service import ConversationService
from app.services.message_service import MessageService
from app.services.feedback_service import FeedbackService
from app.services.ai_service import AIService
from app.services.document_service import DocumentService
from app.services.document_processing_service import DocumentProcessingService

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.workers.celery_app import celery_app

from app.utils.validation.user_validation import UserValidationUtils
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.feedback_validation import FeedbackValidationUtils
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.document_validation import DocumentValidationUtils

from app.core.dependency_injection import AppAutoInjector, AppContainerInjector
from app.database.qdrant import ensure_collection


from app.interfaces import (
    IUserService,
    IConversationService,
    IMessageService,
    IFeedbackService,
    IAuthService,
    IDocumentService,
)


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

    ai_service = providers.Factory(
        AIService,
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
    )

    message_service: providers.Provider[IMessageService] = providers.Factory(
        MessageService,
        message_repository=message_repository,
        conversation_validation_utils=conversation_validation_utils,
        message_validation_utils=message_validation_utils,
        ai_service=ai_service,
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
    )

    document_processing_service = providers.Factory(
        DocumentProcessingService,
        settings=providers.Object(settings),
        celery_app=providers.Object(celery_app),
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
    )

    document_service: providers.Provider[IDocumentService] = providers.Factory(
        DocumentService,
        document_repository=document_repository,
        document_processing_service=document_processing_service,
        document_validation_utils=document_validation_utils,
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


def init_qdrant_collection():
    """Initialize Qdrant collection at application startup."""
    client = container.qdrant_client()
    ensure_collection(
        qdrant_client=client,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.embedding_dimension,
    )
