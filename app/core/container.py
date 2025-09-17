"""
Dependency Injection Container using python-dependency-injector.
"""

from dependency_injector import containers, providers

from app.core.config import settings
from app.database.database import Database
from app.repositories.user import UserRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository
from app.repositories.feedback import FeedbackRepository
from app.services.user_service import UserService
from app.services.conversation_service import ConversationService
from app.services.message_service import MessageService
from app.services.feedback_service import FeedbackService
from app.services.validation_service import (
    UserValidationService,
    ConversationValidationService,
    MessageValidationService,
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
        ]
    )

    # Configuration
    config = providers.Configuration()

    # Database
    db = providers.Singleton(
        Database,
        db_url=settings.database_url,
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

    # Validation services
    user_validation_service = providers.Factory(
        UserValidationService,
        session_factory=db.provided.session,
    )
    conversation_validation_service = providers.Factory(
        ConversationValidationService,
        session_factory=db.provided.session,
    )
    message_validation_service = providers.Factory(
        MessageValidationService,
        session_factory=db.provided.session,
    )

    # Business services
    user_service = providers.Factory(
        UserService,
        user_repository=user_repository,
        user_validation_service=user_validation_service,
    )

    conversation_service = providers.Factory(
        ConversationService,
        conversation_repository=conversation_repository,
        user_repository=user_repository,
        user_validation_service=user_validation_service,
        conversation_validation_service=conversation_validation_service,
    )

    message_service = providers.Factory(
        MessageService,
        message_repository=message_repository,
        conversation_repository=conversation_repository,
        user_repository=user_repository,
        conversation_validation_service=conversation_validation_service,
        message_validation_service=message_validation_service,
    )

    feedback_service = providers.Factory(
        FeedbackService,
        feedback_repository=feedback_repository,
        message_repository=message_repository,
        user_repository=user_repository,
        user_validation_service=user_validation_service,
        message_validation_service=message_validation_service,
    )


# Create global container instance
container = Container()


def get_container() -> Container:
    """Get the global container instance."""
    return container
