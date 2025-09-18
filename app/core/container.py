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
from app.services.user_service import UserService
from app.services.conversation_service import ConversationService
from app.services.message_service import MessageService
from app.services.feedback_service import FeedbackService
from app.utils.user_validation import UserValidationUtils
from app.utils.conversation_validation import ConversationValidationUtils
from app.utils.message_validation import MessageValidationUtils
from app.interfaces import (
    IUserService,
    IConversationService,
    IMessageService,
    IFeedbackService,
    IAuthService,
)
from app.services.auth_service import AuthService


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

    # Business services
    user_service: providers.Provider[IUserService] = providers.Factory(
        UserService,
        user_repository=user_repository,
        user_validation_utils=user_validation_utils,
    )

    conversation_service: providers.Provider[IConversationService] = providers.Factory(
        ConversationService,
        conversation_repository=conversation_repository,
        user_repository=user_repository,
        user_validation_utils=user_validation_utils,
        conversation_validation_utils=conversation_validation_utils,
    )

    message_service: providers.Provider[IMessageService] = providers.Factory(
        MessageService,
        message_repository=message_repository,
        conversation_repository=conversation_repository,
        user_repository=user_repository,
        conversation_validation_utils=conversation_validation_utils,
        message_validation_utils=message_validation_utils,
    )

    feedback_service: providers.Provider[IFeedbackService] = providers.Factory(
        FeedbackService,
        feedback_repository=feedback_repository,
        message_repository=message_repository,
        user_repository=user_repository,
        user_validation_utils=user_validation_utils,
        message_validation_utils=message_validation_utils,
    )

    auth_service: providers.Provider[IAuthService] = providers.Factory(
        AuthService,
        user_service=user_service,
    )


# Create global container instance
container = Container()


def get_container() -> Container:
    """Get the global container instance."""
    return container
