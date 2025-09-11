"""
Dependency Injection Container for managing application dependencies.
"""

from typing import Dict, Type, TypeVar, Callable, Any
from dataclasses import dataclass
from sqlalchemy.orm import Session

from app.repositories.user import UserRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository
from app.repositories.feedback import FeedbackRepository
from app.services.user_service import UserService
from app.services.conversation_service import ConversationService
from app.services.message_service import MessageService
from app.services.feedback_service import FeedbackService
from app.services.validation_service import UserValidationService

T = TypeVar("T")


@dataclass
class ServiceDefinition:
    """Definition for how to create a service instance"""

    service_class: Type
    dependencies: list[str]
    singleton: bool = False


class DIContainer:
    """
    Dependency Injection Container that manages service instantiation and dependency resolution.
    Eliminates manual dependency creation in service constructors.
    """

    def __init__(self):
        self._services: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
        self._session: Session = None

        # Register all services and their dependencies
        self._register_services()

    def _register_services(self):
        """Register all services with their dependency requirements"""

        # Repository registrations - depend only on db session
        self._services["user_repository"] = ServiceDefinition(
            service_class=UserRepository, dependencies=["db"], singleton=False
        )

        self._services["conversation_repository"] = ServiceDefinition(
            service_class=ConversationRepository, dependencies=["db"], singleton=False
        )

        self._services["message_repository"] = ServiceDefinition(
            service_class=MessageRepository, dependencies=["db"], singleton=False
        )

        self._services["feedback_repository"] = ServiceDefinition(
            service_class=FeedbackRepository, dependencies=["db"], singleton=False
        )

        # Validation service registration
        self._services["user_validation_service"] = ServiceDefinition(
            service_class=UserValidationService, dependencies=["db"], singleton=False
        )

        # Business service registrations - depend on repositories/validation services
        self._services["user_service"] = ServiceDefinition(
            service_class=UserService,
            dependencies=["user_repository", "user_validation_service"],
            singleton=False,
        )

        self._services["conversation_service"] = ServiceDefinition(
            service_class=ConversationService,
            dependencies=["conversation_repository", "user_repository"],
            singleton=False,
        )

        self._services["message_service"] = ServiceDefinition(
            service_class=MessageService,
            dependencies=[
                "message_repository",
                "conversation_repository",
                "user_repository",
            ],
            singleton=False,
        )

        self._services["feedback_service"] = ServiceDefinition(
            service_class=FeedbackService,
            dependencies=[
                "feedback_repository",
                "message_repository",
                "user_repository",
            ],
            singleton=False,
        )

    def set_session(self, session: Session):
        """Set the database session for this request context"""
        self._session = session
        self._instances.clear()  # Clear instances for new request

    def get(self, service_name: str) -> Any:
        """
        Get a service instance, resolving all dependencies automatically.

        Args:
            service_name: Name of the service to retrieve

        Returns:
            Fully configured service instance with all dependencies injected
        """
        if service_name == "db":
            if self._session is None:
                raise ValueError("Database session not set. Call set_session() first.")
            return self._session

        # Check if already instantiated (for singletons)
        if service_name in self._instances:
            return self._instances[service_name]

        if service_name not in self._services:
            raise ValueError(f"Service '{service_name}' not registered")

        service_def = self._services[service_name]

        # Resolve all dependencies first
        dependencies = {}
        for dep_name in service_def.dependencies:
            dependencies[dep_name] = self.get(dep_name)

        # Create service instance with resolved dependencies
        if service_name.endswith("_service") and not service_name.endswith(
            "_validation_service"
        ):
            # Business services get dependency injection
            instance = service_def.service_class(container=self, **dependencies)
        else:
            # Repositories and validation services get direct db injection
            instance = service_def.service_class(dependencies["db"])

        # Store singleton instances
        if service_def.singleton:
            self._instances[service_name] = instance

        return instance


# Global container instance
container = DIContainer()


def get_container() -> DIContainer:
    """Get the global DI container instance"""
    return container
