"""
Auto-injection decorators for FastAPI dependency management.
"""

import inspect
import functools
from typing import Dict, Any, Type
from fastapi import Depends
from dependency_injector.wiring import Provide
from dependency_injector import providers

from app.interfaces import (
    IUserService,
    IConversationService,
    IMessageService,
    IFeedbackService,
    IAuthService,
    IDocumentService,
)
from app.interfaces.task_plan_service_interface import ITaskPlanService
from app.repositories.user import UserRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository
from app.repositories.feedback import FeedbackRepository
from app.repositories.document import DocumentRepository
from app.repositories.task_plan import TaskPlanRepository
from app.services.document_processing_service import DocumentProcessingService
from app.services.mcp_service import MCPService
from app.services.jwt_service import JwtService
from app.services.ai_service import AIService
from app.services.provider_service import ProviderService
from app.services.model_config_service import ModelConfigService
from app.utils.validation.user_validation import UserValidationUtils
from app.utils.validation.conversation_validation import (
    ConversationValidationUtils,
)
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.feedback_validation import FeedbackValidationUtils
from app.utils.validation.document_validation import DocumentValidationUtils
from app.utils.validation.task_plan_validation import TaskPlanValidationUtils


class AutoInjector:
    """Base class for auto-injection with wiring_map configuration."""

    wiring_map: Dict[Type, Any] = {}

    @classmethod
    def auto_inject(cls):
        """Decorator factory to auto-wire FastAPI route parameters from wiring_map."""

        def decorator(func):
            sig = inspect.signature(func)
            new_params = []

            for name, param in sig.parameters.items():
                ann = param.annotation
                if ann in cls.wiring_map and (
                    param.default == inspect.Parameter.empty or param.default is None
                ):
                    provider = cls.wiring_map[ann]

                    def create_dependency(provider=provider):
                        return lambda: provider()

                    param = param.replace(default=Depends(create_dependency()))
                new_params.append(param)

            def wrapper_factory():
                if inspect.iscoroutinefunction(func):

                    @functools.wraps(func)
                    async def wrapper(*args, **kwargs):
                        return await func(*args, **kwargs)

                else:

                    @functools.wraps(func)
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs)

                wrapper.__signature__ = sig.replace(parameters=new_params)
                return wrapper

            return wrapper_factory()

        return decorator


class ContainerInjector:
    """Helper for automatic container provider creation."""

    wiring_map: Dict[Type, Any] = {}

    @classmethod
    def inject_container(cls, target_cls):
        """Automatically build a providers.Factory from __init__ annotations."""
        sig = inspect.signature(target_cls.__init__)
        kwargs = {}

        # skip 'self'
        for name, param in list(sig.parameters.items())[1:]:
            ann = param.annotation
            if ann in cls.wiring_map:
                kwargs[name] = cls.wiring_map[ann]
            else:
                raise ValueError(f"No provider registered for type {ann}")

        return providers.Factory(target_cls, **kwargs)


class AppAutoInjector(AutoInjector):
    """Application-specific auto-injector with service mappings."""

    @classmethod
    def setup_wiring_map(cls, container):
        """Setup the wiring map with container providers."""
        container_ref = container

        cls.wiring_map = {
            IUserService: getattr(container_ref, "user_service"),
            IConversationService: getattr(container_ref, "conversation_service"),
            IMessageService: getattr(container_ref, "message_service"),
            IFeedbackService: getattr(container_ref, "feedback_service"),
            IAuthService: getattr(container_ref, "auth_service"),
            IDocumentService: getattr(container_ref, "document_service"),
            ITaskPlanService: getattr(container_ref, "task_plan_service"),
            DocumentProcessingService: getattr(
                container_ref, "document_processing_service"
            ),
            MCPService: getattr(container_ref, "mcp_service"),
            JwtService: getattr(container_ref, "jwt_service"),
            AIService: getattr(container_ref, "ai_service"),
            ProviderService: getattr(container_ref, "provider_service"),
            ModelConfigService: getattr(container_ref, "model_config_service"),
        }

    @classmethod
    def auto_inject(cls):
        """Decorator factory to auto-wire FastAPI route parameters."""
        from app.core.auth import get_current_user_id, get_refresh_token_user_id
        from app.schemas.pagination import (
            MessagePaginationParams,
            ConversationPaginationParams,
        )
        from uuid import UUID

        def decorator(func):
            sig = inspect.signature(func)
            new_params = []

            for name, param in sig.parameters.items():
                ann = param.annotation

                # Handle services from wiring_map
                if ann in cls.wiring_map and (
                    param.default == inspect.Parameter.empty or param.default is None
                ):
                    provider = cls.wiring_map[ann]

                    def create_dependency(provider=provider):
                        return lambda: provider()

                    param = param.replace(default=Depends(create_dependency()))

                # Handle authentication parameters
                elif ann == UUID and param.default == inspect.Parameter.empty:
                    # Auto-inject authentication based on parameter name
                    if name in ["user_id", "current_user_id", "authenticated_user_id"]:
                        param = param.replace(default=Depends(get_current_user_id))
                    elif name in ["refresh_user_id"]:
                        param = param.replace(
                            default=Depends(get_refresh_token_user_id)
                        )

                # Handle pagination parameters
                elif (
                    ann in [MessagePaginationParams, ConversationPaginationParams]
                    and param.default == inspect.Parameter.empty
                ):
                    param = param.replace(default=Depends())

                new_params.append(param)

            def wrapper_factory():
                params_without_default = []
                params_with_default = []

                for param in new_params:
                    if param.default == inspect.Parameter.empty:
                        params_without_default.append(param)
                    else:
                        params_with_default.append(param)

                ordered_params = params_without_default + params_with_default

                if inspect.iscoroutinefunction(func):

                    @functools.wraps(func)
                    async def wrapper(*args, **kwargs):
                        return await func(*args, **kwargs)

                else:

                    @functools.wraps(func)
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs)

                wrapper.__signature__ = sig.replace(parameters=ordered_params)
                return wrapper

            return wrapper_factory()

        return decorator


class AppContainerInjector(ContainerInjector):
    """Application-specific container injector with repository and service mappings."""

    @classmethod
    def setup_wiring_map(cls, container):
        """Setup the wiring map with container providers for automatic injection."""
        container_ref = container

        cls.wiring_map = {
            # Services
            IUserService: getattr(container_ref, "user_service"),
            IConversationService: getattr(container_ref, "conversation_service"),
            IMessageService: getattr(container_ref, "message_service"),
            IFeedbackService: getattr(container_ref, "feedback_service"),
            IAuthService: getattr(container_ref, "auth_service"),
            IDocumentService: getattr(container_ref, "document_service"),
            ITaskPlanService: getattr(container_ref, "task_plan_service"),
            DocumentProcessingService: getattr(
                container_ref, "document_processing_service"
            ),
            JwtService: getattr(container_ref, "jwt_service"),
            AIService: getattr(container_ref, "ai_service"),
            # Repositories
            UserRepository: getattr(container_ref, "user_repository"),
            ConversationRepository: getattr(container_ref, "conversation_repository"),
            MessageRepository: getattr(container_ref, "message_repository"),
            FeedbackRepository: getattr(container_ref, "feedback_repository"),
            DocumentRepository: getattr(container_ref, "document_repository"),
            TaskPlanRepository: getattr(container_ref, "task_plan_repository"),
            # Validation utilities
            UserValidationUtils: getattr(container_ref, "user_validation_utils"),
            ConversationValidationUtils: getattr(
                container_ref, "conversation_validation_utils"
            ),
            MessageValidationUtils: getattr(container_ref, "message_validation_utils"),
            FeedbackValidationUtils: getattr(
                container_ref, "feedback_validation_utils"
            ),
            DocumentValidationUtils: getattr(
                container_ref, "document_validation_utils"
            ),
            TaskPlanValidationUtils: getattr(
                container_ref, "task_plan_validation_utils"
            ),
        }
