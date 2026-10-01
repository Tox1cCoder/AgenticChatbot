"""
Auto-injection decorators for FastAPI dependency management.
"""

import functools
import inspect
from typing import Any

from fastapi import Depends

from app.interfaces import (
    IAuthService,
    IConversationService,
    IDocumentService,
    IFeedbackService,
    IMessageService,
    IModelUsageService,
    IUserService,
)
from app.interfaces.task_plan_service_interface import ITaskPlanService
from app.repositories.hitl_interrupt import HITLInterruptRepository
from app.services.ai_service import AIService
from app.services.custom_agent_service import CustomAgentService
from app.services.document_processing_service import DocumentProcessingService
from app.services.hitl_settings_service import HitlSettingsService
from app.services.jwt_service import JwtService
from app.services.mcp_service import MCPService
from app.services.model_config_service import ModelConfigService
from app.services.project_service import ProjectService
from app.services.provider_service import ProviderService


class AutoInjector:
    wiring_map: dict[type, Any] = {}


class AppAutoInjector(AutoInjector):
    """Resolves annotated FastAPI route parameters from the container."""

    @classmethod
    def setup_wiring_map(cls, container):
        container_ref = container

        cls.wiring_map = {
            IUserService: container_ref.user_service,
            IConversationService: container_ref.conversation_service,
            IMessageService: container_ref.message_service,
            IFeedbackService: container_ref.feedback_service,
            IAuthService: container_ref.auth_service,
            IDocumentService: container_ref.document_service,
            ITaskPlanService: container_ref.task_plan_service,
            IModelUsageService: container_ref.model_usage_service,
            DocumentProcessingService: container_ref.document_processing_service,
            MCPService: container_ref.mcp_service,
            JwtService: container_ref.jwt_service,
            AIService: container_ref.ai_service,
            ProviderService: container_ref.provider_service,
            ModelConfigService: container_ref.model_config_service,
            CustomAgentService: container_ref.custom_agent_service,
            HitlSettingsService: container_ref.hitl_settings_service,
            HITLInterruptRepository: container_ref.hitl_interrupt_repository,
            ProjectService: container_ref.project_service,
        }

    @classmethod
    def auto_inject(cls):
        """Decorator factory to auto-wire FastAPI route parameters."""
        from uuid import UUID

        from app.core.auth import get_current_user_id, get_refresh_token_user_id
        from app.schemas.pagination import (
            ConversationPaginationParams,
            MessagePaginationParams,
        )

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
                        param = param.replace(default=Depends(get_refresh_token_user_id))

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
