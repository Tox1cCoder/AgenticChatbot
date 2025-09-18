"""
Interface definitions for service layer
"""

from __future__ import annotations
from .user_service_interface import IUserService
from .conversation_service_interface import IConversationService
from .message_service_interface import IMessageService
from .feedback_service_interface import IFeedbackService
from .auth_service_interface import IAuthService

__all__ = ["IUserService", "IConversationService", "IMessageService", "IFeedbackService", "IAuthService"]
