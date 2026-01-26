"""
Services package.

Avoid importing service modules at package import time to keep imports lightweight
and prevent optional/heavy dependencies (e.g., vector DB clients) from being
required when only a single service is needed.

Prefer importing services directly:
`from app.services.provider_service import ProviderService`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["UserService", "ConversationService", "MessageService", "FeedbackService"]

if TYPE_CHECKING:
    from app.services.user_service import UserService
    from app.services.conversation_service import ConversationService
    from app.services.message_service import MessageService
    from app.services.feedback_service import FeedbackService


def __getattr__(name: str) -> Any:
    if name == "UserService":
        from app.services.user_service import UserService as _UserService

        return _UserService
    if name == "ConversationService":
        from app.services.conversation_service import (
            ConversationService as _ConversationService,
        )

        return _ConversationService
    if name == "MessageService":
        from app.services.message_service import MessageService as _MessageService

        return _MessageService
    if name == "FeedbackService":
        from app.services.feedback_service import FeedbackService as _FeedbackService

        return _FeedbackService

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
