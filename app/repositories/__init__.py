"""Lazy-loading exports for repository classes."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "UserRepository",
    "ConversationRepository",
    "MessageRepository",
    "FeedbackRepository",
    "DocumentRepository",
    "DocumentChunkRepository",
]

_MODULE_MAP = {
    "UserRepository": "user",
    "ConversationRepository": "conversation",
    "MessageRepository": "message",
    "FeedbackRepository": "feedback",
    "DocumentRepository": "document",
    "DocumentChunkRepository": "document",
}

if TYPE_CHECKING:
    from .user import UserRepository
    from .conversation import ConversationRepository
    from .message import MessageRepository
    from .feedback import FeedbackRepository
    from .document import DocumentRepository, DocumentChunkRepository


def __getattr__(name: str) -> Any:
    """Dynamically load repository classes when accessed via package imports."""

    if name in _MODULE_MAP:
        module = import_module(f"{__name__}.{_MODULE_MAP[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
