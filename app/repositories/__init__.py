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
    "DocumentChunkRepository": "document_chunk",
}

if TYPE_CHECKING:
    from .conversation import ConversationRepository
    from .document import DocumentRepository
    from .document_chunk import DocumentChunkRepository
    from .feedback import FeedbackRepository
    from .message import MessageRepository
    from .user import UserRepository


def __getattr__(name: str) -> Any:
    """Dynamically load repository classes when accessed via package imports."""

    if name in _MODULE_MAP:
        module = import_module(f"{__name__}.{_MODULE_MAP[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
