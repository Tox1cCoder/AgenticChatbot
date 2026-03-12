from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "UserValidationUtils",
    "ConversationValidationUtils",
    "MessageValidationUtils",
    "FeedbackValidationUtils",
]

_MODULE_MAP = {
    "UserValidationUtils": "validation.user_validation",
    "ConversationValidationUtils": "validation.conversation_validation",
    "MessageValidationUtils": "validation.message_validation",
    "FeedbackValidationUtils": "validation.feedback_validation",
}

if TYPE_CHECKING:
    from .validation.conversation_validation import ConversationValidationUtils
    from .validation.feedback_validation import FeedbackValidationUtils
    from .validation.message_validation import MessageValidationUtils
    from .validation.user_validation import UserValidationUtils


def __getattr__(name: str) -> Any:
    """Dynamically load validation utilities when accessed via package imports."""

    if name in _MODULE_MAP:
        module_path = f"{__name__}.{_MODULE_MAP[name]}"
        module = import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
