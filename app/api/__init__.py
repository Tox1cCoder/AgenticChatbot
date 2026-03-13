from app.core.container import Container, setup_auto_injection

# Ensure AppAutoInjector wiring_map is configured even when importing `app.api.*`
# modules directly (e.g. verification scripts that don't import `app.main`).
setup_auto_injection(Container)

from app.api.conversations import router as conversations_router  # noqa: E402
from app.api.feedback import router as feedback_router  # noqa: E402
from app.api.messages import router as messages_router  # noqa: E402
from app.api.users import router as users_router  # noqa: E402

__all__ = [
    "users_router",
    "conversations_router",
    "messages_router",
    "feedback_router",
]
