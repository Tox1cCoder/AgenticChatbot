# Import all routers for easy access
from app.api.health import router as health_router
from app.api.users import router as users_router
from app.api.conversations import router as conversations_router
from app.api.messages import router as messages_router

__all__ = [
    "health_router",
    "users_router",
    "conversations_router",
    "messages_router",
]
