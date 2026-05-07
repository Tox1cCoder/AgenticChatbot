from app.core.container import Container, setup_auto_injection

# Ensure AppAutoInjector wiring_map is configured even when importing `app.api.*`
# modules directly (e.g. verification scripts that don't import `app.main`).
setup_auto_injection(Container)

from app.api.ai_sdk import router as ai_sdk_router  # noqa: E402
from app.api.auth import router as auth_router  # noqa: E402
from app.api.client_devices import router as client_devices_router  # noqa: E402
from app.api.conversations import router as conversations_router  # noqa: E402
from app.api.device_runtime import router as device_runtime_router  # noqa: E402
from app.api.documents import router as documents_router  # noqa: E402
from app.api.feedback import router as feedback_router  # noqa: E402
from app.api.mcp import router as mcp_router  # noqa: E402
from app.api.messages import router as messages_router  # noqa: E402
from app.api.model_config import router as model_config_router  # noqa: E402
from app.api.providers import router as providers_router  # noqa: E402
from app.api.task_plans import router as task_plans_router  # noqa: E402
from app.api.tool_result_blobs import router as tool_result_blobs_router  # noqa: E402
from app.api.users import router as users_router  # noqa: E402
from app.api.widgets import router as widgets_router  # noqa: E402

__all__ = [
    "ai_sdk_router",
    "auth_router",
    "client_devices_router",
    "device_runtime_router",
    "documents_router",
    "users_router",
    "conversations_router",
    "messages_router",
    "feedback_router",
    "mcp_router",
    "model_config_router",
    "providers_router",
    "task_plans_router",
    "tool_result_blobs_router",
    "widgets_router",
]
