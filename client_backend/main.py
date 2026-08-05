"""
Client Backend FastAPI Application.

This is the main entry point for the local client backend that handles:
- Local tool execution (shell, filesystem)
- Local MCP server management
- Local skill loading
- Request proxying to the canonical server backend
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from client_backend import __version__
from client_backend.api.auth import router as auth_router
from client_backend.api.chat_images import router as chat_images_router
from client_backend.api.conversations import router as conversations_router
from client_backend.api.documents import router as documents_router
from client_backend.api.health import router as health_router
from client_backend.api.mcp import router as mcp_router
from client_backend.api.messages import ai_sdk_router
from client_backend.api.messages import router as messages_router
from client_backend.api.proxy import router as proxy_router
from client_backend.api.runtime import router as runtime_router
from client_backend.api.skill_errors import register_skill_exception_handlers
from client_backend.api.skills import router as skills_router
from client_backend.api.web_images import router as web_images_router
from client_backend.core.config import client_settings, initialize_client_environment
from client_backend.core.logging import get_logger, setup_logging
from client_backend.services.local_skills_registry import initialize_skills_registry
from client_backend.services.runtime_bridge import get_runtime_bridge
from client_backend.services.server_api import close_server_client
from client_backend.services.skill_runtime.operations import get_skill_installation_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events.
    """
    initialize_client_environment()
    setup_logging()
    logger.info(
        f"Starting Client Backend v{__version__} on "
        f"{client_settings.backend_host}:{client_settings.backend_port}"
    )
    logger.info(f"Server API URL: {client_settings.server_api_base_url}")
    logger.info(f"Profile root: {client_settings.profile_root}")
    logger.info(f"Device name: {client_settings.device_name}")

    # Startup tasks
    try:
        await initialize_skills_registry()
    except Exception as exc:
        logger.warning("Skills registry initialization failed during startup: %s", exc)

    yield

    # Shutdown tasks
    logger.info("Shutting down Client Backend...")
    # Installation tasks are stopped before the bridge so a half-finished install
    # cannot try to publish a catalog through a closing connection. Their receipts
    # stay on disk and are reconciled by recovery on the next start.
    try:
        await get_skill_installation_service().shutdown()
    except Exception as exc:  # noqa: BLE001 - shutdown must continue regardless
        logger.warning("Skill installation shutdown failed: %s", exc)
    await get_runtime_bridge().stop()
    await close_server_client()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="Kani Client Backend",
        description="Local runtime backend for the Kani Desktop App",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware. Origins are explicit: this process executes local shell
    # commands, filesystem operations, and skill runtimes, so any page must not
    # be able to drive it. `allow_private_network` stays on because a browser on
    # a public origin cannot reach a loopback server without that opt-in.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(client_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_private_network=True,
    )

    compatibility_routers = [
        health_router,
        auth_router,
        conversations_router,
        messages_router,
        documents_router,
        runtime_router,
        mcp_router,
        skills_router,
        chat_images_router,
        web_images_router,
        proxy_router,
    ]

    for api_router in compatibility_routers:
        app.include_router(api_router)
        app.include_router(api_router, prefix="/api")

    app.include_router(ai_sdk_router)

    # Scoped to /skills and /api/skills inside the handlers; every other route
    # keeps FastAPI's default error shape.
    register_skill_exception_handlers(app)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Kani Client Backend",
            "version": __version__,
            "status": "running",
        }

    return app


# Create the application instance
app = create_app()


def main():
    """
    Main entry point for running the client backend.
    """
    import uvicorn

    initialize_client_environment()
    setup_logging()

    uvicorn.run(
        "client_backend.main:app",
        host=client_settings.backend_host,
        port=client_settings.backend_port,
        reload=client_settings.environment == "development",
        log_level=client_settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
