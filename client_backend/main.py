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
from client_backend.api.conversations import router as conversations_router
from client_backend.api.documents import router as documents_router
from client_backend.api.health import router as health_router
from client_backend.api.mcp import router as mcp_router
from client_backend.api.messages import ai_sdk_router
from client_backend.api.messages import router as messages_router
from client_backend.api.proxy import router as proxy_router
from client_backend.api.runtime import router as runtime_router
from client_backend.api.skills import router as skills_router
from client_backend.core.config import client_settings, initialize_client_environment
from client_backend.core.logging import get_logger, setup_logging
from client_backend.services.local_mcp_manager import get_mcp_manager
from client_backend.services.local_skills_registry import initialize_skills_registry
from client_backend.services.runtime_bridge import get_runtime_bridge
from client_backend.services.server_api import close_server_client

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

    try:
        await get_mcp_manager().initialize()
    except Exception as exc:
        logger.warning("MCP manager initialization failed during startup: %s", exc)

    yield

    # Shutdown tasks
    logger.info("Shutting down Client Backend...")
    await get_runtime_bridge().stop()
    await close_server_client()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="Codex Client Backend",
        description="Local runtime backend for the Codex Desktop App",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware - intentionally permissive for local sidecar usage.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
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
        proxy_router,
    ]

    for api_router in compatibility_routers:
        app.include_router(api_router)
        app.include_router(api_router, prefix="/api")

    app.include_router(ai_sdk_router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Codex Client Backend",
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
