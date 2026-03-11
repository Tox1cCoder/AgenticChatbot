import sys
import asyncio
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from datetime import datetime, timezone

from app.core.config import settings
from app.core.container import (
    get_container,
    setup_auto_injection,
)
from app.api import (
    users_router,
    conversations_router,
    messages_router,
    feedback_router,
)
from app.api.documents import router as documents_router
from app.api.mcp import router as mcp_router
from app.api.task_plans import router as task_plans_router
from app.api.ai_sdk import router as ai_sdk_router
from app.api.providers import router as providers_router
from app.api.model_config import router as model_config_router
from app.api.skills import router as skills_router
from app.database.session import get_engine
from app.api.auth import router as auth_router
from app.utils.exception_handler import register_exception_handlers
from app.workers.celery_app import celery_app

from app.core.events import get_event_bus, DocumentEvent
from app.services.document_event_listener import DocumentEventLogger

logger = logging.getLogger(__name__)


async def init_checkpoint_tables():
    """Initialize LangGraph checkpoint tables at application startup."""
    if not settings.enable_langgraph_checkpoints:
        logger.info("LangGraph checkpoints disabled in settings")
        return

    container = get_container()
    checkpoint_manager = container.checkpoint_manager()

    await checkpoint_manager.setup()


async def init_agents():
    """Pre-warm agents by initializing their tools at startup."""
    try:
        container = get_container()
        ai_service = container.ai_service()

        await ai_service.workflow.initialize()

    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")


async def init_skills():
    """Pre-scan skills folder at startup."""
    try:
        from app.ai.skills_registry import get_skills_registry

        registry = get_skills_registry()
        skills = registry.get_all_skills()
        logger.info(
            f"Loaded {len(skills)} skills ({sum(s.enabled for s in skills)} enabled)"
        )
    except Exception as e:
        logger.warning(f"Skills init failed (non-fatal): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    await init_checkpoint_tables()
    await init_agents()
    await init_skills()
    yield
    # Shutdown
    try:
        from app.ai.mcp_integration import get_global_mcp_manager
        mcp_manager = await get_global_mcp_manager()
        if mcp_manager:
            await mcp_manager.cleanup()
            logger.info("MCP sessions closed cleanly")
    except Exception as e:
        logger.debug(f"MCP cleanup during shutdown (non-fatal): {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    container = get_container()

    setup_auto_injection(container)

    container.wire(
        modules=[
            "app.api.auth",
            "app.api.users",
            "app.api.conversations",
            "app.api.messages",
            "app.api.feedback",
            "app.api.documents",
            "app.api.mcp",
            "app.api.task_plans",
            "app.api.ai_sdk",
            "app.api.providers",
            "app.api.model_config",
            "app.api.skills",
        ]
    )

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        debug=settings.api_debug,
        lifespan=lifespan,
    )

    # Attach container to app for dependency injection
    app.container = container

    # Add CORS middleware
    _cors_origins = settings.cors_origins
    if "*" in _cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_origin_regex=".*",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["x-vercel-ai-ui-message-stream"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["x-vercel-ai-ui-message-stream"],
        )

    # Register centralized exception handlers
    register_exception_handlers(app)

    # Include routers
    app.include_router(auth_router)
    app.include_router(users_router)
    app.include_router(conversations_router)
    app.include_router(messages_router)
    app.include_router(feedback_router)
    app.include_router(documents_router)
    app.include_router(mcp_router)
    app.include_router(task_plans_router)
    app.include_router(ai_sdk_router)
    app.include_router(providers_router)
    app.include_router(model_config_router)
    app.include_router(skills_router)

    # Initialize and register event listeners
    event_bus = get_event_bus()
    doc_logger = DocumentEventLogger()
    for evt in [
        DocumentEvent.UPLOAD_STARTED,
        DocumentEvent.PROCESSING_STARTED,
        DocumentEvent.PROCESSING_COMPLETED,
        DocumentEvent.PROCESSING_FAILED,
        DocumentEvent.DELETED,
    ]:
        event_bus.register_listener(evt, doc_logger)

    return app


# Create the FastAPI app instance
app = create_app()

engine = get_engine()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Sample Chatbot API",
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "message": "OK"}


@app.get("/health/celery")
async def health_check_celery():
    """Check Celery worker health"""
    try:
        # Use Celery inspect API to check active workers
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()

        if active_workers:
            worker_names = list(active_workers.keys())
            return {
                "status": "healthy",
                "workers": worker_names,
                "worker_count": len(worker_names),
                "message": f"{len(worker_names)} Celery worker(s) active",
            }
        else:
            return {
                "status": "unhealthy",
                "workers": [],
                "worker_count": 0,
                "message": "No active Celery workers found",
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Failed to connect to Celery",
        }


@app.get("/health/redis")
async def health_check_redis():
    """Check Redis connection health"""
    try:
        redis_client = Redis.from_url(settings.celery_broker_url, decode_responses=True)

        response = redis_client.ping()

        if response:
            redis_client.close()
            return {"status": "healthy", "message": "Redis connection successful"}
        else:
            redis_client.close()
            return {"status": "unhealthy", "message": "Redis ping failed"}
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Failed to connect to Redis",
        }


@app.get("/health/qdrant")
async def health_check_qdrant():
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url)
        collections = client.get_collections()

        collection_exists = any(
            c.name == settings.qdrant_collection_name for c in collections.collections
        )

        if collection_exists:
            info = client.get_collection(settings.qdrant_collection_name)
            return {
                "status": "healthy",
                "collection": settings.qdrant_collection_name,
                "vectors_count": info.vectors_count,
                "message": "Qdrant connection successful",
            }
        else:
            return {
                "status": "unhealthy",
                "message": f"Collection '{settings.qdrant_collection_name}' not found",
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Failed to connect to Qdrant",
        }


@app.get("/health/all")
async def health_check_all():
    """Check health of all services"""
    timestamp = datetime.now(timezone.utc).isoformat()

    celery_health = await health_check_celery()

    redis_health = await health_check_redis()

    qdrant_health = await health_check_qdrant()

    all_healthy = (
        celery_health.get("status") == "healthy"
        and redis_health.get("status") == "healthy"
        and qdrant_health.get("status") == "healthy"
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": timestamp,
        "services": {
            "celery": celery_health,
            "redis": redis_health,
            "qdrant": qdrant_health,
        },
        "message": (
            "All services healthy" if all_healthy else "One or more services unhealthy"
        ),
    }


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
