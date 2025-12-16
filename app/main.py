import sys
import asyncio
import uvicorn
import logging
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
from app.database.session import get_engine
from app.api.auth import router as auth_router
from app.utils.exception_handler import register_exception_handlers
from app.workers.celery_app import celery_app
from app.ai.agents.rag_agent import RAGAgent

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
        ]
    )

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        debug=settings.api_debug,
    )

    # Attach container to app for dependency injection
    app.container = container

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
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

    # Register startup event for checkpoint initialization
    @app.on_event("startup")
    async def startup_event():
        """Initialize checkpoint tables on application startup."""
        await init_checkpoint_tables()

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
    """Check Qdrant connection health"""
    rag_agent = None
    try:
        container = get_container()
        app_settings = container.config()

        rag_agent = RAGAgent(
            settings=app_settings,
            collection_name=app_settings.qdrant_collection_name,
        )

        await rag_agent.initialize()

        status_info = await rag_agent.get_status()

        return {
            "status": status_info.get("status", "unknown"),
            "collection": status_info.get("collection"),
            "vectors_count": status_info.get("vectors_count", 0),
            "message": (
                "Qdrant connection successful"
                if status_info.get("status") == "healthy"
                else "Qdrant connection issues"
            ),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Failed to connect to Qdrant",
        }
    finally:
        if rag_agent:
            try:
                await rag_agent.cleanup()
            except:
                pass


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
