import asyncio
import logging
import sys
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis

from app.api import (
    ai_sdk_router,
    auth_router,
    chat_images_router,
    client_devices_router,
    conversations_router,
    custom_agents_conversation_router,
    custom_agents_router,
    device_runtime_router,
    documents_router,
    feedback_router,
    health_router,
    hitl_router,
    mcp_router,
    messages_router,
    model_config_router,
    model_usage_router,
    providers_router,
    task_plans_router,
    tool_result_blobs_router,
    users_router,
    web_images_router,
    widgets_router,
)
from app.core.build_info import resolve_build_info
from app.core.config import settings
from app.core.container import (
    get_container,
    setup_auto_injection,
)
from app.core.events import DocumentEvent, get_event_bus
from app.database.migrations import upgrade_database
from app.database.session import SessionLocal, get_engine
from app.services.client_device_service import periodic_session_cleanup_task
from app.services.client_runtime_store import close_client_runtime_store
from app.services.document_event_listener import DocumentEventLogger
from app.utils.exception_handler import register_exception_handlers
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)
_client_runtime_cleanup_task: asyncio.Task | None = None


async def init_database_migrations():
    """Apply pending Alembic migrations before serving requests."""
    await asyncio.to_thread(upgrade_database)


async def init_checkpoint_tables():
    """Initialize LangGraph checkpoint tables at application startup."""
    if not settings.enable_langgraph_checkpoints:
        logger.info("LangGraph checkpoints disabled in settings")
        return

    try:
        container = get_container()
        checkpoint_manager = container.checkpoint_manager()
        await checkpoint_manager.setup()
    except Exception as e:
        logger.warning(f"Checkpoint table setup failed (non-fatal): {e}")


async def init_agents():
    """Pre-warm agents by initializing their tools at startup."""
    try:
        container = get_container()
        ai_service = container.ai_service()

        await ai_service.initialize()

    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")


def _log_widget_runtime_status():
    """Log whether Redis-backed widget storage is available."""
    redis_url = settings.redis_url or settings.celery_broker_url
    if not redis_url:
        logger.warning(
            "Widget runtime: no Redis URL configured - "
            "live widget flows require Redis when the widgets MCP server runs out-of-process"
        )
        return
    try:
        r = Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
        r.ping()
        r.close()
        logger.info("Widget runtime: Redis-backed storage active (%s)", redis_url.split("@")[-1])
    except Exception as e:
        logger.warning("Widget runtime: Redis unavailable (%s) - widget flows will be degraded", e)


def _ensure_qdrant_collection():
    """Bootstrap the Qdrant collection through DocumentIndexService.

    Single owner of collection bootstrap after Phase 11. Reads
    ``settings.qdrant_collection_name`` and ``settings.rag_embedding_dimension``
    from the same source as the rest of the indexing path, so a misconfigured
    deployment surfaces immediately at startup.
    """
    try:
        index_service = get_container().document_index_service()
        index_service.ensure_collection()
        logger.info(
            "Qdrant collection ready: %s (dim=%d)",
            settings.qdrant_collection_name,
            settings.rag_embedding_dimension,
        )
    except Exception as exc:
        logger.warning(
            "ensure_collection failed at startup: %s. Indexing will retry on first write.",
            exc,
        )


def _selector_loop_error(loop: Any, *, platform: str = sys.platform) -> str | None:
    """Return an operator-facing error if ``loop`` cannot serve psycopg async pools.

    Pure so it can be tested without owning the running loop.

    uvicorn (0.46+) hard-codes ``ProactorEventLoop`` for non-subprocess launches
    on Windows, ignoring the policy set at the top of this module. psycopg's
    async mode refuses that loop, and the application now opens an async
    SQLAlchemy engine unconditionally (``app.database.async_session``), so on
    that loop *every* request's database call fails while the server still
    reports healthy. ``reload``/``workers`` launches are unaffected — subprocess
    launches get a ``SelectorEventLoop``.
    """
    if platform != "win32":
        return None
    if not isinstance(loop, asyncio.ProactorEventLoop):
        return None
    return (
        "This server is running on a ProactorEventLoop, which psycopg async "
        "pools cannot use. Every database call on the async engine would fail. "
        "Launch with reload/workers (subprocess mode), or start via the uvicorn "
        "Server API after setting asyncio.WindowsSelectorEventLoopPolicy(), e.g.:\n"
        "  asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())\n"
        "  asyncio.run(uvicorn.Server(uvicorn.Config('app.main:app')).serve())"
    )


def _ensure_selector_event_loop():
    """Fail fast when the server loop cannot run psycopg async pools.

    Applies on every Windows launch. It is deliberately not gated on
    ``enable_langgraph_checkpoints``: the async engine that serves ordinary
    requests needs the selector loop even when checkpoints are disabled.
    """
    error = _selector_loop_error(asyncio.get_running_loop())
    if error:
        raise RuntimeError(error)


async def _verify_async_database_ready() -> None:
    """Probe the async engine and check the connection budget at startup.

    Ordinary requests are served by the async engine, so an unusable async pool
    means every request fails while the process still answers ``/health``.
    Failing here — before migrations, agents, and MCP servers start — makes that
    a single clear message instead of a flood of per-request errors.
    """
    from app.database.readiness import resolve_connection_budget, verify_async_engine_ready
    from app.database.session import engine as sync_engine

    readiness = await verify_async_engine_ready()
    if not readiness.ok:
        raise RuntimeError(
            "The async database engine is not usable, so no request could be "
            f"served: {readiness.error}. Check DATABASE_URL, database "
            "reachability, and (on Windows) that the server runs on a "
            "SelectorEventLoop."
        )

    # Reported per worker on purpose. The real worker count belongs to the
    # launch command (uvicorn --workers), and mirroring it into a setting here
    # would create a second number that can silently disagree with it.
    budget = resolve_connection_budget(
        sync_pool=sync_engine.pool.size(),
        sync_overflow=sync_engine.pool._max_overflow,  # noqa: SLF001 - no public accessor
        async_pool=settings.db_pool_size,
        async_overflow=settings.db_max_overflow,
        workers=1,
    )
    description = budget.describe(max_connections=readiness.max_connections)
    if budget.exceeds(max_connections=readiness.max_connections):
        # A warning, not a failure: the worst case is rarely reached, and
        # refusing to boot over a capacity estimate would be worse than serving.
        logger.warning(
            "Over-subscribed %s. Lower DB_POOL_SIZE/DB_MAX_OVERFLOW or raise max_connections.",
            description,
        )
    else:
        logger.info("Verified %s", description)


async def _subscribe_generation_stop_signals() -> None:
    """Attach this worker's registry to the generation stop bus.

    Without it the bus published into a void: a Stop landing on a worker other
    than the streaming one transitioned the row and interrupted nothing. Stop
    is still correct without this — the row is the authority and each worker
    checks it at its own tool and model boundaries — but a worker blocked
    inside a provider call has no check point coming, and this is what reaches
    it.
    """
    from app.services.generation_stop_subscriber import install_generation_stop_subscriber

    try:
        container = get_container()
        await install_generation_stop_subscriber(container.generation_control_bus())
    except Exception as exc:  # noqa: BLE001 - degrades to the durable path
        logger.warning(
            "Generation stop signals are not subscribed on this worker: %s",
            type(exc).__name__,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global _client_runtime_cleanup_task
    # Startup
    _ensure_selector_event_loop()
    await _verify_async_database_ready()
    await init_database_migrations()
    await init_checkpoint_tables()
    await init_agents()
    _log_widget_runtime_status()
    _ensure_qdrant_collection()
    await _subscribe_generation_stop_signals()
    if settings.enable_client_runtime_bridge:
        _client_runtime_cleanup_task = asyncio.create_task(
            periodic_session_cleanup_task(
                SessionLocal,
                interval_seconds=settings.client_runtime_heartbeat_interval_seconds,
            )
        )
    yield
    # Shutdown
    if _client_runtime_cleanup_task is not None:
        _client_runtime_cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await _client_runtime_cleanup_task
        _client_runtime_cleanup_task = None
    try:
        from app.ai.mcp_registry import get_global_mcp_manager

        mcp_manager = await get_global_mcp_manager()
        if mcp_manager:
            await mcp_manager.cleanup()
            logger.info("MCP sessions closed cleanly")
    except Exception as e:
        logger.debug(f"MCP cleanup during shutdown (non-fatal): {e}")
    try:
        await close_client_runtime_store()
    except Exception as e:
        logger.debug(f"Client runtime store cleanup during shutdown (non-fatal): {e}")
    try:
        from app.database.async_session import dispose_async_engine

        await dispose_async_engine()
        logger.info("Async database connections closed cleanly")
    except Exception as e:
        logger.debug(f"Async engine disposal during shutdown (non-fatal): {e}")


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
            "app.api.model_usage",
            "app.api.client_devices",
            "app.api.device_runtime",
            "app.api.widgets",
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
            allow_private_network=True,
            expose_headers=["x-vercel-ai-ui-message-stream"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_private_network=True,
            expose_headers=["x-vercel-ai-ui-message-stream"],
        )

    # Register centralized exception handlers
    register_exception_handlers(app)

    # Include routers
    app.include_router(auth_router)
    app.include_router(users_router)
    app.include_router(conversations_router)
    app.include_router(custom_agents_router)
    app.include_router(custom_agents_conversation_router)
    # AI SDK aliases: same handlers re-mounted under /ai.
    app.include_router(custom_agents_router, prefix="/ai")
    app.include_router(custom_agents_conversation_router, prefix="/ai")
    app.include_router(messages_router)
    app.include_router(feedback_router)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(mcp_router)
    app.include_router(hitl_router)
    app.include_router(task_plans_router)
    app.include_router(ai_sdk_router)
    app.include_router(providers_router)
    app.include_router(model_config_router)
    app.include_router(model_usage_router)
    app.include_router(client_devices_router)
    app.include_router(device_runtime_router)
    app.include_router(tool_result_blobs_router)
    app.include_router(chat_images_router)
    app.include_router(web_images_router)
    app.include_router(widgets_router)

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
    """Basic health check, including the build this process is running.

    The build identity makes a stale process immediately distinguishable from
    current source when an endpoint's behavior is disputed.
    """
    return {"status": "healthy", "message": "OK", **resolve_build_info()}


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
        "message": ("All services healthy" if all_healthy else "One or more services unhealthy"),
    }


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
