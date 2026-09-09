import contextlib
import inspect
import logging
import sys
from importlib.metadata import PackageNotFoundError, version

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg_pool import AsyncConnectionPool

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import (
    AgentTransition,
    OutcomeProvenance,
    PendingTransition,
    PlanningDispatch,
    ResponseOutcome,
    RoutingDecision,
    TurnIdentity,
    WorkerResult,
    WorkerTask,
    WorkflowError,
)
from app.core.config import Settings

_CHECKPOINT_ALLOWED_TYPES = (
    AgentType,
    MessageRole,
    AgentResponse,
    AgentMessage,
    # routing-v2 control-plane contracts. Every Pydantic type stored directly
    # in graph state must round-trip as its own class, never as ``dict``.
    TurnIdentity,
    RoutingDecision,
    AgentTransition,
    PendingTransition,
    OutcomeProvenance,
    ResponseOutcome,
    WorkerTask,
    WorkerResult,
    PlanningDispatch,
    WorkflowError,
)

_CHECKPOINT_ALLOWED_JSON_MODULES: list[tuple[str, ...]] = [
    (*symbol.__module__.split("."), symbol.__name__) for symbol in _CHECKPOINT_ALLOWED_TYPES
]
_CHECKPOINT_ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    (symbol.__module__, symbol.__name__) for symbol in _CHECKPOINT_ALLOWED_TYPES
]

logger = logging.getLogger(__name__)


def _build_checkpoint_serializer() -> JsonPlusSerializer:
    """Build a serializer with strict JSON and MsgPack type allowlists."""
    try:
        parameters = inspect.signature(JsonPlusSerializer).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Cannot inspect the LangGraph checkpoint serializer API") from exc

    if "allowed_msgpack_modules" not in parameters:
        try:
            installed_version = version("langgraph-checkpoint")
        except PackageNotFoundError:
            installed_version = "unknown"
        raise RuntimeError(
            "Unsupported langgraph-checkpoint "
            f"{installed_version}; install langgraph-checkpoint>=4.1.1,<5.0.0 "
            f"in the active interpreter ({sys.executable})"
        )

    return JsonPlusSerializer(
        allowed_json_modules=_CHECKPOINT_ALLOWED_JSON_MODULES,
        allowed_msgpack_modules=_CHECKPOINT_ALLOWED_MSGPACK_MODULES,
    )


class CheckpointManager:
    def __init__(self, db_url: str, settings: Settings):
        """
        Initialize the checkpoint manager.
        """
        self.db_url = db_url
        self.settings = settings
        self.checkpointer: AsyncPostgresSaver | None = None
        self._pool: AsyncConnectionPool | None = None
        self._initialized = False

        if "postgresql+psycopg2://" in self.db_url:
            self.db_url = self.db_url.replace("postgresql+psycopg2://", "postgresql://")

        logger.debug(f"CheckpointManager initialized with schema: {settings.checkpoint_schema}")

    async def setup(self) -> None:
        """Initialize checkpoint persistence and its connection pool."""
        if self._initialized:
            return

        # Validate the dependency before allocating database resources. Older
        # serializers cannot enforce the MsgPack type allowlist.
        serde = _build_checkpoint_serializer()

        try:
            # Get pool configuration from settings
            min_size = getattr(self.settings, "checkpoint_pool_min_size", 2)
            max_size = getattr(self.settings, "checkpoint_pool_max_size", 10)

            # Create connection pool
            self._pool = AsyncConnectionPool(
                self.db_url,
                min_size=min_size,
                max_size=max_size,
                open=False,  # Don't open immediately
            )

            # Open the pool
            await self._pool.open()

            # Create AsyncPostgresSaver with the pool (not a dedicated connection)
            self.checkpointer = AsyncPostgresSaver(self._pool, serde=serde)

            # Run one-time setup using a temporary pooled connection
            async with self._pool.connection() as conn:
                await conn.set_autocommit(True)
                temp_saver = AsyncPostgresSaver(conn)
                await temp_saver.setup()

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to setup checkpoint manager: {e}", exc_info=True)
            raise

    async def health_check(self) -> bool:
        """
        Check if the connection pool is healthy.

        Returns:
            True if healthy, False otherwise
        """
        if not self._pool:
            return False

        try:
            async with self._pool.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"Checkpoint health check failed: {e}")
            return False

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect the connection pool with exponential backoff.

        Returns:
            True if reconnection successful, False otherwise
        """
        import asyncio

        delays = [1, 2, 4]  # Exponential backoff delays in seconds

        for attempt, delay in enumerate(delays, 1):
            try:
                logger.debug(f"Attempting checkpoint reconnection (attempt {attempt}/3)")

                # Close existing pool if present
                if self._pool:
                    with contextlib.suppress(Exception):
                        await self._pool.close()
                    self._pool = None

                # Reset state
                self._initialized = False
                self.checkpointer = None

                # Attempt setup
                await self.setup()

                logger.debug("Checkpoint reconnection successful")
                return True

            except Exception as e:
                logger.warning(f"Checkpoint reconnection attempt {attempt} failed: {e}")
                if attempt < len(delays):
                    await asyncio.sleep(delay)

        logger.error("All checkpoint reconnection attempts failed")
        return False

    def get_checkpointer(self) -> AsyncPostgresSaver | None:
        """
        Get the initialized checkpointer instance.
        """
        if not self._initialized:
            return None

        return self.checkpointer

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete all persisted checkpoint rows for a LangGraph thread."""
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            return False

        if not self._initialized or self.checkpointer is None:
            await self.setup()

        if self.checkpointer is not None:
            async_delete_thread = getattr(self.checkpointer, "adelete_thread", None)
            if callable(async_delete_thread):
                await async_delete_thread(normalized_thread_id)
                return True

        if self._pool is None:
            return False

        # Fixed, code-controlled table list only — never accept table names
        # from caller input. Delete children before parents (writes/blobs
        # reference checkpoints) to stay dependency-safe under FK constraints.
        tables = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")
        schema = getattr(self.settings, "checkpoint_schema", "public") or "public"
        async with self._pool.connection() as conn:
            for table_name in tables:
                await conn.execute(
                    f'DELETE FROM "{schema}"."{table_name}" WHERE thread_id = %s',
                    (normalized_thread_id,),
                )
        return True

    async def cleanup(self) -> None:
        """Cleanup checkpoint manager and close connection pool."""
        try:
            # Close the pool
            if self._pool:
                await self._pool.close()
                self._pool = None

            self._initialized = False
            self.checkpointer = None
            logger.debug("Checkpoint manager cleaned up")
        except Exception as e:
            logger.error(f"Error during checkpoint manager cleanup: {e}", exc_info=True)

    def get_pool_stats(self) -> dict | None:
        """
        Get statistics about the connection pool.

        Returns:
            Dict with pool stats if pool exists, None otherwise
        """
        if not self._pool:
            return None

        try:
            return {
                "min_size": self._pool.min_size,
                "max_size": self._pool.max_size,
                "initialized": self._initialized,
            }
        except Exception:
            return {"initialized": self._initialized}
