import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.core.config import Settings

logger = logging.getLogger(__name__)


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
        """
        Initialize the checkpoint database tables with connection pooling.
        """
        if self._initialized:
            return

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
            self.checkpointer = AsyncPostgresSaver(self._pool)

            # Run one-time setup using a temporary pooled connection.
            # autocommit=True is required because LangGraph's setup() issues
            # CREATE INDEX CONCURRENTLY, which PostgreSQL forbids inside a
            # transaction block.
            async with self._pool.connection() as conn:
                await conn.set_autocommit(True)
                temp_saver = AsyncPostgresSaver(conn)
                await temp_saver.setup()

            self._initialized = True
            logger.debug(
                f"Successfully created checkpoint tables in schema '{self.settings.checkpoint_schema}' "
                f"with pool size {min_size}-{max_size}"
            )

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
                    try:
                        await self._pool.close()
                    except Exception:
                        pass
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
