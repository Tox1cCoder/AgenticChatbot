import logging
from typing import Optional
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import psycopg

from app.core.config import Settings

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, db_url: str, settings: Settings):
        """
        Initialize the checkpoint manager.
        """
        self.db_url = db_url
        self.settings = settings
        self.checkpointer: Optional[AsyncPostgresSaver] = None
        self._conn = None
        self._initialized = False

        if "postgresql+psycopg2://" in self.db_url:
            self.db_url = self.db_url.replace("postgresql+psycopg2://", "postgresql://")

        logger.info(
            f"CheckpointManager initialized with schema: {settings.checkpoint_schema}"
        )

    async def setup(self) -> None:
        """
        Initialize the checkpoint database tables.
        """
        if self._initialized:
            return

        try:
            # Create persistent async connection
            self._conn = await psycopg.AsyncConnection.connect(self.db_url)

            # Create AsyncPostgresSaver with the connection
            self.checkpointer = AsyncPostgresSaver(self._conn)

            # Create checkpoint tables in the database
            await self.checkpointer.setup()

            self._initialized = True
            logger.info(
                f"Successfully created checkpoint tables in schema '{self.settings.checkpoint_schema}'"
            )

        except Exception as e:
            raise

    def get_checkpointer(self) -> Optional[AsyncPostgresSaver]:
        """
        Get the initialized checkpointer instance.
        """
        if not self._initialized:
            return None

        return self.checkpointer

    async def cleanup(self) -> None:
        """Cleanup checkpoint manager and close database connection."""
        try:
            if self._conn:
                await self._conn.close()
                self._conn = None

            self._initialized = False
            self.checkpointer = None
        except Exception as e:
            logger.error(f"Error during checkpoint manager cleanup: {e}", exc_info=True)
