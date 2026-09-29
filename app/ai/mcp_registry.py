"""
MCP Registry Module - Single Source of Truth for MCPManager.

This module provides a unified registry for MCPManager instances, ensuring
both the DI container and agent code share the exact same MCPManager instance.

Key features:
- Single MCPManager instance shared across the application
- Config file mtime tracking with automatic reload on changes
- Tools generation versioning for cache invalidation
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .mcp_integration import MCPManager

logger = logging.getLogger(__name__)


class MCPRegistry:
    """
    Central registry for MCPManager instance.

    Ensures a single MCPManager is shared between:
    - DI Container (Container.mcp_manager)
    - Agent initialization (get_global_mcp_manager)

    Also provides:
    - Config file change detection (mtime tracking)
    - Tools generation versioning for cache invalidation
    """

    _instance: Optional["MCPManager"] = None
    _initialized: bool = False
    _init_lock: asyncio.Lock = asyncio.Lock()

    # Config file tracking
    _config_mtime: float = 0.0
    _config_path: str | None = None

    # Tools generation version (incremented on reload, enable/disable, add/remove)
    _tools_generation: int = 0

    @classmethod
    def get_config_path(cls) -> str:
        if cls._config_path:
            return cls._config_path
        return str(Path(__file__).parent / "mcp_config.json")

    @classmethod
    def get_tools_generation(cls) -> int:
        """
        Get the current tools generation version.

        Agents can compare this with their cached version to detect
        when tools need to be refreshed.
        """
        return cls._tools_generation

    @classmethod
    def increment_tools_generation(cls) -> int:
        """
        Increment and return the new tools generation version.

        Called when:
        - Config is reloaded
        - A server is enabled/disabled
        - A server is added/removed
        - Tools are explicitly refreshed
        """
        cls._tools_generation += 1
        logger.debug("MCP tools generation incremented to %d", cls._tools_generation)
        return cls._tools_generation

    @classmethod
    def _check_config_changed(cls) -> bool:
        """
        Check if the config file has been modified since last load.

        Returns True if config should be reloaded.
        """
        config_path = cls.get_config_path()
        try:
            current_mtime = os.path.getmtime(config_path)
            if current_mtime > cls._config_mtime:
                logger.debug(
                    "MCP config file changed: mtime %f -> %f",
                    cls._config_mtime,
                    current_mtime,
                )
                return True
        except OSError:
            pass
        return False

    @classmethod
    def _update_config_mtime(cls) -> None:
        config_path = cls.get_config_path()
        try:
            cls._config_mtime = os.path.getmtime(config_path)
        except OSError:
            cls._config_mtime = time.time()

    @classmethod
    def get_manager_sync(cls) -> Optional["MCPManager"]:
        """
        Get the MCPManager instance synchronously (without initialization).

        Returns None if not yet initialized. Use this for DI container
        providers that need synchronous access.
        """
        return cls._instance

    @classmethod
    async def get_manager_async(cls, auto_reload: bool = True) -> "MCPManager":
        """Get or create the shared, initialized MCPManager.

        With ``auto_reload`` a changed config file mtime triggers a reload.
        """
        if cls._instance is None or not cls._initialized:
            async with cls._init_lock:
                if cls._instance is None or not cls._initialized:
                    from .mcp_integration import MCPManager

                    cls._instance = MCPManager(config_path=cls.get_config_path())
                    await cls._instance.initialize()
                    await cls._instance.get_tools()

                    cls._initialized = True
                    cls._update_config_mtime()
                    cls.increment_tools_generation()

                    logger.info("MCP Registry initialized with shared MCPManager instance")
                    return cls._instance

        # Outside the lock on purpose: asyncio.Lock is not reentrant, and
        # reload_config takes it. Checking here while holding it deadlocked a
        # caller that lost the initialization race to a config change.
        if auto_reload and cls._check_config_changed():
            await cls.reload_config(only_if_changed=True)
        return cls._instance

    @classmethod
    async def reload_config(cls, *, only_if_changed: bool = False) -> None:
        """Refresh tools from all enabled servers and bump the tools generation."""
        if cls._instance is None:
            return

        async with cls._init_lock:
            # Concurrent callers all observe the same new mtime; only the first
            # to get the lock should tear the sessions down and rebuild them.
            if only_if_changed and not cls._check_config_changed():
                return
            logger.info("Reloading MCP configuration...")

            # Reload configuration and tools
            await cls._instance.reload_tools()

            cls._update_config_mtime()
            cls.increment_tools_generation()

            logger.info(
                "MCP configuration reloaded (generation=%d)",
                cls._tools_generation,
            )

    @classmethod
    def notify_server_change(cls) -> None:
        """Called by MCPManager when servers are enabled/disabled/added/removed."""
        cls.increment_tools_generation()


async def get_global_mcp_manager() -> "MCPManager":
    """The primary entry point for agents to get the shared MCPManager."""
    return await MCPRegistry.get_manager_async()


def get_mcp_tools_generation() -> int:
    """
    Get the current tools generation version.

    Agents can use this to check if they need to refresh their tool cache.
    """
    return MCPRegistry.get_tools_generation()
