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
        """Get the default config path."""
        if cls._config_path:
            return cls._config_path
        return str(Path(__file__).parent / "mcp_config.json")

    @classmethod
    def set_config_path(cls, path: str) -> None:
        """Override the config path."""
        cls._config_path = path

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
        """Update the tracked config mtime to current."""
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
        """
        Get or create the shared MCPManager instance.

        Args:
            auto_reload: If True, check config mtime and reload if changed.

        Returns:
            The shared MCPManager instance, fully initialized.
        """
        # Fast path: already initialized
        if cls._instance is not None and cls._initialized:
            # Check for config changes if auto_reload enabled
            if auto_reload and cls._check_config_changed():
                await cls.reload_config()
            return cls._instance

        async with cls._init_lock:
            # Double-check after acquiring lock
            if cls._instance is not None and cls._initialized:
                if auto_reload and cls._check_config_changed():
                    await cls.reload_config()
                return cls._instance

            # Create new instance
            from .mcp_integration import MCPManager

            cls._instance = MCPManager(config_path=cls.get_config_path())
            await cls._instance.initialize()

            # Pre-load tools
            await cls._instance.get_tools()

            cls._initialized = True
            cls._update_config_mtime()
            cls.increment_tools_generation()

            logger.info("MCP Registry initialized with shared MCPManager instance")

        return cls._instance

    @classmethod
    async def reload_config(cls) -> None:
        """
        Reload MCP configuration from file.

        This refreshes tools from all enabled servers and increments
        the tools generation.
        """
        if cls._instance is None:
            return

        async with cls._init_lock:
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
    async def reset(cls) -> None:
        """
        Reset the registry, cleaning up the current manager.

        Used for testing or full application restart.
        """
        async with cls._init_lock:
            if cls._instance is not None:
                await cls._instance.cleanup()

            cls._instance = None
            cls._initialized = False
            cls._config_mtime = 0.0
            cls._tools_generation = 0

            logger.debug("MCP Registry reset")

    @classmethod
    def notify_server_change(cls) -> None:
        """
        Notify the registry that a server config has changed.

        Called by MCPManager when servers are enabled/disabled/added/removed.
        """
        cls.increment_tools_generation()


# =============================================================================
# Backward-compatible module-level functions
# =============================================================================


async def get_global_mcp_manager() -> "MCPManager":
    """
    Get or create a singleton MCPManager instance.

    This function is the primary entry point for agents to get MCP tools.
    It delegates to MCPRegistry for unified instance management.

    Returns:
        MCPManager: The global MCP manager instance with tools pre-loaded.
    """
    return await MCPRegistry.get_manager_async()


async def reset_global_mcp_manager() -> None:
    """
    Reset the global MCP manager.

    Used for testing or when a full refresh is needed.
    """
    await MCPRegistry.reset()


def get_mcp_tools_generation() -> int:
    """
    Get the current tools generation version.

    Agents can use this to check if they need to refresh their tool cache.
    """
    return MCPRegistry.get_tools_generation()
