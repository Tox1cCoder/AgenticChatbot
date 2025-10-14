"""
MCP Integration Module
Provides MCP server client management and tool loading
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages MCP server connections and tool loading"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MCP Manager

        Args:
            config_path: Path to MCP configuration JSON file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self.client: Optional[MultiServerMCPClient] = None
        self._tools: List[BaseTool] = []

    def _get_default_config_path(self) -> str:
        """Get default config path relative to this file"""
        return str(Path(__file__).parent / "mcp_config.json")

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse MCP configuration from JSON file"""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded MCP config from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.warning(
                f"MCP config file not found at {self.config_path}. Using empty config."
            )
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MCP config JSON: {e}")
            return {}

    def _build_server_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Build MultiServerMCPClient configuration from loaded config

        Returns:
            Dictionary compatible with MultiServerMCPClient format
        """
        mcp_servers = self.config.get("mcp_servers", {})
        server_config = {}

        for server_name, server_info in mcp_servers.items():
            if not server_info.get("enabled", True):
                logger.info(f"Skipping disabled MCP server: {server_name}")
                continue

            transport = server_info.get("transport", "stdio")

            if transport == "stdio":
                # Convert relative paths to absolute
                args = server_info.get("args", [])
                abs_args = []
                for arg in args:
                    if arg.endswith(".py") and not os.path.isabs(arg):
                        # Make path absolute relative to project root
                        abs_path = os.path.abspath(arg)
                        abs_args.append(abs_path)
                    else:
                        abs_args.append(arg)

                server_config[server_name] = {
                    "transport": transport,
                    "command": server_info.get("command", "python"),
                    "args": abs_args,
                }

                # Pass environment variables to subprocess if specified
                if "env" in server_info:
                    server_config[server_name]["env"] = server_info["env"]
            elif transport in ["streamable_http", "sse"]:
                server_config[server_name] = {
                    "transport": transport,
                    "url": server_info.get("url", ""),
                }
                if "headers" in server_info:
                    server_config[server_name]["headers"] = server_info["headers"]
            else:
                logger.warning(f"Unknown transport type for {server_name}: {transport}")
                continue

            logger.info(f"Configured MCP server: {server_name} ({transport})")

        return server_config

    async def initialize(self) -> None:
        """Initialize MCP client and load configuration"""
        # Set environment variables for MCP servers to use
        from app.core.config import settings

        if settings.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
            logger.info("Set TAVILY_API_KEY environment variable for MCP servers")

        self.config = self._load_config()
        server_config = self._build_server_config()

        if not server_config:
            logger.warning(
                "No MCP servers configured. MCP tools will not be available."
            )
            return

        try:
            self.client = MultiServerMCPClient(server_config)
            logger.info("MCP client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            self.client = None

    async def get_tools(self) -> List[BaseTool]:
        """
        Get all tools from configured MCP servers

        Returns:
            List of LangChain BaseTool instances
        """
        if not self.client:
            logger.warning("MCP client not initialized. Returning empty tool list.")
            return []

        if self._tools:
            return self._tools

        try:
            self._tools = await self.client.get_tools()
            logger.info(f"Loaded {len(self._tools)} MCP tools")
            return self._tools
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            return []

    async def get_server_tools(self, server_name: str) -> List[BaseTool]:
        """
        Get tools from a specific MCP server

        Args:
            server_name: Name of the MCP server

        Returns:
            List of LangChain BaseTool instances from that server
        """
        if not self.client:
            logger.warning("MCP client not initialized")
            return []

        try:
            async with self.client.session(server_name) as session:
                from langchain_mcp_adapters.tools import load_mcp_tools

                tools = await load_mcp_tools(session)
                logger.info(f"Loaded {len(tools)} tools from {server_name}")
                return tools
        except Exception as e:
            logger.error(f"Failed to load tools from {server_name}: {e}")
            return []

    async def cleanup(self) -> None:
        """Cleanup MCP client resources"""
        if self.client:
            self._tools = []
            self.client = None
            logger.info("MCP client cleaned up")
