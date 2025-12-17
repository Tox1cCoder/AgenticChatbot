"""URL parser for MCP server configuration.

This module provides utilities to parse MCP server URLs and convert them
into appropriate server configurations. Supports both local npx-based
servers and remote HTTP servers.
"""

from typing import Any, Dict
from app.core.exceptions.mcp import ServerConfigurationError


def generate_server_name_from_url(url: str) -> str:
    """Generate a unique server name from a URL.

    Args:
        url: The MCP server URL (npx command or HTTP URL)

    Returns:
        A generated server name

    Examples:
        >>> generate_server_name_from_url("npx @smithery/cli@latest run @ThinkFar/clear-thought-mcp --playground")
        'clear-thought-mcp'
        >>> generate_server_name_from_url("https://server.smithery.ai/reddit/mcp")
        'reddit-mcp'
    """
    url = url.strip()

    # Handle npx URLs
    if url.startswith("npx "):
        # Extract package name from npx command
        # Pattern: npx @smithery/cli@latest run @ThinkFar/clear-thought-mcp --playground
        parts = url.split()
        for i, part in enumerate(parts):
            if part == "run" and i + 1 < len(parts):
                package = parts[i + 1]
                # Extract the actual package name after @org/
                if "/" in package:
                    package_name = package.split("/")[-1]
                else:
                    package_name = package.lstrip("@")
                # Clean up any remaining special characters
                return package_name.replace("@", "-")
        # Fallback if pattern doesn't match
        return "npx-server"

    # Handle HTTP/HTTPS URLs
    elif url.startswith("http://") or url.startswith("https://"):
        # Extract server name from URL path
        # https://server.smithery.ai/reddit/mcp -> reddit-mcp
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            path = parsed.path.strip("/")
            if path:
                # Use the last two segments or the entire path
                segments = path.split("/")
                if len(segments) >= 2:
                    return "-".join(segments[-2:])
                else:
                    return segments[-1]
            else:
                # Use domain name if no path
                domain = parsed.netloc.replace(".", "-")
                return f"{domain}-mcp"
        except Exception:
            return "http-server"

    # Unknown format
    return "unknown-server"


def parse_mcp_url(url: str) -> Dict[str, Any]:
    """Parse an MCP server URL and generate appropriate configuration.

    Args:
        url: The MCP server URL (npx command or HTTP URL)

    Returns:
        A dictionary containing the server configuration

    Raises:
        ServerConfigurationError: If the URL format is invalid
    """
    url = url.strip()

    if not url:
        raise ServerConfigurationError(
            detail="URL cannot be empty", error_code="INVALID_URL"
        )

    # Handle npx URLs
    if url.startswith("npx "):
        parts = url.split()

        if len(parts) < 2:
            raise ServerConfigurationError(
                detail="Invalid npx command format. Expected: npx <package> [args...]",
                error_code="INVALID_NPX_FORMAT",
            )

        # Extract command and args
        command = parts[0]  # "npx"
        args = parts[1:]

        filtered_args = []
        flags_to_remove = {
            "--playground",
            "--verbose",
            "-v",
            "--debug",
            "--interactive",
        }

        for arg in args:
            # Skip flags that cause non-JSON output
            if arg not in flags_to_remove:
                filtered_args.append(arg)

        env_vars = {
            "NODE_NO_WARNINGS": "1",  # Suppress Node.js warnings
        }

        return {
            "transport": "stdio",
            "command": command,
            "args": filtered_args,
            "env": env_vars,
        }

    # Handle HTTP/HTTPS URLs
    elif url.startswith("http://") or url.startswith("https://"):
        # Validate URL format
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)

            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL structure")

            return {"transport": "streamable_http", "url": url}
        except Exception as e:
            raise ServerConfigurationError(
                detail=f"Invalid HTTP URL format: {str(e)}",
                error_code="INVALID_HTTP_URL",
            )

    # Unknown URL format
    else:
        raise ServerConfigurationError(
            detail="URL must start with 'npx ', 'http://', or 'https://'",
            error_code="UNSUPPORTED_URL_FORMAT",
        )
