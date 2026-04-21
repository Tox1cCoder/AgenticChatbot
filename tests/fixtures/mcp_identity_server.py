import json
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Scope Probe")


@mcp.tool()
def get_identity() -> str:
    """
    Return the connected sidecar runtime identity marker for scope verification.
    """
    payload = {
        "identity": os.environ.get("MCP_IDENTITY", "unknown"),
        "description": "connected sidecar runtime identity marker",
    }
    return json.dumps(payload)


if __name__ == "__main__":
    mcp.run(transport="stdio")
