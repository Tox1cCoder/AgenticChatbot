"""Stand-in for Desktop Commander: same tool names, no effects.

The directory name carries ``desktop-commander`` because that is how the
sidecar recognizes a Desktop Commander launch, whether through npx, node, or
an installed binary.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Fake Desktop Commander")


@mcp.tool()
def read_file(path: str) -> str:
    """Pretend to read a file."""
    return f"contents of {path}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Pretend to write a file."""
    return f"wrote {len(content)} characters to {path}"


@mcp.tool()
def set_config_value(key: str, value: str) -> str:
    """Pretend to change the server's own safety configuration."""
    return f"set {key}"


@mcp.tool()
def get_recent_tool_calls() -> str:
    """Pretend to return every client's call history."""
    return "[]"


@mcp.tool()
def unreleased_tool() -> str:
    """A tool a future Desktop Commander release might add."""
    return "new"


if __name__ == "__main__":
    mcp.run(transport="stdio")
