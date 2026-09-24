"""Stand-in for Desktop Commander: same tool names, no effects.

The directory name carries ``desktop-commander`` because that is how the
sidecar recognizes a Desktop Commander launch, whether through npx, node, or
an installed binary.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Fake Desktop Commander")


@mcp.tool()
def read_file(path: str, isUrl: bool = False) -> str:  # noqa: N803 - Desktop Commander's name
    """Pretend to read a file."""
    return f"contents of {path}"


@mcp.tool()
def read_multiple_files(paths: list[str]) -> str:
    """Pretend to read several files."""
    return f"contents of {len(paths)} files"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Pretend to write a file."""
    return f"wrote {len(content)} characters to {path}"


@mcp.tool()
def edit_block(file_path: str, old_string: str, new_string: str) -> str:
    """Pretend to edit a file."""
    return f"edited {file_path}"


@mcp.tool()
def move_file(source: str, destination: str) -> str:
    """Pretend to move a file."""
    return f"moved {source} to {destination}"


@mcp.tool()
def list_directory(path: str) -> str:
    """Pretend to list a folder."""
    return f"listing of {path}"


@mcp.tool()
def start_search(path: str, pattern: str, includeHidden: bool = False) -> str:  # noqa: N803
    """Pretend to search a folder."""
    return f"searching {path} for {pattern}"


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
