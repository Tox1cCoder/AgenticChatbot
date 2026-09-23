import asyncio
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Stateful Probe")

# In-process state stands in for what a real server keeps between calls, such
# as the processes Desktop Commander starts and later reads from.
_calls = {"count": 0}


@mcp.tool()
def increment() -> int:
    """Count calls served by this server process."""
    _calls["count"] += 1
    return _calls["count"]


@mcp.tool()
def process_id() -> int:
    """Return the server's own process id."""
    return os.getpid()


@mcp.tool()
async def wait(seconds: float) -> str:
    """Sleep, to outlast a caller's timeout."""
    await asyncio.sleep(seconds)
    return "done"


@mcp.tool()
def crash(marker_path: str) -> str:
    """Record the call in ``marker_path``, then exit in the middle of it."""
    with open(marker_path, "a", encoding="utf-8") as marker:
        marker.write("crash\n")
    os._exit(3)


if __name__ == "__main__":
    mcp.run(transport="stdio")
