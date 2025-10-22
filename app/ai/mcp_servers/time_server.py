import json
from datetime import datetime, timezone as dt_timezone

from mcp.server.fastmcp import FastMCP

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # pragma: no cover - zoneinfo should exist on supported versions
    ZoneInfo = None  # type: ignore[assignment]

mcp = FastMCP("Time")


@mcp.tool()
def get_current_time(timezone: str = "UTC", format: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """
    Return the current time for the requested IANA timezone.

    Args:
        timezone: IANA timezone identifier (e.g. 'UTC', 'America/New_York').
        format: Optional strftime-compatible format for the human_readable field.
    """
    tzinfo = None
    resolved_timezone = timezone

    if ZoneInfo is not None:
        try:
            tzinfo = ZoneInfo(timezone)
        except Exception:
            tzinfo = None

    if tzinfo is None:
        tzinfo = dt_timezone.utc
        resolved_timezone = "UTC"

    now = datetime.now(tzinfo)

    try:
        human_readable = now.strftime(format)
    except Exception:
        human_readable = now.strftime("%Y-%m-%d %H:%M:%S %Z")

    payload = {
        "requested_timezone": timezone,
        "timezone": resolved_timezone,
        "iso": now.isoformat(),
        "epoch_seconds": now.timestamp(),
        "human_readable": human_readable,
    }

    if resolved_timezone != timezone:
        payload[
            "note"
        ] = "Requested timezone was not found. Returned time in UTC instead."

    return json.dumps(payload)


if __name__ == "__main__":
    mcp.run(transport="stdio")
