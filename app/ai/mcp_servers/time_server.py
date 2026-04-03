import json
from datetime import datetime
from datetime import timezone as dt_timezone
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Time")


@mcp.tool()
def get_current_time(
    timezone: str = "Asia/Ho_Chi_Minh", format: str = "%Y-%m-%d %H:%M:%S %Z"
) -> str:
    """
    Return the current time for the requested IANA timezone.

    Args:
        timezone: IANA timezone identifier string. Must be a valid IANA timezone name.
            Common timezones: 'UTC', 'America/New_York', 'Europe/London', 'Asia/Tokyo', 'Asia/Ho_Chi_Minh'

        format: Optional strftime-compatible format string for the human_readable field.
            Format examples:
                '%Y-%m-%d %H:%M:%S' for '2024-01-15 14:30:00'
                '%I:%M %p' for '02:30 PM'
                '%A, %B %d, %Y' for 'Monday, January 15, 2024'
                '%Y-%m-%d %I:%M %p %Z' for '2024-01-15 02:30 PM EST'
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
        payload["note"] = "Requested timezone was not found. Returned time in UTC instead."

    return json.dumps(payload)


if __name__ == "__main__":
    mcp.run(transport="stdio")
