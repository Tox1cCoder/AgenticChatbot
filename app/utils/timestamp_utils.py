"""
Shared utilities for factory timestamp generation
"""

from datetime import datetime, timezone


class TimestampUtils:
    """Utility class for consistent timestamp generation across factories"""

    @staticmethod
    def now() -> datetime:
        """Generate current UTC timestamp for entity creation/updates"""
        return datetime.now(timezone.utc)

    @staticmethod
    def get_timestamp_dict(created_at: datetime = None, updated_at: datetime = None) -> dict:
        """Generate timestamp dictionary with consistent defaults"""
        now = TimestampUtils.now()
        return {
            "created_at": created_at or now,
            "updated_at": updated_at or now,
        }
