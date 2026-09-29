"""Label coercion shared by every metrics module in this package."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any


def bounded_label(value: Any, allowed: Collection[str]) -> str:
    """Normalize ``value`` and return it only if allowlisted, else ``"other"``.

    Prometheus keeps one series per distinct label value forever, so anything
    outside a closed set -- an id, a URL, a caller's mistake -- must collapse.
    """
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else "other"
