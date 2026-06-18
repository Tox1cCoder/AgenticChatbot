"""Shared Gemini API retry helpers.

Extracted from document_processing_service.py so that any service that calls
the Gemini API can detect 429 / RESOURCE_EXHAUSTED errors and honour the
``retryDelay`` hint without duplicating the parsing logic.
"""

from __future__ import annotations

import re

from google.genai import errors as genai_errors


def is_rate_limit_error(error: genai_errors.ClientError) -> bool:
    """Return True if *error* is a 429 / RESOURCE_EXHAUSTED rate-limit error."""
    status = (error.status or "").upper() if isinstance(error.status, str) else ""
    return error.code == 429 or status == "RESOURCE_EXHAUSTED"


def parse_retry_delay(error: genai_errors.ClientError) -> float | None:
    """Parse the suggested retry delay from a 429 ClientError.

    Checks ``error.details`` for a ``retryDelay`` / ``retry_delay`` field that
    may be an int/float (seconds), a string with a unit suffix (``"30s"``,
    ``"500ms"``, ``"2m"``), or a dict with ``seconds``/``nanos`` keys.

    Falls back to scanning the error message for ``"retry in Xs"`` text.

    Returns the delay in seconds, or *None* if no hint can be found.
    """
    details = getattr(error, "details", None)
    detail_entries: list = []
    if isinstance(details, dict):
        error_block = details.get("error")
        if isinstance(error_block, dict):
            detail_entries = error_block.get("details") or []
        if not detail_entries:
            detail_entries = details.get("details") or []
    elif isinstance(details, list):
        detail_entries = details

    for entry in detail_entries or []:
        if not isinstance(entry, dict):
            continue
        retry_value = entry.get("retryDelay") or entry.get("retry_delay")
        if retry_value is None:
            continue

        parsed_value: float | None = None
        if isinstance(retry_value, (int, float)):
            parsed_value = float(retry_value)
        elif isinstance(retry_value, str):
            match = re.match(r"([\d\.]+)\s*([a-zA-Z]*)", retry_value.strip())
            if match:
                amount_str, unit = match.groups()
                try:
                    amount = float(amount_str)
                except ValueError:
                    amount = None
                if amount is not None:
                    unit = unit.lower()
                    if unit in ("", "s", "sec", "secs", "second", "seconds"):
                        parsed_value = amount
                    elif unit in ("ms", "millisecond", "milliseconds"):
                        parsed_value = amount / 1000.0
                    elif unit in ("m", "min", "mins", "minute", "minutes"):
                        parsed_value = amount * 60.0
        elif isinstance(retry_value, dict):
            seconds = retry_value.get("seconds")
            nanos = retry_value.get("nanos", 0)
            if seconds is not None or nanos:
                parsed_value = float(seconds or 0) + float(nanos) / 1_000_000_000

        if parsed_value is not None:
            return parsed_value

    # Fallback: scan error message for "retry in Xs"
    message = getattr(error, "message", "")
    if isinstance(message, str):
        match = re.search(r"retry in\s+([\d\.]+)s", message, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

    return None
