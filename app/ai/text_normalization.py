"""Shared text normalization helpers for routing and tool discovery."""

from __future__ import annotations


def tokenize_text(text: str, *, preserve_underscore: bool = False) -> list[str]:
    """Split text into lowercase alphanumeric tokens.

    Args:
        text: Input text.
        preserve_underscore: If True, underscores are kept as part of tokens.
    """
    if not text:
        return []

    tokens: list[str] = []
    current: list[str] = []

    for char in text.lower():
        if char.isalnum() or (preserve_underscore and char == "_"):
            current.append(char)
            continue

        if current:
            tokens.append("".join(current))
            current.clear()

    if current:
        tokens.append("".join(current))

    return tokens


def sanitize_identifier(value: str | None, *, fallback: str = "tool") -> str:
    """Normalize a string into a safe lowercase identifier."""
    raw_value = str(value or "").strip().lower()
    if not raw_value:
        return fallback

    normalized: list[str] = []
    last_was_separator = False

    for char in raw_value:
        if (char.isascii() and char.isalnum()) or char == "_":
            normalized.append(char)
            last_was_separator = False
            continue

        if normalized and not last_was_separator:
            normalized.append("_")
            last_was_separator = True

    result = "".join(normalized).strip("_")
    return result or fallback
