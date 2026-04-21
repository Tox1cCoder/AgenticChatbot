"""Shared text normalization helpers for routing and tool discovery."""

from __future__ import annotations

# Common English stopwords that carry no signal for tool search
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "it",
        "this",
        "that",
        "be",
        "as",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "can",
        "use",
        "get",
        "set",
        "new",
        "my",
        "me",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "its",
        "all",
        "so",
        "if",
        "up",
        "out",
        "into",
        "via",
        "also",
    }
)

# Common English suffixes for lightweight morphological normalization
_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("tion", ""),
    ("tions", ""),
    ("ings", ""),
    ("ing", ""),
    ("tion", ""),
    ("ated", "ate"),
    ("ates", "ate"),
    ("ers", "er"),
    ("ies", "y"),
    ("es", ""),
    ("s", ""),
)


def _normalize_token(token: str) -> str:
    """Apply lightweight morphological normalization (suffix trimming).

    Only applied to tokens longer than 4 characters to avoid over-stemming.
    """
    if len(token) <= 4:
        return token
    for suffix, replacement in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)] + replacement
    return token


def tokenize_text(
    text: str,
    *,
    preserve_underscore: bool = False,
    filter_stopwords: bool = False,
    normalize: bool = False,
) -> list[str]:
    """Split text into lowercase alphanumeric tokens.

    Args:
        text: Input text.
        preserve_underscore: If True, underscores are kept as part of tokens.
        filter_stopwords: If True, remove common English stopwords.
        normalize: If True, apply lightweight suffix normalization.
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

    if filter_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    if normalize:
        tokens = [_normalize_token(t) for t in tokens]

    return tokens


def filter_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords from a token list."""
    return [t for t in tokens if t not in _STOPWORDS]


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
