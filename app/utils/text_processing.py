"""
Text processing utilities for RAG pipeline.

Provides functions for token counting, sentence splitting, smart chunking,
text truncation, and page range extraction.
"""

import re
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

import tiktoken
from nltk.tokenize import sent_tokenize


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    """
    if not text:
        return 0

    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    """
    if not text:
        return []

    return sent_tokenize(text)


def truncate_text(text: str, max_chars: int, add_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum number of characters, preserving word boundaries.

    Args:
        text: The text to truncate
        max_chars: Maximum number of characters
        add_ellipsis: Whether to add "..." when truncated

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_chars:
        return text

    # Truncate at word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")

    if last_space > 0:
        truncated = truncated[:last_space]

    if add_ellipsis:
        truncated += "..."

    return truncated


def extract_page_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract page range from text containing [PAGE X] markers.

    Args:
        text: The text containing page markers

    Returns:
        Tuple of (first_page, last_page). Returns (None, None) if no markers found.
    """
    # Find all page markers in format [PAGE X]
    page_pattern = r"\[PAGE\s+(\d+)\]"
    matches = re.findall(page_pattern, text)

    if not matches:
        return (None, None)

    page_numbers = [int(m) for m in matches]
    return (min(page_numbers), max(page_numbers))


def calculate_text_overlap(text1: str, text2: str) -> int:
    """
    Calculate the character overlap between two text chunks.
    """
    if not text1 or not text2:
        return 0

    # Find the longest suffix of text1 that is a prefix of text2
    max_overlap = min(len(text1), len(text2))

    for i in range(max_overlap, 0, -1):
        if text1[-i:] == text2[:i]:
            return i

    return 0


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing line breaks.
    """
    if not text:
        return ""

    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)

    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def validate_persona(persona: str | None, max_length: int = 8000) -> str | None:
    """
    Validate persona text.

    Args:
        persona: The persona text to validate
        max_length: Maximum allowed length

    Returns:
        The validated persona or None if empty

    Raises:
        ValueError: If persona exceeds max_length
    """
    if persona is None or not persona.strip():
        return None

    if len(persona) > max_length:
        raise ValueError(f"Persona exceeds maximum length of {max_length} characters")

    return persona


def sanitize_persona(persona: str | None) -> str | None:
    """
    Sanitize and truncate persona text.

    Args:
        persona: The persona text to sanitize

    Returns:
        Cleaned and truncated persona or None if empty
    """
    if persona is None or not persona.strip():
        return None

    # Clean the persona text
    cleaned = clean_text(persona)

    if len(cleaned) > 8000:
        cleaned = truncate_text(cleaned, 8000, add_ellipsis=False)

    return cleaned if cleaned else None


def fix_markdown_code_blocks(text: str) -> str:
    """
    Fix markdown code blocks that are missing newlines before opening fences.

    Ensures proper rendering by adding newline before ``` if preceded by non-whitespace.
    """
    if not text:
        return text

    # Pattern: non-whitespace character followed by ``` (code fence)
    # Replace with: the character, newline, then the code fence
    fixed = re.sub(r"([^\n\s])(```)", r"\1\n\2", text)

    return fixed
