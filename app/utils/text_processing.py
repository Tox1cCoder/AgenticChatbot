"""
Text processing utilities for RAG pipeline.

Provides functions for token counting, sentence splitting, smart chunking,
text truncation, and page range extraction.
"""

import re
import logging
from typing import List, Tuple

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


def extract_page_range(text: str) -> Tuple[int, int]:
    """
    Extract page range from text containing [PAGE X] markers.

    Args:
        text: The text containing page markers

    Returns:
        Tuple of (first_page, last_page). Returns (1, 1) if no markers found.
    """
    # Find all page markers in format [PAGE X]
    page_pattern = r"\[PAGE\s+(\d+)\]"
    matches = re.findall(page_pattern, text)

    if not matches:
        return (1, 1)

    page_numbers = [int(m) for m in matches]
    return (min(page_numbers), max(page_numbers))


def create_chunks(
    text: str, max_chunk_size: int, overlap_size: int, by_sentences: bool = True
) -> List[str]:
    """
    Create text chunks with optional sentence-aware splitting.

    Args:
        text: The text to chunk
        max_chunk_size: Maximum size of each chunk in characters
        overlap_size: Number of characters to overlap between chunks
        by_sentences: If True, preserve sentence boundaries

    Returns:
        List of text chunks
    """
    if not text:
        return []

    if len(text) <= max_chunk_size:
        return [text]

    chunks = []

    if by_sentences:
        # Split by sentences and combine into chunks
        sentences = split_into_sentences(text)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed max size
            if current_length + sentence_length > max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                # Find sentences from the end to include for overlap
                overlap_sentences = []
                overlap_length = 0

                for prev_sentence in reversed(current_chunk):
                    if overlap_length + len(prev_sentence) <= overlap_size:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_length += len(prev_sentence)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

    else:
        # Character-based chunking with overlap
        start = 0

        while start < len(text):
            end = start + max_chunk_size

            # If not at the end, try to break at a space
            if end < len(text):
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos

            chunks.append(text[start:end].strip())

            # Move start position with overlap
            start = end - overlap_size

            # Ensure we're making progress
            if start <= chunks[-1].find(text[start : start + 10]):
                start = end

    return chunks


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
