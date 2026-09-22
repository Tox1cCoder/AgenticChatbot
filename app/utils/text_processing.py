"""
Text processing utilities for RAG pipeline.

Provides functions for token counting, sentence splitting, smart chunking,
text truncation, and page range extraction.
"""

import logging
import re

logger = logging.getLogger(__name__)

from nltk.tokenize import sent_tokenize  # noqa: E402


def split_into_sentences(text: str) -> list[str]:
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


def extract_page_range(text: str) -> tuple[int | None, int | None]:
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


PROJECT_INSTRUCTION_HEADER = "Project instructions:"
CONVERSATION_INSTRUCTION_HEADER = (
    "Conversation-specific instructions:"
)


def compose_system_instruction(
    project_instructions: str | None,
    persona_prompt: str | None,
    user_memory: str | None = None,
) -> str | None:
    """Combine a project's instructions, a conversation's persona, and memory.

    Each instruction part is sanitized independently against its own
    8000-character cap. Composing first and truncating after would
    silently discard the persona, because the project text leads —
    never call :func:`sanitize_persona` on the value returned here.

    ``user_memory`` is pre-fenced untrusted reference data, not an
    instruction, so it is appended verbatim and last, and it never
    causes the instruction headers to appear. It deliberately skips
    :func:`sanitize_persona`: that function is for authored prompts,
    and stripping the fence would turn remembered text into standing
    instructions.

    Headers are added only when both instruction parts are present, so
    a conversation with no project and no memory renders
    byte-identically to how it rendered before projects existed.
    """
    project = sanitize_persona(project_instructions)
    persona = sanitize_persona(persona_prompt)

    if project and persona:
        instructions = (
            f"{PROJECT_INSTRUCTION_HEADER}\n{project}\n\n"
            f"{CONVERSATION_INSTRUCTION_HEADER}\n{persona}"
        )
    else:
        instructions = project or persona

    memory = (user_memory or "").strip()
    if not memory:
        return instructions
    if not instructions:
        return memory
    return f"{instructions}\n\n{memory}"


def fix_markdown_code_blocks(text: str) -> str:
    """
    Fix markdown code blocks that are missing newlines before opening fences.

    """
    if not text:
        return text

    fixed = re.sub(r"([^\n\s])(```)", r"\1\n\2", text)

    return fixed
