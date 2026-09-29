"""Prompt-text normalization: persona/instruction sanitizing and markdown fixes."""

import re


def truncate_text(text: str, max_chars: int, add_ellipsis: bool = True) -> str:
    """Truncate to ``max_chars``, backing off to the last word boundary."""
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


def clean_text(text: str) -> str:
    """Collapse runs of spaces and of 3+ newlines, then strip."""
    if not text:
        return ""
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_persona(persona: str | None) -> str | None:
    """Clean authored prompt text and cap it at 8000 characters; blank -> None."""
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
    """Insert the newline a model often omits before a ``` fence."""
    if not text:
        return text
    return re.sub(r"([^\n\s])(```)", r"\1\n\2", text)
