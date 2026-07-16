from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REMOVED_FILES = (
    "app/ai/summarization_middleware.py",
    "app/ai/conversation_summarizer.py",
    "app/ai/memory.py",
    "app/repositories/conversation_memory_summary.py",
    "tests/test_conversation_summarizer.py",
    "tests/test_conversation_memory_summary_repository.py",
)

FORBIDDEN_TEXT = (
    "app.ai.memory",
    "app.ai.conversation_summarizer",
    "app.ai.summarization_middleware",
    "app.repositories.conversation_memory_summary",
    "MemoryManager",
    "get_memory_manager",
    "history_summary",
    "history_summary_updated_at",
    "summary_cursor_message_id",
    "conversation_summarized",
    "last_summarized_message_id",
    "_summary_refresh_pending",
    "_summary_refresh_active",
    "_schedule_summary_refresh",
    "_summary_refresh_runner",
    "refresh_summary_after_turn",
    "compact_tool_messages_for_retry",
    "memory_summary_min_unsummarized_messages",
    "memory_summary_min_unsummarized_tokens",
    "memory_summary_keep_messages",
    "memory_summary_max_tokens",
    "memory_summary_timeout_seconds",
    "enable_summarization",
    "summarization_trigger_tokens",
    "summarization_trigger_messages",
    "summarization_trigger_fraction",
    "summarization_model_context_size",
    "summarization_keep_messages",
    "summarization_model",
    "summarization_max_summary_tokens",
    "summarization_timeout_seconds",
    "MEMORY_SUMMARY_",
    "ENABLE_SUMMARIZATION",
    "SUMMARIZATION_",
)


def _scanned_files():
    for base in (ROOT / "app", ROOT / "tests"):
        for path in base.rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith("app/alembic/versions/"):
                continue
            if path.name == Path(__file__).name:
                continue
            yield path
    yield ROOT / "README.md"
    yield ROOT / ".env.example"


def test_scoped_legacy_files_are_deleted() -> None:
    remaining = [path for path in REMOVED_FILES if (ROOT / path).exists()]

    assert remaining == []


def test_removed_symbols_and_environment_names_are_absent() -> None:
    matches = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8")
        for symbol in FORBIDDEN_TEXT:
            if symbol in text:
                matches.append(f"{path.relative_to(ROOT).as_posix()}: {symbol}")

    assert matches == []


def test_no_duplicate_public_or_four_character_token_estimators() -> None:
    matches = []
    four_character = re.compile(r"len\([^\n]+\)\s*//\s*4")
    for path in _scanned_files():
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^def estimate_tokens\(", text, flags=re.MULTILINE):
            matches.append(f"{path.relative_to(ROOT).as_posix()}: public estimate_tokens")
        if four_character.search(text):
            matches.append(f"{path.relative_to(ROOT).as_posix()}: len(...) // 4")

    assert matches == []
