from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool


def create_user_memory_tools(*, repository: Any, user_id: str | None):
    if not user_id:
        return []

    def remember_memory(content: str, source: str = "explicit_user_request") -> str:
        """Save a memory item when the user explicitly asks to remember something."""
        text = str(content or "").strip()
        if not text:
            return "Error: memory content is required."
        repository.create(user_id=user_id, content=text, source=source)
        return "Memory saved."

    def list_memories(limit: int = 20) -> str:
        """List the current user's saved memory items, most recent first."""
        rows = repository.list_for_user(user_id, limit=max(1, min(int(limit or 20), 50)))
        if not rows:
            return "No saved memories."
        return "\n".join(
            f"- {row['content'] if isinstance(row, dict) else row.content}" for row in rows
        )

    def forget_memory(memory_id: str) -> str:
        """Delete a saved memory item by id for the current user."""
        deleted = repository.delete_for_user(memory_id, user_id)
        return "Memory removed." if deleted else "Memory not found."

    return [
        StructuredTool.from_function(remember_memory),
        StructuredTool.from_function(list_memories),
        StructuredTool.from_function(forget_memory),
    ]
