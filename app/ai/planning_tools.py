from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import tool

from .schemas import TodoAction, TodoItem, WriteTodosInput


def create_write_todos_tool():
    """
    Expose the write_todos schema to the LLM.
    """

    @tool(args_schema=WriteTodosInput)
    def write_todos(
        action: TodoAction,
        todos: Optional[List[TodoItem]] = None,
        todo: Optional[TodoItem] = None,
        todo_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        action_value = action.value if hasattr(action, "value") else str(action)
        if action == TodoAction.SET_TODOS:
            return f"Set {len(todos or [])} todos."
        if action == TodoAction.ADD_TODO:
            desc = getattr(todo, "description", None) if todo else None
            return f"Added todo: {desc or 'unknown'}"
        if action == TodoAction.UPDATE_TODO:
            ref = getattr(todo, "id", None) if todo else None
            return f"Updated todo: {ref or 'unknown'}"
        if action == TodoAction.REMOVE_TODO:
            return f"Removed todo: {todo_id or 'unknown'}"
        if action == TodoAction.START_TODO:
            return f"Started todo: {todo_id or 'unknown'}"
        if action == TodoAction.COMPLETE_TODO:
            msg = f"Completed todo: {todo_id or 'unknown'}"
            if reason:
                msg += f" ({reason})"
            return msg
        return f"Unsupported action: {action_value}"

    return write_todos
