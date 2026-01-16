from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .schemas import TodoAction, TodoStatus


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_action(action: Any) -> str:
    if hasattr(action, "value"):
        return str(action.value).strip().lower()
    if isinstance(action, str):
        return action.strip().lower()
    return str(action or "").strip().lower()


def coerce_todo_item(raw_todo: Any, *, fallback_order: int) -> Dict[str, Any]:
    if hasattr(raw_todo, "model_dump"):
        todo = raw_todo.model_dump()
    elif isinstance(raw_todo, dict):
        todo = dict(raw_todo)
    else:
        todo = {"description": str(raw_todo)}

    todo_id = todo.get("id")
    todo["id"] = str(todo_id) if todo_id else str(uuid.uuid4())

    status = todo.get("status", TodoStatus.PENDING)
    status_value = status.value if hasattr(status, "value") else str(status)
    todo["status"] = status_value or TodoStatus.PENDING.value

    order = todo.get("order", fallback_order)
    try:
        todo["order"] = int(order)
    except (TypeError, ValueError):
        todo["order"] = fallback_order

    if "description" in todo and todo["description"] is not None:
        todo["description"] = str(todo["description"])
    else:
        todo["description"] = ""

    return todo


def find_next_ready_task(
    todos: List[Dict[str, Any]], start_index: int = 0
) -> Optional[int]:
    for i in range(start_index, len(todos)):
        raw_status = todos[i].get("status", TodoStatus.PENDING.value)
        status = raw_status.value if hasattr(raw_status, "value") else raw_status
        if status == TodoStatus.PENDING.value:
            return i
    return None


def apply_write_todos_action(
    *,
    todos: List[Dict[str, Any]],
    current_task_index: Optional[int],
    tool_args: Any,
    max_todos: int = 50,
) -> Tuple[List[Dict[str, Any]], Optional[int], str, str]:
    """
    Apply a single write_todos tool call to an in-memory todos list.

    Returns: (todos, current_task_index, result_message, normalized_action)
    """
    args = _as_dict(tool_args)
    action = _normalize_action(args.get("action"))
    result = ""

    if action == TodoAction.SET_TODOS.value:
        new_todos = args.get("todos", [])
        if not isinstance(new_todos, list):
            return todos, current_task_index, "Error: Invalid todos payload", action

        if len(new_todos) > max_todos:
            msg = (
                f"Error: Plan exceeds maximum of {max_todos} todos "
                f"(requested {len(new_todos)}). Please reduce the number of tasks."
            )
            return todos, current_task_index, msg, action

        coerced = [coerce_todo_item(t, fallback_order=i) for i, t in enumerate(new_todos)]
        current_task_index = 0 if coerced else None
        return coerced, current_task_index, f"Set {len(coerced)} todos in the plan.", action

    if action == TodoAction.ADD_TODO.value:
        new_todo_raw = args.get("todo")
        if not new_todo_raw:
            return todos, current_task_index, "Error: No todo provided for ADD_TODO", action

        new_todo = coerce_todo_item(new_todo_raw, fallback_order=len(todos))
        todos.append(new_todo)
        if len(todos) > max_todos:
            todos = todos[:max_todos]
        return (
            todos,
            current_task_index,
            f"Added todo: {new_todo.get('description', 'unknown')}",
            action,
        )

    if action == TodoAction.COMPLETE_TODO.value:
        todo_id = args.get("todo_id")
        if not todo_id:
            return todos, current_task_index, "Error: todo_id required for COMPLETE_TODO", action

        for i, todo in enumerate(todos):
            if str(todo.get("id")) == str(todo_id):
                todo["status"] = TodoStatus.COMPLETED.value
                result = f"Completed todo: {todo.get('description', todo_id)}"
                if current_task_index is not None and i == current_task_index:
                    current_task_index = find_next_ready_task(todos, i)
                return todos, current_task_index, result, action
        return todos, current_task_index, f"Todo with id {todo_id} not found", action

    if action == TodoAction.START_TODO.value:
        todo_id = args.get("todo_id")
        if not todo_id:
            return todos, current_task_index, "Error: todo_id required for START_TODO", action

        for i, todo in enumerate(todos):
            if str(todo.get("id")) == str(todo_id):
                todo["status"] = TodoStatus.IN_PROGRESS.value
                current_task_index = i
                return (
                    todos,
                    current_task_index,
                    f"Started todo: {todo.get('description', todo_id)}",
                    action,
                )
        return todos, current_task_index, f"Todo with id {todo_id} not found", action

    if action == TodoAction.UPDATE_TODO.value:
        updated_todo = _as_dict(args.get("todo"))
        todo_id = updated_todo.get("id")
        if not todo_id:
            return todos, current_task_index, "Error: todo.id required for UPDATE_TODO", action

        for i, todo in enumerate(todos):
            if str(todo.get("id")) == str(todo_id):
                todos[i] = {**todo, **updated_todo}
                return todos, current_task_index, f"Updated todo: {todo_id}", action
        return todos, current_task_index, f"Todo with id {todo_id} not found", action

    if action == TodoAction.REMOVE_TODO.value:
        todo_id = args.get("todo_id")
        if not todo_id:
            return todos, current_task_index, "Error: todo_id required for REMOVE_TODO", action

        for i, todo in enumerate(todos):
            if str(todo.get("id")) == str(todo_id):
                todos.pop(i)
                if current_task_index is not None:
                    if i < current_task_index:
                        current_task_index -= 1
                    elif i == current_task_index:
                        current_task_index = find_next_ready_task(todos, max(0, i - 1))
                return todos, current_task_index, f"Removed todo: {todo_id}", action
        return todos, current_task_index, f"Todo with id {todo_id} not found", action

    return todos, current_task_index, f"Unknown action: {action}", action

