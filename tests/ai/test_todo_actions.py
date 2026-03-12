"""Unit tests for app.ai.todo_actions.apply_write_todos_action."""

import uuid

from app.ai.todo_actions import apply_write_todos_action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_todo(
    description: str = "Task", status: str = "pending", order: int = 0
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "description": description,
        "status": status,
        "order": order,
    }


def _todo_ids(todos):
    return [t["id"] for t in todos]


# ---------------------------------------------------------------------------
# SET_TODOS
# ---------------------------------------------------------------------------


class TestSetTodos:
    def test_set_todos_replaces_list(self):
        original = [_make_todo("old")]
        new = [_make_todo("new-1"), _make_todo("new-2")]
        todos, idx, msg, action = apply_write_todos_action(
            todos=original,
            current_task_index=0,
            tool_args={"action": "set_todos", "todos": new},
        )
        assert len(todos) == 2
        assert todos[0]["description"] == "new-1"
        assert action == "set_todos"

    def test_set_todos_exceeds_max_returns_error(self):
        big_list = [_make_todo(f"t{i}") for i in range(5)]
        todos, idx, msg, action = apply_write_todos_action(
            todos=[],
            current_task_index=None,
            tool_args={"action": "set_todos", "todos": big_list},
            max_todos=3,
        )
        assert "Error" in msg
        # Original list unchanged
        assert todos == []

    def test_set_todos_non_list_payload_returns_error(self):
        todos, idx, msg, action = apply_write_todos_action(
            todos=[],
            current_task_index=None,
            tool_args={"action": "set_todos", "todos": "not-a-list"},
        )
        assert "Error" in msg

    def test_set_todos_tracks_first_pending_as_current(self):
        items = [
            {"id": "a", "description": "done", "status": "completed", "order": 0},
            {"id": "b", "description": "pending", "status": "pending", "order": 1},
        ]
        todos, idx, msg, action = apply_write_todos_action(
            todos=[],
            current_task_index=None,
            tool_args={"action": "set_todos", "todos": items},
        )
        assert idx == 1


# ---------------------------------------------------------------------------
# ADD_TODO
# ---------------------------------------------------------------------------


class TestAddTodo:
    def test_add_todo_appends(self):
        existing = [_make_todo("existing")]
        new_todo = {
            "id": str(uuid.uuid4()),
            "description": "appended",
            "status": "pending",
            "order": 1,
        }
        todos, idx, msg, action = apply_write_todos_action(
            todos=existing,
            current_task_index=0,
            tool_args={"action": "add_todo", "todo": new_todo},
        )
        assert len(todos) == 2
        assert todos[-1]["description"] == "appended"

    def test_add_todo_missing_payload_returns_error(self):
        todos, idx, msg, action = apply_write_todos_action(
            todos=[],
            current_task_index=None,
            tool_args={"action": "add_todo"},
        )
        assert "Error" in msg

    def test_add_todo_respects_max_todos(self):
        existing = [_make_todo(f"t{i}") for i in range(3)]
        new_todo = {
            "id": str(uuid.uuid4()),
            "description": "overflow",
            "status": "pending",
            "order": 3,
        }
        todos, idx, msg, action = apply_write_todos_action(
            todos=existing,
            current_task_index=None,
            tool_args={"action": "add_todo", "todo": new_todo},
            max_todos=3,
        )
        assert len(todos) == 3


# ---------------------------------------------------------------------------
# COMPLETE_TODO
# ---------------------------------------------------------------------------


class TestCompleteTodo:
    def test_complete_marks_done(self):
        t1 = _make_todo("step 1", "in_progress", 0)
        t2 = _make_todo("step 2", "pending", 1)
        todos, idx, msg, action = apply_write_todos_action(
            todos=[t1, t2],
            current_task_index=0,
            tool_args={"action": "complete_todo", "todo_id": t1["id"]},
        )
        assert todos[0]["status"] == "completed"
        assert idx == 1  # advances to next pending

    def test_complete_unknown_id_leaves_list_unchanged(self):
        t = _make_todo("step 1")
        todos, idx, msg, action = apply_write_todos_action(
            todos=[t],
            current_task_index=0,
            tool_args={"action": "complete_todo", "todo_id": "nonexistent"},
        )
        assert todos[0]["status"] == "pending"
        assert "not found" in msg.lower()

    def test_complete_requires_todo_id(self):
        todos, idx, msg, action = apply_write_todos_action(
            todos=[_make_todo()],
            current_task_index=0,
            tool_args={"action": "complete_todo"},
        )
        assert "Error" in msg


# ---------------------------------------------------------------------------
# START_TODO
# ---------------------------------------------------------------------------


class TestStartTodo:
    def test_start_sets_in_progress(self):
        t = _make_todo("do it", "pending", 0)
        todos, idx, msg, action = apply_write_todos_action(
            todos=[t],
            current_task_index=None,
            tool_args={"action": "start_todo", "todo_id": t["id"]},
        )
        assert todos[0]["status"] == "in_progress"
        assert idx == 0

    def test_start_resets_previous_in_progress(self):
        t1 = _make_todo("step 1", "in_progress", 0)
        t2 = _make_todo("step 2", "pending", 1)
        todos, idx, msg, action = apply_write_todos_action(
            todos=[t1, t2],
            current_task_index=0,
            tool_args={"action": "start_todo", "todo_id": t2["id"]},
        )
        assert todos[0]["status"] == "pending"  # reset
        assert todos[1]["status"] == "in_progress"
        assert idx == 1


# ---------------------------------------------------------------------------
# UPDATE_TODO
# ---------------------------------------------------------------------------


class TestUpdateTodo:
    def test_update_description(self):
        t = _make_todo("original", "pending", 0)
        todos, idx, msg, action = apply_write_todos_action(
            todos=[t],
            current_task_index=0,
            tool_args={
                "action": "update_todo",
                "todo": {"id": t["id"], "description": "updated"},
            },
        )
        assert todos[0]["description"] == "updated"

    def test_update_missing_todo_id_returns_error(self):
        todos, idx, msg, action = apply_write_todos_action(
            todos=[_make_todo()],
            current_task_index=0,
            tool_args={"action": "update_todo", "todo": {"description": "no id"}},
        )
        assert "Error" in msg


# ---------------------------------------------------------------------------
# REMOVE_TODO
# ---------------------------------------------------------------------------


class TestRemoveTodo:
    def test_remove_by_id(self):
        t1 = _make_todo("step 1", "pending", 0)
        t2 = _make_todo("step 2", "pending", 1)
        todos, idx, msg, action = apply_write_todos_action(
            todos=[t1, t2],
            current_task_index=1,
            tool_args={"action": "remove_todo", "todo_id": t1["id"]},
        )
        assert len(todos) == 1
        assert todos[0]["description"] == "step 2"
        # current_task_index decremented
        assert idx == 0

    def test_remove_unknown_id_returns_not_found(self):
        t = _make_todo()
        todos, idx, msg, action = apply_write_todos_action(
            todos=[t],
            current_task_index=0,
            tool_args={"action": "remove_todo", "todo_id": "nope"},
        )
        assert len(todos) == 1
        assert "not found" in msg.lower()


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------


def test_unknown_action_returns_error():
    todos, idx, msg, action = apply_write_todos_action(
        todos=[],
        current_task_index=None,
        tool_args={"action": "fly_away"},
    )
    assert "Unknown action" in msg
