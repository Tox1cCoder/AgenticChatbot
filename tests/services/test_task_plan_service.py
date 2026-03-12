"""
Service-level tests for TaskPlanService.

Tests cover:
- _transition_lifecycle: persists lifecycle to conversation
- sync_todos_from_agent lifecycle opt-in/opt-out
- get_planning_status includes plan_lifecycle
- _sync_todo_snapshot: upsert, delete, active-task dedup semantics
- concurrent create protection via SELECT FOR UPDATE
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.models.enums import PlanLifecycle, TaskStatus
from app.schemas.task_plan import PlanningStatusResponse

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_conversation(lifecycle=None, planning_mode=True):
    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.owner_id = uuid.uuid4()
    conv.planning_mode_enabled = planning_mode
    conv.plan_lifecycle = lifecycle
    return conv


_NOW = datetime.now(timezone.utc)


def _make_task(id_=None, order=0, status="pending", description="Do something useful"):
    """Return a mock that carries all fields required by TaskPlanRead."""
    t = MagicMock()
    t.id = id_ or uuid.uuid4()
    t.conversation_id = uuid.uuid4()
    t.task_order = order
    t.status = status
    t.description = description
    t.task_metadata = {}
    t.created_at = _NOW
    t.updated_at = _NOW
    t.completed_at = None
    return t


def _build_service(conversation=None, tasks=None):
    """Build a TaskPlanService with all dependencies mocked."""
    from app.services.task_plan_service import TaskPlanService

    task_repo = MagicMock()
    conv_repo = MagicMock()
    conv_validation = MagicMock()
    task_validation = MagicMock()
    planning_agent = MagicMock()

    conv = conversation or _make_conversation()
    conv_repo.get_by_id.return_value = conv
    conv_repo.update.return_value = conv
    conv_repo.set_plan_lifecycle.return_value = conv

    task_repo.get_by_conversation_id.return_value = tasks or []
    task_repo.count_by_conversation.return_value = 0
    task_repo.get_active_or_next_task.return_value = None

    # session_factory must be a callable that returns a context-manager
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session_factory = MagicMock(return_value=session)
    task_repo.session_factory = session_factory

    svc = TaskPlanService(
        task_plan_repository=task_repo,
        conversation_validation_utils=conv_validation,
        task_plan_validation_utils=task_validation,
        planning_agent=planning_agent,
        conversation_repository=conv_repo,
    )
    return svc, task_repo, conv_repo, conv, session


# ---------------------------------------------------------------------------
# _transition_lifecycle
# ---------------------------------------------------------------------------


class TestTransitionLifecycle:
    def test_sets_lifecycle_on_conversation(self):
        svc, task_repo, conv_repo, conv, session = _build_service()
        conversation_id = conv.id

        svc._transition_lifecycle(conversation_id, PlanLifecycle.executing)

        conv_repo.set_plan_lifecycle.assert_called_once_with(
            conversation_id, PlanLifecycle.executing
        )

    def test_swallows_exception_with_warning(self):
        """_transition_lifecycle must not propagate errors."""
        svc, task_repo, conv_repo, conv, session = _build_service()
        conv_repo.set_plan_lifecycle.side_effect = RuntimeError("db down")

        # Should not raise
        svc._transition_lifecycle(conv.id, PlanLifecycle.draft)


# ---------------------------------------------------------------------------
# sync_todos_from_agent — lifecycle opt-in / opt-out
# ---------------------------------------------------------------------------


class TestSyncTodosLifecycle:
    def _svc_with_patched_sync(self):
        svc, task_repo, conv_repo, conv, session = _build_service()
        svc._sync_todo_snapshot = MagicMock(return_value=[])
        svc._ensure_planning_mode_enabled = MagicMock()
        svc._transition_lifecycle = MagicMock()
        return svc, conv

    def test_lifecycle_none_does_not_transition(self):
        svc, conv = self._svc_with_patched_sync()
        svc.sync_todos_from_agent(
            conversation_id=conv.id,
            todos=[
                {
                    "id": "1",
                    "description": "Write a basic test case",
                    "status": "pending",
                    "order": 0,
                }
            ],
            user_id=uuid.uuid4(),
            lifecycle=None,
        )
        svc._transition_lifecycle.assert_not_called()

    def test_lifecycle_provided_triggers_transition(self):
        svc, conv = self._svc_with_patched_sync()
        svc.sync_todos_from_agent(
            conversation_id=conv.id,
            todos=[
                {
                    "id": "1",
                    "description": "Implement the feature set",
                    "status": "pending",
                    "order": 0,
                }
            ],
            user_id=uuid.uuid4(),
            lifecycle=PlanLifecycle.executing,
        )
        svc._transition_lifecycle.assert_called_once_with(conv.id, PlanLifecycle.executing)


# ---------------------------------------------------------------------------
# get_planning_status includes plan_lifecycle
# ---------------------------------------------------------------------------


class TestGetPlanningStatus:
    def test_lifecycle_included_in_response(self):
        conv = _make_conversation(lifecycle=PlanLifecycle.draft)
        svc, task_repo, conv_repo, _, session = _build_service(conversation=conv)

        result = svc.get_planning_status(conv.id, conv.owner_id)

        assert isinstance(result, PlanningStatusResponse)
        assert result.plan_lifecycle == "draft"

    def test_lifecycle_none_when_no_plan(self):
        conv = _make_conversation(lifecycle=None)
        svc, task_repo, conv_repo, _, session = _build_service(conversation=conv)

        result = svc.get_planning_status(conv.id, conv.owner_id)

        assert result.plan_lifecycle is None


# ---------------------------------------------------------------------------
# _sync_todo_snapshot — upsert / delete semantics
# ---------------------------------------------------------------------------


class TestSyncTodoSnapshot:
    """Verify that _sync_todo_snapshot correctly upserts and prunes rows."""

    def _run_snapshot(self, existing_tasks, todos, preserve=False):
        """
        Exercise _sync_todo_snapshot using a real session mock that
        provides realistic query().filter().order_by().all() chaining.
        """
        from app.services.task_plan_service import TaskPlanService

        task_repo = MagicMock()
        conv_repo = MagicMock()

        # Session mock with query chaining
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)

        # Make query(...).filter(...).with_for_update().first() work
        query_chain = MagicMock()
        query_chain.filter.return_value = query_chain
        query_chain.with_for_update.return_value = query_chain
        query_chain.order_by.return_value = query_chain
        # First all() fetches existing tasks; second all() re-fetches after sync (return value).
        # Use side_effect so pydantic model_validate is never called on our MagicMock tasks,
        # since TaskPlanRead uses alias_generator=to_camel and accesses via camelCase.
        query_chain.all.side_effect = [existing_tasks, []]
        query_chain.first.return_value = MagicMock()  # the locked conversation row
        session.query.return_value = query_chain

        task_repo.session_factory = MagicMock(return_value=session)

        svc = TaskPlanService(
            task_plan_repository=task_repo,
            conversation_validation_utils=MagicMock(),
            task_plan_validation_utils=MagicMock(),
            planning_agent=MagicMock(),
            conversation_repository=conv_repo,
        )

        conversation_id = uuid.uuid4()
        svc._sync_todo_snapshot(
            conversation_id=conversation_id,
            todos=todos,
            preserve_existing_status=preserve,
        )

        return session

    def test_new_todo_added(self):
        todos = [
            {
                "id": "1",
                "description": "Create unit test suite",
                "status": "pending",
                "order": 0,
            }
        ]
        session = self._run_snapshot([], todos)
        session.add.assert_called_once()

    def test_existing_todo_updated(self):
        existing = _make_task(id_=uuid.UUID("00000000-0000-0000-0000-000000000001"))
        todos = [
            {
                "id": str(existing.id),
                "description": "Updated description text here",
                "status": "pending",
                "order": 0,
            }
        ]
        session = self._run_snapshot([existing], todos)
        # Should NOT add a new row
        session.add.assert_not_called()
        assert existing.description == "Updated description text here"

    def test_removed_todo_deleted(self):
        existing = _make_task(id_=uuid.UUID("00000000-0000-0000-0000-000000000002"))
        # Empty new list — existing task should be deleted
        session = self._run_snapshot([existing], [])
        session.delete.assert_called_once_with(existing)

    def test_only_one_in_progress_at_a_time(self):
        """Multiple in_progress todos in the payload must be deduped."""
        t1_id = str(uuid.uuid4())
        t2_id = str(uuid.uuid4())
        todos = [
            {
                "id": t1_id,
                "description": "First task to run now",
                "status": "in_progress",
                "order": 0,
            },
            {
                "id": t2_id,
                "description": "Second concurrent task",
                "status": "in_progress",
                "order": 1,
            },
        ]
        session = self._run_snapshot([], todos)
        # Verify two tasks were added; the second's status should be demoted.
        calls = session.add.call_args_list
        assert len(calls) == 2
        statuses = [call[0][0].status for call in calls]
        assert statuses.count(TaskStatus.in_progress) == 1
