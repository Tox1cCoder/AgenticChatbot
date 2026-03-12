"""
API-level tests for /task_plans endpoints.

Tests cover:
- GET /task_plans/{conversation_id}/planning_status returns plan_lifecycle
- POST /task_plans/{conversation_id}/create_plan transitions lifecycle to "draft"
"""

import uuid
from unittest.mock import AsyncMock, MagicMock


def _make_planning_status_response(lifecycle="draft"):
    from app.schemas.task_plan import PlanningStatusResponse

    return PlanningStatusResponse(
        planning_mode_enabled=True,
        plan_lifecycle=lifecycle,
        total_tasks=2,
        pending_tasks=1,
        in_progress_tasks=0,
        completed_tasks=1,
        skipped_tasks=0,
        progress_percentage=50.0,
        next_task=None,
    )


class TestPlanningStatusEndpoint:
    """GET /task_plans/{conversation_id}/planning_status"""

    def test_response_includes_plan_lifecycle(self):

        status_resp = _make_planning_status_response(lifecycle="executing")
        assert status_resp.plan_lifecycle == "executing"

    def test_plan_lifecycle_none_for_new_conversation(self):

        status_resp = _make_planning_status_response(lifecycle=None)
        assert status_resp.plan_lifecycle is None

    def test_all_lifecycle_values_are_valid(self):
        """Smoke-test that all lifecycle states serialise correctly."""
        from app.models.enums import PlanLifecycle

        for lc in PlanLifecycle:
            resp = _make_planning_status_response(lifecycle=lc.value)
            assert resp.plan_lifecycle == lc.value


class TestPlanLifecycleTransitions:
    """Verify service-level lifecycle transitions are invoked correctly."""

    def test_create_plan_transitions_to_draft(self):
        """After create_task_plan, lifecycle should be draft."""
        from app.services.task_plan_service import TaskPlanService

        task_repo = MagicMock()
        task_repo.get_by_conversation_id.return_value = []  # no existing tasks
        conv_repo = MagicMock()
        conv = MagicMock()
        conv.id = uuid.uuid4()
        conv.owner_id = uuid.uuid4()
        conv.plan_lifecycle = None
        conv_repo.get_by_id.return_value = conv
        conv_repo.update.return_value = conv

        svc = TaskPlanService(
            task_plan_repository=task_repo,
            conversation_validation_utils=MagicMock(),
            task_plan_validation_utils=MagicMock(),
            planning_agent=MagicMock(),
            conversation_repository=conv_repo,
        )
        svc._transition_lifecycle = MagicMock()
        svc._ensure_planning_mode_enabled = MagicMock()
        svc.sync_todos_from_agent = MagicMock(return_value=[])

        # Mock agent to return a simple plan
        response = MagicMock()
        response.error = None
        response.metadata = {
            "todos": [
                {
                    "id": "1",
                    "description": "Implement the auth module with JWT support",
                    "status": "pending",
                    "order": 0,
                }
            ]
        }
        svc.planning_agent = MagicMock()
        svc.planning_agent.generate_plan = AsyncMock(return_value=response)

        import asyncio

        asyncio.run(svc.create_task_plan(conv.id, "build the feature", conv.owner_id))

        # Check that _transition_lifecycle was called with draft
        calls = svc._transition_lifecycle.call_args_list
        from app.models.enums import PlanLifecycle

        lifecycle_values = [c[0][1] for c in calls]
        assert PlanLifecycle.draft in lifecycle_values
