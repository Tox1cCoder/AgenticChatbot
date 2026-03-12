"""
Service-level tests for MessageService.

Tests cover:
- _prepare_planning_context includes plan_lifecycle in its result dict
- _run_plan_execution_loop passes plan_lifecycle to generate_bot_response
"""

import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest


def _make_agent_response(content="ok"):
    """Return a minimal AgentResponse-like mock."""
    resp = MagicMock()
    resp.message = MagicMock()
    resp.message.content = content
    resp.metadata = {}  # no interrupt, no todos
    return resp


# ---------------------------------------------------------------------------
# _prepare_planning_context
# ---------------------------------------------------------------------------


class TestPreparePlanningContext:
    """test _prepare_planning_context (async method)."""

    @pytest.mark.asyncio
    async def test_plan_lifecycle_included(self):
        from app.services.message_service import MessageService

        svc = object.__new__(MessageService)
        # task_plan_service=None causes the method to return base result early,
        # which still includes plan_lifecycle from the parameter.
        svc.task_plan_service = None

        ctx = await svc._prepare_planning_context(
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            message_content="hello",
            planning_mode_enabled=True,
            plan_lifecycle="executing",
        )
        assert ctx["plan_lifecycle"] == "executing"

    @pytest.mark.asyncio
    async def test_plan_lifecycle_none_when_omitted(self):
        from app.services.message_service import MessageService

        svc = object.__new__(MessageService)
        svc.task_plan_service = None

        ctx = await svc._prepare_planning_context(
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            message_content="hello",
            planning_mode_enabled=False,
        )
        # plan_lifecycle defaults to None when not passed
        assert ctx.get("plan_lifecycle") is None


# ---------------------------------------------------------------------------
# _run_plan_execution_loop passes plan_lifecycle
# ---------------------------------------------------------------------------


class TestRunPlanExecutionLoop:
    @pytest.mark.asyncio
    async def test_plan_lifecycle_forwarded_to_ai(self):
        from app.services.message_service import MessageService

        svc = object.__new__(MessageService)

        mock_ai = MagicMock()
        mock_ai.generate_bot_response = AsyncMock(return_value=_make_agent_response())
        svc.ai_service = mock_ai
        svc.task_plan_service = None
        svc._sync_todos_to_database = MagicMock()

        await svc._run_plan_execution_loop(
            message_content="do the work",
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            sanitized_persona="assistant",
            planning_mode_enabled=True,
            has_existing_plan=True,
            current_task_context=None,
            existing_tasks_dict=[],
            attachments=[],
            model_request=None,
            persona="assistant",
            plan_lifecycle="executing",
        )

        call_kwargs = mock_ai.generate_bot_response.call_args
        assert call_kwargs is not None
        all_kwargs = dict(call_kwargs.kwargs)
        assert all_kwargs.get("plan_lifecycle") == "executing", (
            f"plan_lifecycle='executing' not found in kwargs: {all_kwargs}"
        )

    @pytest.mark.asyncio
    async def test_interrupt_transitions_lifecycle_to_paused(self):
        from app.models.enums import PlanLifecycle
        from app.services.message_service import MessageService

        svc = object.__new__(MessageService)

        interrupt_payload = {"interrupt_id": "abc123"}
        response = _make_agent_response()
        response.metadata = {"interrupt": interrupt_payload}

        mock_ai = MagicMock()
        mock_ai.generate_bot_response = AsyncMock(return_value=response)
        svc.ai_service = mock_ai
        svc.task_plan_service = MagicMock()
        svc._set_plan_lifecycle = MagicMock()

        _, _, interrupt = await svc._run_plan_execution_loop(
            message_content="continue",
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            sanitized_persona="assistant",
            planning_mode_enabled=True,
            has_existing_plan=True,
            current_task_context=None,
            existing_tasks_dict=[],
            attachments=[],
            model_request=None,
            persona="assistant",
            plan_lifecycle="executing",
        )

        assert interrupt == interrupt_payload
        svc._set_plan_lifecycle.assert_called_once()
        assert svc._set_plan_lifecycle.call_args.args[2] == PlanLifecycle.paused


class TestPlanLifecycleSync:
    def test_sync_response_plan_state_derives_executing_from_todos(self):
        from app.models.enums import PlanLifecycle
        from app.services.message_service import MessageService

        svc = object.__new__(MessageService)
        svc.task_plan_service = MagicMock()
        svc._sync_todos_to_database = MagicMock()
        svc._set_plan_lifecycle = MagicMock()

        response = _make_agent_response()
        response.metadata = {
            "todos": [
                {
                    "id": "1",
                    "description": "Implement the feature end to end",
                    "status": "in_progress",
                    "order": 0,
                }
            ]
        }

        synced = svc._sync_response_plan_state(
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            bot_response=response,
            current_lifecycle="draft",
        )

        assert synced is True
        svc._sync_todos_to_database.assert_called_once()
        assert (
            svc._sync_todos_to_database.call_args.kwargs["lifecycle"]
            == PlanLifecycle.executing
        )
        svc._set_plan_lifecycle.assert_not_called()
