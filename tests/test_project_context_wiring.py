"""Coverage for the `project_context_service is not None` branch at all
three call sites Task 5 touched.

`tests/test_project_context_service.py` proves the resolver itself is
correct in isolation. It does not prove anything is wired to it: nothing
there constructs `MessageService` or `AIService` with a real
`project_context_service`, so the wired branches at
`MessageService._get_conversation_context`,
`MessageService._build_user_message_workflow_request`, and
`AIService._prepare_request` had no coverage — deleting the wiring, or
re-adding the old `sanitize_persona(persona)` re-truncation this task
removed, would break nothing. These tests close that gap with a stub
resolver, asserting a long (16060-character) composed instruction survives
each site untouched.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.models.enums import MessageRole
from app.schemas.message import MessageRead
from app.schemas.workflow import (
    WorkflowExecutionRequest,
    WorkflowResponse,
    WorkflowResponseMessage,
)
from app.services.ai_service import AIService
from app.services.event_streaming.events import make_event
from app.services.message_service import MessageService

LONG_INSTRUCTION = "Z" * 16060


def _message_row(*, conversation_id, sender: int, content: str) -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=uuid4(),
        conversation_id=conversation_id,
        sender=sender,
        content=content,
        message_metadata={},
        feedback=None,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


class _StubResolver:
    """Minimal stand-in for ProjectContextService: ignores the conversation
    it is given and always returns the same fixed instruction, so these
    tests only need to prove the value flows through unmodified.

    Both twins are provided: the streaming turn awaits the async one so the
    lookups do not block the event loop."""

    def __init__(self, value: str) -> None:
        self.value = value

    def resolve_system_instruction(self, conversation):
        return self.value

    async def aresolve_system_instruction(self, conversation):
        return self.value


def test_get_conversation_context_returns_the_wired_resolvers_full_string():
    """Site 1: message_service.py's resume path assigns this return value
    directly to `sanitized_persona` with no further sanitization. If the
    wired branch silently truncated (or if the old sanitize_persona(persona)
    re-truncation were reintroduced), this would come back at 8000 chars."""
    conversation = SimpleNamespace(persona_prompt="ignored", project_id=uuid4())
    service = MessageService.__new__(MessageService)
    service.conversation_validation_utils = SimpleNamespace(
        conversation_repository=SimpleNamespace(get_by_id=lambda _id: conversation)
    )
    service.project_context_service = _StubResolver(LONG_INSTRUCTION)

    user_id, persona = service._get_conversation_context(uuid4(), user_id=uuid4())

    assert persona == LONG_INSTRUCTION
    assert len(persona) == 16060


@pytest.mark.asyncio
async def test_build_user_message_workflow_request_uses_the_wired_resolver():
    """Site 2: the outgoing WorkflowExecutionRequest's persona field must
    carry the resolver's full string, not a sanitize_persona-truncated one."""
    service = MessageService(
        message_repository=SimpleNamespace(),
        conversation_validation_utils=SimpleNamespace(),
        message_validation_utils=SimpleNamespace(),
        ai_service=SimpleNamespace(),
        project_context_service=_StubResolver(LONG_INSTRUCTION),
    )
    message_create_data = SimpleNamespace(
        conversation_id=uuid4(),
        content="hello",
        device_id=None,
        attachments=None,
        model_config_field=None,
        inline_rich_response_v1=False,
    )

    _, sanitized_persona, request = await service._build_user_message_workflow_request(
        message_create_data=message_create_data,
        user_id=uuid4(),
        conversation=None,
    )

    assert sanitized_persona == LONG_INSTRUCTION
    assert request.persona == LONG_INSTRUCTION
    assert len(request.persona) == 16060


def test_ai_service_prepare_request_uses_the_wired_resolver():
    """Site 3: AIService._prepare_request must hand the resolver's full
    string to the workflow request's persona field, not a sanitized one."""
    service = AIService.__new__(AIService)
    service.conversation_repository = SimpleNamespace(
        get_by_id=lambda _id: SimpleNamespace(persona_prompt="ignored")
    )
    service.project_context_service = _StubResolver(LONG_INSTRUCTION)
    request = WorkflowExecutionRequest(message="hi", conversation_id=str(uuid4()))

    result = service._prepare_request(request)

    assert result.persona == LONG_INSTRUCTION
    assert len(result.persona) == 16060


@pytest.mark.asyncio
async def test_resume_message_creation_stream_persists_the_wired_instruction():
    """Regression pin for the historical bug directly: runs the real
    ``resume_message_creation_stream`` — the method containing the collapsed
    call site at message_service.py ~line 2530 — end to end, without
    monkeypatching ``_get_conversation_context`` away as the other resume
    tests in test_message_service_event_streaming.py do. If line 2530 ever
    grows a re-sanitize call again (``sanitized_persona =
    sanitize_persona(sanitized_persona)``), this test catches it because the
    persisted persona would come back truncated to 8000 characters instead
    of the full 16060.
    """
    conversation_id = uuid4()
    user_id = uuid4()
    bot_message_id = uuid4()
    persisted: list[dict] = []

    conversation = SimpleNamespace(persona_prompt="ignored", project_id=uuid4())
    service = MessageService.__new__(MessageService)
    service.hitl_interrupt_repository = None
    service.conversation_validation_utils = SimpleNamespace(
        conversation_repository=SimpleNamespace(get_by_id=lambda _id: conversation)
    )
    service.project_context_service = _StubResolver(LONG_INSTRUCTION)
    service._validate_and_claim_interrupt_resume = lambda **_kwargs: None
    service._revalidate_resume_custom_agent = lambda *_args: None
    service._audit_interrupt_resume_decisions = lambda **_kwargs: None
    service._resolve_custom_agents_state = lambda *_args: {}
    service._clear_redis_interrupt = lambda *_args: None
    service._sync_response_plan_state = lambda **_kwargs: False

    async def resume_source(**_kwargs):
        yield make_event(
            "complete",
            sequence=1,
            data={
                "response": WorkflowResponse(
                    message=WorkflowResponseMessage(content="resumed"),
                    metadata={},
                )
            },
        )

    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_args: None,
        resume_interrupted_execution_stream=resume_source,
    )
    service._persist_completed_workflow_response = AsyncMock(
        side_effect=lambda **kwargs: (
            persisted.append(kwargs)
            or MessageRead.model_validate(
                _message_row(
                    conversation_id=conversation_id,
                    sender=MessageRole.assistant.value,
                    content="resumed",
                )
            )
        )
    )
    service._compact_checkpoint_after_persist = AsyncMock()

    events = [
        event
        async for event in service.resume_message_creation_stream(
            thread_id=str(conversation_id),
            conversation_id=conversation_id,
            user_id=user_id,
            decisions=[],
            bot_message_id=bot_message_id,
        )
    ]

    assert events[-1].type == "complete"
    assert len(persisted) == 1
    assert persisted[0]["sanitized_persona"] == LONG_INSTRUCTION
    assert len(persisted[0]["sanitized_persona"]) == 16060


# ---------------------------------------------------------------------------
# Container-wiring regression guards (Task 7 fix round, finding B1).
#
# None of the tests above construct MessageService, ConversationService, or
# AIService through the actual container — they build doubles with __new__
# or pass a stub resolver directly. That proves the *branches* inside those
# classes are correct, but nothing proves the container's Factory providers
# still pass ``project_context_service``/``project_service`` at all. Deleting
# any one of those three keyword arguments from app/core/container.py leaves
# every test above green (their doubles never touch the container) and
# leaves tests/test_projects_api.py green (it builds a bare FastAPI() and
# never constructs MessageService or AIService either) — production would
# silently fall back to persona-only prompts while every other project
# feature (CRUD, attach/detach, agent seeding) kept working normally.
#
# These follow the same source-introspection pattern as
# test_the_container_wires_a_durable_turn_coordinator_into_message_service
# and test_the_container_wires_the_generation_control_service in
# tests/test_workflow_concurrency.py: read the container's own source and
# assert the wiring keyword is actually present in the right provider block.
# ---------------------------------------------------------------------------


def test_the_streaming_turn_awaits_the_async_resolver():
    """The sync resolver takes a sync engine checkout. Calling it from
    ``_build_user_message_workflow_request`` blocks the event loop before the
    first token for every project or memory conversation, which is what
    tests/test_preflight_has_no_blocking_db.py forbids."""
    import inspect

    source = inspect.getsource(MessageService._build_user_message_workflow_request)

    assert "await self.project_context_service.aresolve_system_instruction" in source
    assert "self.project_context_service.resolve_system_instruction" not in source


def test_the_container_wires_project_context_service_into_message_service():
    """Without this, every turn on a project conversation silently reverts
    to persona-only prompts: project instructions stop reaching the model,
    while project CRUD and conversation attach/detach keep working fine."""
    import inspect

    from app.core import container as container_module

    assert "project_context_service" in inspect.signature(MessageService.__init__).parameters

    source = inspect.getsource(container_module)
    message_service_block = source.split("message_service: providers.Provider")[1].split(
        "feedback_service"
    )[0]
    assert "project_context_service=project_context_service" in message_service_block


def test_the_container_wires_project_service_into_conversation_service():
    """Without this, ConversationService.create_conversation can never
    validate a caller-supplied projectId or seed the project's default
    agents onto a newly created conversation — project_id is stored but
    silently inert, and no test elsewhere in the suite would notice."""
    import inspect

    from app.core import container as container_module
    from app.services.conversation_service import ConversationService

    assert "project_service" in inspect.signature(ConversationService.__init__).parameters

    source = inspect.getsource(container_module)
    conversation_service_block = source.split("conversation_service: providers.Provider")[
        1
    ].split("custom_agent_service = providers.Factory")[0]
    assert "project_service=project_service" in conversation_service_block


def test_the_container_wires_project_context_service_into_ai_service():
    """Without this, AIService._prepare_request falls back to the
    conversation's raw persona_prompt: project instructions never reach the
    model on this path even though the HTTP surface for projects works."""
    import inspect

    from app.core import container as container_module

    assert "project_context_service" in inspect.signature(AIService.__init__).parameters

    source = inspect.getsource(container_module)
    create_ai_service_block = source.split("def _create_ai_service")[1].split(
        "ai_service = providers.ThreadSafeSingleton"
    )[0]
    assert (
        "project_context_service=container.project_context_service()" in create_ai_service_block
    )
