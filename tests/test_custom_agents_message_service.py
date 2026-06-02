"""Message-service / workflow-request tests for custom agents."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import delete

from app.ai.schemas import WorkflowExecutionRequest as AIWorkflowExecutionRequest
from app.core.config import settings
from app.core.exceptions import CustomAgentInUseError
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.user import User
from app.repositories.custom_agent import CustomAgentRepository
from app.schemas.custom_agent import CustomAgentCreate
from app.schemas.workflow import WorkflowExecutionRequest
from app.services.custom_agent_service import CustomAgentService
from app.services.generation_registry import GenerationRegistry
from app.utils.validation.conversation_validation import ConversationValidationUtils


def test_workflow_request_defaults_custom_agents_to_empty():
    """Existing callers that omit custom_agents still validate, defaulting to {}."""
    req = WorkflowExecutionRequest(message="hello")
    assert req.custom_agents == {}

    ai_req = AIWorkflowExecutionRequest(message="hello")
    assert ai_req.custom_agents == {}


class _FakeModelConfig:
    def validate_provider_model(self, user_id, provider_type, model, *, allow_custom_model=False):
        return None


@pytest.fixture
def paused_env():
    db = Database(settings.database_url)
    session_factory = db.session
    owner_id = uuid4()
    conversation_id = uuid4()
    with session_factory() as s:
        s.add(
            User(
                id=owner_id,
                username=f"u_{owner_id.hex[:12]}",
                email=f"{owner_id.hex[:12]}@test.local",
                password_hash="x",
            )
        )
        s.add(Conversation(id=conversation_id, owner_id=owner_id, title="t"))
        s.commit()

    repo = CustomAgentRepository(session_factory)
    service = CustomAgentService(
        repo,
        ConversationValidationUtils(session_factory),
        _FakeModelConfig(),
        generation_registry=GenerationRegistry(),
    )
    agent = service.create_agent(
        owner_id,
        CustomAgentCreate(
            name="Paused One",
            prompt="prompt",
            provider_type="openai",
            model="gpt-4.1-mini",
        ),
    )
    try:
        yield service, owner_id, conversation_id, agent
    finally:
        with session_factory() as s:
            s.execute(
                delete(ConversationCustomAgent).where(ConversationCustomAgent.owner_id == owner_id)
            )
            s.execute(delete(CustomAgent).where(CustomAgent.owner_id == owner_id))
            s.execute(delete(Conversation).where(Conversation.owner_id == owner_id))
            s.execute(delete(User).where(User.id == owner_id))
            s.commit()


class _FakeCustomAgentSvc:
    def __init__(self, state):
        self._state = state

    def build_runtime_state(self, owner_id, conversation_id):
        return dict(self._state)


def _bare_message_service(custom_agent_service):
    from app.services.message_service import MessageService

    svc = MessageService.__new__(MessageService)
    svc.custom_agent_service = custom_agent_service
    return svc


def test_resolve_custom_agents_state_uses_service():
    rid = f"custom_agent:{uuid4()}"
    svc = _bare_message_service(_FakeCustomAgentSvc({rid: {"name": "X"}}))
    out = svc._resolve_custom_agents_state(uuid4(), uuid4())
    assert rid in out
    # No service wired -> empty (conversations without custom agents unchanged).
    assert _bare_message_service(None)._resolve_custom_agents_state(uuid4(), uuid4()) == {}


def test_agent_selected_event_adds_name_for_custom_only():
    from app.services.message_service import MessageService

    rid = f"custom_agent:{uuid4()}"
    custom = MessageService._agent_selected_event(rid, {rid: {"name": "Analyst"}})
    assert custom["agent"] == rid
    assert custom["agent_name"] == "Analyst"
    # Base agents are emitted unchanged (no agent_name key).
    base = MessageService._agent_selected_event("chat_agent", {rid: {"name": "Analyst"}})
    assert base == {"type": "agent_selected", "agent": "chat_agent"}


def test_resume_revalidation_conflicts_when_custom_agent_detached():
    from app.core.exceptions import CustomHTTPException
    from app.services.generation_registry import get_generation_registry

    rid = f"custom_agent:{uuid4()}"
    owner = uuid4()
    conv = uuid4()
    registry = get_generation_registry()
    msg_id = uuid4()
    registry.register(msg_id, conv, owner, selected_agent=rid)
    registry.mark_paused(msg_id)
    try:
        # Agent no longer attached -> resume conflict (409).
        detached = _bare_message_service(_FakeCustomAgentSvc({}))
        with pytest.raises(CustomHTTPException) as exc_info:
            detached._revalidate_resume_custom_agent(owner, conv)
        assert exc_info.value.status_code == 409

        # Still attached -> no conflict.
        attached = _bare_message_service(_FakeCustomAgentSvc({rid: {"name": "X"}}))
        attached._revalidate_resume_custom_agent(owner, conv)
    finally:
        registry._store.pop(str(msg_id), None)


def test_resume_lock_blocks_delete_for_paused_custom_agent(paused_env):
    """A paused HITL run keeps the custom agent locked against delete/edit."""
    service, owner_id, conversation_id, agent = paused_env
    runtime_id = f"custom_agent:{agent.id}"

    # Simulate a streaming turn that selected this custom agent and then paused
    # on a HITL interrupt (mark_paused keeps the entry as a lock token).
    msg_id = uuid4()
    service.generation_registry.register(
        msg_id, conversation_id, owner_id, selected_agent=runtime_id
    )
    service.generation_registry.mark_paused(msg_id)

    with pytest.raises(CustomAgentInUseError):
        service.delete_agent(owner_id, agent.id)

    # Resume resolves the paused run -> lock released -> delete now succeeds.
    service.generation_registry.clear_paused_for_conversation(owner_id, conversation_id)
    service.delete_agent(owner_id, agent.id)


def test_agent_selected_event_still_adds_name_for_custom_only():
    from app.services.message_service import MessageService

    rid = f"custom_agent:{uuid4()}"
    custom = MessageService._agent_selected_event(rid, {rid: {"name": "Analyst"}})

    assert custom == {
        "type": "agent_selected",
        "agent": rid,
        "agent_name": "Analyst",
    }
    assert MessageService._agent_selected_event("chat_agent", {rid: {"name": "Analyst"}}) == {
        "type": "agent_selected",
        "agent": "chat_agent",
    }


def test_attach_selected_agent_metadata_adds_canonical_custom_agent():
    from app.services.message_service import MessageService

    rid = f"custom_agent:{uuid4()}"
    metadata = {}

    MessageService._attach_selected_agent_metadata(
        metadata,
        rid,
        {rid: {"id": rid.split(":", 1)[1], "runtime_agent_id": rid, "name": "Analyst"}},
    )

    assert metadata["agent"] == {
        "id": rid,
        "kind": "custom",
        "name": "Analyst",
        "custom_agent_id": rid.split(":", 1)[1],
        "source": "response",
    }


def test_attach_selected_agent_metadata_noops_without_selected_agent():
    from app.services.message_service import MessageService

    metadata = {"stopped": True}

    MessageService._attach_selected_agent_metadata(metadata, None, {})

    assert metadata == {"stopped": True}
