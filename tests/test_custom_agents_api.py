"""HTTP-level tests for the custom-agent API (canonical + /ai alias)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from dependency_injector import providers
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.api.custom_agents import conversation_router, router
from app.core.auth import get_current_user_id
from app.core.config import settings
from app.core.container import Container
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.user import User
from app.services.generation_registry import get_generation_registry


class _FakeProviderService:
    def get_cached_provider_models(self, user_id, provider_type):
        return []


class _FakeModelConfig:
    provider_service = _FakeProviderService()

    def validate_provider_model(
        self,
        user_id,
        provider_type,
        model,
        *,
        allow_custom_model=False,
        reasoning_effort=None,
    ):
        if (provider_type, model) != ("openai", "gpt-4.1-mini"):
            raise ValueError(f"invalid {provider_type}/{model}")
        return reasoning_effort


def _build_app(user_id):
    app = FastAPI()
    app.include_router(router)
    app.include_router(conversation_router)
    app.include_router(router, prefix="/ai")
    app.include_router(conversation_router, prefix="/ai")
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    return app


@pytest.fixture
def api():
    db = Database(settings.database_url)
    sf = db.session
    owner_id = uuid4()
    other_id = uuid4()
    conversation_id = uuid4()
    with sf() as s:
        for uid in (owner_id, other_id):
            s.add(
                User(
                    id=uid,
                    username=f"u_{uid.hex[:12]}",
                    email=f"{uid.hex[:12]}@test.local",
                    password_hash="x",
                )
            )
        s.add(Conversation(id=conversation_id, owner_id=owner_id, title="t"))
        s.commit()

    Container.model_config_service.override(providers.Object(_FakeModelConfig()))
    owner_client = TestClient(_build_app(owner_id))
    other_client = TestClient(_build_app(other_id))
    try:
        yield owner_client, other_client, owner_id, other_id, conversation_id
    finally:
        Container.model_config_service.reset_override()
        with sf() as s:
            for uid in (owner_id, other_id):
                s.execute(
                    delete(ConversationCustomAgent).where(ConversationCustomAgent.owner_id == uid)
                )
                s.execute(delete(CustomAgent).where(CustomAgent.owner_id == uid))
                s.execute(delete(Conversation).where(Conversation.owner_id == uid))
                s.execute(delete(User).where(User.id == uid))
            s.commit()


def _create_body(name="Data Analyst"):
    return {
        "name": name,
        "prompt": "You are a precise data analyst.",
        "provider_type": "openai",
        "model": "gpt-4.1-mini",
    }


def test_crud_flow(api):
    owner, _other, _oid, _otid, _cid = api

    # Create
    r = owner.post("/custom-agents", json=_create_body())
    assert r.status_code == 201, r.text
    agent = r.json()["data"]
    agent_id = agent["id"]
    assert agent["slug"] == "data-analyst"
    assert agent["runtimeAgentId"] == f"custom_agent:{agent_id}"

    # List
    r = owner.get("/custom-agents")
    assert r.status_code == 200
    assert len(r.json()["data"]) == 1

    # Get
    r = owner.get(f"/custom-agents/{agent_id}")
    assert r.status_code == 200

    # Update
    r = owner.patch(f"/custom-agents/{agent_id}", json={"prompt": "Updated"})
    assert r.status_code == 200
    assert r.json()["data"]["prompt"] == "Updated"

    # Delete
    r = owner.delete(f"/custom-agents/{agent_id}")
    assert r.status_code == 200
    assert owner.get(f"/custom-agents/{agent_id}").status_code == 404


def test_validation_returns_400(api):
    owner = api[0]
    body = _create_body()
    body["model"] = "nonexistent"
    r = owner.post("/custom-agents", json=body)
    assert r.status_code == 400
    assert r.json()["detail"] and "nonexistent" in r.json()["detail"]


def test_ownership_403_and_404(api):
    owner, other, _oid, _otid, _cid = api
    agent_id = owner.post("/custom-agents", json=_create_body()).json()["data"]["id"]

    # Other user cannot see/modify it -> 403
    assert other.get(f"/custom-agents/{agent_id}").status_code == 403
    assert other.patch(f"/custom-agents/{agent_id}", json={"prompt": "x"}).status_code == 403
    # Missing agent -> 404
    assert owner.get(f"/custom-agents/{uuid4()}").status_code == 404


def test_attach_detach_and_ai_alias(api):
    owner, _other, _oid, _otid, cid = api
    agent_id = owner.post("/custom-agents", json=_create_body()).json()["data"]["id"]

    # Attach via canonical PUT
    r = owner.put(
        f"/conversations/{cid}/custom-agents",
        json={"customAgentIds": [agent_id]},
    )
    assert r.status_code == 200
    assert [a["id"] for a in r.json()["data"]] == [agent_id]

    # Read back through the /ai alias
    r = owner.get(f"/ai/conversations/{cid}/custom-agents")
    assert r.status_code == 200
    assert [a["id"] for a in r.json()["data"]] == [agent_id]

    # Detach
    r = owner.put(f"/conversations/{cid}/custom-agents", json={"customAgentIds": []})
    assert r.status_code == 200
    assert r.json()["data"] == []


def test_conflict_409_when_active(api):
    owner, _other, oid, _otid, cid = api
    agent_id = owner.post("/custom-agents", json=_create_body()).json()["data"]["id"]

    registry = get_generation_registry()
    entry = registry.register(uuid4(), cid, oid, selected_agent=f"custom_agent:{agent_id}")
    try:
        assert owner.delete(f"/custom-agents/{agent_id}").status_code == 409
        assert owner.patch(f"/custom-agents/{agent_id}", json={"prompt": "x"}).status_code == 409
    finally:
        # Clean the singleton so other tests are unaffected.
        for key, value in list(registry._store.items()):
            if value is entry:
                registry._store.pop(key, None)


@pytest.mark.asyncio
async def test_ai_sdk_chat_uses_attached_custom_agents():
    """The shared workflow-request builder (used by Streamlit + AI SDK chat)
    resolves attached custom agents into the request's custom_agents map."""
    from types import SimpleNamespace

    from app.schemas.message import MessageCreate
    from app.schemas.workflow import WorkflowPlanningContext
    from app.services.message_service import MessageService

    rid = f"custom_agent:{uuid4()}"
    owner = uuid4()
    conv = uuid4()

    class _FakeCustomAgentSvc:
        def build_runtime_state(self, owner_id, conversation_id):
            return {
                rid: {
                    "id": rid.split(":", 1)[1],
                    "runtime_agent_id": rid,
                    "model_agent_key": "custom",
                    "name": "Analyst",
                    "prompt": "p",
                    "model_request": {"provider_type": "openai", "model": "gpt-4.1-mini"},
                    "tool_refs": [],
                    "skill_refs": [],
                    "agent_order": 0,
                }
            }

    svc = MessageService.__new__(MessageService)
    svc.custom_agent_service = _FakeCustomAgentSvc()

    async def _planning(**_kwargs):
        return WorkflowPlanningContext()

    svc._prepare_planning_context = _planning
    svc._extract_message_execution_inputs = lambda _mc: ([], None)
    svc._coerce_plan_lifecycle = lambda _v: None

    conversation = SimpleNamespace(
        owner_id=owner, persona_prompt=None, planning_mode_enabled=False, plan_lifecycle=None
    )
    message = MessageCreate(conversation_id=conv, content="hello")

    _ruid, _persona, request = await svc._build_user_message_workflow_request(
        message_create_data=message, user_id=owner, conversation=conversation
    )
    assert rid in request.custom_agents
    assert request.custom_agents[rid]["name"] == "Analyst"


def test_options_endpoint(api):
    owner = api[0]
    r = owner.get("/custom-agents/options")
    assert r.status_code == 200
    data = r.json()["data"]
    assert {"providers", "serverDefaultTools", "clientTools", "skills"} <= set(data.keys())
    assert isinstance(data["serverDefaultTools"], list)


def test_contextual_reads_and_options_serialize_camel_case_contract(api):
    owner = api[0]
    agent_id = owner.post("/custom-agents", json=_create_body()).json()["data"]["id"]

    listed = owner.get("/ai/custom-agents?deviceId=desktop-1").json()["data"][0]
    fetched = owner.get(f"/ai/custom-agents/{agent_id}?deviceId=desktop-1").json()["data"]
    options = owner.get("/ai/custom-agents/options?deviceId=desktop-1").json()["data"]

    assert listed["availability"]["status"] == "ready"
    assert fetched["availability"]["missingTools"] == []
    assert options["deviceSnapshot"]["status"] == "unavailable"
    assert "toolCatalogVersion" in options["deviceSnapshot"]
