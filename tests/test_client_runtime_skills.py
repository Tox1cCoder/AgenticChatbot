from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.agents.base_agent import BaseAgent
from app.ai.agents.router import Router
from app.ai.schemas import AgentType
from app.ai.skills_tool import create_activate_skill_tool, get_available_skill_summaries
from app.ai.tool_context import tool_execution_context


def _server_registry_stub(*skills):
    skills_by_name = {skill.name: skill for skill in skills}

    def _get_skill(name: str):
        if name not in skills_by_name:
            raise KeyError(name)
        return skills_by_name[name]

    return SimpleNamespace(
        get_active_skills=lambda: list(skills_by_name.values()),
        get_skill=_get_skill,
    )


class _GatewayStub:
    def __init__(self):
        self.calls: list[dict] = []

    async def dispatch_tool_call(
        self,
        *,
        request_id: str,
        tool_name: str,
        qualified_tool_id: str,
        arguments: dict,
        timeout_seconds: int,
    ) -> dict:
        self.calls.append(
            {
                "request_id": request_id,
                "tool_name": tool_name,
                "qualified_tool_id": qualified_tool_id,
                "arguments": arguments,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "success": True,
            "result": "── Skill: demo ──\n\nUse the client-side instructions.\n\n── End Skill: demo ──",
        }


class _DummyAgent(BaseAgent):
    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "dummy_agent"

    def _get_base_system_prompt(self) -> str:
        return "BASE PROMPT"


@pytest.mark.asyncio
async def test_activate_skill_tool_dispatches_to_client_runtime(monkeypatch):
    device_id = uuid4()
    gateway = _GatewayStub()
    session = SimpleNamespace(
        device_id=device_id,
        user_id="user-123",
        session_id="session-123",
        websocket=gateway,
        skill_catalog={
            "skills": [
                {
                    "name": "demo",
                    "description": "Device-only skill",
                    "enabled": True,
                }
            ]
        },
    )

    monkeypatch.setattr(
        "app.ai.skills_tool.ClientDeviceService.lookup_active_session",
        lambda queried_device_id: session if queried_device_id == device_id else None,
    )
    monkeypatch.setattr(
        "app.ai.skills_tool.get_skills_registry",
        lambda: _server_registry_stub(),
    )

    summaries = get_available_skill_summaries(user_id="user-123", device_id=str(device_id))
    tool = create_activate_skill_tool(user_id="user-123", device_id=str(device_id))

    with tool_execution_context(user_id="user-123", device_id=str(device_id)):
        result = await tool.ainvoke({"skill_name": "demo"})

    assert summaries == [
        {
            "name": "demo",
            "lookup_name": "demo",
            "description": "Device-only skill",
            "category": None,
            "tags": [],
            "content_length": None,
            "source": "client",
        }
    ]
    assert "Use the client-side instructions." in result
    assert gateway.calls[0]["tool_name"] == "activate_skill"
    assert gateway.calls[0]["qualified_tool_id"] == "native::activate_skill"
    assert gateway.calls[0]["arguments"] == {"skill_name": "demo"}


@pytest.mark.asyncio
async def test_activate_skill_tool_falls_back_to_server_registry(monkeypatch):
    server_skill = SimpleNamespace(
        name="server_demo",
        description="Server-side skill",
        content="Use the canonical backend instructions.",
        enabled=True,
    )

    monkeypatch.setattr(
        "app.ai.skills_tool.get_skills_registry",
        lambda: _server_registry_stub(server_skill),
    )
    monkeypatch.setattr(
        "app.ai.skills_tool.ClientDeviceService.lookup_active_session",
        lambda queried_device_id: None,
    )

    summaries = get_available_skill_summaries(user_id="user-123", device_id=None)
    tool = create_activate_skill_tool(user_id="user-123", device_id=None)
    result = await tool.ainvoke({"skill_name": "server_demo"})

    assert summaries == [
        {
            "name": "server_demo",
            "lookup_name": "server_demo",
            "description": "Server-side skill",
            "category": None,
            "tags": [],
            "content_length": len("Use the canonical backend instructions."),
            "source": "server",
        }
    ]
    assert "Use the canonical backend instructions." in result
    assert "Skill: server_demo" in result


def test_base_agent_uses_client_runtime_skill_catalog(monkeypatch):
    monkeypatch.setattr(BaseAgent, "_init_gemini", lambda self: None)
    monkeypatch.setattr(
        "app.ai.agents.base_agent.get_available_skill_summaries",
        lambda *, user_id, device_id: (
            [
                {
                    "name": "demo",
                    "lookup_name": "demo",
                    "description": "Client-side skill",
                    "source": "client",
                }
            ]
            if user_id == "user-123" and device_id == "device-123"
            else []
        ),
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.create_activate_skill_tool",
        lambda *, user_id, device_id: SimpleNamespace(name="activate_skill"),
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.get_client_runtime_tools",
        lambda *, user_id, device_id: [],
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda agent_key: False,
    )

    agent = _DummyAgent()
    agent.tools = []

    tools = agent._get_tools_for_binding(user_id="user-123", device_id="device-123")
    prompt = agent._build_system_prompt(
        None,
        False,
        user_id="user-123",
        device_id="device-123",
    )
    prompt_without_device_skills = agent._build_system_prompt(
        None,
        False,
        user_id="user-123",
        device_id=None,
    )

    assert [tool.name for tool in tools] == ["activate_skill"]
    assert "Available Skills" in prompt
    assert "demo" in prompt
    assert "[client]" in prompt
    assert "activate_skill" in prompt
    assert "Available Skills" not in prompt_without_device_skills


def test_base_agent_uses_server_skill_catalog_without_device(monkeypatch):
    monkeypatch.setattr(BaseAgent, "_init_gemini", lambda self: None)
    monkeypatch.setattr(
        "app.ai.agents.base_agent.get_available_skill_summaries",
        lambda *, user_id, device_id: (
            [
                {
                    "name": "server_demo",
                    "lookup_name": "server_demo",
                    "description": "Server-side skill",
                    "source": "server",
                }
            ]
            if user_id == "user-123" and device_id is None
            else []
        ),
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.create_activate_skill_tool",
        lambda *, user_id, device_id: SimpleNamespace(name="activate_skill"),
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.get_client_runtime_tools",
        lambda *, user_id, device_id: [],
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda agent_key: False,
    )

    agent = _DummyAgent()
    agent.tools = []

    tools = agent._get_tools_for_binding(user_id="user-123", device_id=None)
    prompt = agent._build_system_prompt(
        None,
        False,
        user_id="user-123",
        device_id=None,
    )

    assert [tool.name for tool in tools] == ["activate_skill"]
    assert "server_demo" in prompt
    assert "[server]" in prompt


def test_router_prompt_uses_client_runtime_skill_catalog(monkeypatch):
    monkeypatch.setattr(
        "app.ai.agents.router.get_available_skill_summaries",
        lambda *, user_id, device_id: (
            [
                {
                    "name": "demo",
                    "lookup_name": "demo",
                    "description": "Client-side skill",
                    "source": "client",
                }
            ]
            if user_id == "user-123" and device_id == "device-123"
            else []
        ),
    )

    router = Router()
    prompt = router._build_prompt(
        content="Use the demo skill",
        persona=None,
        available_agents=["chat_agent", "rag_agent"],
        has_documents=False,
        planning_mode_enabled=False,
        has_existing_plan=False,
        user_id="user-123",
        device_id="device-123",
    )

    assert "Active skills available for this request" in prompt
    assert "demo" in prompt
    assert "[client]" in prompt


def test_router_prompt_uses_server_skill_catalog_without_device(monkeypatch):
    monkeypatch.setattr(
        "app.ai.agents.router.get_available_skill_summaries",
        lambda *, user_id, device_id: (
            [
                {
                    "name": "server_demo",
                    "lookup_name": "server_demo",
                    "description": "Server-side skill",
                    "source": "server",
                }
            ]
            if user_id == "user-123" and device_id is None
            else []
        ),
    )

    router = Router()
    prompt = router._build_prompt(
        content="Use the server demo skill",
        persona=None,
        available_agents=["chat_agent", "rag_agent"],
        has_documents=False,
        planning_mode_enabled=False,
        has_existing_plan=False,
        user_id="user-123",
        device_id=None,
    )

    assert "Active skills available for this request" in prompt
    assert "server_demo" in prompt
    assert "[server]" in prompt
