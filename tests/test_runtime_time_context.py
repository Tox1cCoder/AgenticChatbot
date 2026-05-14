from __future__ import annotations

from datetime import datetime, timezone

from app.ai.agents.base_agent import BaseAgent
from app.ai.agents.router import Router
from app.ai.schemas import AgentType


class _DummyAgent(BaseAgent):
    def _init_gemini(self) -> None:
        self.gemini_client = None
        self.langchain_model = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "dummy_agent"

    def _get_base_system_prompt(self) -> str:
        return "Dummy system prompt."


def test_runtime_time_context_block_anchors_utc_and_configured_timezone():
    from app.ai.time_context import build_runtime_time_context_block

    now = datetime(2026, 5, 14, 1, 30, tzinfo=timezone.utc)

    block = build_runtime_time_context_block(
        now=now,
        timezone_name="Asia/Bangkok",
    )

    assert "Runtime Time Context" in block
    assert "2026-05-14T01:30:00+00:00" in block
    assert "Asia/Bangkok" in block
    assert "Thursday, May 14, 2026" in block
    assert "2026-05-14T08:30:00+07:00" in block
    assert "changing external facts" in block


def test_runtime_time_context_block_falls_back_to_utc_for_invalid_timezone():
    from app.ai.time_context import build_runtime_time_context_block

    now = datetime(2026, 5, 14, 1, 30, tzinfo=timezone.utc)

    block = build_runtime_time_context_block(
        now=now,
        timezone_name="Invalid/Timezone",
    )

    assert "timezone: UTC" in block
    assert "2026-05-14T01:30:00+00:00" in block
    assert "fallback: configured timezone was invalid" in block


def test_base_agent_build_system_prompt_injects_runtime_time_context(monkeypatch):
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_runtime_time_context_block",
        lambda: "RUNTIME TIME BLOCK",
    )
    agent = _DummyAgent(agent_config_key="chat")

    prompt = agent._build_system_prompt(persona=None, has_tool_context=False)

    assert "Dummy system prompt." in prompt
    assert "RUNTIME TIME BLOCK" in prompt
    assert prompt.index("Dummy system prompt.") < prompt.index("RUNTIME TIME BLOCK")


def test_base_agent_full_system_prompt_injects_runtime_time_context(monkeypatch):
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_runtime_time_context_block",
        lambda: "RUNTIME TIME BLOCK",
    )
    agent = _DummyAgent(agent_config_key="chat")

    prompt = agent._get_full_system_prompt()

    assert "Dummy system prompt." in prompt
    assert "RUNTIME TIME BLOCK" in prompt


def test_router_prompt_injects_runtime_time_context(monkeypatch):
    monkeypatch.setattr(
        Router,
        "_init_gemini",
        lambda self: setattr(self, "gemini_client", None),
    )
    monkeypatch.setattr(
        "app.ai.agents.router.build_runtime_time_context_block",
        lambda: "RUNTIME TIME BLOCK",
    )
    router = Router()

    prompt = router._build_prompt(
        content="create an analysis about a pro team play this season",
        persona=None,
        available_agents=["chat_agent", "search_agent"],
        has_documents=False,
        planning_mode_enabled=False,
        has_existing_plan=False,
        user_id=None,
        device_id=None,
    )

    assert "RUNTIME TIME BLOCK" in prompt
    assert "create an analysis about a pro team play this season" in prompt
