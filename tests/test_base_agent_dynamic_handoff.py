"""Base-agent dynamic delegation: a base agent must be able to receive a
graph-injected hand_off tool (targeting attached custom agents) and a dynamic
delegation prompt, and to surface a multi-agent awareness block."""

from __future__ import annotations

from app.ai.agents.base_agent import BaseAgent
from app.ai.hand_off_tool import create_hand_off_tool
from app.ai.schemas import AgentType


class _DummyBase(BaseAgent):
    def _init_gemini(self) -> None:
        self.gemini_client = None
        self.langchain_model = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "chat_agent"

    def _get_base_system_prompt(self) -> str:
        return "Base system prompt."


def _agent() -> _DummyBase:
    return _DummyBase(agent_config_key="chat")


def test_build_delegation_suffix_static_without_descriptions():
    suffix = _agent()._build_delegation_suffix()
    assert "hand_off" in suffix
    # Falls back to the canonical static suffix when no dynamic targets given.
    assert "Available targets" in suffix


def test_build_delegation_suffix_dynamic_lists_custom_targets():
    descriptions = {
        "search_agent": "Current events, news, web search, and fact-checking.",
        "custom_agent:abc": "Legal Reviewer: Reviews contract and policy questions.",
    }
    suffix = _agent()._build_delegation_suffix(descriptions)

    assert "custom_agent:abc" in suffix
    assert "Legal Reviewer" in suffix
    assert "search_agent" in suffix


def test_build_system_prompt_injects_multi_agent_activity():
    block = 'MULTI-AGENT SYSTEM:\nYou are "Chat Agent".'
    prompt = _agent()._build_system_prompt(
        persona=None,
        has_tool_context=False,
        multi_agent_activity=block,
    )
    assert block in prompt


def test_build_system_prompt_uses_dynamic_delegation_targets():
    descriptions = {"custom_agent:abc": "Legal Reviewer: contracts."}
    prompt = _agent()._build_system_prompt(
        persona=None,
        has_tool_context=False,
        handoff_target_descriptions=descriptions,
    )
    assert "custom_agent:abc" in prompt
    assert "Legal Reviewer" in prompt


def test_injected_handoff_tool_wins_over_static():
    """A hand_off passed via internal_tools (the graph's dynamic one) must
    replace the static base-only hand_off, not be deduped away by it."""
    dynamic = create_hand_off_tool(
        ["search_agent", "custom_agent:abc"],
        {"custom_agent:abc": "Legal Reviewer: contracts."},
    )
    tools = _agent()._get_tools_for_binding(internal_tools=[dynamic])

    handoffs = [t for t in tools if getattr(t, "name", None) == "hand_off"]
    assert len(handoffs) == 1
    assert "custom_agent:abc" in handoffs[0].description
