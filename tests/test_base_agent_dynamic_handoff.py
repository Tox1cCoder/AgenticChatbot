"""Base-agent dynamic delegation: a base agent must be able to receive a
graph-injected hand_off tool (targeting attached custom agents) and a dynamic
delegation prompt, and to surface a multi-agent awareness block."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


def test_hand_off_tool_schema_takes_a_target_and_a_reason():
    """``reason`` is part of the transition record, so the model supplies it.

    Anything else is refused: the schema is the only way a target is chosen,
    and the tool call id is injected rather than model-authored.
    """
    schema = create_hand_off_tool(
        source_agent_id="chat_agent", allowed_targets=["search_agent"]
    ).args_schema

    decided = schema.model_validate({"target_agent": "search_agent", "reason": "current info"})
    assert decided.target_agent == "search_agent"
    assert decided.reason == "current info"

    with pytest.raises(ValidationError):
        schema.model_validate({"target_agent": "search_agent", "unexpected": "field"})


def test_build_delegation_suffix_is_empty_without_live_targets():
    suffix = _agent()._build_delegation_suffix()
    assert suffix == ""


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


def test_build_system_prompt_includes_currency_markdown_guidance_once():
    prompt = _agent()._build_system_prompt(persona=None, has_tool_context=False)

    assert "write dollar prices as \\$150; reserve $...$ for LaTeX" in prompt
    assert prompt.count("write dollar prices as") == 1


def test_build_system_prompt_uses_dynamic_delegation_targets():
    descriptions = {"custom_agent:abc": "Legal Reviewer: contracts."}
    prompt = _agent()._build_system_prompt(
        persona=None,
        has_tool_context=False,
        include_hand_off=True,
        handoff_target_descriptions=descriptions,
    )
    assert "custom_agent:abc" in prompt
    assert "Legal Reviewer" in prompt


def test_build_system_prompt_omits_delegation_without_a_bound_handoff_tool():
    prompt = _agent()._build_system_prompt(persona=None, has_tool_context=False)

    assert "INTER-AGENT DELEGATION" not in prompt


def test_injected_handoff_tool_is_the_only_bound_handoff():
    """The graph-injected handoff is the only handoff BaseAgent can bind."""
    dynamic = create_hand_off_tool(
        source_agent_id="chat_agent",
        allowed_targets=["search_agent", "custom_agent:abc"],
        target_descriptions={"custom_agent:abc": "Legal Reviewer: contracts."},
    )
    tools = _agent()._get_tools_for_binding(internal_tools=[dynamic])

    handoffs = [t for t in tools if getattr(t, "name", None) == "hand_off"]
    assert len(handoffs) == 1
    assert "custom_agent:abc" in handoffs[0].description
