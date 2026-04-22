import pytest

from app.ai.agents.router import Router
from app.ai.schemas import AgentMessage, MessageRole


@pytest.mark.asyncio
async def test_route_message_uses_llm_result_for_device_action_queries(monkeypatch):
    monkeypatch.setattr(
        Router,
        "_init_gemini",
        lambda self: setattr(self, "gemini_client", None),
    )
    router = Router()

    async def fake_call_llm(prompt: str, available_agents: list[str]) -> str | None:
        assert "open this song on youtube" in prompt
        assert "chat_agent" in available_agents
        return "search_agent"

    monkeypatch.setattr(router, "_call_llm", fake_call_llm)

    result = await router.route_message(
        AgentMessage(role=MessageRole.USER, content="open this song on youtube"),
        ["chat_agent", "search_agent"],
    )

    assert result == "search_agent"


def test_extract_agent_name_handles_punctuation_without_regex():
    assert (
        Router._extract_agent_name(
            "search_agent.",
            ["chat_agent", "search_agent"],
        )
        == "search_agent"
    )


def test_extract_agent_name_handles_surrounding_text_without_regex():
    assert (
        Router._extract_agent_name(
            "Selected agent: chat_agent",
            ["chat_agent", "search_agent"],
        )
        == "chat_agent"
    )


@pytest.mark.asyncio
async def test_route_message_honors_llm_non_canvas_decision_for_product_names(monkeypatch):
    monkeypatch.setattr(
        Router,
        "_init_gemini",
        lambda self: setattr(self, "gemini_client", None),
    )
    router = Router()

    async def fake_call_llm(prompt: str, available_agents: list[str]) -> str | None:
        assert "Use Figma to design this icon" in prompt
        return "chat_agent"

    monkeypatch.setattr(router, "_call_llm", fake_call_llm)

    result = await router.route_message(
        AgentMessage(role=MessageRole.USER, content="Use Figma to design this icon"),
        ["chat_agent", "search_agent", "canvas_agent"],
    )

    assert result == "chat_agent"


@pytest.mark.asyncio
async def test_canvas_agent_routed_for_explicit_website_request(monkeypatch):
    """An explicit website creation request must still route to canvas_agent."""
    monkeypatch.setattr(
        Router,
        "_init_gemini",
        lambda self: setattr(self, "gemini_client", None),
    )
    router = Router()

    async def fake_call_llm(prompt: str, available_agents: list[str]) -> str | None:
        return "canvas_agent"

    monkeypatch.setattr(router, "_call_llm", fake_call_llm)

    result = await router.route_message(
        AgentMessage(role=MessageRole.USER, content="Create a website for my business"),
        ["chat_agent", "search_agent", "canvas_agent"],
    )

    assert result == "canvas_agent"


@pytest.mark.asyncio
async def test_canvas_agent_honors_llm_decision_for_browser_artifact_request(monkeypatch):
    """Router should not second-guess the LLM with local phrase matching."""
    monkeypatch.setattr(
        Router,
        "_init_gemini",
        lambda self: setattr(self, "gemini_client", None),
    )
    router = Router()

    async def fake_call_llm(prompt: str, available_agents: list[str]) -> str | None:
        return "canvas_agent"

    monkeypatch.setattr(router, "_call_llm", fake_call_llm)

    result = await router.route_message(
        AgentMessage(
            role=MessageRole.USER,
            content="Build an online ordering experience for my restaurant",
        ),
        ["chat_agent", "search_agent", "canvas_agent"],
    )

    assert result == "canvas_agent"


def test_router_prompt_uses_semantic_canvas_liveui_boundary():
    from app.ai.prompts import ROUTER_SYSTEM_PROMPT

    assert "Do not rely on exact phrase matching" in ROUTER_SYSTEM_PROMPT
    assert "standalone artifacts rendered in the canvas panel" in ROUTER_SYSTEM_PROMPT
    assert "LiveUI widgets are for compact in-chat aids" in ROUTER_SYSTEM_PROMPT
    assert "Examples:" not in ROUTER_SYSTEM_PROMPT
