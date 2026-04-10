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


# ---------------------------------------------------------------------------
# Phase 4: Router canvas_agent guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_canvas_agent_not_routed_for_product_name_without_web_intent(monkeypatch):
    """A query that contains a product/app name but no explicit web-creation
    keywords must NOT be routed to canvas_agent."""
    monkeypatch.setattr(
        Router,
        "_init_gemini",
        lambda self: setattr(self, "gemini_client", None),
    )
    router = Router()

    # LLM incorrectly returns canvas_agent for a product name query
    async def fake_call_llm(prompt: str, available_agents: list[str]) -> str | None:
        return "canvas_agent"

    monkeypatch.setattr(router, "_call_llm", fake_call_llm)

    # "Use Figma" — a product name, no explicit website creation intent
    result = await router.route_message(
        AgentMessage(role=MessageRole.USER, content="Use Figma to design this icon"),
        ["chat_agent", "search_agent", "canvas_agent"],
    )

    # The guard should override canvas_agent -> chat_agent
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


def test_has_web_creation_intent_detects_website():
    assert Router._has_web_creation_intent("build me a website") is True


def test_has_web_creation_intent_rejects_product_name():
    assert Router._has_web_creation_intent("open this song on Figma") is False


def test_has_web_creation_intent_detects_landing_page():
    assert Router._has_web_creation_intent("Create a landing page for my startup") is True
