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
