"""Safe-by-agent thinking level (latency fix).

chat_agent handles routing/handoff and simple conversation, none of which need
deep reasoning. It runs at a lower Gemini ``thinking_level`` than the global
default so those calls stop paying the full high-thinking latency, while
search/rag agents (which do document synthesis) keep the global level.

Precedence for the resolved level:
    explicit per-request override (reasoning_effort) > per-agent config > global
"""

from __future__ import annotations

from app.ai import agent_config
from app.core.config import settings


def test_chat_agent_resolves_to_configured_lower_thinking_level(monkeypatch):
    monkeypatch.setitem(agent_config.AGENT_CONFIG["chat"], "thinking_level", "minimal")
    assert agent_config._resolve_thinking_level("chat", None) == "minimal"


def test_search_agent_keeps_global_thinking_level():
    # search has no per-agent thinking_level -> falls back to the global default
    assert agent_config._resolve_thinking_level("search", None) == settings.thinking_level


def test_explicit_thinking_override_beats_per_agent_default(monkeypatch):
    monkeypatch.setitem(agent_config.AGENT_CONFIG["chat"], "thinking_level", "minimal")
    assert agent_config._resolve_thinking_level("chat", "high") == "high"


def test_create_langchain_model_builds_chat_with_lower_thinking(monkeypatch):
    captured: dict = {}

    def fake_cgg(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_config, "ReasoningNormalizedChatGoogleGenerativeAI", fake_cgg)
    monkeypatch.setattr(agent_config, "get_api_key", lambda **kw: "fake-key")
    monkeypatch.setattr(agent_config.settings, "enable_thinking", True)
    monkeypatch.setitem(agent_config.AGENT_CONFIG["chat"], "thinking_level", "low")

    agent_config.create_langchain_model(agent_type="chat", model_override="gemini-3-flash-preview")
    assert captured.get("thinking_level") == "low"


def test_create_langchain_model_leaves_search_at_global_thinking(monkeypatch):
    captured: dict = {}

    def fake_cgg(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_config, "ReasoningNormalizedChatGoogleGenerativeAI", fake_cgg)
    monkeypatch.setattr(agent_config, "get_api_key", lambda **kw: "fake-key")
    monkeypatch.setattr(agent_config.settings, "enable_thinking", True)

    agent_config.create_langchain_model(
        agent_type="search", model_override="gemini-3-flash-preview"
    )
    assert captured.get("thinking_level") == settings.thinking_level
