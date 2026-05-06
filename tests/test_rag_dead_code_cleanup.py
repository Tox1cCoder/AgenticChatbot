from pathlib import Path

RAG_AGENT_SOURCE = Path("app/ai/agents/rag_agent.py")


def test_rag_agent_does_not_use_langchain_create_agent() -> None:
    source = RAG_AGENT_SOURCE.read_text(encoding="utf-8")

    assert "from langchain.agents import create_agent" not in source
    assert "create_agent(" not in source


def test_rag_agent_has_no_legacy_generation_helpers() -> None:
    source = RAG_AGENT_SOURCE.read_text(encoding="utf-8")

    legacy_helpers = [
        "def _create_agent_executor(",
        "def _generate_with_tools_stream(",
        "def _generate_with_tools(",
        "def _generate_with_vision(",
        "def _generate(",
        "def _verify_citations(",
        "def _augment_prompt_with_tool_context(",
    ]

    for helper in legacy_helpers:
        assert helper not in source
