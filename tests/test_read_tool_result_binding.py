from __future__ import annotations

from types import SimpleNamespace

from app.ai.agents.base_agent import BaseAgent
from app.ai.schemas import AgentType


class _BindingTestAgent(BaseAgent):
    def _init_gemini(self) -> None:
        self.gemini_client = None
        self.langchain_model = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "binding-test"

    def _get_base_system_prompt(self) -> str:
        return "binding-test"


def _bound_names(monkeypatch, *, offload_enabled: bool) -> list[str]:
    agent = _BindingTestAgent(agent_config_key="search")
    agent.tools = []
    agent.mcp_manager = None
    monkeypatch.setattr(
        "app.ai.agents.base_agent.settings.tool_result_offload_enabled",
        offload_enabled,
        raising=False,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_deferred_tool_list",
        lambda **kwargs: list(kwargs.get("internal_tools") or []),
    )
    monkeypatch.setattr(agent, "_get_client_runtime_tools", lambda **kwargs: [])
    monkeypatch.setattr(
        agent, "_get_skills_internal_tools", lambda **kwargs: [SimpleNamespace(name="noop")]
    )
    return [tool.name for tool in agent._get_tools_for_binding(conversation_id="c1")]


class _StubContainer:
    """Stands in for ``app.core.container.Container``; the repositories are opaque."""

    fail = False

    def user_memory_repository(self):
        if self.fail:
            raise RuntimeError("memory repository down")
        return "memory-repository"

    def conversation_search_repository(self):
        if self.fail:
            raise RuntimeError("search repository down")
        return "search-repository"


def _bind_with_repository_tools(monkeypatch, *, user_id, fail=False, internal_tools=None):
    agent = _BindingTestAgent(agent_config_key="chat")
    agent.tools = []
    agent.mcp_manager = None
    calls: list[tuple[str, dict]] = []
    stub_container = type("_Container", (_StubContainer,), {"fail": fail})
    monkeypatch.setattr("app.core.container.Container", stub_container)
    for flag in (
        "enable_user_memory_tools",
        "enable_conversation_search_tools",
        "tool_result_offload_enabled",
    ):
        monkeypatch.setattr(f"app.ai.agents.base_agent.settings.{flag}", True, raising=False)
    monkeypatch.setattr(
        "app.ai.agents.base_agent.create_user_memory_tools",
        lambda **kwargs: calls.append(("memory", kwargs)) or [SimpleNamespace(name="memory")],
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.create_conversation_search_tools",
        lambda **kwargs: calls.append(("search", kwargs)) or [SimpleNamespace(name="history")],
    )
    for factory in ("create_web_search_tool", "create_web_open_tool"):
        monkeypatch.setattr(
            f"app.ai.agents.base_agent.{factory}",
            lambda tool_scope, _name=factory: SimpleNamespace(name=_name, scope=tool_scope),
        )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_deferred_tool_list",
        lambda **kwargs: list(kwargs.get("internal_tools") or []),
    )
    monkeypatch.setattr(agent, "_get_client_runtime_tools", lambda **kwargs: [])
    monkeypatch.setattr(
        agent, "_get_skills_internal_tools", lambda **kwargs: [SimpleNamespace(name="skill")]
    )
    tools = agent._get_tools_for_binding(
        conversation_id="c1",
        user_id=user_id,
        internal_tools=internal_tools,
    )
    return [tool.name for tool in tools], calls


def test_internal_tools_merge_in_source_order_and_the_first_name_wins(monkeypatch):
    names, calls = _bind_with_repository_tools(
        monkeypatch,
        user_id="u1",
        internal_tools=[SimpleNamespace(name="hand_off"), SimpleNamespace(name="skill")],
    )

    assert names == [
        "skill",
        "read_tool_result",
        "create_web_search_tool",
        "create_web_open_tool",
        "hand_off",
        "memory",
        "history",
    ]
    assert calls == [
        ("memory", {"repository": "memory-repository", "user_id": "u1", "conversation_id": "c1"}),
        ("search", {"repository": "search-repository", "user_id": "u1", "conversation_id": "c1"}),
    ]


def test_repository_tools_need_a_user(monkeypatch):
    names, calls = _bind_with_repository_tools(monkeypatch, user_id=None)

    assert "memory" not in names
    assert "history" not in names
    assert calls == []


def test_an_unavailable_repository_binds_nothing_from_it(monkeypatch):
    names, calls = _bind_with_repository_tools(monkeypatch, user_id="u1", fail=True)

    assert "memory" not in names
    assert "history" not in names
    assert "skill" in names
    assert calls == []


def test_read_tool_result_is_bound_when_offload_is_enabled(monkeypatch):
    assert "read_tool_result" in _bound_names(monkeypatch, offload_enabled=True)


def test_read_tool_result_is_absent_when_offload_is_disabled(monkeypatch):
    assert "read_tool_result" not in _bound_names(monkeypatch, offload_enabled=False)


def test_tool_context_prompt_points_at_the_reader():
    from app.ai.prompts import TOOL_CONTEXT_SUFFIX

    assert "read_tool_result" in TOOL_CONTEXT_SUFFIX


def test_tool_context_prompt_asks_for_an_objective_not_a_page():
    """A bullet that says "read the rest" is an instruction to page. The reader
    now answers a question, so the prompt has to ask one."""
    from app.ai.prompts import TOOL_CONTEXT_SUFFIX

    assert "objective" in TOOL_CONTEXT_SUFFIX
    assert "read the rest" not in TOOL_CONTEXT_SUFFIX
    assert "offset" not in TOOL_CONTEXT_SUFFIX


def test_tool_context_prompt_offload_bullet_is_tool_neutral():
    """The offload bullet must read correctly for read_file, SQL tools, and
    client skills, not only for a search tool. It must also agree in register
    with the offload notice text in tool_result_blob_service.py, which already
    says "do not re-run the tool".
    """
    from app.ai.prompts import TOOL_CONTEXT_SUFFIX

    assert "search" not in TOOL_CONTEXT_SUFFIX.lower()
    assert "do not re-run the tool" in TOOL_CONTEXT_SUFFIX.lower()
