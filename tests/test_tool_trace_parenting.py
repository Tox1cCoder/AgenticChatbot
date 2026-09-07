"""A conversation-owned provider call belongs under its product tool.

The product runs tool calls through its own pipeline rather than the framework
tool node, so nothing here is guaranteed by LangChain composition alone. What
carries the ancestry is ``langchain_core``'s ``var_child_runnable_config``
contextvar: every ``Runnable`` sets it to its own child config while it runs,
and both hops the pipeline makes -- ``asyncio.create_task`` in
``invoke_tool_attempt`` and ``asyncio.to_thread`` in ``invoke_tool`` -- copy the
current context.

The consequence reads backwards, which is why it is worth stating: passing
**no** config to a nested call is what parents it correctly. Handing it the
caller's own config replaces the active child context with the parent's, and
whether that shows up as a broken tree depends on something outside this
repository -- with LangSmith tracing enabled, ``langsmith``'s current-run-tree
contextvar supplies the right parent anyway and hides the mistake; with tracing
disabled there is no such fallback and the provider run comes out a sibling of
the product tool. That is the argument against threading a config through the
pipeline: unnecessary when tracing is on, wrong when it is off.

``test_ancestry_does_not_depend_on_langsmith_tracing_being_enabled`` pins the
harder half. The rest run in whatever tracing state the suite is configured
with.

Assertions compare run ids rather than checking for a non-null parent: the
flattened topology has a non-null parent too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from langsmith.run_helpers import tracing_context

from app.ai.research_budget import reset_research_budget
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.tool_execution import execute_tool_calls
from app.ai.web_tools import (
    create_image_search_tool,
    create_web_open_tool,
    create_web_search_tool,
)
from app.ai.workflow.middleware import SpecialistToolScope, ToolExecutionMiddleware

CONVERSATION_ID = "66666666-6666-6666-6666-666666666666"


@dataclass(frozen=True)
class _Start:
    name: str
    run_id: UUID
    parent_run_id: UUID | None


class _RecordingHandler(BaseCallbackHandler):
    """Records the tool run tree as the callback manager reports it."""

    def __init__(self) -> None:
        self.starts: list[_Start] = []

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.starts.append(
            _Start(str((serialized or {}).get("name") or ""), run_id, parent_run_id)
        )

    def starts_named(self, name: str) -> list[_Start]:
        return [start for start in self.starts if start.name == name]

    def only(self, name: str) -> _Start:
        matches = self.starts_named(name)
        assert len(matches) == 1, f"expected one {name} run, saw {len(matches)}"
        return matches[0]


class _TracedProvider(BaseTool):
    """A provider double that is a real tool, so it produces a real run.

    ``tests/test_web_tools.py`` uses a bare object for this, which is right for
    argument mapping and useless here: an object with an ``ainvoke`` method
    creates no run to be the child of anything.
    """

    name: str = "traced_provider"
    description: str = "Returns a fixed payload without contacting anything."
    payload: str = "provider-ok"

    def _run(self, **kwargs: Any) -> str:
        return self.payload

    async def _arun(self, **kwargs: Any) -> str:
        return self.payload


def _product_calling(provider: BaseTool) -> StructuredTool:
    async def _product(query: str) -> str:
        return await provider.ainvoke({"query": query})

    return StructuredTool.from_function(
        coroutine=_product, name="product_tool", description="Calls one provider."
    )


async def _in_a_traced_node(
    handler: _RecordingHandler,
    *,
    tool_map: dict[str, Any],
    call: dict[str, Any],
) -> Any:
    """Run one tool call the way a graph node does, with callbacks attached."""

    async def node(_value: Any, config: dict[str, Any]) -> Any:
        return await execute_tool_calls(tool_calls=[call], tool_map=tool_map)

    return await RunnableLambda(node, name="tool_node").ainvoke(
        "go", config={"callbacks": [handler]}
    )


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"id": "call-1", "name": name, "args": args}


async def test_a_nested_provider_call_is_a_child_of_its_product_tool():
    provider = _TracedProvider(name="provider_tool")
    handler = _RecordingHandler()

    await _in_a_traced_node(
        handler,
        tool_map={"product_tool": _product_calling(provider)},
        call=_call("product_tool", query="q"),
    )

    assert handler.only("provider_tool").parent_run_id == handler.only("product_tool").run_id


async def test_the_product_tool_itself_is_not_a_root():
    provider = _TracedProvider(name="provider_tool")
    handler = _RecordingHandler()

    await _in_a_traced_node(
        handler,
        tool_map={"product_tool": _product_calling(provider)},
        call=_call("product_tool", query="q"),
    )

    assert handler.only("product_tool").parent_run_id is not None
    assert all(start.parent_run_id is not None for start in handler.starts)


async def test_ancestry_does_not_depend_on_langsmith_tracing_being_enabled():
    """The harder case: no current run tree to fall back on.

    With tracing enabled, ``langsmith`` supplies a parent from its own
    contextvar even when a caller hands a nested tool the wrong config, which
    hides the mistake. With tracing disabled, the LangChain child-config
    contextvar this pipeline relies on is the only thing carrying ancestry.
    Pinning the disabled case keeps the guarantee true in local development and
    CI, and stops a forwarded-config regression from hiding behind a live
    tracer.
    """
    provider = _TracedProvider(name="provider_tool")
    handler = _RecordingHandler()

    with tracing_context(enabled=False):
        await _in_a_traced_node(
            handler,
            tool_map={"product_tool": _product_calling(provider)},
            call=_call("product_tool", query="q"),
        )

    assert handler.only("provider_tool").parent_run_id == handler.only("product_tool").run_id


async def test_a_traced_synchronous_product_parents_its_provider_exactly():
    """A sync tool's body runs in a worker thread, and ancestry crosses it.

    ``StructuredTool`` built from a plain function has no ``coroutine``, so
    LangChain runs ``_run`` in an executor. Both runs exist here, so this is
    the synchronous case stated exactly rather than as "has some parent".
    """
    provider = _TracedProvider(name="provider_tool")

    def _product(query: str) -> str:
        return provider.invoke({"query": query})

    product = StructuredTool.from_function(
        func=_product, name="product_tool", description="Calls one provider synchronously."
    )
    handler = _RecordingHandler()

    with tracing_context(enabled=False):
        await _in_a_traced_node(
            handler,
            tool_map={"product_tool": product},
            call=_call("product_tool", query="q"),
        )

    assert handler.only("provider_tool").parent_run_id == handler.only("product_tool").run_id


async def test_an_untraced_thread_hop_does_not_orphan_the_provider():
    """The ``asyncio.to_thread`` branch of ``invoke_tool``, which few tools take.

    A bare object with ``invoke`` and no ``ainvoke`` is not a ``Runnable``, so
    it produces no run of its own and there is no product id to compare
    against. What is assertable -- and what matters -- is that the provider
    inherits the ambient run instead of opening a root.
    """
    provider = _TracedProvider(name="provider_tool")

    class _SyncProduct:
        name = "sync_product"

        def invoke(self, args: dict[str, Any]) -> str:
            return provider.invoke({"query": args["query"]})

    handler = _RecordingHandler()

    with tracing_context(enabled=False):
        await _in_a_traced_node(
            handler,
            tool_map={"sync_product": _SyncProduct()},
            call=_call("sync_product", query="q"),
        )

    assert handler.starts_named("product_tool") == []
    assert handler.only("provider_tool").parent_run_id is not None


async def test_the_specialist_middleware_path_parents_the_provider_exactly():
    """The production entry point, not a stand-in for it.

    Every other test here calls ``execute_tool_calls`` from a traced node.
    A real turn arrives through ``ToolExecutionMiddleware.awrap_tool_call``,
    which resolves the tool map, takes the execution context and then calls the
    same executor. Ancestry is decided in that path, so the regression belongs
    on it.
    """
    provider = _TracedProvider(name="provider_tool")

    async def _product(query: str) -> str:
        return await provider.ainvoke({"query": query})

    product = StructuredTool.from_function(
        coroutine=_product, name="product_tool", description="Calls one provider."
    )
    scope = SpecialistToolScope(
        agent=None,
        agent_key="chat",
        conversation_id=CONVERSATION_ID,
        user_id="user-1",
        device_id="device-1",
    )
    scope.offer([product])
    middleware = ToolExecutionMiddleware(scope=scope, tool_factory=_no_tools)
    request = SimpleNamespace(
        tool_call={"id": "call-1", "name": "product_tool", "args": {"query": "q"}},
        tool=SimpleNamespace(name="product_tool", metadata={}),
        state={},
        runtime=SimpleNamespace(context=None, config=None),
    )
    handler = _RecordingHandler()

    async def _framework_handler(_request: Any) -> Any:
        raise AssertionError("the product pipeline must run the tool, not the framework")

    async def node(_value: Any, config: dict[str, Any]) -> Any:
        # The real ToolRuntime carries the tools node's own config. Handing the
        # double the genuine article is what makes this test able to fail if
        # someone starts forwarding it into the executor.
        request.runtime.config = config
        return await middleware.awrap_tool_call(request, _framework_handler)

    with tracing_context(enabled=False):
        message = await RunnableLambda(node, name="tools").ainvoke(
            "go", config={"callbacks": [handler]}
        )

    assert message.status == "success"
    assert handler.only("provider_tool").parent_run_id == handler.only("product_tool").run_id


async def _no_tools() -> list[Any]:
    return []


async def test_every_retry_attempt_reaches_the_provider_beneath_its_own_run():
    """Each attempt is a fresh task, and each carries the ancestry with it."""
    provider = _TracedProvider(name="provider_tool")
    attempts = 0

    async def _product(query: str) -> str:
        nonlocal attempts
        attempts += 1
        result = await provider.ainvoke({"query": query})
        if attempts == 1:
            raise ConnectionError("connection reset")
        return result

    product = StructuredTool.from_function(
        coroutine=_product,
        name="product_tool",
        description="Fails once, then succeeds.",
        metadata={"application_execution_policy": {"max_attempts": 2, "retry_safe": True}},
    )
    handler = _RecordingHandler()

    with tracing_context(enabled=False):
        await _in_a_traced_node(
            handler, tool_map={"product_tool": product}, call=_call("product_tool", query="q")
        )

    assert attempts == 2
    product_runs = {start.run_id for start in handler.starts_named("product_tool")}
    provider_parents = {start.parent_run_id for start in handler.starts_named("provider_tool")}
    assert len(product_runs) == 2
    assert provider_parents == product_runs


# --- The three product web tools -------------------------------------------


@pytest.fixture(autouse=True)
def _clean_tool_state(monkeypatch):
    from app.ai import web_tools

    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        web_tools.settings, "remote_image_enrichment_enabled", True, raising=False
    )
    monkeypatch.setattr(web_tools.settings, "inline_rich_response_enabled", True, raising=False)
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


async def _run_product_tool(handler: _RecordingHandler, tool: Any, **args: Any) -> Any:
    with (
        tool_execution_context(conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"),
        selected_image_sink(),
        tracing_context(enabled=False),
    ):
        return await _in_a_traced_node(
            handler,
            tool_map={tool.name: tool},
            call=_call(tool.name, **args),
        )


def _search_payload() -> str:
    return json.dumps(
        {
            "results": [
                {
                    "title": "Aurora 4.2 release notes",
                    "url": "https://vendor.example/aurora/4.2",
                    "content": "Aurora 4.2 shipped on 14 March 2026.",
                    "score": 0.9,
                }
            ]
        }
    )


def _extract_payload() -> str:
    return json.dumps(
        {
            "results": [
                {
                    "url": "https://example.com/a",
                    "raw_content": "Aurora 4.2 was released on 14 March 2026.",
                }
            ],
            "failed_results": [],
        }
    )


async def test_web_search_parents_its_provider_call():
    provider = _TracedProvider(name="tavily_search", payload=_search_payload())
    handler = _RecordingHandler()

    await _run_product_tool(
        handler,
        create_web_search_tool(tavily_tool=provider),
        query="aurora release notes",
        objective="Find the stated release date",
    )

    assert handler.only("tavily_search").parent_run_id == handler.only("web_search").run_id


async def test_web_open_parents_its_provider_call():
    provider = _TracedProvider(name="tavily_extract", payload=_extract_payload())
    handler = _RecordingHandler()

    await _run_product_tool(
        handler,
        create_web_open_tool(extract_tool=provider),
        urls=["https://example.com/a"],
        question="Which release date is stated?",
    )

    assert handler.only("tavily_extract").parent_run_id == handler.only("web_open").run_id


async def test_image_search_parents_its_provider_call():
    provider = _TracedProvider(name="brave_image_search", payload=json.dumps({"results": []}))
    handler = _RecordingHandler()

    await _run_product_tool(
        handler,
        create_image_search_tool(brave_tool=provider),
        query="aurora instrument cluster",
    )

    assert (
        handler.only("brave_image_search").parent_run_id == handler.only("image_search").run_id
    )


async def test_a_turn_using_all_three_product_tools_creates_no_provider_root():
    """The offline half of the live LangSmith canary."""
    handler = _RecordingHandler()
    tools = [
        (
            create_web_search_tool(
                tavily_tool=_TracedProvider(name="tavily_search", payload=_search_payload())
            ),
            {"query": "aurora release notes", "objective": "Find the stated release date"},
        ),
        (
            create_web_open_tool(
                extract_tool=_TracedProvider(name="tavily_extract", payload=_extract_payload())
            ),
            {"urls": ["https://example.com/a"], "question": "Which release date is stated?"},
        ),
        (
            create_image_search_tool(
                brave_tool=_TracedProvider(
                    name="brave_image_search", payload=json.dumps({"results": []})
                )
            ),
            {"query": "aurora instrument cluster"},
        ),
    ]

    for tool, args in tools:
        await _run_product_tool(handler, tool, **args)

    provider_names = {"tavily_search", "tavily_extract", "brave_image_search"}
    observed = {start.name for start in handler.starts}
    assert provider_names <= observed
    assert [start.name for start in handler.starts if start.parent_run_id is None] == []
