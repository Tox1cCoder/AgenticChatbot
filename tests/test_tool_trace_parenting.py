"""Nested provider calls remain children of their product tool trace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from langsmith.run_helpers import tracing_context

from app.ai.tool_execution import execute_tool_calls


@dataclass(frozen=True)
class _Start:
    name: str
    run_id: UUID
    parent_run_id: UUID | None


class _RecordingHandler(BaseCallbackHandler):
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

    def only(self, name: str) -> _Start:
        matches = [start for start in self.starts if start.name == name]
        assert len(matches) == 1
        return matches[0]


class _Provider(BaseTool):
    name: str = "provider_tool"
    description: str = "A traced provider double."

    def _run(self, **kwargs: Any) -> str:
        return "ok"

    async def _arun(self, **kwargs: Any) -> str:
        return "ok"


@pytest.mark.asyncio
async def test_nested_provider_is_child_of_product_tool_without_langsmith() -> None:
    provider = _Provider()

    async def product(query: str) -> str:
        return await provider.ainvoke({"query": query})

    tool = StructuredTool.from_function(
        coroutine=product,
        name="product_tool",
        description="Calls one provider.",
    )
    handler = _RecordingHandler()

    async def node(_value: Any, config: dict[str, Any]) -> Any:
        return await execute_tool_calls(
            tool_calls=[{"id": "c1", "name": "product_tool", "args": {"query": "q"}}],
            tool_map={"product_tool": tool},
        )

    with tracing_context(enabled=False):
        await RunnableLambda(node, name="tool_node").ainvoke(
            "go", config={"callbacks": [handler]}
        )

    assert handler.only("provider_tool").parent_run_id == handler.only("product_tool").run_id


def test_suite_does_not_trace() -> None:
    from langsmith.utils import tracing_is_enabled

    assert tracing_is_enabled() is False
