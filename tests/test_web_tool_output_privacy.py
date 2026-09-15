from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.ai.tool_context import tool_execution_context
from app.ai.web_tools import create_web_search_tool


@pytest.mark.asyncio
async def test_public_web_tool_output_contains_no_candidate_descriptor() -> None:
    bundle = SimpleNamespace(
        status="success",
        mode="quick",
        operation_index=1,
        sources=(),
        images=(
            SimpleNamespace(
                candidate_id="I1",
                delivery_url="/web-images/private",
                digest="deadbeef",
            ),
        ),
        failures=(),
        reused=False,
        omitted_source_count=0,
        omitted_image_count=0,
    )

    class Session:
        mode = "quick"
        budget = SimpleNamespace(search_calls=1)

        async def search(self, request):
            return bundle

    with tool_execution_context(web_research_session=Session()):
        raw = await create_web_search_tool().ainvoke(
            {"query": "subject", "objective": "verify subject"}
        )

    public = json.loads(raw)
    assert "images" not in public
    assert "I1" not in raw
    assert "/web-images/" not in raw
    assert "deadbeef" not in raw
