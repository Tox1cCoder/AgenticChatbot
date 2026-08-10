from __future__ import annotations

import asyncio
import inspect
import json

import pytest
from pydantic import ValidationError

from app.ai.research_budget import reset_research_budget
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.web_research_tool import create_web_research_tool

CONVERSATION_ID = "11111111-1111-1111-1111-111111111111"

TAVILY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "index": 1,
                "title": "LoL: T1 completed 2026 LCK roster",
                "url": "https://sheepesports.example/t1",
                "content": "T1 finalized its roster.",
                "score": 0.88,
            }
        ],
        "total_results": 1,
        "answer": "T1 is a South Korean esports organization.",
        "provider": "tavily",
        "operation": "search",
        "query": "T1 roster 2026",
    }
)


class _FakeTool:
    def __init__(self, name: str, payload: str, delay: float = 0.0):
        self.name = name
        self.payload = payload
        self.delay = delay
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.payload


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        "app.ai.web_research_tool.settings.remote_image_enrichment_enabled",
        True,
        raising=False,
    )
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


def _brave_payload(*, count: int = 1, confidence: str = "high") -> str:
    return json.dumps(
        {
            "query": "T1 team photo",
            "provider": "brave_image_search",
            "images": [
                {
                    "url": f"https://imgs.search.brave.com/display-{index}.jpg",
                    "original_image_url": f"https://origin.example/team-{index}.jpg",
                    "thumbnail_url": f"https://imgs.search.brave.com/thumb-{index}.jpg",
                    "confidence": confidence,
                    "result_rank": index,
                    "provider": "brave_image_search",
                    "mime_type": "image/jpeg",
                    "title": f"T1 roster {index}",
                    "description": f"T1 roster {index}",
                    "width": 995,
                    "height": 565,
                    "source_url": "https://sheepesports.example/t1",
                }
                for index in range(count)
            ],
            "total_results": count,
        }
    )


def _tool(tavily: _FakeTool | None, brave: _FakeTool | None):
    return create_web_research_tool(tavily_tool=tavily, brave_tool=brave)


async def _run(tool, **kwargs):
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), selected_image_sink() as sink:
        raw = await tool.ainvoke(kwargs)
    return json.loads(raw), sink


@pytest.mark.asyncio
async def test_web_research_offers_provider_selected_images_without_a_verifier():
    brave = _FakeTool("brave_image_search", _brave_payload(confidence="high"))
    tool = create_web_research_tool(
        tavily_tool=_FakeTool("tavily_search", TAVILY_PAYLOAD),
        brave_tool=brave,
    )

    with selected_image_sink() as sink:
        await tool.ainvoke({"query": "T1 roster", "image_query": "T1 team photo"})

    assert sink
    assert brave.calls == [{"query": "T1 team photo"}]


def test_web_research_has_no_verifier_dependencies():
    parameters = inspect.signature(create_web_research_tool).parameters

    assert "verifier_model" not in parameters
    assert "web_image_service" not in parameters
    assert "recorder" not in parameters


@pytest.mark.asyncio
async def test_gallery_intent_returns_one_grid_item_holding_every_selected_image():
    _, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", _brave_payload(count=4)),
        ),
        query="T1 roster 2026",
        image_query="T1 team photo",
        image_intent="gallery",
    )

    assert len(sink) == 1
    assert sink[0]["type"] == "image_group"
    assert len(sink[0]["payload"]["items"]) == 4


@pytest.mark.asyncio
async def test_figure_intent_caps_at_two_selected_images():
    _, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", _brave_payload(count=4)),
        ),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert len(sink) == 2
    assert all(item["type"] == "image" for item in sink)


@pytest.mark.asyncio
async def test_missing_image_query_uses_the_factual_query_for_images():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)
    brave = _FakeTool("brave_image_search", _brave_payload())

    _, sink = await _run(_tool(tavily, brave), query="cho t thông tin về T1")

    assert brave.calls == [{"query": "cho t thông tin về T1"}]
    assert sink


@pytest.mark.asyncio
async def test_skip_images_prevents_the_brave_call():
    brave = _FakeTool("brave_image_search", _brave_payload())

    payload, sink = await _run(
        _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), brave),
        query="explain big-O notation",
        skip_images=True,
    )

    assert brave.calls == []
    assert sink == []
    assert payload["results"]


@pytest.mark.asyncio
async def test_news_topic_and_time_range_reach_tavily():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)

    await _run(
        _tool(tavily, None),
        query="latest T1 match",
        topic="news",
        time_range="week",
        skip_images=True,
    )

    assert tavily.calls == [
        {
            "query": "latest T1 match",
            "topic": "news",
            "time_range": "week",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [("topic", "sports"), ("time_range", "quarter")],
)
async def test_unsupported_tavily_controls_are_rejected_before_the_provider(field, value):
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)

    with pytest.raises(ValidationError):
        await _run(
            _tool(tavily, None),
            query="latest T1 match",
            skip_images=True,
            **{field: value},
        )

    assert tavily.calls == []


@pytest.mark.asyncio
async def test_tavily_error_is_not_reported_as_successful_research():
    tavily = _FakeTool(
        "tavily_search",
        json.dumps(
            {
                "provider": "tavily",
                "operation": "search",
                "error": "rate limited",
                "retryable": True,
            }
        ),
    )

    payload, _ = await _run(
        _tool(tavily, None),
        query="latest T1 match",
        skip_images=True,
    )

    assert payload["status"] == "error"
    assert payload["retryable"] is True
    assert "research" not in payload


@pytest.mark.asyncio
async def test_providers_run_concurrently():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD, delay=0.3)
    brave = _FakeTool("brave_image_search", _brave_payload(), delay=0.3)

    started = asyncio.get_running_loop().time()
    await _run(_tool(tavily, brave), query="T1 roster 2026", image_query="T1 team photo")
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.55, "tavily and brave must overlap"


@pytest.mark.asyncio
async def test_near_duplicate_query_reuses_the_first_result():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)
    tool = _tool(tavily, _FakeTool("brave_image_search", _brave_payload()))

    await _run(tool, query="T1 League of Legends Esports team news roster 2026")
    payload, _ = await _run(tool, query="T1 League of Legends team overview roster news 2026")

    assert len(tavily.calls) == 1
    assert payload["research"]["reused"] is True


@pytest.mark.asyncio
async def test_second_image_query_reuses_the_first_selected_images():
    brave = _FakeTool("brave_image_search", _brave_payload())
    tool = _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), brave)

    _, first = await _run(tool, query="T1 roster 2026", image_query="T1 team photo")
    _, second = await _run(tool, query="T1 sponsors 2026", image_query="T1 jersey photo")

    assert len(brave.calls) == 1
    assert [item["id"] for item in second] == [item["id"] for item in first]


@pytest.mark.asyncio
async def test_brave_failure_yields_a_normal_text_answer():
    class _Broken:
        name = "brave_image_search"

        async def ainvoke(self, args):
            raise RuntimeError("brave down")

    payload, sink = await _run(
        _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), _Broken()),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert sink == []
    assert payload["results"]


@pytest.mark.asyncio
async def test_tavily_failure_is_reported_as_a_research_error():
    class _Broken:
        name = "tavily_search"

        async def ainvoke(self, args):
            raise RuntimeError("tavily down")

    payload, _ = await _run(
        _tool(_Broken(), _FakeTool("brave_image_search", _brave_payload())),
        query="T1 roster 2026",
    )

    assert payload["status"] == "error"
    assert payload["retryable"] is True


@pytest.mark.asyncio
async def test_tavily_failure_cancels_a_live_image_task_cleanly():
    class _FailingTavily:
        name = "tavily_search"

        async def ainvoke(self, args):
            raise RuntimeError("tavily down")

    class _SlowBrave:
        name = "brave_image_search"

        async def ainvoke(self, args):
            await asyncio.sleep(0.2)
            return _brave_payload()

    tasks_before = asyncio.all_tasks()
    payload, sink = await _run(
        _tool(_FailingTavily(), _SlowBrave()),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert payload["status"] == "error"
    assert sink == []

    leftover = asyncio.all_tasks() - tasks_before - {asyncio.current_task()}
    if leftover:
        await asyncio.gather(*leftover, return_exceptions=True)
    assert all(task.done() for task in leftover)
    assert all(task.cancelled() for task in leftover)


@pytest.mark.asyncio
async def test_disabled_remote_image_flag_skips_the_image_path_entirely(monkeypatch):
    monkeypatch.setattr(
        "app.ai.web_research_tool.settings.remote_image_enrichment_enabled",
        False,
        raising=False,
    )
    brave = _FakeTool("brave_image_search", _brave_payload())

    _, sink = await _run(
        _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), brave),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert brave.calls == []
    assert sink == []


def test_tool_identity_is_internal():
    tool = _tool(None, None)

    assert tool.name == "web_research"
    assert tool.metadata["qualified_tool_id"] == "internal::web_research"


@pytest.mark.asyncio
async def test_a_request_without_the_rich_capability_skips_the_image_path():
    brave = _FakeTool("brave_image_search", _brave_payload())
    tool = _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), brave)

    with tool_execution_context(
        conversation_id=CONVERSATION_ID,
        user_id="u1",
        agent_key="search",
        rich_response_capable=False,
    ), selected_image_sink() as sink:
        raw = await tool.ainvoke(
            {"query": "T1 roster 2026", "image_query": "T1 team photo"}
        )

    assert brave.calls == []
    assert sink == []
    assert json.loads(raw)["results"], "the text answer must be unaffected"


@pytest.mark.asyncio
async def test_an_unstated_capability_keeps_the_image_path_open():
    brave = _FakeTool("brave_image_search", _brave_payload())

    _, sink = await _run(
        _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), brave),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert len(brave.calls) == 1
    assert len(sink) == 1
