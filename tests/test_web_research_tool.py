from __future__ import annotations

import asyncio
import json
import re
from unittest.mock import Mock

import pytest

from app.ai import web_research_tool as wrt
from app.ai.research_budget import reset_research_budget
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.verified_image_sink import verified_image_sink
from app.ai.visual_verifier import VisualCandidateDecision, VisualVerificationResult
from app.ai.web_research_tool import create_web_research_tool
from app.services.web_image_service import FetchedWebImage

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

BRAVE_PAYLOAD = json.dumps(
    {
        "query": "T1 League of Legends team photo",
        "provider": "brave_image_search",
        "images": [
            {
                "url": "https://cdn.example/portrait.jpg",
                "provider": "brave_image_search",
                "mime_type": "image/jpeg",
                "title": "Moi",
                "description": "Moi",
                "width": 1080,
                "height": 1600,
                "source_url": "https://sheepesports.example/t1",
            },
            {
                "url": "https://cdn.example/team.jpg",
                "provider": "brave_image_search",
                "mime_type": "image/jpeg",
                "title": "T1 roster",
                "description": "T1 roster",
                "width": 995,
                "height": 565,
                "source_url": "https://sheepesports.example/t1",
            },
        ],
        "total_results": 2,
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


class _FakeImageService:
    def __init__(self):
        self.fetched: list[str] = []

    async def fetch_url(self, url: str, *, provider: str = "other") -> FetchedWebImage:
        self.fetched.append(url)
        return FetchedWebImage(
            content=b"bytes", media_type="image/jpeg", width=995, height=565
        )


class _ApproveOnlyTeamPhoto:
    """Approves whichever candidate line mentions the team, rejects the portrait."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        text = messages[0].content[0]["text"]
        decisions = []
        # Match only candidate lines; the prompt's instruction bullets also
        # start with "- " and must not be read as candidate ids.
        for candidate_id, line in re.findall(r"^- (c\d+): (.*)$", text, flags=re.MULTILINE):
            is_portrait = "Moi" in line
            decisions.append(
                VisualCandidateDecision(
                    candidate_id=candidate_id,
                    depicts_requested_subject=not is_portrait,
                    materially_supports_answer=not is_portrait,
                    confidence=0.95,
                    content_kind="portrait" if is_portrait else "photo",
                )
            )
        return VisualVerificationResult(decisions=decisions)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        "app.ai.web_research_tool.settings.vision_image_verification_enabled",
        True,
        raising=False,
    )
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


def _tool(tavily, brave, service, verifier):
    return create_web_research_tool(
        tavily_tool=tavily,
        brave_tool=brave,
        web_image_service=service,
        verifier_model=verifier,
    )


async def _run(tool, **kwargs):
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), verified_image_sink() as sink:
        raw = await tool.ainvoke(kwargs)
    return json.loads(raw), sink


@pytest.mark.asyncio
async def test_only_the_verified_team_photo_is_offered():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)

    payload, sink = await _run(
        _tool(tavily, brave, _FakeImageService(), _ApproveOnlyTeamPhoto()),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
    )

    assert len(sink) == 1
    assert sink[0]["payload"]["url"] == "https://cdn.example/team.jpg"
    assert "portrait.jpg" not in json.dumps(sink)
    assert "images" not in payload
    assert "portrait.jpg" not in json.dumps(payload)
    assert payload["answer"].startswith("T1 is a South Korean")


def _brave_payload(count: int) -> str:
    """A Brave result with ``count`` distinct, verifiable team photos."""
    return json.dumps(
        {
            "query": "T1 League of Legends team photo",
            "provider": "brave_image_search",
            "images": [
                {
                    "url": f"https://cdn.example/team-{index}.jpg",
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


@pytest.mark.asyncio
async def test_gallery_intent_returns_one_grid_item_holding_every_survivor():
    verifier = _ApproveOnlyTeamPhoto()

    _, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", _brave_payload(4)),
            _FakeImageService(),
            verifier,
        ),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
        image_intent="gallery",
    )

    assert len(sink) == 1
    assert sink[0]["type"] == "image_group"
    assert len(sink[0]["payload"]["items"]) == 4
    assert verifier.calls == 1


@pytest.mark.asyncio
async def test_gallery_candidates_reach_the_verifier_individually():
    """Grouping before verification would hide images and cap discovery."""

    seen: list[str] = []

    class _Recorder:
        calls = 0

        async def ainvoke(self, messages):
            text = messages[0].content[0]["text"]
            seen.extend(re.findall(r"^- (c\d+): ", text, flags=re.MULTILINE))
            return VisualVerificationResult(decisions=[])

    await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", _brave_payload(5)),
            _FakeImageService(),
            _Recorder(),
        ),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
        image_intent="gallery",
    )

    assert len(seen) == 5, "every candidate must be judged on its own pixels"


@pytest.mark.asyncio
async def test_figure_intent_caps_at_two_individual_items():
    _, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", _brave_payload(4)),
            _FakeImageService(),
            _ApproveOnlyTeamPhoto(),
        ),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
    )

    assert len(sink) == 2
    assert all(item["type"] == "image" for item in sink)


@pytest.mark.asyncio
async def test_gallery_with_a_single_survivor_is_not_a_one_cell_grid():
    _, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", _brave_payload(1)),
            _FakeImageService(),
            _ApproveOnlyTeamPhoto(),
        ),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
        image_intent="gallery",
    )

    assert len(sink) == 1
    assert sink[0]["type"] == "image"


@pytest.mark.asyncio
async def test_no_image_query_skips_brave_and_the_verifier():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)
    verifier = _ApproveOnlyTeamPhoto()

    payload, sink = await _run(
        _tool(tavily, brave, _FakeImageService(), verifier), query="explain big-O notation"
    )

    assert brave.calls == []
    assert verifier.calls == 0
    assert sink == []
    assert payload["results"]


@pytest.mark.asyncio
async def test_providers_run_concurrently():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD, delay=0.3)
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD, delay=0.3)

    started = asyncio.get_running_loop().time()
    await _run(
        _tool(tavily, brave, _FakeImageService(), _ApproveOnlyTeamPhoto()),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.55, "tavily and brave must overlap"


@pytest.mark.asyncio
async def test_near_duplicate_query_reuses_the_first_result():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)
    tool = _tool(tavily, _FakeTool("brave_image_search", BRAVE_PAYLOAD), _FakeImageService(), None)

    await _run(tool, query="T1 League of Legends Esports team news roster 2026")
    payload, _ = await _run(tool, query="T1 League of Legends team overview roster news 2026")

    assert len(tavily.calls) == 1
    assert payload["research"]["reused"] is True


@pytest.mark.asyncio
async def test_second_image_query_does_not_launch_another_brave_call():
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)
    tool = _tool(
        _FakeTool("tavily_search", TAVILY_PAYLOAD),
        brave,
        _FakeImageService(),
        _ApproveOnlyTeamPhoto(),
    )

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
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _Broken(),
            _FakeImageService(),
            _ApproveOnlyTeamPhoto(),
        ),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert sink == []
    assert payload["results"]


@pytest.mark.asyncio
async def test_verifier_returning_nothing_yields_a_text_answer():
    class _RejectAll:
        async def ainvoke(self, messages):
            return VisualVerificationResult(decisions=[])

    _, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", BRAVE_PAYLOAD),
            _FakeImageService(),
            _RejectAll(),
        ),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert sink == []


@pytest.mark.asyncio
async def test_tavily_failure_is_reported_as_a_research_error():
    class _Broken:
        name = "tavily_search"

        async def ainvoke(self, args):
            raise RuntimeError("tavily down")

    payload, _ = await _run(
        _tool(_Broken(), _FakeTool("brave_image_search", BRAVE_PAYLOAD), _FakeImageService(), None),
        query="T1 roster 2026",
    )

    assert payload["status"] == "error"
    assert payload["retryable"] is True


@pytest.mark.asyncio
async def test_tavily_failure_cancels_a_live_image_task_cleanly():
    """The image path must be genuinely in flight, not merely absent, when the
    search fails — otherwise image_task.cancel() is never exercised at all."""

    class _FailingTavily:
        name = "tavily_search"

        async def ainvoke(self, args):
            raise RuntimeError("tavily down")

    class _SlowBrave:
        name = "brave_image_search"

        async def ainvoke(self, args):
            await asyncio.sleep(0.2)
            return BRAVE_PAYLOAD

    tasks_before = asyncio.all_tasks()

    payload, sink = await _run(
        _tool(_FailingTavily(), _SlowBrave(), _FakeImageService(), _ApproveOnlyTeamPhoto()),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert payload["status"] == "error"
    assert payload["retryable"] is True
    assert sink == []

    # image_task.cancel() only schedules cancellation; the loop must run once
    # more to unwind it. Gather (not bare-await) any leftover task so its
    # CancelledError is retrieved here rather than logged as "Task exception
    # was never retrieved" whenever the task object is later garbage collected.
    leftover = asyncio.all_tasks() - tasks_before - {asyncio.current_task()}
    if leftover:
        await asyncio.gather(*leftover, return_exceptions=True)
    assert all(task.done() for task in leftover), "image task left pending after cancel"
    assert all(task.cancelled() for task in leftover), "image task did not honor cancellation"


@pytest.mark.asyncio
async def test_outer_deadline_expiry_records_timeout_once(monkeypatch):
    """The whole-path deadline, not just the verifier's own timeout, must be
    observable: a Brave call that outlives it should show up as ``timeout``,
    not silently as no metric at all."""

    metrics = type("Metrics", (), {"record_verification_outcome": Mock()})()
    monkeypatch.setattr(wrt, "rich_image_metrics", metrics)
    monkeypatch.setattr(wrt.settings, "image_verification_deadline_seconds", 0.02, raising=False)

    class _SlowBrave:
        async def ainvoke(self, args):
            await asyncio.sleep(0.3)
            return BRAVE_PAYLOAD

    payload, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _SlowBrave(),
            _FakeImageService(),
            _ApproveOnlyTeamPhoto(),
        ),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert sink == []
    assert payload["results"]
    metrics.record_verification_outcome.assert_called_once()
    assert metrics.record_verification_outcome.call_args.kwargs["outcome"] == "timeout"


@pytest.mark.asyncio
async def test_disabled_flag_skips_the_image_path_entirely(monkeypatch):
    monkeypatch.setattr(
        "app.ai.web_research_tool.settings.vision_image_verification_enabled",
        False,
        raising=False,
    )
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)

    _, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            brave,
            _FakeImageService(),
            _ApproveOnlyTeamPhoto(),
        ),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert brave.calls == []
    assert sink == []


def test_tool_identity_is_internal():
    tool = _tool(None, None, None, None)

    assert tool.name == "web_research"
    assert tool.metadata["qualified_tool_id"] == "internal::web_research"


@pytest.mark.asyncio
async def test_a_request_without_the_rich_capability_skips_the_image_path():
    """No marker inventory reaches a non-rich answer, so the work is provably wasted.

    Injection is already gated downstream in ``graph.py``: candidates offered
    for a request that never advertised ``inline_rich_response_v1`` are
    discarded. Running Brave, the thumbnail batch and a billed vision call to
    produce them anyway costs money and up to the full image deadline.
    """

    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)
    verifier = _ApproveOnlyTeamPhoto()
    tool = _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), brave, _FakeImageService(), verifier)

    with tool_execution_context(
        conversation_id=CONVERSATION_ID,
        user_id="u1",
        agent_key="search",
        rich_response_capable=False,
    ), verified_image_sink() as sink:
        raw = await tool.ainvoke(
            {"query": "T1 roster 2026", "image_query": "T1 League of Legends team photo"}
        )

    assert brave.calls == []
    assert verifier.calls == 0
    assert sink == []
    assert json.loads(raw)["results"], "the text answer must be unaffected"


@pytest.mark.asyncio
async def test_an_unstated_capability_keeps_the_image_path_open():
    """Omission must not disable the feature.

    This gate skips provably-wasted work; it is not the correctness boundary.
    A caller that forgets to thread the flag should behave exactly as before,
    so the default is open and only an explicit False closes it.
    """

    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)

    tool = _tool(
        _FakeTool("tavily_search", TAVILY_PAYLOAD),
        brave,
        _FakeImageService(),
        _ApproveOnlyTeamPhoto(),
    )

    _, sink = await _run(
        tool,
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
    )

    assert len(brave.calls) == 1
    assert len(sink) == 1
