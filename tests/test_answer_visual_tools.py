"""The server-owned ``create_answer_visual`` tool and where it is bound."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.ai.answer_visual_tools import (
    CREATE_ANSWER_VISUAL_TOOL_NAME,
    create_answer_visual_tool,
)
from app.ai.tool_context import tool_execution_context
from app.ai.tool_execution import execute_tool_calls
from app.ai.web_research.contracts import ProviderSource
from app.ai.web_research.source_registry import SourceRegistry
from app.services.ai_service import AIService

_STEPS = {
    "variant": "steps",
    "title": "Set the default search engine",
    "steps": [{"title": "Open settings"}, {"title": "Choose Google"}],
}


def _session() -> SimpleNamespace:
    registry = SourceRegistry(max_sources=8)
    registry.admit(
        (
            ProviderSource(
                provider="tavily",
                url="https://bench.example/review",
                title="Bench review",
                rank=1,
                query_index=1,
            ),
        )
    )
    return SimpleNamespace(source_registry=registry)


async def _call(args: dict, *, call_id: str = "call-v1", session=None):
    tool = create_answer_visual_tool()
    with tool_execution_context(
        conversation_id="conv-1",
        logical_turn_id="turn-1",
        web_research_session=session if session is not None else _session(),
    ):
        return await execute_tool_calls(
            tool_calls=[{"id": call_id, "name": CREATE_ANSWER_VISUAL_TOOL_NAME, "args": args}],
            tool_map={CREATE_ANSWER_VISUAL_TOOL_NAME: tool},
        )


@pytest.mark.asyncio
async def test_tool_returns_a_marker_and_keeps_the_descriptor_out_of_model_text():
    outputs, artifacts, _images = await _call({"visual": _STEPS})

    content = json.loads(outputs[0]["content"])
    [candidate] = artifacts[0]["_rich_item_candidates"]
    assert content["status"] == "ok"
    assert content["marker"] == f"<!--rich:{candidate['id']}-->"
    assert "fallback_markdown" not in outputs[0]["content"]
    assert candidate["type"] == "answer_visual"
    assert candidate["payload"]["fallback_markdown"].startswith("**Set the default")


@pytest.mark.asyncio
async def test_tool_resolves_web_evidence_from_the_turn_session():
    visual = {
        "variant": "bars",
        "title": "FPS",
        "unit": "FPS",
        "basis": "measured",
        "bars": [{"label": "Stock", "value": 100, "evidence": [{"kind": "web", "id": "S1"}]}],
    }

    _outputs, artifacts, _images = await _call({"visual": visual})

    [candidate] = artifacts[0]["_rich_item_candidates"]
    assert candidate["payload"]["evidence"][0]["url"] == "https://bench.example/review"
    assert candidate["_internal"]["web_source_ids"] == ["S1"]


@pytest.mark.asyncio
async def test_unknown_evidence_returns_a_retryable_error_and_no_candidate():
    visual = {
        "variant": "bars",
        "title": "FPS",
        "unit": "FPS",
        "basis": "measured",
        "bars": [{"label": "Stock", "value": 100, "evidence": [{"kind": "web", "id": "S7"}]}],
    }

    outputs, artifacts, _images = await _call({"visual": visual})

    content = json.loads(outputs[0]["content"])
    assert content["status"] == "error"
    assert content["retryable"] is True
    assert "S7" in content["message"]
    assert "_rich_item_candidates" not in artifacts[0]


@pytest.mark.asyncio
async def test_invalid_shape_returns_a_concise_retryable_error():
    outputs, artifacts, _images = await _call(
        {"visual": {**_STEPS, "steps": [{"title": f"s{n}"} for n in range(13)]}}
    )

    content = json.loads(outputs[0]["content"])
    assert content["status"] == "error"
    assert content["retryable"] is True
    assert "_rich_item_candidates" not in artifacts[0]


@pytest.mark.asyncio
async def test_replayed_call_reuses_its_id():
    _o1, first, _i1 = await _call({"visual": _STEPS}, call_id="call-a")
    _o2, second, _i2 = await _call({"visual": _STEPS}, call_id="call-b")

    assert (
        first[0]["_rich_item_candidates"][0]["id"] == second[0]["_rich_item_candidates"][0]["id"]
    )


def test_tool_end_stream_never_publishes_a_visual_or_a_generic_render():
    model_text = json.dumps({"status": "ok", "rich_item_id": "visual:x", "marker": "m"})

    items = AIService._build_tool_end_rich_items(
        render={"version": 1, "type": "table", "title": "x"},
        result=model_text,
        tool_call_id="call-v1",
        tool_name=CREATE_ANSWER_VISUAL_TOOL_NAME,
    )

    assert items == []


# ---------------------------------------------------------------------------
# Evidence cited only inside a visual
# ---------------------------------------------------------------------------


def _bars_candidate(item_id: str = "visual:abc") -> dict:
    from app.core.answer_visuals import ResolvedEvidence, build_answer_visual

    item = build_answer_visual(
        {
            "variant": "bars",
            "title": "FPS",
            "unit": "FPS",
            "basis": "measured",
            "bars": [{"label": "Stock", "value": 100, "evidence": [{"kind": "web", "id": "S1"}]}],
        },
        resolve_evidence=lambda ref: ResolvedEvidence(
            title="Bench review", url="https://bench.example/review"
        ),
        turn_key="turn-1",
    )
    item["id"] = item_id
    return item


class _PublishingSession:
    def __init__(self):
        self.source_registry = _session().source_registry

    def published_sources(self, cited_source_ids=()):
        return tuple(
            record
            for record in self.source_registry.records
            if record.source_id in set(cited_source_ids)
        )


def _outcome_with_visual(text: str, candidate: dict):
    from langchain_core.messages import AIMessage

    from app.ai.schemas import AgentType
    from app.ai.workflow.specialists import SpecialistFactory

    factory = SpecialistFactory.__new__(SpecialistFactory)
    return factory._to_outcome(
        SimpleNamespace(agent_type=AgentType.SEARCH, output_policy_ids=("public_content",)),
        SimpleNamespace(agent_id="search_agent", extras={}, messages=[], conversation_id="c"),
        [AIMessage(content=text)],
        SimpleNamespace(
            artifacts=[{"tool_call_id": "v1", "_rich_item_candidates": [candidate]}], images=[]
        ),
        web_research_session=_PublishingSession(),
    )


def test_a_source_cited_only_inside_a_referenced_visual_is_published():
    outcome = _outcome_with_visual("Results:\n\n<!--rich:visual:abc-->", _bars_candidate())

    assert [record["source_id"] for record in outcome.provenance.web_sources] == ["S1"]
    assert outcome.response.metadata["web_sources"][0]["source_id"] == "S1"


def test_an_unreferenced_visual_publishes_no_source():
    outcome = _outcome_with_visual("No visual here.", _bars_candidate())

    assert outcome.provenance.web_sources == ()


@pytest.mark.asyncio
async def test_visual_citations_satisfy_the_web_evidence_policy():
    from app.ai.workflow.finalization import WebEvidencePolicy

    outcome = _outcome_with_visual("Results:\n\n<!--rich:visual:abc-->", _bars_candidate())

    await WebEvidencePolicy().validate(outcome, {})


@pytest.mark.asyncio
async def test_an_unreferenced_visual_does_not_satisfy_the_web_evidence_policy():
    from app.ai.workflow.finalization import OutputValidationError, WebEvidencePolicy

    outcome = _outcome_with_visual("No visual here.", _bars_candidate())
    outcome = outcome.model_copy(
        update={
            "provenance": outcome.provenance.model_copy(
                update={"web_sources": ({"source_id": "S1", "url": "https://bench.example/review"},)}
            )
        }
    )

    with pytest.raises(OutputValidationError):
        await WebEvidencePolicy().validate(outcome, {})


def test_finalization_admits_at_most_six_visuals_per_answer(monkeypatch):
    from app.core.config import settings
    from app.core.response_constants import build_bot_metadata
    from app.schemas.workflow import WorkflowResponse, WorkflowResponseMessage

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    candidates = [_bars_candidate(f"visual:v{n}") for n in range(7)]
    content = "\n\n".join(f"<!--rich:visual:v{n}-->" for n in range(7))

    metadata = build_bot_metadata(
        WorkflowResponse(
            message=WorkflowResponseMessage(content=content),
            metadata={"_rich_item_candidates": candidates},
        )
    )

    assert [item["id"] for item in metadata["rich_items"]] == [f"visual:v{n}" for n in range(6)]
    assert {"code": "answer_visual_limit", "id": "visual:v6"} in metadata[
        "rich_reference_warnings"
    ]
    assert all("_internal" not in item for item in metadata["rich_items"])


def test_finalization_bounds_total_visual_bytes(monkeypatch):
    import app.core.answer_visuals as answer_visuals
    from app.core.config import settings
    from app.core.response_constants import build_bot_metadata
    from app.schemas.workflow import WorkflowResponse, WorkflowResponseMessage

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    [single] = build_bot_metadata(
        WorkflowResponse(
            message=WorkflowResponseMessage(content="<!--rich:visual:a-->"),
            metadata={"_rich_item_candidates": [_bars_candidate("visual:a")]},
        )
    )["rich_items"]
    one = len(json.dumps(single, ensure_ascii=False).encode())
    monkeypatch.setattr(answer_visuals, "ANSWER_VISUAL_MAX_TOTAL_BYTES", one + 10)

    metadata = build_bot_metadata(
        WorkflowResponse(
            message=WorkflowResponseMessage(content="<!--rich:visual:a-->\n<!--rich:visual:b-->"),
            metadata={
                "_rich_item_candidates": [
                    _bars_candidate("visual:a"),
                    _bars_candidate("visual:b"),
                ]
            },
        )
    )

    assert [item["id"] for item in metadata["rich_items"]] == ["visual:a"]


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def _gate(monkeypatch, *, rich: bool = True, visuals: bool = True):
    from app.core.config import settings

    monkeypatch.setattr(settings, "inline_rich_response_enabled", rich)
    monkeypatch.setattr(settings, "answer_visuals_enabled", visuals)


def _bound(
    model_config_key: str = "chat",
    *,
    capable: bool = True,
    visuals: bool = True,
    worker: bool = False,
) -> list[str]:
    from app.ai.workflow.specialists import with_answer_authoring_tools

    definition = SimpleNamespace(model_config_key=model_config_key)
    request = SimpleNamespace(
        state={"context": {"inline_rich_response_v1": capable, "answer_visuals_v1": visuals}},
        extras={"worker": True} if worker else {},
    )
    tools = with_answer_authoring_tools([SimpleNamespace(name="web_search")], definition, request)
    return [tool.name for tool in tools]


@pytest.mark.parametrize("key", ["chat", "search"])
def test_public_chat_and_search_answers_bind_the_tool(monkeypatch, key):
    _gate(monkeypatch)

    assert _bound(key) == ["web_search", CREATE_ANSWER_VISUAL_TOOL_NAME]


@pytest.mark.parametrize("key", ["rag", "canvas", "custom", "image_generator", "planning"])
def test_other_agents_do_not_bind_the_tool(monkeypatch, key):
    _gate(monkeypatch)

    assert CREATE_ANSWER_VISUAL_TOOL_NAME not in _bound(key)


def test_private_planning_workers_do_not_bind_the_tool(monkeypatch):
    _gate(monkeypatch)

    assert CREATE_ANSWER_VISUAL_TOOL_NAME not in _bound(worker=True)


def test_rollout_gate_off_does_not_bind_the_tool(monkeypatch):
    _gate(monkeypatch, visuals=False)

    assert CREATE_ANSWER_VISUAL_TOOL_NAME not in _bound()


def test_inline_rich_kill_switch_does_not_bind_the_tool(monkeypatch):
    _gate(monkeypatch, rich=False)

    assert CREATE_ANSWER_VISUAL_TOOL_NAME not in _bound()


def test_non_rich_request_does_not_bind_the_tool(monkeypatch):
    _gate(monkeypatch)

    assert CREATE_ANSWER_VISUAL_TOOL_NAME not in _bound(capable=False)


def test_client_without_visual_capability_does_not_bind_the_tool(monkeypatch):
    """A live stream cannot substitute a fallback it has never seen, so a
    visual is only authored for a client that will render it."""
    _gate(monkeypatch)

    assert CREATE_ANSWER_VISUAL_TOOL_NAME not in _bound(visuals=False)


def test_binding_does_not_duplicate_an_existing_tool(monkeypatch):
    from app.ai.workflow.specialists import with_answer_authoring_tools

    _gate(monkeypatch)
    existing = [SimpleNamespace(name=CREATE_ANSWER_VISUAL_TOOL_NAME)]
    request = SimpleNamespace(
        state={"context": {"inline_rich_response_v1": True, "answer_visuals_v1": True}},
        extras={},
    )

    tools = with_answer_authoring_tools(
        existing, SimpleNamespace(model_config_key="chat"), request
    )

    assert [tool.name for tool in tools] == [CREATE_ANSWER_VISUAL_TOOL_NAME]
