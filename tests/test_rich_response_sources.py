"""Tests for the candidate-registration helpers introduced in response_format.md
Task 3 (image candidates from tool results, tool_render candidates, widget
deduplication, RAG document image candidates).
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from app.ai.mcp_servers import tavily_server
from app.ai.tool_execution import (
    build_image_candidates_from_tool_result,
    build_live_widget_candidate_from_tool_result,
    build_tool_render_candidate,
    extract_images_from_tool_result,
)

# ---------------------------------------------------------------------------
# Tavily-style image candidates
# ---------------------------------------------------------------------------


def _tavily_payload():
    return json.dumps(
        {
            "images": [
                {
                    "url": "https://img.test/photo-1.png",
                    "description": "An airfoil cross section",
                },
                {
                    "url": "https://img.test/photo-2.jpg",
                    "description": "Streamlines around the wing",
                },
            ]
        }
    )


def test_image_candidates_carry_stable_ids_and_provenance():
    payload = _tavily_payload()
    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="call_7", tool_name="tavily_search"
    )
    ids = [c["id"] for c in candidates]
    assert ids == ["image:tool:call_7:0", "image:tool:call_7:1"]
    for candidate in candidates:
        assert candidate["type"] == "image"
        assert candidate["display_policy"] == "inline_only"
        assert candidate["source"] == "web_search"
        assert candidate["provenance"]["tool_call_id"] == "call_7"
        assert candidate["provenance"]["tool"] == "tavily_search"
        assert "url" in candidate["payload"]


def test_image_candidate_inventory_view_excludes_base64():
    inline_data_payload = json.dumps(
        {
            "images": [
                {
                    "data": "QUJDRA==",
                    "mime_type": "image/png",
                    "description": "Inline raster",
                }
            ]
        }
    )
    candidates = build_image_candidates_from_tool_result(
        inline_data_payload, tool_call_id="call_inline", tool_name="image_search"
    )
    # The candidate may include the data payload for selection, but the
    # bounded summary line returned for inventory must not include it.
    from app.core.rich_response import build_rich_item_inventory_block

    block = build_rich_item_inventory_block(
        candidates, max_items=5, max_chars=2400, summary_chars=180
    )
    assert "QUJDRA==" not in block


def test_legacy_extract_images_from_tool_result_still_returns_url_dicts():
    images = extract_images_from_tool_result(_tavily_payload())
    assert images and images[0]["url"].startswith("https://img.test/")
    assert "description" in images[0]


def test_image_candidates_skip_entries_missing_url_and_data():
    payload = json.dumps(
        {
            "images": [
                {"description": "no source"},
                {"url": "https://img.test/good.png"},
            ]
        }
    )
    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="call_1", tool_name="tavily_search"
    )
    assert len(candidates) == 1
    assert candidates[0]["payload"]["url"] == "https://img.test/good.png"


def test_image_candidates_default_mime_when_absent():
    payload = json.dumps(
        {
            "images": [
                {"url": "https://img.test/photo.png", "description": "A png"},
                {"url": "https://img.test/photo.jpg", "description": "A jpeg"},
            ]
        }
    )
    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="call_2", tool_name="tavily_search"
    )
    assert candidates[0]["payload"]["mime_type"] == "image/png"
    assert candidates[1]["payload"]["mime_type"] == "image/jpeg"


def test_candidate_provenance_keeps_source_title_and_query():
    payload = json.dumps(
        {
            "provider": "tavily",
            "query": "apple park cupertino aerial",
            "images": [
                {
                    "url": "https://cdn.example.com/park.jpg",
                    "description": "Aerial view of Apple Park",
                    "source_url": "https://example.com/apple-park",
                    "source_title": "Inside Apple Park",
                    "source_domain": "example.com",
                    "result_rank": 0,
                    "result_score": 0.91,
                    "width": 1200,
                    "height": 800,
                }
            ],
        }
    )
    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="call_1", tool_name="tavily_search"
    )
    provenance = candidates[0]["provenance"]
    assert provenance["source_title"] == "Inside Apple Park"
    assert provenance["query"] == "apple park cupertino aerial"


def test_missing_tavily_score_is_none_not_zero():
    normalized = tavily_server._normalize_search_response(
        query="q",
        response={"results": [{"url": "https://e.com/a", "title": "A", "images": ["https://e.com/i.jpg"]}]},
        include_images=True,
    )
    assert normalized["images"][0]["result_score"] is None


# ---------------------------------------------------------------------------
# Brave Image Search candidates (image_search.md Phase 6)
# ---------------------------------------------------------------------------


def _brave_payload():
    return json.dumps(
        {
            "query": "spain architecture",
            "provider": "brave_image_search",
            "images": [
                {
                    "url": "https://img.test/direct-1.jpg",
                    "thumbnail_url": "https://img.test/thumb-1.jpg",
                    "source_url": "https://example.com/page",
                    "title": "Sagrada Familia exterior",
                    "description": "Sagrada Familia exterior",
                    "mime_type": "image/jpeg",
                    "width": 1200,
                    "height": 800,
                    "source_domain": "example.com",
                    "provider": "brave_image_search",
                }
            ],
            "total_results": 1,
        }
    )


def test_brave_image_candidate_identity_and_payload():
    candidates = build_image_candidates_from_tool_result(
        _brave_payload(), tool_call_id="call_b", tool_name="brave_image_search"
    )
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["type"] == "image"
    # Provider identity lives in provenance; source may remain the generic value.
    assert cand["provenance"]["tool"] == "brave_image_search"
    assert cand["source"] == "tool_image"
    # source_url stays in the public payload (ImagePayload accepts it).
    assert cand["payload"]["url"] == "https://img.test/thumb-1.jpg"
    assert cand["payload"]["source_url"] == "https://example.com/page"
    assert cand["payload"]["width"] == 1200
    assert cand["payload"]["height"] == 800


def test_brave_image_candidate_keeps_provider_urls_in_provenance():
    [cand] = build_image_candidates_from_tool_result(
        _brave_payload(), tool_call_id="call_b", tool_name="brave_image_search"
    )
    prov = cand["provenance"]
    assert prov["thumbnail_url"] == "https://img.test/thumb-1.jpg"
    assert prov["original_image_url"] == "https://img.test/direct-1.jpg"
    assert prov["source_domain"] == "example.com"
    assert prov["provider"] == "brave_image_search"
    for forbidden in ("thumbnail_url", "original_image_url", "source_domain", "provider"):
        assert forbidden not in cand["payload"]


def test_remote_candidates_reject_insecure_duplicate_and_known_tiny_images(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "rich_image_min_width_px", 320)
    monkeypatch.setattr(settings, "rich_image_min_height_px", 180)
    payload = json.dumps(
        {
            "images": [
                {"url": "http://img.test/insecure.jpg", "width": 800, "height": 600},
                {"url": "https://img.test/tiny.jpg", "width": 100, "height": 100},
                {"url": "https://img.test/good.jpg", "width": 800, "height": 600},
                {"url": "https://img.test/good.jpg", "width": 800, "height": 600},
            ]
        }
    )

    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="call_filter", tool_name="brave_image_search"
    )

    assert [candidate["payload"]["url"] for candidate in candidates] == [
        "https://img.test/good.jpg"
    ]


def test_remote_candidates_obey_configured_candidate_cap(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "rich_image_candidate_max_count", 2)
    payload = json.dumps(
        {
            "images": [
                {"url": f"https://img.test/{index}.jpg", "width": 800, "height": 600}
                for index in range(4)
            ]
        }
    )

    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="call_cap", tool_name="brave_image_search"
    )

    assert len(candidates) == 2


def test_candidate_selection_records_bounded_provider_counts(monkeypatch):
    from app.ai import tool_execution

    metrics = type(
        "Metrics",
        (),
        {
            "record_discovery": Mock(),
            "record_selection": Mock(),
            "record_candidate": Mock(),
        },
    )()
    monkeypatch.setattr(tool_execution, "rich_image_metrics", metrics)

    build_image_candidates_from_tool_result(
        _brave_payload(), tool_call_id="call_metrics", tool_name="brave_image_search"
    )

    metrics.record_discovery.assert_called_once_with(provider="brave", result_count=1)
    metrics.record_selection.assert_called_once_with(provider="brave", outcome="selected")
    metrics.record_candidate.assert_called_once_with(provider="brave", outcome="eligible")


def test_brave_image_candidate_validates_against_public_schema():
    from app.core.rich_response import validate_public_rich_item

    [cand] = build_image_candidates_from_tool_result(
        _brave_payload(), tool_call_id="call_b", tool_name="brave_image_search"
    )
    # Passes the discriminated-union validation only if payload has no forbidden
    # extras and the mime/url contract holds.
    validate_public_rich_item(cand)


# ---------------------------------------------------------------------------
# Tool render candidates
# ---------------------------------------------------------------------------


def test_chart_render_creates_tool_render_candidate():
    render = {
        "version": 1,
        "type": "chart",
        "title": "Quarterly Revenue",
        "data": {"x": [1, 2, 3], "y": [10, 20, 30]},
    }
    candidate = build_tool_render_candidate(
        render, tool_call_id="call_chart", tool_name="chart_tool"
    )
    assert candidate is not None
    assert candidate["id"] == "tool:call_chart"
    assert candidate["type"] == "tool_render"
    assert candidate["display_policy"] == "inline_or_append"
    assert candidate["payload"]["render"] == render
    assert candidate["provenance"]["tool"] == "chart_tool"


def test_widget_render_is_not_duplicated_as_tool_render():
    """Live-widget tools have their own widget:<id> candidate; the static
    tool_render candidate would duplicate the inline experience."""
    render = {
        "version": 1,
        "type": "live_widget",
        "widget_id": "w-1",
    }
    candidate = build_tool_render_candidate(
        render, tool_call_id="call_widget", tool_name="widget_create"
    )
    assert candidate is None


def test_widget_result_creates_dedicated_live_widget_candidate():
    candidate = build_live_widget_candidate_from_tool_result(
        json.dumps(
            {
                "widget_id": "w-1",
                "session_id": "conv-1",
                "widget_type": "chart",
                "title": "Pressure comparison",
                "status": "active",
                "version": 1,
                "state": {"large": "not public"},
            }
        ),
        tool_name="widget_create",
    )

    assert candidate is not None
    assert candidate["id"] == "widget:w-1"
    assert candidate["type"] == "live_widget"
    assert candidate["display_policy"] == "inline_or_append"
    assert "state" not in str(candidate["payload"])


def test_error_and_text_renders_do_not_create_tool_render_candidate():
    assert (
        build_tool_render_candidate(
            {"version": 1, "type": "error", "error": "boom"},
            tool_call_id="c1",
            tool_name="something",
        )
        is None
    )
    assert (
        build_tool_render_candidate(
            {"version": 1, "type": "text", "text": "Just text"},
            tool_call_id="c2",
            tool_name="something",
        )
        is None
    )


def test_mcp_app_render_creates_tool_render_candidate():
    render = {
        "version": 1,
        "type": "mcp_app",
        "template_uri": "ui://canva/presentation-viewer.html",
    }
    candidate = build_tool_render_candidate(
        render, tool_call_id="call_app", tool_name="canva_create_presentation"
    )
    assert candidate is not None
    assert candidate["id"] == "tool:call_app"
    assert candidate["payload"]["render"]["template_uri"].startswith("ui://")


# ---------------------------------------------------------------------------
# RAG document image candidates
# ---------------------------------------------------------------------------


def test_rag_document_images_become_candidates_with_stable_ids():
    from app.ai.rag_tool_actions import register_document_image_candidates

    context: dict = {}
    added = register_document_image_candidates(
        context=context,
        images=[
            {
                "id": "img-1",
                "data": "QUJDRA==",
                "mime_type": "image/png",
                "caption": "A scatter plot",
                "page_number": 4,
            },
            {
                "id": "img-2",
                "data": "QUJDRA==",
                "mime_type": "image/jpeg",
                "caption": "A line chart",
                "page_number": 5,
            },
        ],
    )
    assert added == 2
    ids = [c["id"] for c in context["rich_item_candidates"]]
    assert ids == ["image:document:img-1", "image:document:img-2"]


def test_rag_document_image_candidates_reject_disallowed_mime():
    from app.ai.rag_tool_actions import register_document_image_candidates

    context: dict = {}
    added = register_document_image_candidates(
        context=context,
        images=[
            {
                "id": "img-svg",
                "data": "<svg/>",
                "mime_type": "image/svg+xml",
                "caption": "An SVG",
            },
        ],
    )
    assert added == 0
    assert context.get("rich_item_candidates", []) == []


def test_rag_document_image_candidates_deduplicate_by_id():
    from app.ai.rag_tool_actions import register_document_image_candidates

    context: dict = {}
    image = {
        "id": "img-1",
        "data": "QUJDRA==",
        "mime_type": "image/png",
        "caption": "A figure",
    }
    register_document_image_candidates(context=context, images=[image])
    added = register_document_image_candidates(context=context, images=[image])
    assert added == 0
    assert len(context["rich_item_candidates"]) == 1


# ---------------------------------------------------------------------------
# execute_tool_calls candidate attachment
# ---------------------------------------------------------------------------


class _TavilyStyleTool:
    name = "tavily_search"

    async def ainvoke(self, args):
        return json.dumps(
            {
                "images": [
                    {
                        "url": "https://img.test/photo-1.png",
                        "description": "Airfoil cross section",
                    }
                ]
            }
        )


class _WidgetTool:
    name = "widget_create"

    async def ainvoke(self, args):
        return json.dumps(
            {
                "widget_id": "w-created",
                "session_id": args.get("session_id", "conv-1"),
                "widget_type": "chart",
                "title": "Created chart",
                "status": "active",
                "version": 1,
                "state": {"labels": ["private-state-not-in-candidate"]},
            }
        )


@pytest.mark.asyncio
async def test_execute_tool_calls_attaches_rich_candidates_to_artifact():
    from app.ai.tool_execution import execute_tool_calls

    outputs, artifacts, _images = await execute_tool_calls(
        tool_calls=[{"id": "call_99", "name": "tavily_search", "args": {"query": "wings"}}],
        tool_map={"tavily_search": _TavilyStyleTool()},
    )
    candidates = artifacts[0].get("_rich_item_candidates", [])
    image_candidates = [c for c in candidates if c["type"] == "image"]
    assert image_candidates and image_candidates[0]["id"] == "image:tool:call_99:0"
    # Result content sent to the model must still be the JSON text from the
    # tool; image bytes/data must not have been spliced into it.
    assert outputs[0]["content"].startswith("{")
    assert "_rich_item_candidates" not in outputs[0]


@pytest.mark.asyncio
async def test_execute_tool_calls_attaches_dedicated_widget_candidate_to_artifact():
    from app.ai.tool_execution import execute_tool_calls

    _outputs, artifacts, _images = await execute_tool_calls(
        tool_calls=[
            {
                "id": "call_widget",
                "name": "widget_create",
                "args": {"session_id": "conv-1"},
            }
        ],
        tool_map={"widget_create": _WidgetTool()},
    )

    [candidate] = artifacts[0]["_rich_item_candidates"]
    assert candidate["id"] == "widget:w-created"
    assert candidate["type"] == "live_widget"
    assert "private-state-not-in-candidate" not in str(candidate)
