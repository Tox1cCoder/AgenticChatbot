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
    image_aspect_ratio_ok,
    is_junk_image_url,
    order_tavily_images,
)
from app.core.rich_image_selection import (
    ImageSelectionPolicy,
    select_rich_item_candidates,
)
from app.core.rich_response import sanitize_public_rich_item

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

    # Single candidate: image_max_items is generous so it never interferes
    # with the base64-exclusion behavior this test actually exercises.
    block = build_rich_item_inventory_block(
        candidates, max_items=5, max_chars=2400, summary_chars=180, image_max_items=10
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
    # A single result has the same deliberate-search intent as a Brave group.
    assert cand["provenance"]["tool"] == "brave_image_search"
    assert cand["source"] == "image_search"
    # source_url stays in the public payload (ImagePayload accepts it).
    assert cand["payload"]["url"] == "https://img.test/thumb-1.jpg"
    assert cand["payload"]["source_url"] == "https://example.com/page"
    assert cand["payload"]["width"] == 1200
    assert cand["payload"]["height"] == 800


def test_brave_image_candidate_keeps_safe_provider_provenance_only():
    [cand] = build_image_candidates_from_tool_result(
        _brave_payload(), tool_call_id="call_b", tool_name="brave_image_search"
    )
    prov = cand["provenance"]
    assert prov["thumbnail_url"] == "https://img.test/thumb-1.jpg"
    assert "original_image_url" not in prov
    assert prov["source_domain"] == "example.com"
    assert prov["provider"] == "brave_image_search"
    for forbidden in ("thumbnail_url", "original_image_url", "source_domain", "provider"):
        assert forbidden not in cand["payload"]
    public = sanitize_public_rich_item(cand)
    assert "original_image_digests" not in public["provenance"]


def test_brave_thumbnails_of_same_original_image_are_deduplicated() -> None:
    payload = json.dumps(
        {
            "query": "team roster",
            "provider": "brave_image_search",
            "images": [
                {
                    "url": "https://origin.example/shared.jpg",
                    "thumbnail_url": f"https://thumbs.example/variant-{index}.jpg",
                    "source_url": f"https://pages.example/article-{index}",
                    "description": "team roster",
                    "mime_type": "image/jpeg",
                    "width": 1200,
                    "height": 800,
                    "provider": "brave_image_search",
                }
                for index in range(2)
            ],
        }
    )
    candidates = build_image_candidates_from_tool_result(
        payload,
        tool_call_id="call-original-dedupe",
        tool_name="brave_image_search",
    )

    selected = select_rich_item_candidates(
        candidates,
        policy=ImageSelectionPolicy(
            max_items=2,
            min_width_px=320,
            min_height_px=180,
            min_aspect_ratio=0.2,
            max_aspect_ratio=5.0,
        ),
    )

    assert len(selected) == 1
    assert selected[0]["type"] == "image"
    assert selected[0]["payload"]["url"] == (
        "https://thumbs.example/variant-0.jpg"
    )


def test_brave_originals_are_deduplicated_before_group_cell_cap() -> None:
    payload = json.dumps(
        {
            "query": "team roster",
            "provider": "brave_image_search",
            "images": [
                *[
                    {
                        "url": "https://origin.example/shared.jpg",
                        "thumbnail_url": f"https://thumbs.example/variant-{index}.jpg",
                        "description": "team roster",
                        "mime_type": "image/jpeg",
                        "width": 1200,
                        "height": 800,
                        "provider": "brave_image_search",
                    }
                    for index in range(3)
                ],
                {
                    "url": "https://origin.example/unique.jpg",
                    "thumbnail_url": "https://thumbs.example/unique.jpg",
                    "description": "team roster unique view",
                    "mime_type": "image/jpeg",
                    "width": 1200,
                    "height": 800,
                    "provider": "brave_image_search",
                },
            ],
        }
    )
    [candidate] = build_image_candidates_from_tool_result(
        payload,
        tool_call_id="call-pre-cap-dedupe",
        tool_name="brave_image_search",
    )

    assert candidate["type"] == "image_group"
    assert [cell["url"] for cell in candidate["payload"]["items"]] == [
        "https://thumbs.example/variant-0.jpg",
        "https://thumbs.example/unique.jpg",
    ]


def test_malformed_provider_url_rejects_only_that_candidate() -> None:
    payload = json.dumps(
        {
            "provider": "brave_image_search",
            "images": [
                {
                    "url": "https://[malformed",
                    "description": "broken",
                    "width": 1200,
                    "height": 800,
                },
                {
                    "url": "https://media.example/valid.jpg",
                    "description": "valid",
                    "width": 1200,
                    "height": 800,
                },
            ],
        }
    )

    candidates = build_image_candidates_from_tool_result(
        payload,
        tool_call_id="call-malformed-url",
        tool_name="brave_image_search",
    )

    assert [candidate["payload"]["url"] for candidate in candidates] == [
        "https://media.example/valid.jpg"
    ]


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


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (2000, 200, False),   # wide hero strip, ratio 10.0
        (300, 1600, False),   # ratio 0.1875, strictly below the minimum
        (300, 1500, True),    # lower boundary (ratio 0.2) is accepted
        (1000, 200, True),    # upper boundary (ratio 5.0) is accepted
        (1200, 800, True),    # ordinary photo
        (800, 2600, True),    # tall infographic, ratio ~0.31
        (2200, 500, True),    # panorama, ratio 4.4
        (None, 800, True),    # unknown dimensions never reject
        (1200, None, True),
        (0, 0, True),         # nonsense dimensions are not a rejection signal
    ],
)
def test_aspect_ratio_gate(width, height, expected):
    assert image_aspect_ratio_ok(width, height, minimum=0.2, maximum=5.0) is expected


@pytest.mark.parametrize(
    "url",
    [
        "https://e.com/favicon.ico",
        "https://e.com/assets/sprite-v2.png",
        "https://e.com/img/spacer.gif",
        "https://e.com/t/1x1.png",
        "https://e.com/pixel.gif",
        "https://e.com/avatars/default.png",
        "https://e.com/default-avatar.jpg",
    ],
)
def test_junk_urls_are_rejected(url):
    assert is_junk_image_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://e.com/new-logo-reveal.jpg",
        "https://e.com/logos/brand.png",
        "https://e.com/photos/apple-park.jpg",
        "https://e.com/movies/avatar-poster.jpg",
        "https://e.com/users/avatar/12.jpg",
        "https://e.com/avatars/me.png",
        "https://e.com/diagram-1x100.jpg",
    ],
)
def test_legitimate_urls_including_logos_are_accepted(url):
    assert is_junk_image_url(url) is False


def test_wide_strip_candidate_is_rejected_end_to_end():
    payload = json.dumps(
        {
            "provider": "brave_image_search",
            "query": "apple park",
            "images": [
                {"url": "https://e.com/strip.jpg", "width": 2000, "height": 200},
                {"url": "https://e.com/ok.jpg", "width": 1200, "height": 800},
            ],
        }
    )
    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="c1", tool_name="brave_image_search"
    )
    urls = json.dumps(candidates)
    assert "strip.jpg" not in urls
    assert "ok.jpg" in urls


def test_remote_candidates_obey_configured_candidate_cap(monkeypatch):
    """The per-image candidate_cap still bounds the loop's output even though a
    Brave result with 2+ eligible candidates now collapses into one group
    (Task 8): the 4 source images are trimmed to 2 candidates before grouping,
    so surplus images never reach the group's cells."""
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

    assert len(candidates) == 1
    assert candidates[0]["type"] == "image_group"
    assert len(candidates[0]["payload"]["items"]) == 2


def test_candidate_selection_records_bounded_provider_counts(monkeypatch):
    from app.ai import tool_execution

    metrics = type(
        "Metrics",
        (),
        {
            "record_discovery": Mock(),
            "record_candidate": Mock(),
        },
    )()
    monkeypatch.setattr(tool_execution, "rich_image_metrics", metrics)

    build_image_candidates_from_tool_result(
        _brave_payload(), tool_call_id="call_metrics", tool_name="brave_image_search"
    )

    metrics.record_discovery.assert_called_once_with(provider="brave", result_count=1)
    metrics.record_candidate.assert_called_once_with(provider="brave", outcome="eligible")


def test_candidate_survives_record_candidate_raising_on_acceptance(monkeypatch):
    """Telemetry is best-effort: a raising counter must not drop an eligible
    candidate or fail the surrounding tool result."""
    from app.ai import tool_execution

    metrics = type(
        "Metrics",
        (),
        {
            "record_discovery": Mock(),
            "record_candidate": Mock(side_effect=RuntimeError("boom")),
        },
    )()
    monkeypatch.setattr(tool_execution, "rich_image_metrics", metrics)

    candidates = build_image_candidates_from_tool_result(
        _brave_payload(), tool_call_id="call_metrics", tool_name="brave_image_search"
    )

    assert len(candidates) == 1


def test_rejection_survives_record_candidate_raising(monkeypatch):
    """Same guarantee on the reject path (_reject helper), not just acceptance."""
    from app.ai import tool_execution

    metrics = type(
        "Metrics",
        (),
        {
            "record_discovery": Mock(),
            "record_candidate": Mock(side_effect=RuntimeError("boom")),
        },
    )()
    monkeypatch.setattr(tool_execution, "rich_image_metrics", metrics)
    payload = json.dumps({"images": [{"url": "http://insecure.test/a.jpg"}]})

    assert (
        build_image_candidates_from_tool_result(
            payload, tool_call_id="call_metrics", tool_name="brave_image_search"
        )
        == []
    )


def test_tavily_malformed_entry_is_counted_as_rejected(monkeypatch):
    """The Tavily-only pre-filter used to drop non-dict entries silently
    before the counting loop ran, so a Brave malformed entry was counted but
    an equivalent Tavily one was not. Both providers must record the same
    outcome for the same shape of bad input, exactly once (no double count)."""
    from app.ai import tool_execution

    metrics = type(
        "Metrics",
        (),
        {
            "record_discovery": Mock(),
            "record_candidate": Mock(),
        },
    )()
    monkeypatch.setattr(tool_execution, "rich_image_metrics", metrics)
    payload = json.dumps(
        {
            "provider": "tavily",
            "images": [
                "not-a-dict",
                {"url": "https://img.test/good.jpg", "description": "ok"},
            ],
        }
    )

    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="call_1", tool_name="tavily_search"
    )

    assert len(candidates) == 1
    malformed_calls = [
        call
        for call in metrics.record_candidate.call_args_list
        if call.kwargs.get("outcome") == "rejected_malformed"
    ]
    assert len(malformed_calls) == 1
    assert malformed_calls[0].kwargs["provider"] == "tavily"


def test_brave_image_candidate_validates_against_public_schema():
    from app.core.rich_response import validate_public_rich_item

    [cand] = build_image_candidates_from_tool_result(
        _brave_payload(), tool_call_id="call_b", tool_name="brave_image_search"
    )
    # Passes the discriminated-union validation only if payload has no forbidden
    # extras and the mime/url contract holds.
    validate_public_rich_item(cand)


# ---------------------------------------------------------------------------
# Image group collapsing (Task 8)
# ---------------------------------------------------------------------------


def _brave_group_payload(count):
    return json.dumps(
        {
            "provider": "brave_image_search",
            "query": "red panda photo",
            "images": [
                {
                    "url": f"https://e.com/{n}.jpg",
                    "thumbnail_url": f"https://cdn.brave.com/{n}.jpg",
                    "mime_type": "image/jpeg",
                    "width": 1200,
                    "height": 800,
                    "source_url": f"https://e.com/page-{n}",
                    "description": f"red panda {n}",
                }
                for n in range(count)
            ],
        }
    )


def test_single_eligible_brave_candidate_stays_an_image_item():
    candidates = build_image_candidates_from_tool_result(
        _brave_group_payload(1), tool_call_id="c1", tool_name="brave_image_search"
    )
    assert len(candidates) == 1
    assert candidates[0]["type"] == "image"


def test_multiple_brave_candidates_collapse_into_one_group():
    candidates = build_image_candidates_from_tool_result(
        _brave_group_payload(4), tool_call_id="c1", tool_name="brave_image_search"
    )
    assert len(candidates) == 1
    group = candidates[0]
    assert group["type"] == "image_group"
    assert group["id"] == "imagegroup:tool:c1"
    assert group["source"] == "image_search"
    assert len(group["payload"]["items"]) == 3  # capped
    assert group["provenance"]["query"] == "red panda photo"


def test_group_cells_prefer_brave_thumbnails():
    candidates = build_image_candidates_from_tool_result(
        _brave_group_payload(2), tool_call_id="c1", tool_name="brave_image_search"
    )
    urls = [cell["url"] for cell in candidates[0]["payload"]["items"]]
    assert all(url.startswith("https://cdn.brave.com/") for url in urls)


def test_group_alt_text_comes_from_the_image_query():
    candidates = build_image_candidates_from_tool_result(
        _brave_group_payload(2), tool_call_id="c1", tool_name="brave_image_search"
    )
    assert "red panda photo" in candidates[0]["alt_text"]


def test_group_provenance_provider_is_never_none():
    """_brave_group_payload() images carry no per-image "provider" key, so a
    group built from ``first_provenance.get("provider")`` would silently get
    None here. The group's provenance must instead carry the caller's
    reliable, already-classified ``metric_provider`` value ("brave"), which
    is guaranteed non-None whenever a group is emitted."""
    candidates = build_image_candidates_from_tool_result(
        _brave_group_payload(2), tool_call_id="c1", tool_name="brave_image_search"
    )
    provider = candidates[0]["provenance"]["provider"]
    assert provider
    assert provider == "brave"


def test_tavily_images_are_never_grouped():
    payload = json.dumps(
        {
            "provider": "tavily",
            "query": "chip rules",
            "images": [
                {"url": "https://e.com/a.jpg", "description": "a", "source_url": "https://e.com/1"},
                {"url": "https://e.com/b.jpg", "description": "b", "source_url": "https://e.com/2"},
            ],
        }
    )
    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="c1", tool_name="tavily_search"
    )
    assert {c["type"] for c in candidates} == {"image"}


def test_group_is_not_emitted_when_all_candidates_are_ineligible():
    payload = json.dumps(
        {
            "provider": "brave_image_search",
            "query": "x",
            "images": [
                {"url": "http://e.com/a.jpg"},
                {"url": "https://e.com/favicon.ico"},
            ],
        }
    )
    assert (
        build_image_candidates_from_tool_result(
            payload, tool_call_id="c1", tool_name="brave_image_search"
        )
        == []
    )


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


class _ManyWebImagesTool:
    def __init__(self, *, provider: str) -> None:
        self.provider = provider

    async def ainvoke(self, args):
        return json.dumps(
            {
                "provider": self.provider,
                "query": args.get("query", "team roster"),
                "images": [
                    {
                        "url": f"https://img.test/photo-{index}.jpg",
                        "thumbnail_url": f"https://img.test/thumb-{index}.jpg",
                        "source_url": f"https://pages.test/article-{index}",
                        "description": f"team roster photo {index}",
                        "width": 1200,
                        "height": 800,
                        "provider": self.provider,
                        "result_rank": index,
                    }
                    for index in range(114)
                ],
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


class _ImageContentTool:
    name = "diagram_tool"

    async def ainvoke(self, _args):
        return {
            "content": [
                {
                    "type": "image",
                    "data": "QUJDRA==",
                    "mimeType": "image/png",
                    "description": "Generated diagram",
                }
            ]
        }


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
    assert _images == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "provider"),
    [
        ("tavily_search", "tavily"),
        ("brave_image_search", "brave_image_search"),
    ],
)
async def test_typed_web_provider_raw_images_do_not_enter_legacy_gallery(
    tool_name: str,
    provider: str,
) -> None:
    from app.ai.tool_execution import execute_tool_calls
    from app.core.config import settings

    _outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[
            {
                "id": f"call-{provider}",
                "name": tool_name,
                "args": {"query": "team roster"},
            }
        ],
        tool_map={tool_name: _ManyWebImagesTool(provider=provider)},
    )

    assert len(artifacts[0]["_rich_item_candidates"]) <= (
        settings.rich_image_candidate_max_count
    )
    assert images == []


@pytest.mark.asyncio
async def test_non_search_tool_image_content_keeps_existing_capture_path() -> None:
    from app.ai.tool_execution import execute_tool_calls

    _outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[
            {
                "id": "call-diagram",
                "name": "diagram_tool",
                "args": {},
            }
        ],
        tool_map={"diagram_tool": _ImageContentTool()},
    )

    assert images == [
        {
            "data": "QUJDRA==",
            "mime": "image/png",
            "description": "Generated diagram",
        }
    ]
    assert artifacts[0]["_rich_item_candidates"][0]["source"] == "tool_image"


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


# ---------------------------------------------------------------------------
# Tavily image ordering (Task 5)
# ---------------------------------------------------------------------------


def test_source_bound_images_precede_query_level():
    ordered = order_tavily_images(
        [
            {"url": "q", "query_level": True},
            {"url": "s", "result_rank": 3, "result_score": 0.1},
        ]
    )
    assert [i["url"] for i in ordered] == ["s", "q"]


def test_higher_score_then_lower_rank_wins():
    ordered = order_tavily_images(
        [
            {"url": "a", "result_rank": 2, "result_score": 0.5},
            {"url": "b", "result_rank": 0, "result_score": 0.9},
            {"url": "c", "result_rank": 1, "result_score": 0.9},
        ]
    )
    assert [i["url"] for i in ordered] == ["b", "c", "a"]


def test_absent_score_sorts_after_any_numeric_score():
    ordered = order_tavily_images(
        [
            {"url": "none", "result_rank": 0, "result_score": None},
            {"url": "low", "result_rank": 9, "result_score": 0.01},
        ]
    )
    assert [i["url"] for i in ordered] == ["low", "none"]


def test_ordering_is_stable_for_equivalent_candidates():
    ordered = order_tavily_images(
        [
            {"url": "first", "result_rank": 1, "result_score": 0.5},
            {"url": "second", "result_rank": 1, "result_score": 0.5},
        ]
    )
    assert [i["url"] for i in ordered] == ["first", "second"]


def test_ordering_never_drops_a_candidate():
    images = [{"url": str(n)} for n in range(7)]
    assert len(order_tavily_images(images)) == 7
