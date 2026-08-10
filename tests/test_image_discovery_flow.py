"""Deterministic selection of normalized Brave image results."""

from __future__ import annotations

import json
from unittest.mock import ANY, Mock

import pytest

from app.ai import image_discovery_flow
from app.ai.image_discovery_flow import discover_images, select_brave_candidates


def _payload(
    candidates: list[tuple[str, int]],
    *,
    offensive: bool = False,
    original_dimensions: tuple[int, int] | None = (1200, 800),
    thumbnail_dimensions: tuple[int, int] | None = (500, 281),
) -> str:
    """Return the normalized Brave shape with independently visible ranks."""

    images: list[dict[str, object]] = []
    for confidence, rank in candidates:
        image: dict[str, object] = {
            "url": f"https://imgs.search.brave.com/display-{rank}.jpg",
            "original_image_url": f"https://origin.example/image-{rank}.jpg",
            "thumbnail_url": f"https://imgs.search.brave.com/thumb-{rank}.jpg",
            "confidence": confidence,
            "result_rank": rank,
            "provider": "brave_image_search",
            "mime_type": "image/jpeg",
            "title": f"T1 roster {rank}",
            "description": f"T1 roster {rank}",
            "source_url": f"https://source.example/{rank}",
        }
        if original_dimensions is not None:
            image["width"], image["height"] = original_dimensions
        if thumbnail_dimensions is not None:
            image["thumbnail_width"], image["thumbnail_height"] = thumbnail_dimensions
        images.append(image)
    return json.dumps(
        {
            "query": "T1 team photo",
            "provider": "brave_image_search",
            "safety": {"might_be_offensive": offensive},
            "images": images,
            "total_results": len(images),
        }
    )


def _error_payload(error_type: str, *, retryable: bool) -> str:
    return json.dumps(
        {
            "error": "Brave image search returned an error.",
            "error_type": error_type,
            "retryable": retryable,
            "provider": "brave_image_search",
            "images": [],
            "total_results": 0,
        }
    )


class _Tool:
    def __init__(self, result: str) -> None:
        self.result = result

    async def ainvoke(self, args: dict[str, str]) -> str:
        return self.result


@pytest.fixture
def metrics(monkeypatch):
    fake = type("Metrics", (), {"record_discovery_outcome": Mock()})()
    monkeypatch.setattr(image_discovery_flow, "rich_image_metrics", fake)
    return fake


def test_high_confidence_wins_in_provider_order():
    selected = select_brave_candidates(
        _payload([("medium", 1), ("high", 2), ("high", 3)]),
        image_query="T1 team photo",
    )

    assert [item["provenance"]["result_rank"] for item in selected] == [2, 3]


def test_medium_is_used_only_when_no_high_candidate_survives():
    selected = select_brave_candidates(
        _payload([("medium", 1), ("low", 2), ("medium", 3)]),
        image_query="T1 team photo",
    )

    assert [item["provenance"]["result_rank"] for item in selected] == [1, 3]


def test_later_high_confidence_candidate_wins_past_builder_cap(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "rich_image_candidate_max_count", 8)
    selected = select_brave_candidates(
        _payload([*( ("medium", rank) for rank in range(1, 9)), ("high", 9)]),
        image_query="T1 team photo",
    )

    assert [item["provenance"]["result_rank"] for item in selected] == [9]


@pytest.mark.parametrize("confidence", ["low", "", "unknown"])
def test_low_missing_and_unknown_confidence_are_rejected(confidence):
    assert select_brave_candidates(
        _payload([(confidence, 1)]), image_query="T1 team photo"
    ) == []


def test_offensive_response_selects_nothing():
    assert select_brave_candidates(
        _payload([("high", 1)], offensive=True), image_query="art"
    ) == []


def test_thumbnail_dimensions_do_not_trigger_source_minimum_rejection():
    selected = select_brave_candidates(
        _payload(
            [("high", 1)],
            original_dimensions=None,
            thumbnail_dimensions=(200, 112),
        ),
        image_query="T1 team photo",
    )
    assert len(selected) == 1


def test_extreme_thumbnail_aspect_is_rejected_when_original_dimensions_are_unknown():
    selected = select_brave_candidates(
        _payload(
            [("high", 1)],
            original_dimensions=None,
            thumbnail_dimensions=(500, 20),
        ),
        image_query="T1 team photo",
    )

    assert selected == []


def test_gallery_groups_selected_candidates_after_confidence_selection():
    selected = select_brave_candidates(
        _payload([("high", 1), ("high", 2), ("medium", 3)]),
        image_query="T1 roster",
        image_intent="gallery",
    )
    assert len(selected) == 1
    assert selected[0]["type"] == "image_group"
    assert len(selected[0]["payload"]["items"]) == 2


@pytest.mark.asyncio
async def test_structured_brave_error_is_search_failure(metrics):
    selected = await discover_images(
        brave_tool=_Tool(_error_payload("rate_limit", retryable=True)),
        image_query="T1 team photo",
    )

    assert selected == []
    metrics.record_discovery_outcome.assert_called_once_with(
        outcome="search_failure", duration_seconds=ANY
    )
