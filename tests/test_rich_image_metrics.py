"""Rich-image telemetry must remain bounded and content-free."""

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.health import create_health_router
from app.observability.rich_images import RichImageMetrics


def test_rich_image_metrics_are_bounded_and_content_free():
    metrics = RichImageMetrics()
    secret = f"https://secret.example/{uuid4()}"

    metrics.record_discovery(provider="brave_image_search", result_count=6)
    metrics.record_selection(provider="tavily", outcome="selected")
    metrics.record_selection(provider=secret, outcome=secret)
    metrics.record_fetch(provider="brave", outcome="success", duration_seconds=0.2)
    metrics.record_fetch(provider=secret, outcome=secret, duration_seconds=0.1)

    payload = metrics.render().decode("utf-8")

    assert "rich_image_discovery_results" in payload
    assert "rich_image_selections_total" in payload
    assert "rich_image_fetches_total" in payload
    assert "rich_image_fetch_duration_seconds" in payload
    assert 'provider="brave"' in payload
    assert 'provider="tavily"' in payload
    assert 'provider="other"' in payload
    assert secret not in payload


def test_health_router_exposes_rich_image_metrics():
    metrics = RichImageMetrics()
    metrics.record_fetch(provider="tavily", outcome="timeout", duration_seconds=0.1)
    app = FastAPI()
    app.include_router(create_health_router(rich_image_metrics=metrics))

    response = TestClient(app).get("/metrics/rich-images")

    assert response.status_code == 200
    assert "rich_image_fetches_total" in response.text


def test_stage_counters_are_bounded_and_content_free():
    metrics = RichImageMetrics()
    metrics.record_candidate(provider="tavily", outcome="rejected_aspect_ratio")
    metrics.record_candidate(provider="brave", outcome="not-a-real-outcome")
    metrics.record_presentation(provider="brave", count=3)
    metrics.record_anchor(provider="brave", outcome="fallback_anchored")
    metrics.record_anchor(provider="brave", outcome="unplaced")
    metrics.record_final_selection(provider="brave", count=1)
    body = metrics.render().decode()

    assert 'outcome="rejected_aspect_ratio"' in body
    assert 'outcome="other"' in body
    assert 'outcome="fallback_anchored"' in body
    assert 'outcome="unplaced"' in body
    assert "rich_image_presented_total" in body
    assert "rich_image_final_selection_total" in body
    for forbidden in ("query", "caption", "http", "conversation"):
        assert f'{forbidden}="' not in body


def test_old_selection_counter_still_emits_during_compatibility_window():
    metrics = RichImageMetrics()
    metrics.record_selection(provider="tavily", outcome="selected")
    assert "rich_image_selections_total" in metrics.render().decode()
