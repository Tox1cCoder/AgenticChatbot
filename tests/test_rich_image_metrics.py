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
