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
    metrics.record_candidate(provider="tavily", outcome="eligible")
    metrics.record_candidate(provider=secret, outcome=secret)
    metrics.record_fetch(provider="brave", outcome="success", duration_seconds=0.2)
    metrics.record_fetch(provider=secret, outcome=secret, duration_seconds=0.1)

    payload = metrics.render().decode("utf-8")

    assert "rich_image_discovery_results" in payload
    assert "rich_image_candidates_total" in payload
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


def test_deprecated_selection_counter_is_removed():
    metrics = RichImageMetrics()
    assert not hasattr(metrics, "record_selection")
    assert "rich_image_selections_total" not in metrics.render().decode()


def test_presentation_and_final_selection_ignore_non_positive_counts():
    metrics = RichImageMetrics()
    metrics.record_presentation(provider="brave", count=0)
    metrics.record_presentation(provider="brave", count=-1)
    metrics.record_final_selection(provider="brave", count=0)
    metrics.record_final_selection(provider="brave", count=-1)

    body = metrics.render().decode()

    # Non-positive counts must never touch .labels(): no sample line is
    # created for the label combination at all (not even a 0.0 one).
    assert 'rich_image_presented_total{provider="brave"}' not in body
    assert 'rich_image_final_selection_total{provider="brave"}' not in body


def test_record_anchor_bounds_unknown_outcome_to_other():
    metrics = RichImageMetrics()
    metrics.record_anchor(provider="brave", outcome="not-a-real-outcome")

    body = metrics.render().decode()

    assert 'outcome="other"' in body
    assert "not-a-real-outcome" not in body


def test_registration_outcomes_are_recorded_and_bounded():
    """Registration is one of the stages the rollout is supposed to watch, and a
    cell-level failure inside a group that keeps its siblings is invisible
    without it."""
    metrics = RichImageMetrics()
    for outcome in ("registered", "reused", "skipped_scheme", "failed"):
        metrics.record_registration(provider="brave", outcome=outcome)
    metrics.record_registration(provider="tavily", outcome="https://leak.test/secret.jpg")

    body = metrics.render().decode()

    assert "rich_image_registrations_total" in body
    for outcome in ("registered", "reused", "skipped_scheme", "failed"):
        assert f'outcome="{outcome}"' in body
    assert 'outcome="other"' in body
    assert "leak.test" not in body
