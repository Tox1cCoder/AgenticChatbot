"""Rich-image telemetry must remain bounded and content-free."""

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from app.api.health import create_health_router
from app.observability.rich_images import RichImageMetrics


def test_selector_duration_metric_has_no_content_labels() -> None:
    metrics = RichImageMetrics(registry=CollectorRegistry())

    metrics.record_selection_duration(0.004)

    samples = list(metrics.selector_duration.collect())[0].samples
    count = next(sample for sample in samples if sample.name.endswith("_count"))
    assert count.value == 1
    assert count.labels == {}


def test_rich_image_metrics_are_bounded_and_content_free():
    metrics = RichImageMetrics()
    secret = f"https://secret.example/{uuid4()}"

    metrics.record_fetch(provider="brave_image_search", outcome="success", duration_seconds=0.2)
    metrics.record_fetch(provider="tavily", outcome="timeout", duration_seconds=0.3)
    metrics.record_fetch(provider=secret, outcome=secret, duration_seconds=0.1)

    payload = metrics.render().decode("utf-8")

    assert "rich_image_fetches_total" in payload
    assert "rich_image_fetch_duration_seconds" in payload
    assert 'provider="brave"' in payload
    assert 'provider="tavily"' in payload
    assert 'provider="other"' in payload
    assert 'outcome="other"' in payload
    assert secret not in payload
    assert "rich_image_" + "verification" not in payload


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
    metrics.record_presentation(provider="brave", count=3)
    metrics.record_final_selection(provider="brave", count=1)
    body = metrics.render().decode()

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


def test_unwired_stage_metrics_are_not_exposed():
    """Discovery, candidate and anchor collectors were never recorded, so they
    read zero forever; an always-zero series looks like a healthy stage."""
    body = RichImageMetrics().render().decode()

    for name in (
        "rich_image_discovery_results",
        "rich_image_candidates_total",
        "rich_image_anchor_outcomes_total",
        "rich_image_discovery_outcome_total",
        "rich_image_discovery_duration_seconds",
    ):
        assert name not in body


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
