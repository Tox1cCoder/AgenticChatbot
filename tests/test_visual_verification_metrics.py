from __future__ import annotations

from app.observability.rich_images import RichImageMetrics


def test_verification_stages_are_counted():
    metrics = RichImageMetrics()

    metrics.record_verification(stage="discovered", count=6)
    metrics.record_verification(stage="approved", count=1)
    rendered = metrics.render().decode("utf-8")

    assert 'stage="discovered"' in rendered
    assert 'stage="approved"' in rendered


def test_an_unknown_stage_is_bucketed_not_recorded_verbatim():
    metrics = RichImageMetrics()

    metrics.record_verification(stage="league of legends t1 roster", count=1)
    rendered = metrics.render().decode("utf-8")

    assert "league" not in rendered
    assert 'stage="other"' in rendered


def test_an_unknown_outcome_is_bucketed():
    metrics = RichImageMetrics()

    metrics.record_verification_outcome(outcome="portrait of Moi", duration_seconds=0.2)
    rendered = metrics.render().decode("utf-8")

    assert "Moi" not in rendered
    assert 'outcome="other"' in rendered
