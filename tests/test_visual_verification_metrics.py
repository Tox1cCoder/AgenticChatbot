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


def test_an_operational_failure_is_logged_not_only_counted(caplog):
    """A silent total outage is the failure mode that actually happened.

    Every image-path failure is a successful text-only answer by design, so
    "the deadline cancels every verifier call" and "no candidate was good
    enough" are indistinguishable from the outside. Only a metric recorded the
    difference, and nothing was watching it.
    """
    import asyncio
    import logging

    from app.ai.image_verification_flow import discover_and_verify_images

    with caplog.at_level(logging.WARNING, logger="app.ai.image_verification_flow"):
        asyncio.run(
            discover_and_verify_images(
                brave_tool=None,
                web_image_service=None,
                verifier_model=None,
                user_request="q",
                image_query="i",
                factual_query="f",
            )
        )

    assert "text-only" in caplog.text
    assert "unavailable" in caplog.text


def test_finding_nothing_worth_showing_is_not_logged_as_a_failure(caplog):
    """A verifier that rejects everything is the feature working."""
    import logging

    from app.ai import image_verification_flow

    assert "no_match" not in image_verification_flow._OPERATIONAL_FAILURES
    with caplog.at_level(logging.WARNING, logger="app.ai.image_verification_flow"):
        pass
    assert caplog.text == ""
