"""Operational limits for deterministic rich-image selection and delivery."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _settings(**overrides):
    return Settings(
        _env_file=None,
        secret_key="test-secret",
        environment="development",
        **overrides,
    )


def test_rich_image_selection_defaults_are_bounded():
    settings = _settings()

    assert settings.rich_image_candidate_max_count == 8
    assert settings.rich_image_min_width_px == 320
    assert settings.rich_image_min_height_px == 180


def test_web_image_delivery_defaults_are_render_time_bounded():
    settings = _settings()

    assert settings.web_image_fetch_connect_timeout_seconds == pytest.approx(2.0)
    assert settings.web_image_fetch_read_timeout_seconds == pytest.approx(5.0)
    assert settings.web_image_fetch_max_redirects == 3
    assert settings.web_image_fetch_max_bytes == 5 * 1024 * 1024
    assert settings.web_image_fetch_max_pixels == 25_000_000


@pytest.mark.parametrize(
    "field",
    (
        "rich_image_candidate_max_count",
        "rich_image_min_width_px",
        "rich_image_min_height_px",
    ),
)
def test_rich_image_selection_limits_must_be_positive(field):
    with pytest.raises(ValidationError, match="positive"):
        _settings(**{field: 0})


@pytest.mark.parametrize(
    "field",
    (
        "web_image_fetch_connect_timeout_seconds",
        "web_image_fetch_read_timeout_seconds",
        "web_image_fetch_max_bytes",
        "web_image_fetch_max_pixels",
    ),
)
def test_web_image_delivery_positive_limits_reject_zero(field):
    with pytest.raises(ValidationError):
        _settings(**{field: 0})


def test_web_image_redirect_limit_is_bounded():
    assert _settings(web_image_fetch_max_redirects=0).web_image_fetch_max_redirects == 0
    with pytest.raises(ValidationError):
        _settings(web_image_fetch_max_redirects=6)
