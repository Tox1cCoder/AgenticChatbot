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
