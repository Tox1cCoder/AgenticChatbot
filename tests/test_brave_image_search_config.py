"""Config contract for Brave Image Search settings (image_search.md Phase 2).

These are operational budget/safety limits, not behavior hardcoding. The
``safesearch`` default must be a Brave-supported value (``off`` or ``strict``).
"""

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


def test_brave_image_search_defaults(monkeypatch):
    # ``_env_file=None`` suppresses the .env file but not os.environ, and .env is
    # loaded into the process environment at import. Without these delenvs this
    # asserts whatever the current machine is configured with, not what ships.
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_IMAGE_SEARCH_DEFAULT_COUNT", raising=False)
    monkeypatch.delenv("BRAVE_IMAGE_SEARCH_MAX_COUNT", raising=False)
    monkeypatch.delenv("BRAVE_IMAGE_SEARCH_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("BRAVE_IMAGE_SEARCH_DEFAULT_SAFESEARCH", raising=False)
    settings = _settings()
    assert settings.brave_search_api_key == ""
    assert settings.brave_image_search_default_count == 6
    assert settings.brave_image_search_max_count == 10
    assert settings.brave_image_search_timeout_seconds == pytest.approx(2.5)
    assert settings.brave_image_search_default_safesearch == "strict"


def test_brave_image_search_default_safesearch_accepts_supported_values():
    for value in ("off", "strict"):
        settings = _settings(brave_image_search_default_safesearch=value)
        assert settings.brave_image_search_default_safesearch == value


def test_brave_image_search_default_safesearch_rejects_unsupported_value():
    with pytest.raises(ValidationError):
        _settings(brave_image_search_default_safesearch="moderate")


def test_brave_image_search_count_and_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        _settings(brave_image_search_default_count=0)
    with pytest.raises(ValidationError):
        _settings(brave_image_search_max_count=0)
    with pytest.raises(ValidationError):
        _settings(brave_image_search_timeout_seconds=0)
