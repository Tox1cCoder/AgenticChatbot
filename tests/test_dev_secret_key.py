import pytest
from pydantic import ValidationError

from app.core.config import Settings, _load_or_create_dev_secret_key

_STRONG_KEY = "k" * 32
_ORIGINS = ["https://chat.example.com"]


def _settings(**overrides) -> Settings:
    # Every field the checks read is explicit: `.env` is loaded into the
    # process environment, so an omitted one would come from the developer's
    # machine rather than from the test.
    values = {
        "secret_key": _STRONG_KEY,
        "cors_origins": _ORIGINS,
        "environment": "production",
        "api_debug": False,
        "langsmith_tracing": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_dev_secret_key_is_generated_and_persisted(tmp_path):
    key_path = tmp_path / ".dev_secret_key"

    first = _load_or_create_dev_secret_key(key_path)

    assert first
    assert key_path.read_text(encoding="utf-8").strip() == first


def test_dev_secret_key_is_stable_across_calls(tmp_path):
    """A server restart must reuse the persisted key, not mint a new one that
    would invalidate every previously issued token."""
    key_path = tmp_path / ".dev_secret_key"

    first = _load_or_create_dev_secret_key(key_path)
    second = _load_or_create_dev_secret_key(key_path)

    assert first == second


def test_dev_secret_key_reuses_existing_file(tmp_path):
    key_path = tmp_path / ".dev_secret_key"
    key_path.write_text("preexisting-key-value", encoding="utf-8")

    assert _load_or_create_dev_secret_key(key_path) == "preexisting-key-value"


@pytest.mark.parametrize("environment", ["production", "staging"])
def test_a_short_secret_key_is_refused_outside_development(environment):
    with pytest.raises(ValidationError, match="at least 32 characters"):
        _settings(environment=environment, secret_key="k" * 31)


@pytest.mark.parametrize("origins", [[], ["*"], ["https://chat.example.com", "*"], [""]])
def test_any_origin_cors_is_refused_outside_development(origins):
    with pytest.raises(ValidationError, match="cors_origins"):
        _settings(cors_origins=origins)


def test_a_hardened_production_configuration_is_accepted():
    settings = _settings()

    assert settings.secret_key == _STRONG_KEY
    assert settings.cors_origins == _ORIGINS


def test_development_keeps_its_permissive_defaults():
    """Local setup must not change: a short key and an empty CORS list stay valid."""
    settings = _settings(environment="development", secret_key="short", cors_origins=[])

    assert settings.secret_key == "short"
    assert settings.cors_origins == []
