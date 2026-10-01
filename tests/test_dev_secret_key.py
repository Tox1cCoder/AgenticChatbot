"""Key configuration: optional in every environment, validated when it is set.

An unset key is not a weak key: ``app.core.server_secrets`` resolves it from the
``server_secrets`` table. An explicitly configured one is used as given, so it
is the one that must be strong.
"""

import base64

import pytest
from pydantic import ValidationError

from app.core.config import Settings

_STRONG_KEY = "k" * 32
_ORIGINS = ["https://chat.example.com"]
_FERNET_KEY = base64.urlsafe_b64encode(b"f" * 32).decode()


def _settings(**overrides) -> Settings:
    # Every field the checks read is explicit: `.env` is loaded into the
    # process environment, so an omitted one would come from the developer's
    # machine rather than from the test.
    values = {
        "secret_key": _STRONG_KEY,
        "model_encryption_key": _FERNET_KEY,
        "cors_origins": _ORIGINS,
        "environment": "production",
        "api_debug": False,
        "langsmith_tracing": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize("environment", ["development", "staging", "production"])
@pytest.mark.parametrize("unset", ["", "   "])
def test_an_unset_secret_key_is_left_for_the_database(environment, unset):
    settings = _settings(environment=environment, secret_key=unset)

    assert settings.secret_key == ""


def test_the_placeholder_secret_key_counts_as_unset_in_development():
    assert _settings(environment="development", secret_key="secret-key").secret_key == ""


@pytest.mark.parametrize("environment", ["production", "staging"])
@pytest.mark.parametrize("weak", ["k" * 31, "secret-key"])
def test_a_short_secret_key_is_refused_outside_development(environment, weak):
    with pytest.raises(ValidationError, match="at least 32 characters"):
        _settings(environment=environment, secret_key=weak)


@pytest.mark.parametrize("environment", ["development", "production"])
def test_the_model_encryption_key_is_optional(environment):
    assert _settings(environment=environment, model_encryption_key="").model_encryption_key == ""


@pytest.mark.parametrize(
    "invalid",
    ["not-a-fernet-key", base64.urlsafe_b64encode(b"short").decode(), "%%%%"],
)
def test_a_configured_model_encryption_key_must_be_a_fernet_key(invalid):
    with pytest.raises(ValidationError, match="MODEL_ENCRYPTION_KEY"):
        _settings(model_encryption_key=invalid)


def test_a_configured_model_encryption_key_is_kept_as_given():
    assert _settings(model_encryption_key=f" {_FERNET_KEY} ").model_encryption_key == _FERNET_KEY


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


@pytest.mark.parametrize("spelling", ["Development", " development ", "DEVELOPMENT"])
def test_the_environment_name_is_normalized_once_for_every_check(spelling):
    """The security checks compared the raw value while the production check
    lowered it, so ``Development`` was a deployment to one and not the other."""
    settings = _settings(environment=spelling, secret_key="short", cors_origins=[])

    assert settings.environment == "development"
    assert settings.secret_key == "short"
