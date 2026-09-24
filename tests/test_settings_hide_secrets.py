"""Secret settings never appear when a settings object is printed.

A settings object reaches text more often than anyone intends: a traceback,
a pytest failure, or an error like ``Settings(...) has no attribute ...``
prints its repr. That repr used to include every API key and the database
password. Fields are found by name so a key added later is covered too.
"""

from __future__ import annotations

import re

import pytest

from app.core.config import Settings
from client_backend.core.config import ClientSettings

_SECRET_NAME = re.compile(r"api_key|secret|password|encryption_key")

# URLs that carry a password in their authority part.
_CREDENTIAL_URLS = {
    Settings: {"database_url", "redis_url", "celery_broker_url", "celery_result_backend"},
    ClientSettings: set(),
}


def _secret_fields(model) -> set[str]:
    named = {name for name in model.model_fields if _SECRET_NAME.search(name)}
    return named | _CREDENTIAL_URLS[model]


@pytest.mark.parametrize("model", [Settings, ClientSettings])
def test_printing_settings_never_shows_a_secret(model):
    fields = _secret_fields(model)
    assert fields, "the name pattern no longer finds any secret field"
    values = {name: f"scheme://user:SENTINEL-{name}@host/db" for name in fields}

    printed = model.model_construct(**values)

    assert "SENTINEL" not in repr(printed)
    assert "SENTINEL" not in str(printed)
