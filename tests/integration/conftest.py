"""Refuse to run integration tests against the application's own database.

Every module in this directory calls ``Base.metadata.create_all`` and seeds
rows. Pointed at the app database, that is not merely untidy:

* ``create_all`` builds tables and indexes from the *models*, bypassing
  Alembic. That is how ``ix_chat_images_sha256`` came to exist on a database
  whose migrations never created it, and it kept ``alembic check`` dirty until
  ``c9d0e1f2a3b4`` removed it.
* ``test_model_usage_repository_postgres.py`` and
  ``test_conversation_compaction_postgres.py`` assert over aggregates that
  include every row in the table, so real data makes them fail for reasons
  that have nothing to do with the code under test.
* the seed/cleanup fixtures delete by primary key, so a bug in one is a delete
  against production rows.

The guard is unconditional rather than per-module. A module that only touches
its own ``uuid4`` rows today is one edit away from not doing so, and the cost
of being wrong is someone's data.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy.engine import make_url


def _same_database(left: str, right: str) -> bool:
    """Whether two URLs address the same host/port/database.

    Driver prefix and credentials are ignored on purpose: ``postgresql://`` and
    ``postgresql+psycopg://`` pointing at the same database are the same
    database, and a different password does not make it a different one.
    """
    try:
        first, second = make_url(left), make_url(right)
    except Exception:  # noqa: BLE001 - an unparseable URL is not a match
        return False
    return (
        (first.host or "") == (second.host or "")
        and (first.port or 5432) == (second.port or 5432)
        and (first.database or "") == (second.database or "")
    )


def pytest_collection_modifyitems(config, items):
    """Fail the integration suite outright when it is aimed at the app database."""
    test_url = os.getenv("TEST_DATABASE_URL")
    if not test_url:
        return

    from app.core.config import settings

    app_url = getattr(settings, "database_url", "") or ""
    if not app_url or not _same_database(test_url, app_url):
        return

    target = make_url(test_url)
    raise pytest.UsageError(
        "TEST_DATABASE_URL points at the application database "
        f"({target.host}:{target.port or 5432}/{target.database}). These tests run "
        "Base.metadata.create_all and delete seeded rows, so they must never run there. "
        "Create a dedicated database (for example `createdb chatbot_test`) and set "
        "TEST_DATABASE_URL to it."
    )
