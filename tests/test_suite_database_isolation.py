"""The default suite must never be bound to the application database."""

from __future__ import annotations

import os
import pathlib
import re

import pytest
from sqlalchemy.engine import make_url

from tests import database_isolation

TESTS_ROOT = pathlib.Path(__file__).parent

# Ways a test module reaches the database the settings object points at.
DATABASE_ACCESS = re.compile(
    r"Database\(settings\.database_url\)"
    r"|create_engine\(settings\.database_url\)"
    r"|AsyncSessionLocal"
    r"|\bSessionLocal\(\)"
)
# Gates that skip a test when no dedicated test database is configured. The two
# fixtures depend on ``require_test_database``, so each one is a real gate.
DATABASE_GATE = re.compile(r"\brequires?_test_database\b|\brequire_async_db\b")


def test_settings_database_is_the_test_database_when_one_is_configured():
    test_url = os.getenv("TEST_DATABASE_URL")
    if not test_url:
        pytest.skip("TEST_DATABASE_URL is not set; DB-backed modules are skipped instead")

    from app.core.config import settings

    assert make_url(settings.database_url).database == make_url(test_url).database


def test_db_backed_modules_are_gated():
    """``tests/integration`` is excluded: its modules build engines from TEST_DATABASE_URL."""
    offenders = []
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        if "integration" in path.relative_to(TESTS_ROOT).parts:
            continue
        source = path.read_text("utf-8")
        if DATABASE_ACCESS.search(source) and not DATABASE_GATE.search(source):
            offenders.append(path.relative_to(TESTS_ROOT).as_posix())

    assert offenders == []


def test_a_missing_test_database_skips_with_the_reason(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)

    with pytest.raises(pytest.skip.Exception) as skipped:
        database_isolation.skip_unless_test_database()

    assert "TEST_DATABASE_URL" in str(skipped.value)


@pytest.mark.parametrize(
    "fixture_name", ["require_test_database", "require_async_db", "seeded_conversation_id"]
)
def test_the_database_fixtures_skip_without_a_test_database(monkeypatch, request, fixture_name):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)

    with pytest.raises(pytest.skip.Exception, match="TEST_DATABASE_URL"):
        request.getfixturevalue(fixture_name)


def test_the_async_probe_does_not_connect_without_a_test_database(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)

    def _refuse_to_connect():
        raise AssertionError("the probe opened a session with no test database configured")

    monkeypatch.setattr("app.database.async_session.AsyncSessionLocal", _refuse_to_connect)

    assert database_isolation.async_test_database_available() is False


def test_a_test_database_that_is_the_app_database_is_refused():
    app_url = "postgresql+psycopg://someone:secret@localhost:5432/chatbot"
    same_database_other_driver = "postgresql://other:pw@localhost/chatbot"

    with pytest.raises(pytest.UsageError) as refused:
        database_isolation.refuse_application_database(same_database_other_driver, app_url)

    assert "secret" not in str(refused.value)
    assert "pw" not in str(refused.value)


def test_a_separate_test_database_is_accepted():
    database_isolation.refuse_application_database(
        "postgresql://someone@localhost:5432/chatbot_test",
        "postgresql://someone@localhost:5432/chatbot",
    )
