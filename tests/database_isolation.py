"""Keep tests that open real database sessions off the application database.

``tests/conftest.py`` calls :func:`bind_suite_to_test_database` before any ``app``
import. With ``TEST_DATABASE_URL`` set, ``DATABASE_URL`` is rebound to it, so the
whole suite (``settings``, ``SessionLocal``, ``AsyncSessionLocal``) uses the test
database. Without it, ``settings`` still points at the application database, so
every test that opens a session must be gated:

* ``pytestmark = requires_test_database`` for a module that is DB-backed throughout;
* the ``require_test_database`` fixture (``tests/conftest.py``) as a parameter of a
  DB fixture in a module that also holds pure tests.

``tests/test_suite_database_isolation.py`` fails for any module that reaches the
database without one of these gates.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest
from sqlalchemy.engine import make_url

TEST_DATABASE_ENV = "TEST_DATABASE_URL"
SKIP_REASON = "needs TEST_DATABASE_URL (a dedicated database, never the app database)"

# The application database as configured before the rebind: an explicit
# DATABASE_URL in the environment. ``None`` means it comes from ``.env`` or the
# settings default, which :func:`application_database_url` resolves lazily.
_application_database_env: str | None = None
_bound = False


def configured_test_database_url() -> str | None:
    return os.getenv(TEST_DATABASE_ENV) or None


requires_test_database = pytest.mark.skipif(
    configured_test_database_url() is None,
    reason=SKIP_REASON,
)


def skip_unless_test_database() -> None:
    if configured_test_database_url() is None:
        pytest.skip(SKIP_REASON)


def bind_suite_to_test_database() -> None:
    """Point ``DATABASE_URL`` at the test database. Call before importing ``app``."""
    global _application_database_env, _bound
    if _bound:
        return
    _bound = True
    _application_database_env = os.environ.get("DATABASE_URL") or None
    test_url = configured_test_database_url()
    if test_url:
        # load_dotenv never overrides an existing variable, and pydantic-settings
        # ranks the environment above the dotenv file, so this wins over .env.
        os.environ["DATABASE_URL"] = test_url


def application_database_url() -> str:
    """The database the application uses, resolved as ``app.core.config`` would."""
    if _application_database_env:
        return _application_database_env
    from dotenv import dotenv_values

    from app.core.config import Settings, dotenv_path

    if dotenv_path.exists():
        for key, value in dotenv_values(dotenv_path).items():
            if key.upper() == "DATABASE_URL" and value:
                return value
    return Settings.model_fields["database_url"].default


def same_database(left: str, right: str) -> bool:
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


def refuse_application_database(test_url: str, app_url: str) -> None:
    """Raise ``UsageError`` when the configured test database is the app database."""
    if not app_url or not same_database(test_url, app_url):
        return
    target = make_url(test_url)
    raise pytest.UsageError(
        "TEST_DATABASE_URL points at the application database "
        f"({target.host}:{target.port or 5432}/{target.database}). Tests seed rows, "
        "delete them by id, and some run Base.metadata.create_all, so they must never "
        "run there. Create a dedicated database (for example `createdb chatbot_test`) "
        "and set TEST_DATABASE_URL to it."
    )


def async_test_database_available() -> bool:
    """Probe the async engine, but only once it is bound to a test database."""
    if configured_test_database_url() is None:
        return False

    from sqlalchemy import text

    from app.database import async_session

    async def probe() -> bool:
        try:
            async with async_session.AsyncSessionLocal() as session:
                await session.execute(text("select 1"))
            return True
        except Exception:  # noqa: BLE001 - any failure means "not reachable"
            return False

    if sys.platform == "win32":
        return asyncio.run(probe(), loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(probe())
