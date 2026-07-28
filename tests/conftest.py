"""Shared pytest configuration and fixtures.

This module is auto-loaded by pytest and provides shared utilities for all tests.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import sys
import tempfile

import pytest

# Set this before pytest imports application modules. Client settings otherwise
# fall through to the developer's live LOCALAPPDATA profile, where enabled MCP
# processes must never be started by the test suite.
_PYTEST_RUNTIME = tempfile.TemporaryDirectory(
    prefix="sample-chatbot-pytest-",
    ignore_cleanup_errors=True,
)
os.environ["CLIENT_PROFILE_ROOT"] = _PYTEST_RUNTIME.name
atexit.register(_PYTEST_RUNTIME.cleanup)


class FakePopen:
    """Minimal subprocess.Popen stub that records the command and supports wait()."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.pid = 99999

    def wait(self):
        return 0

    def terminate(self):
        pass

    def poll(self):
        return 0


# Export FakePopen for use in test modules
__all__ = ["FakePopen"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "selector_event_loop: run this async test on a SelectorEventLoop "
        "(required for async psycopg on Windows)",
    )


def pytest_asyncio_loop_factories(config, item):
    """Choose the event loop implementation per test, Windows only.

    Two Windows constraints pull in opposite directions and cannot be satisfied
    by one loop type:

    * async psycopg refuses to run on a ``ProactorEventLoop`` and raises
      ``InterfaceError`` at connect time, so database tests need a
      ``SelectorEventLoop`` (``app/main.py:440`` sets the equivalent policy in
      production); and
    * ``SelectorEventLoop`` cannot spawn subprocesses, which
      ``tests/client_backend/test_skill_execution_engine.py`` genuinely does.

    Proactor therefore stays the default — preserving existing behavior — and
    async database tests opt in with ``pytest.mark.selector_event_loop``.

    Returning exactly one factory matters: pytest-asyncio parametrizes across
    every factory returned, so a two-entry mapping would run each async test
    twice.
    """
    if sys.platform != "win32":
        return {"default": asyncio.new_event_loop}
    if item.get_closest_marker("selector_event_loop"):
        return {"selector": asyncio.SelectorEventLoop}
    return {"proactor": asyncio.ProactorEventLoop}


@pytest.fixture(scope="session")
def _async_db_available() -> bool:
    """Probe the async engine once per session."""
    from sqlalchemy import text

    from app.database.async_session import AsyncSessionLocal

    async def probe() -> bool:
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(text("select 1"))
            return True
        except Exception:
            return False

    if sys.platform == "win32":
        return asyncio.run(probe(), loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(probe())


@pytest.fixture
def require_async_db(_async_db_available: bool) -> None:
    """Skip a test that needs a reachable PostgreSQL."""
    if not _async_db_available:
        pytest.skip("async PostgreSQL is not reachable")
