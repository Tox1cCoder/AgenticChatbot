"""Startup must fail fast when the loop cannot serve the async engine.

The async SQLAlchemy engine is created unconditionally and psycopg's async mode
cannot run on a Windows ProactorEventLoop. Before this, the check was skipped
whenever ``enable_langgraph_checkpoints`` was false — so with that flag off the
app booted "healthy" and then failed on every request's database call, and the
error message told operators to disable checkpoints, which made it worse.
"""

from __future__ import annotations

import asyncio

import pytest

from app.main import _selector_loop_error

_proactor_factory = getattr(asyncio, "ProactorEventLoop", None)
requires_proactor = pytest.mark.skipif(
    _proactor_factory is None, reason="ProactorEventLoop exists only on Windows"
)


@pytest.fixture
def proactor_loop():
    loop = _proactor_factory()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture
def selector_loop():
    loop = asyncio.SelectorEventLoop()
    try:
        yield loop
    finally:
        loop.close()


@requires_proactor
@pytest.mark.parametrize("checkpoints_enabled", [True, False])
def test_proactor_loop_is_rejected_regardless_of_checkpoint_setting(
    monkeypatch, proactor_loop, checkpoints_enabled
):
    """The async engine needs a selector loop whether or not checkpoints are on."""
    from app.core import config

    monkeypatch.setattr(config.settings, "enable_langgraph_checkpoints", checkpoints_enabled)

    message = _selector_loop_error(proactor_loop, platform="win32")

    assert message is not None
    assert "ProactorEventLoop" in message


@requires_proactor
def test_error_message_does_not_advise_disabling_checkpoints(proactor_loop):
    """That advice used to be an escape hatch; it no longer helps and misleads."""
    message = _selector_loop_error(proactor_loop, platform="win32")

    assert message is not None
    assert "ENABLE_LANGGRAPH_CHECKPOINTS" not in message
    # It must still tell the operator how to launch correctly.
    assert "WindowsSelectorEventLoopPolicy" in message


def test_selector_loop_is_accepted(selector_loop):
    assert _selector_loop_error(selector_loop, platform="win32") is None


@requires_proactor
def test_non_windows_platforms_are_never_rejected(proactor_loop):
    """Only Windows has the Proactor/psycopg conflict."""
    assert _selector_loop_error(proactor_loop, platform="linux") is None
