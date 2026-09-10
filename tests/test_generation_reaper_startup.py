"""The reaper is wired into the process lifecycle, and cannot block it.

The reclaim itself is asserted against PostgreSQL in
``tests/integration/test_generation_repository_postgres.py`` -- which rows the
predicate touches is a database guarantee and is not observable against a
fake. What is checkable here is the part that made the feature inert when it
was missing: that the lifespan actually calls it, at both ends, and that a
reaper that fails does not take the application down with it.
"""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace

import pytest

from app import main
from app.core.producer_identity import current_producer_token


class _Repository:
    """Records what the reaper asked of it."""

    def __init__(self, tokens: list[str | None] | None = None) -> None:
        self.tokens = tokens or []
        self.failed: list[tuple[list[str], str]] = []

    async def aget_active_producer_tokens(self) -> list[str | None]:
        return list(self.tokens)

    async def afail_active_by_producer(self, producer_tokens, *, terminal_reason) -> int:
        self.failed.append((list(producer_tokens), terminal_reason))
        return len(list(producer_tokens))


class _Exploding:
    async def aget_active_producer_tokens(self):
        raise RuntimeError("the database is unreachable")

    async def afail_active_by_producer(self, producer_tokens, *, terminal_reason):
        raise RuntimeError("the database is unreachable")


def _with_repository(monkeypatch: pytest.MonkeyPatch, repository: object) -> None:
    monkeypatch.setattr(
        main,
        "get_container",
        lambda: SimpleNamespace(generation_repository=lambda: repository),
    )


# ----------------------------------------------------------------------
# wiring
# ----------------------------------------------------------------------


def test_the_startup_sweep_runs_before_the_application_serves():
    """A conversation blocked by an abandoned turn must be usable on boot."""
    startup, _, shutdown = inspect.getsource(main.lifespan).partition("yield")

    assert "await _reclaim_orphaned_generations()" in startup
    assert "await _reclaim_orphaned_generations()" not in shutdown


def test_shutdown_terminalizes_this_workers_own_turns():
    """The reload case is resolved by the process that knows it is leaving."""
    _, _, shutdown = inspect.getsource(main.lifespan).partition("yield")

    assert "await _terminalize_own_generations()" in shutdown


# ----------------------------------------------------------------------
# behaviour
# ----------------------------------------------------------------------


async def test_the_startup_helper_reclaims_a_departed_workers_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unparseable token stands in for a producer this host cannot judge."""
    repository = _Repository(tokens=["definitely-not-a-token"])
    _with_repository(monkeypatch, repository)

    await main._reclaim_orphaned_generations()

    assert repository.failed == []


async def test_the_shutdown_helper_names_this_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _Repository()
    _with_repository(monkeypatch, repository)

    await main._terminalize_own_generations()

    assert repository.failed == [([current_producer_token()], "producer_shutdown")]


async def test_a_failing_sweep_does_not_stop_the_application_starting(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Blocked conversations are worth a warning, never a refusal to boot."""
    _with_repository(monkeypatch, _Exploding())

    with caplog.at_level(logging.WARNING, logger="app.main"):
        await main._reclaim_orphaned_generations()

    assert any("RuntimeError" in record.getMessage() for record in caplog.records)


async def test_a_failing_shutdown_terminalization_does_not_break_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_repository(monkeypatch, _Exploding())

    await main._terminalize_own_generations()
