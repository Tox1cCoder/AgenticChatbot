"""API shutdown closes the process-wide clients startup opened.

The checkpoint manager's psycopg pool, the generation stop bus's Redis
subscriber and the Qdrant client were left open at exit. The sidecar's
equivalent is ``tests/client_backend/test_lifespan_shutdown.py``.
"""

from __future__ import annotations

import logging

import pytest

from app import main


class _Closable:
    def __init__(self, *, fail: bool = False) -> None:
        self.closed = 0
        self._fail = fail

    def _close(self) -> None:
        self.closed += 1
        if self._fail:
            raise ConnectionError("redis://:hunter2@10.0.0.5 went away")


class _CheckpointManager(_Closable):
    async def setup(self) -> None:
        return None

    async def cleanup(self) -> None:
        self._close()


class _Bus(_Closable):
    async def close(self) -> None:
        self._close()


class _Qdrant(_Closable):
    def close(self) -> None:
        self._close()


class _IndexService:
    def ensure_collection(self) -> None:
        return None


class _Container:
    def __init__(self, *, failing_bus: bool = False) -> None:
        self.checkpoint = _CheckpointManager()
        self.bus = _Bus(fail=failing_bus)
        self.qdrant = _Qdrant()
        self.resolved: list[str] = []

    def _resolve(self, name: str, instance):
        self.resolved.append(name)
        return instance

    def checkpoint_manager(self):
        return self._resolve("checkpoint_manager", self.checkpoint)

    def generation_control_bus(self):
        return self._resolve("generation_control_bus", self.bus)

    def qdrant_client(self):
        return self._resolve("qdrant_client", self.qdrant)

    def document_index_service(self):
        return _IndexService()


async def _noop(*_args, **_kwargs) -> None:
    return None


@pytest.fixture
def lifespan_with(monkeypatch):
    """Run the real lifespan against a fake container, database work stubbed."""
    import app.ai.mcp_registry as mcp_registry
    import app.database.async_session as async_session
    import app.services.generation_stop_subscriber as stop_subscriber

    for name in (
        "_verify_async_database_ready",
        "init_database_migrations",
        "_reclaim_orphaned_generations",
        "init_agents",
        "_terminalize_own_generations",
        "close_client_runtime_store",
    ):
        monkeypatch.setattr(main, name, _noop)
    monkeypatch.setattr(main, "_ensure_selector_event_loop", lambda: None)
    monkeypatch.setattr(main, "_log_widget_runtime_status", lambda: None)
    monkeypatch.setattr(main.settings, "enable_client_runtime_bridge", False)
    monkeypatch.setattr(mcp_registry, "get_global_mcp_manager", _noop)
    monkeypatch.setattr(async_session, "dispose_async_engine", _noop)
    monkeypatch.setattr(stop_subscriber, "install_generation_stop_subscriber", _noop)

    def _use(container: _Container, *, checkpoints: bool = True) -> None:
        monkeypatch.setattr(main.settings, "enable_langgraph_checkpoints", checkpoints)
        monkeypatch.setattr(main, "get_container", lambda: container)

    return _use


async def test_shutdown_closes_the_clients_startup_opened(lifespan_with):
    container = _Container()
    lifespan_with(container)

    async with main.lifespan(main.app):
        assert container.bus.closed == 0

    assert container.checkpoint.closed == 1
    assert container.bus.closed == 1
    assert container.qdrant.closed == 1


async def test_one_failing_close_does_not_skip_the_others(lifespan_with, caplog):
    container = _Container(failing_bus=True)
    lifespan_with(container)

    with caplog.at_level(logging.WARNING, logger="app.main"):
        async with main.lifespan(main.app):
            pass

    assert container.bus.closed == 1
    assert container.checkpoint.closed == 1
    assert container.qdrant.closed == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("generation_control_bus" in m and "ConnectionError" in m for m in messages)
    assert not any("hunter2" in m for m in messages)


async def test_shutdown_never_builds_a_client_startup_did_not(lifespan_with):
    """With checkpoints disabled startup never builds the manager, so shutdown
    must not construct one just to close it."""
    container = _Container()
    lifespan_with(container, checkpoints=False)

    async with main.lifespan(main.app):
        pass

    assert "checkpoint_manager" not in container.resolved
    assert container.checkpoint.closed == 0
    assert container.bus.closed == 1


async def test_a_second_shutdown_does_not_close_twice(lifespan_with):
    container = _Container()
    lifespan_with(container)

    async with main.lifespan(main.app):
        pass
    await main._close_startup_clients()

    assert container.bus.closed == 1
    assert container.qdrant.closed == 1


async def test_checkpoint_manager_is_closed_even_when_its_setup_failed(lifespan_with):
    """A failed setup can leave a pool half-built; cleanup is what releases it."""
    container = _Container()

    async def _failing_setup() -> None:
        raise RuntimeError("database unreachable")

    container.checkpoint.setup = _failing_setup
    lifespan_with(container)

    async with main.lifespan(main.app):
        pass

    assert container.checkpoint.closed == 1
