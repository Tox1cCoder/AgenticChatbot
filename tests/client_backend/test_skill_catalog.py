"""Freshness, generation, and degraded-sync behavior of the skill catalog.

Three invariants a frontend cache depends on: the generation identifies which of
two responses is newer, concurrent readers do not each trigger a rescan, and a
publication failure never turns a committed local install into a failure.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from client_backend.services.skill_catalog import (
    SkillCatalogService,
    close_skill_catalog_service,
    get_skill_catalog_service,
)

USER_ID = "user-a"


class _ReadinessStub:
    def evaluate_readiness(self, skill):
        return SimpleNamespace(status="ready", setup_status="not_required")


class _RegistryStub:
    """Counts refreshes so shared-scan behavior is observable."""

    def __init__(self) -> None:
        self._skills: dict[str, SimpleNamespace] = {}
        self.refresh_calls = 0
        self.initialize_calls = 0
        self.add_skill("demo")

    def add_skill(self, name: str, *, enabled: bool = True) -> None:
        self._skills[name] = SimpleNamespace(
            name=name,
            description=f"{name} description",
            enabled=enabled,
            path=Path("/profile") / name / "SKILL.md",
            source_hash="a" * 64,
        )

    async def initialize(self) -> None:
        self.initialize_calls += 1

    async def refresh(self) -> int:
        self.refresh_calls += 1
        return 0

    def get_all_skills(self):
        return list(self._skills.values())

    def _resolve_current_user_id(self) -> str:
        return USER_ID


class _BridgeStub:
    def __init__(self) -> None:
        self.connected = True
        self.device_id = "device-123"
        self.refresh_calls = 0
        self.raise_on_refresh: Exception | None = None

    def is_connected(self) -> bool:
        return self.connected

    def get_registered_device_id(self):
        return self.device_id

    async def refresh_catalogs(self) -> None:
        self.refresh_calls += 1
        if self.raise_on_refresh is not None:
            raise self.raise_on_refresh


@dataclass
class _CatalogEnv:
    service: SkillCatalogService
    registry: _RegistryStub
    bridge: _BridgeStub
    state_path: Path

    def add_skill(self, name: str, *, enabled: bool = True) -> None:
        self.registry.add_skill(name, enabled=enabled)

    def new_service(self) -> SkillCatalogService:
        """A second service over the same persisted state, as after a restart."""
        return SkillCatalogService(registry=self.registry, bridge_factory=lambda: self.bridge)


@pytest.fixture()
def catalog_env(tmp_path, monkeypatch) -> _CatalogEnv:
    catalog_globals = SkillCatalogService.__init__.__globals__
    locks_globals = catalog_globals["profile_lock"].__wrapped__.__globals__
    state_path = tmp_path / "catalog_state.json"

    monkeypatch.setitem(catalog_globals, "get_skill_catalog_state_path", lambda _user: state_path)
    monkeypatch.setitem(
        catalog_globals,
        "SkillRuntimeManager",
        _ReadinessStub,
    )
    monkeypatch.setitem(locks_globals, "get_skill_locks_root", lambda _user: tmp_path / "locks")

    registry = _RegistryStub()
    bridge = _BridgeStub()
    service = SkillCatalogService(registry=registry, bridge_factory=lambda: bridge)
    monkeypatch.setattr(
        catalog_globals["client_settings"],
        "skill_catalog_freshness_seconds",
        1.0,
    )
    return _CatalogEnv(service=service, registry=registry, bridge=bridge, state_path=state_path)


@pytest.mark.asyncio
async def test_catalog_generation_changes_only_when_projection_changes(catalog_env):
    first = await catalog_env.service.snapshot(force=True)
    second = await catalog_env.service.snapshot(force=True)
    catalog_env.add_skill("new-skill")
    third = await catalog_env.service.snapshot(force=True)

    assert second["catalogGeneration"] == first["catalogGeneration"]
    assert third["catalogGeneration"] == first["catalogGeneration"] + 1


@pytest.mark.asyncio
async def test_generation_advances_when_a_skill_is_toggled(catalog_env):
    first = await catalog_env.service.snapshot(force=True)
    catalog_env.add_skill("demo", enabled=False)
    second = await catalog_env.service.snapshot(force=True)

    assert second["catalogGeneration"] == first["catalogGeneration"] + 1
    assert second["enabledCount"] == 0


@pytest.mark.asyncio
async def test_generation_survives_a_restart(catalog_env):
    first = await catalog_env.service.snapshot(force=True)
    catalog_env.add_skill("second-skill")
    await catalog_env.service.snapshot(force=True)

    restarted = catalog_env.new_service()
    catalog_env.add_skill("third-skill")
    after_restart = await restarted.snapshot(force=True)

    assert after_restart["catalogGeneration"] == first["catalogGeneration"] + 2


@pytest.mark.asyncio
async def test_reconnect_alone_does_not_advance_the_generation(catalog_env):
    """A bridge reconnect changes deviceId and sync status, not what is installed."""
    first = await catalog_env.service.snapshot(force=True, sync=True)
    catalog_env.bridge.connected = False
    second = await catalog_env.service.snapshot(force=True, sync=True)
    catalog_env.bridge.connected = True
    third = await catalog_env.service.snapshot(force=True, sync=True)

    assert first["catalogGeneration"] == second["catalogGeneration"] == third["catalogGeneration"]
    assert second["catalogSyncStatus"] == "disconnected"
    assert third["catalogSyncStatus"] == "synced"


@pytest.mark.asyncio
async def test_concurrent_freshness_checks_share_one_scan(catalog_env):
    await asyncio.gather(*(catalog_env.service.snapshot(force=False) for _ in range(10)))

    assert catalog_env.registry.refresh_calls == 1


@pytest.mark.asyncio
async def test_catalog_scan_waits_for_the_skills_mutation_lock(catalog_env, monkeypatch):
    catalog_globals = SkillCatalogService.__init__.__globals__
    assert "SKILLS_MUTATION_SCOPE" in catalog_globals, (
        "catalog scans must share the installed-bundle mutation scope"
    )
    mutation_scope = catalog_globals["SKILLS_MUTATION_SCOPE"]
    original_lock = catalog_globals["profile_lock"]
    attempted = asyncio.Event()

    @asynccontextmanager
    async def observed_lock(user_id: str, scope: str):
        attempted.set()
        async with original_lock(user_id, scope):
            yield

    monkeypatch.setitem(catalog_globals, "profile_lock", observed_lock)
    async with original_lock(USER_ID, mutation_scope):
        snapshot_task = asyncio.create_task(catalog_env.service.snapshot(force=True))
        await asyncio.wait_for(attempted.wait(), timeout=1)
        assert not snapshot_task.done()

    snapshot = await snapshot_task
    assert snapshot["skills"]


@pytest.mark.asyncio
async def test_bounded_freshness_window_is_reused_then_expires(catalog_env, monkeypatch):
    catalog_globals = SkillCatalogService.__init__.__globals__
    clock = {"now": 1000.0}
    monkeypatch.setitem(
        catalog_globals,
        "time",
        SimpleNamespace(monotonic=lambda: clock["now"]),
    )

    await catalog_env.service.snapshot(force=False)
    await catalog_env.service.snapshot(force=False)
    assert catalog_env.registry.refresh_calls == 1

    clock["now"] += 1.5
    await catalog_env.service.snapshot(force=False)
    assert catalog_env.registry.refresh_calls == 2


@pytest.mark.asyncio
async def test_force_bypasses_the_freshness_window(catalog_env):
    await catalog_env.service.snapshot(force=False)
    await catalog_env.service.snapshot(force=True)

    assert catalog_env.registry.refresh_calls == 2


@pytest.mark.asyncio
async def test_sync_failure_returns_committed_catalog_as_pending(catalog_env):
    catalog_env.bridge.raise_on_refresh = RuntimeError("offline")

    snapshot = await catalog_env.service.after_mutation(sync=True)

    assert snapshot["catalogSyncStatus"] == "pending"
    assert snapshot["skills"]
    assert snapshot["totalCount"] == 1


@pytest.mark.asyncio
async def test_disconnected_bridge_reports_disconnected_without_calling_it(catalog_env):
    catalog_env.bridge.connected = False

    snapshot = await catalog_env.service.after_mutation(sync=True)

    assert snapshot["catalogSyncStatus"] == "disconnected"
    assert catalog_env.bridge.refresh_calls == 0


@pytest.mark.asyncio
async def test_pending_sync_status_persists_and_can_be_retried(catalog_env):
    catalog_env.bridge.raise_on_refresh = RuntimeError("offline")
    await catalog_env.service.after_mutation(sync=True)

    reloaded = catalog_env.new_service()
    assert (await reloaded.snapshot(force=True))["catalogSyncStatus"] == "pending"

    catalog_env.bridge.raise_on_refresh = None
    retried = await reloaded.retry_pending_sync()

    assert retried["catalogSyncStatus"] == "synced"


@pytest.mark.asyncio
async def test_retry_is_a_no_op_once_synced(catalog_env):
    await catalog_env.service.after_mutation(sync=True)
    calls_after_sync = catalog_env.bridge.refresh_calls

    await catalog_env.service.retry_pending_sync()

    assert catalog_env.bridge.refresh_calls == calls_after_sync


@pytest.mark.asyncio
async def test_snapshot_shape_keeps_existing_list_fields(catalog_env):
    snapshot = await catalog_env.service.snapshot(force=True)

    assert snapshot["deviceId"] == "device-123"
    assert snapshot["totalCount"] == len(snapshot["skills"])
    assert snapshot["enabledCount"] == 1
    entry = snapshot["skills"][0]
    assert set(entry) == {
        "name",
        "description",
        "enabled",
        "folderPath",
        "sourceHash",
        "commandCapable",
        "runtimeStatus",
        "setupStatus",
    }


@pytest.mark.asyncio
async def test_skills_are_sorted_so_the_projection_hash_is_stable(catalog_env):
    catalog_env.add_skill("alpha")
    catalog_env.add_skill("zulu")
    first = await catalog_env.service.snapshot(force=True)

    names = [entry["name"] for entry in first["skills"]]
    assert names == sorted(names)

    generation = first["catalogGeneration"]
    assert (await catalog_env.service.snapshot(force=True))["catalogGeneration"] == generation


@pytest.mark.asyncio
async def test_unreadable_state_is_discarded_rather_than_crashing(catalog_env):
    catalog_env.state_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_env.state_path.write_text("not json", encoding="utf-8")

    snapshot = await catalog_env.service.snapshot(force=True)

    assert snapshot["catalogGeneration"] == 1


def test_service_singleton_is_resettable():
    first = get_skill_catalog_service()
    assert get_skill_catalog_service() is first

    close_skill_catalog_service()

    assert get_skill_catalog_service() is not first
    close_skill_catalog_service()
