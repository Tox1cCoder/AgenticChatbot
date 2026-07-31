"""The device's skill catalog: one fresh projection, generation-tagged.

Three problems this service exists to solve, none of which the registry can solve
on its own:

**Freshness without rescanning per request.** ``GET /skills`` and every mutation
response must reflect what is actually installed, but a full rescan hashes every
bundle. A short freshness window plus a single shared refresh means a burst of
concurrent requests triggers one scan, not one per request.

**Cache correctness for the frontend.** Responses arrive out of order, so a
client needs to know which of two payloads is newer. The generation is a
persisted integer that increments only when the *projection* changes -- not on
every scan -- so an unchanged catalog keeps its number and a stale response can be
recognized and dropped.

**Local success independent of the network.** Publishing the catalog to the
canonical server can fail while the local install is already committed and
working. Synchronization is therefore best-effort and reported as
``catalogSyncStatus``, never as failure of the mutation that triggered it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import get_skill_catalog_state_path
from client_backend.services.local_skills_registry import get_skills_registry
from client_backend.services.runtime_bridge import get_runtime_bridge
from client_backend.services.skill_runtime.locks import profile_lock
from client_backend.services.skill_runtime.manager import SkillRuntimeManager
from client_backend.services.skill_runtime.state import atomic_write_json, read_json_object

logger = get_logger(__name__)

_STATE_VERSION = 1


def skill_summary(skill, readiness_manager: SkillRuntimeManager | None = None) -> dict[str, Any]:
    """Project one skill into the wire shape shared by every catalog response.

    ``folderPath`` is the one deliberately local value here: it predates this
    service and the frontend contract keeps it as display-only text.
    """
    manager = readiness_manager or SkillRuntimeManager()
    readiness = manager.evaluate_readiness(skill)
    return {
        "name": skill.name,
        "description": skill.description,
        "enabled": skill.enabled,
        "folderPath": str(skill.path.parent),
        "sourceHash": getattr(skill, "source_hash", None),
        "commandCapable": readiness.status == "ready",
        "runtimeStatus": readiness.status,
        "setupStatus": readiness.setup_status,
    }


class SkillCatalogService:
    """Own catalog freshness, the persisted generation, and bridge publication."""

    def __init__(self, registry=None, bridge_factory=None) -> None:
        self._registry = registry
        self._bridge_factory = bridge_factory or get_runtime_bridge
        self._refresh_lock = asyncio.Lock()
        self._fresh_until: float = 0.0

    def _get_registry(self):
        return self._registry if self._registry is not None else get_skills_registry()

    async def snapshot(self, force: bool = False, sync: bool = False) -> dict[str, Any]:
        """Return the current catalog, rescanning only when it is stale.

        Args:
            force: Rescan regardless of the freshness window.
            sync: Also publish the projection to the runtime bridge.
        """
        await self._refresh_if_stale(force=force)
        catalog = await self._project()
        if sync:
            catalog = await self._synchronize(catalog)
        else:
            catalog["catalogSyncStatus"] = self._persisted_sync_status()
        return catalog

    async def after_mutation(self, sync: bool = True) -> dict[str, Any]:
        """Return the catalog produced by a completed local mutation.

        Always rescans: the caller just changed what is installed, so the
        freshness window is meaningless here.
        """
        return await self.snapshot(force=True, sync=sync)

    async def retry_pending_sync(self) -> dict[str, Any]:
        """Re-attempt publication for a catalog committed while offline."""
        if self._persisted_sync_status() == "synced":
            return await self.snapshot(force=False, sync=False)
        return await self.snapshot(force=False, sync=True)

    async def _refresh_if_stale(self, *, force: bool) -> None:
        """Rescan at most once for any number of concurrent callers.

        The deadline is re-checked after the lock is taken so waiters observe the
        scan the holder just completed instead of repeating it. ``time.monotonic``
        is used rather than wall time so a clock adjustment cannot extend or
        collapse the window.
        """
        if not force and time.monotonic() < self._fresh_until:
            return
        async with self._refresh_lock:
            if not force and time.monotonic() < self._fresh_until:
                return
            registry = self._get_registry()
            await registry.initialize()
            await registry.refresh()
            self._fresh_until = time.monotonic() + float(
                client_settings.skill_catalog_freshness_seconds
            )

    async def _project(self) -> dict[str, Any]:
        """Build the payload and advance the generation if the content changed."""
        registry = self._get_registry()
        manager = SkillRuntimeManager()
        skills = sorted(
            (skill_summary(skill, manager) for skill in registry.get_all_skills()),
            key=lambda entry: str(entry["name"]),
        )
        bridge = self._bridge_factory()
        generation = await self._advance_generation(skills)
        return {
            "deviceId": bridge.get_registered_device_id(),
            "catalogGeneration": generation,
            "catalogSyncStatus": self._persisted_sync_status(),
            "skills": skills,
            "totalCount": len(skills),
            "enabledCount": sum(1 for skill in skills if skill["enabled"]),
        }

    async def _advance_generation(self, skills: list[dict[str, Any]]) -> int:
        """Increment the persisted generation only when the projection differs.

        Hashes only the skill list. ``deviceId`` and ``catalogSyncStatus`` are
        excluded deliberately: a bridge reconnect changes both without changing
        what is installed, and bumping the generation for that would invalidate
        every client cache for no reason.
        """
        digest = hashlib.sha256(
            json.dumps(skills, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        user_id = self._resolve_user_id()
        if user_id is None:
            # No profile to persist against; report a stable placeholder rather
            # than inventing a generation that cannot survive a restart.
            return 0

        async with profile_lock(user_id, "catalog"):
            path = get_skill_catalog_state_path(user_id)
            state = self._read_state(path)
            if state.get("projectionHash") == digest:
                return int(state.get("generation") or 0)
            generation = int(state.get("generation") or 0) + 1
            atomic_write_json(
                path,
                {
                    "version": _STATE_VERSION,
                    "generation": generation,
                    "projectionHash": digest,
                    "syncStatus": state.get("syncStatus") or "pending",
                },
            )
            return generation

    async def _synchronize(self, catalog: dict[str, Any]) -> dict[str, Any]:
        """Publish to the runtime bridge without ever failing the caller."""
        bridge = self._bridge_factory()
        if not bridge.is_connected() or not bridge.get_registered_device_id():
            self._persist_sync_status("disconnected")
            catalog["catalogSyncStatus"] = "disconnected"
            return catalog
        try:
            await bridge.refresh_catalogs()
        except Exception as exc:  # noqa: BLE001 - the local commit already happened
            logger.warning("skill catalog synchronization failed: %s", exc)
            self._persist_sync_status("pending")
            catalog["catalogSyncStatus"] = "pending"
            return catalog
        self._persist_sync_status("synced")
        catalog["catalogSyncStatus"] = "synced"
        catalog["deviceId"] = bridge.get_registered_device_id()
        return catalog

    def _resolve_user_id(self) -> str | None:
        return self._get_registry()._resolve_current_user_id()

    @staticmethod
    def _read_state(path) -> dict[str, Any]:
        try:
            return read_json_object(path) or {}
        except Exception:
            logger.warning("discarding unreadable skill catalog state", exc_info=True)
            return {}

    def _persisted_sync_status(self) -> str:
        user_id = self._resolve_user_id()
        if user_id is None:
            return "disconnected"
        state = self._read_state(get_skill_catalog_state_path(user_id))
        status = str(state.get("syncStatus") or "pending")
        return status if status in {"synced", "pending", "disconnected"} else "pending"

    def _persist_sync_status(self, status: str) -> None:
        user_id = self._resolve_user_id()
        if user_id is None:
            return
        path = get_skill_catalog_state_path(user_id)
        state = self._read_state(path)
        if state.get("syncStatus") == status:
            return
        atomic_write_json(
            path,
            {
                "version": _STATE_VERSION,
                "generation": int(state.get("generation") or 0),
                "projectionHash": state.get("projectionHash"),
                "syncStatus": status,
            },
        )

    def invalidate(self) -> None:
        """Force the next snapshot to rescan; used on profile transitions."""
        self._fresh_until = 0.0


_catalog_service: SkillCatalogService | None = None


def get_skill_catalog_service() -> SkillCatalogService:
    """Return the process-wide catalog service."""
    global _catalog_service
    if _catalog_service is None:
        _catalog_service = SkillCatalogService()
    return _catalog_service


def close_skill_catalog_service() -> None:
    """Drop the cached service so the next call rebuilds its state."""
    global _catalog_service
    _catalog_service = None
