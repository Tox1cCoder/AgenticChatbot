# Custom-Agent Device Capability Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make account-wide Custom Agent MCP/skill selections resolve safely against the active sidecar, report missing capabilities without disabling the agent, and eliminate false-ready empty catalog snapshots.

**Architecture:** Treat saved client references as account-wide desired capabilities keyed by exact logical identity, then resolve them into the requesting device's current session/catalog identity before strict authorization and dispatch. A shared pure resolver drives management availability and runtime warnings, while the sidecar publishes readiness only after both catalogs sync. Skills and HITL remain device-local and receive explicit two-device regression coverage.

**Tech Stack:** Python 3.10+ (verified with the repository's Python 3.13 virtualenv), FastAPI, Pydantic v2, SQLAlchemy 2/JSONB, LangChain tools, Redis-backed runtime sessions, Streamlit, pytest, Ruff.

---

## Spec Kit Inputs

- Design specification: `docs/superpowers/specs/2026-07-23-custom-agent-device-capability-resolution-design.md`
- Existing frontend contract: `plans/CUSTOM_AGENTS_FE_CONTRACT.md`
- Existing device/HITL contract: `plans/SKILLS_MCP_HITL_FE_CONTRACT.md`
- Existing reconnect behavior: `tests/test_custom_agent_client_tool_resync.py`
- Existing device isolation: `tests/test_multi_sidecar_hardening.py`, `tests/test_client_tool_isolation.py`, `tests/test_hitl_settings_device_isolation.py`

## Plan Audit Corrections

This plan was re-checked against the codebase on 2026-07-23. The following details
are intentional corrections to the first draft and are requirements for execution:

1. A server session is not a ready device snapshot until **both** catalog version
   counters are greater than zero. A connected-but-not-yet-synced session returns an
   unavailable snapshot with empty client options, so a concurrent direct API request
   cannot label the transient empty catalogs as ready.
2. Management reads load snapshot identity before and after their tool/skill reads.
   They return a ready snapshot only when those identities agree, preventing options
   from being cached under metadata belonging to a different reconnect/resync.
3. The resolver returns only live, current-session client refs for runtime binding.
   Missing saved refs stay in persistence/read payloads, but never enter the runtime
   allowlist. This yields one resolver warning per missing logical capability instead
   of a second, differently worded strict-filter warning for the same absence.
4. A **new** client-tool selection must match the full active option identity
   `(server_name, qualified_tool_id, device_id, session_id, catalog_version,
   tool_instance_id, tool_name)`. Stable logical identity is used only to retain an
   already-saved intent. This prevents a fabricated binding from passing validation
   merely because its server/tool pair exists.
5. Existing legacy `source="server"` skill refs may be retained and resolved as
   client refs, but newly submitted legacy-source refs are not accepted as current
   options. Tool and skill selections are both deduplicated by normalized logical
   identity.
6. Test fixtures distinguish the model-callable name (`client__csv__profile`) from
   the real qualified ID (`csv::profile`), matching `client_runtime_tools.py` and
   `local_mcp_manager.py`.

## Summary

The implementation has four boundaries:

1. The sidecar readiness event moves behind successful MCP and skill catalog synchronization.
2. A pure resolver converts saved logical MCP/skill selections into an active-device availability result and current exact tool refs.
3. Custom Agent management routes expose snapshot identity and structured `ready`/`degraded`/`device_unavailable` status.
4. Runtime binding uses the same resolver, retains exact dispatch validation, and surfaces missing capabilities in the model prompt and `custom_agent_warnings` metadata.

No database migration is needed. Existing JSONB refs already contain `server_name`, `qualified_tool_id`, and skill lookup identities. Device/session/catalog/instance fields remain last-bound compatibility metadata and are replaced only in an in-memory runtime ref.

## Technical Context

| Concern | Decision |
|---|---|
| Persistence | Keep `custom_agents.tool_refs` and `skill_refs` JSONB unchanged. |
| Stable MCP identity | Exact `(server_name, qualified_tool_id)`. No fuzzy or bare-name matching. |
| Stable skill identity | Client source plus compatible `lookup_name`/`name`; legacy saved `source="server"` normalizes to client because server-owned skills were removed. |
| Runtime authorization | Rebuild only resolved refs with current device/session/catalog/instance identity, then retain the existing five-field strict matcher and sidecar validation. Missing refs never enter the runtime allowlist. |
| New-selection validation | Require a full exact active-option identity. Stable logical keys authorize retention only when that intent already exists on the same agent. |
| Missing capability | Skip capability, mark agent degraded, emit one warning per logical key before and after execution, continue agent run. |
| Device ownership | Reuse `ClientDeviceService.lookup_active_session()` plus exact `session.user_id` comparison. |
| API compatibility | Add `deviceSnapshot` and `availability`; preserve current fields and aliases. |
| Snapshot readiness | `ready` requires caller ownership and both catalog versions `> 0`; otherwise client options are empty and the snapshot is `unavailable`. |
| Cache boundary | `(deviceId, sessionId, toolCatalogVersion, skillCatalogVersion)` read consistently before/after catalog materialization. Never cache an unavailable snapshot as ready data. |
| Test database | Existing Custom Agent service/API tests use configured PostgreSQL because JSONB is required. |

## Constitution Check

- **Client ownership:** PASS. Skill content and MCP execution remain on the active sidecar.
- **Fail-closed execution:** PASS. Logical resolution never bypasses exact runtime metadata checks.
- **Multi-device isolation:** PASS. Callers supply request-scoped catalogs and the resolver independently rejects MCP candidates whose `device_id` differs; another active device is never scanned.
- **Backward compatibility:** PASS. Changes are additive and old exact refs remain parseable.
- **Graceful degradation:** PASS. Missing optional local capabilities do not disable prompt/model/server-tool behavior.
- **Secrets and paths:** PASS. Availability contains names and stable IDs only.
- **Schema migration:** NOT REQUIRED. No relational or JSON shape rewrite is necessary.

## Public Contract

Add these camel-cased response shapes in `app/schemas/custom_agent.py`:

```python
CapabilityAvailabilityStatus = Literal["ready", "degraded", "device_unavailable"]
DeviceSnapshotStatus = Literal["ready", "unavailable"]


class DeviceCatalogSnapshot(_CamelModel):
    device_id: str | None = None
    session_id: str | None = None
    tool_catalog_version: int | None = None
    skill_catalog_version: int | None = None
    status: DeviceSnapshotStatus


class MissingClientTool(_CamelModel):
    server_name: str
    qualified_tool_id: str
    tool_name: str | None = None


class MissingClientSkill(_CamelModel):
    lookup_name: str
    name: str


class CustomAgentAvailability(_CamelModel):
    status: CapabilityAvailabilityStatus
    device_id: str | None = None
    session_id: str | None = None
    missing_tools: list[MissingClientTool] = Field(default_factory=list)
    missing_skills: list[MissingClientSkill] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

Extend existing models:

```python
class CustomAgentRead(_CamelModel):
    # Existing fields remain unchanged.
    availability: CustomAgentAvailability | None = None


class CustomAgentOptions(_CamelModel):
    # Existing fields remain unchanged.
    device_snapshot: DeviceCatalogSnapshot
```

`DeviceCatalogSnapshot.status="unavailable"` means there is no complete,
caller-owned snapshot to consume. It covers an offline/unowned device and the short
window where a WebSocket session exists but either catalog version is still zero.
The version fields may therefore be `0` (sync in progress) or `null` (no session);
only `status="ready"` authorizes use/caching of the returned client option lists.

Example degraded response:

```json
{
  "availability": {
    "status": "degraded",
    "deviceId": "device-b",
    "sessionId": "session-b",
    "missingTools": [{
      "serverName": "desktop-commander",
      "qualifiedToolId": "desktop-commander::read_file",
      "toolName": "read_file"
    }],
    "missingSkills": [{
      "lookupName": "kobo-library",
      "name": "kobo-library"
    }],
    "warnings": [
      "Selected MCP tool 'desktop-commander::read_file' is not available on this device.",
      "Selected skill 'kobo-library' is not available on this device."
    ]
  }
}
```

## Source Map

### Create

- `app/services/custom_agent_capability_resolver.py`: pure stable-identity matching, exact ref rebasing, missing sets, warnings, and availability status.
- `tests/test_custom_agent_capability_resolver.py`: resolver unit tests independent of database/runtime singletons.

### Modify

- `client_backend/services/runtime_bridge.py`: readiness ordering.
- `tests/client_backend/test_runtime_bridge.py`: readiness race and failure regressions.
- `app/schemas/custom_agent.py`: snapshot and availability response models.
- `app/services/custom_agent_service.py`: snapshot lookup, contextual reads/options, portable validation, stable deduplication.
- `app/api/custom_agents.py`: accept device context for list/single reads.
- `app/ai/custom_agent_runtime.py`: delegate portable rebasing to the shared resolver.
- `app/ai/agents/custom_agent.py`: resolve MCP/skills together and inject warnings into prompt/metadata.
- `tests/test_custom_agents_service.py`: management, validation, dedupe, and status tests.
- `tests/test_custom_agents_api.py`: camel-case API contract tests.
- `tests/test_custom_agent_client_tool_resync.py`: machine-A selection rebinding to machine B.
- `tests/test_custom_agents_tools.py`: strict runtime and missing-skill warnings.
- `tests/test_custom_agents_graph.py`: warning prompt/metadata behavior.
- `client_backend/api/proxy.py`: no production logic change expected; list/single stamping is locked by tests.
- `tests/client_backend/test_custom_agents_proxy.py`: options/list/read cache-context boundary.
- `demo.py`: stable cross-device matching, degraded hints, and preservation of missing refs.
- `tests/test_demo_custom_agents.py`: Streamlit helper regressions.
- `tests/test_hitl_settings_device_isolation.py`: same-server-name device policy isolation.
- `tests/test_hitl_turn_policy_injection.py`, `tests/test_hitl_gate_policy.py`: adjacent verification that the selected device policy reaches the execution gate.
- `tests/test_skill_device_isolation.py`: same-user client skill isolation.
- `plans/CUSTOM_AGENTS_FE_CONTRACT.md`: frontend state, cache, matching, and warnings.
- `README.md`: concise account-wide intent/device-local execution behavior.

## Dependency Graph

```text
T001 readiness race

T002 schemas ──> T004 management availability ──> T007 API/UI contract
                  ^
T003 resolver ────┼──> T005 validation/dedup
                  └──> T006 runtime warnings

T004 + T006 + T007 ──> T008 isolation audit ──> T009 final verification
```

Tasks T001 and T002 can run independently. T003 is the shared implementation boundary and must land before T004–T006. T007 depends on the response schema and service behavior. T008 is evidence-only hardening after behavior is green.

---

### Task 1 (T001): Publish sidecar readiness only after catalog synchronization

**Files:**
- Modify: `tests/client_backend/test_runtime_bridge.py`
- Modify: `client_backend/services/runtime_bridge.py:387-439`

- [ ] **Step 1: Write failing readiness-order tests**

Append these tests:

```python
@pytest.mark.asyncio
async def test_initial_ready_event_waits_for_catalog_sync(monkeypatch):
    bridge = RuntimeBridgeService(server_client=_ServerClientStub())
    sync_started = asyncio.Event()
    release_sync = asyncio.Event()

    async def _blocked_refresh():
        sync_started.set()
        await release_sync.wait()

    monkeypatch.setattr(bridge, "refresh_catalogs", _blocked_refresh)
    task = asyncio.create_task(bridge._sync_initial_catalogs_and_mark_ready())
    await sync_started.wait()

    assert bridge._connected_event.is_set() is False

    release_sync.set()
    await task
    assert bridge._connected_event.is_set() is True


@pytest.mark.asyncio
async def test_failed_initial_catalog_sync_never_publishes_ready(monkeypatch):
    bridge = RuntimeBridgeService(server_client=_ServerClientStub())

    async def _failed_refresh():
        raise RuntimeError("catalog upload failed")

    monkeypatch.setattr(bridge, "refresh_catalogs", _failed_refresh)

    with pytest.raises(RuntimeError, match="catalog upload failed"):
        await bridge._sync_initial_catalogs_and_mark_ready()

    assert bridge._connected_event.is_set() is False
```

- [ ] **Step 2: Run the tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/client_backend/test_runtime_bridge.py -k "initial_ready_event or failed_initial_catalog" -q
```

Expected: both tests fail because `_sync_initial_catalogs_and_mark_ready` does not exist.

- [ ] **Step 3: Add the readiness boundary and move the event**

Add this method to `RuntimeBridgeService`:

```python
async def _sync_initial_catalogs_and_mark_ready(self) -> None:
    """Publish readiness only after both device catalogs are durable upstream."""
    self._connected_event.clear()
    await self.refresh_catalogs()
    self._connected_event.set()
```

Change `_connect_and_serve()` so the handshake state update is followed by:

```python
self._set_state(
    status=RuntimeStatus.CONNECTED,
    connected_at=now,
    last_heartbeat=now,
    session_id=self._session_id,
    error_message=None,
)

await self._sync_initial_catalogs_and_mark_ready()
self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
```

Remove the earlier `self._connected_event.set()` and direct `await self.refresh_catalogs()` statements. Keep the `finally` block clearing the event.

- [ ] **Step 4: Run focused bridge tests and verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/client_backend/test_runtime_bridge.py -q
```

Expected: all runtime bridge tests pass.

- [ ] **Step 5: Commit**

```powershell
git add client_backend/services/runtime_bridge.py tests/client_backend/test_runtime_bridge.py
git commit -m "fix: wait for sidecar catalogs before readiness"
```

### Task 2 (T002): Add typed snapshot and availability contracts

**Files:**
- Modify: `tests/test_custom_agents_service.py:42-139`
- Modify: `app/schemas/custom_agent.py`

- [ ] **Step 1: Write failing schema serialization tests**

Add imports for `CustomAgentAvailability`, `CustomAgentOptions`, and `DeviceCatalogSnapshot`, then add:

```python
def test_custom_agent_availability_serializes_camel_case():
    value = CustomAgentAvailability(
        status="degraded",
        device_id="device-b",
        session_id="session-b",
        missing_tools=[
            {
                "server_name": "desktop-commander",
                "qualified_tool_id": "desktop-commander::read_file",
                "tool_name": "read_file",
            }
        ],
        missing_skills=[{"lookup_name": "kobo-library", "name": "kobo-library"}],
        warnings=["missing"],
    )

    payload = value.model_dump(mode="json", by_alias=True)

    assert payload["deviceId"] == "device-b"
    assert payload["missingTools"][0]["qualifiedToolId"] == (
        "desktop-commander::read_file"
    )
    assert payload["missingSkills"][0]["lookupName"] == "kobo-library"


def test_custom_agent_options_requires_explicit_device_snapshot():
    options = CustomAgentOptions(
        device_snapshot=DeviceCatalogSnapshot(status="unavailable")
    )

    payload = options.model_dump(mode="json", by_alias=True)

    assert payload["deviceSnapshot"] == {
        "deviceId": None,
        "sessionId": None,
        "toolCatalogVersion": None,
        "skillCatalogVersion": None,
        "status": "unavailable",
    }
```

- [ ] **Step 2: Run the schema tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_custom_agents_service.py -k "availability_serializes or explicit_device_snapshot" -q
```

Expected: import/validation failures for the missing response models and field.

- [ ] **Step 3: Implement the public Pydantic models**

In `app/schemas/custom_agent.py`, import `Literal` if not already present and add exactly the contract models from the **Public Contract** section. Extend `CustomAgentRead` with:

```python
availability: CustomAgentAvailability | None = None
```

Extend `CustomAgentOptions` with:

```python
device_snapshot: DeviceCatalogSnapshot
```

- [ ] **Step 4: Run the schema tests and verify GREEN**

Run the Step 2 command. Expected: both tests pass.

- [ ] **Step 5: Commit**

```powershell
git add app/schemas/custom_agent.py tests/test_custom_agents_service.py
git commit -m "feat: define custom agent availability contract"
```

### Task 3 (T003): Build the shared exact capability resolver

**Files:**
- Create: `app/services/custom_agent_capability_resolver.py`
- Create: `tests/test_custom_agent_capability_resolver.py`

- [ ] **Step 1: Write failing resolver tests**

Create the test file:

```python
from app.services.custom_agent_capability_resolver import (
    client_tool_binding_key,
    client_tool_logical_key,
    resolve_custom_agent_capabilities,
    skill_refs_match,
)


SAVED_TOOL = {
    "type": "client",
    "device_id": "device-a",
    "session_id": "session-a",
    "catalog_version": "1",
    "tool_instance_id": "instance-a",
    "server_name": "desktop-commander",
    "qualified_tool_id": "desktop-commander::read_file",
    "tool_name": "read_file",
}


def _live_tool(**overrides):
    value = {
        "type": "client",
        "device_id": "device-b",
        "session_id": "session-b",
        "catalog_version": "4",
        "tool_instance_id": "instance-b",
        "server_name": "desktop-commander",
        "qualified_tool_id": "desktop-commander::read_file",
        "tool_name": "read_file",
    }
    value.update(overrides)
    return value


def test_same_logical_mcp_rebinds_to_current_device_identity():
    result = resolve_custom_agent_capabilities(
        selected_tool_refs=[SAVED_TOOL],
        selected_skill_refs=[],
        live_tool_refs=[_live_tool()],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=True,
    )

    assert result.status == "ready"
    assert result.missing_tools == []
    assert result.resolved_client_tool_refs[0]["device_id"] == "device-b"
    assert result.resolved_client_tool_refs[0]["session_id"] == "session-b"
    assert result.resolved_client_tool_refs[0]["tool_instance_id"] == "instance-b"


def test_same_tool_name_under_different_server_is_missing():
    result = resolve_custom_agent_capabilities(
        selected_tool_refs=[SAVED_TOOL],
        selected_skill_refs=[],
        live_tool_refs=[
            _live_tool(
                server_name="other-server",
                qualified_tool_id="other-server::read_file",
            )
        ],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=True,
    )

    assert result.status == "degraded"
    assert result.resolved_client_tool_refs == []
    assert result.missing_tools[0]["qualified_tool_id"] == (
        "desktop-commander::read_file"
    )


def test_missing_skill_degrades_and_legacy_server_source_matches_client_once():
    missing = resolve_custom_agent_capabilities(
        selected_tool_refs=[],
        selected_skill_refs=[
            {"source": "client", "lookup_name": "kobo-library", "name": "kobo-library"}
        ],
        live_tool_refs=[],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=True,
    )
    compatible = resolve_custom_agent_capabilities(
        selected_tool_refs=[],
        selected_skill_refs=[
            {"source": "server", "lookup_name": "kobo-library", "name": "kobo-library"}
        ],
        live_tool_refs=[],
        live_skill_refs=[
            {"source": "client", "lookup_name": "kobo-library", "name": "kobo-library"}
        ],
        request_device_id="device-b",
        device_available=True,
    )

    assert missing.status == "degraded"
    assert missing.missing_skills[0]["lookup_name"] == "kobo-library"
    assert compatible.status == "ready"
    assert missing.warnings == [
        "Selected skill 'kobo-library' is not available on this device."
    ]


def test_local_dependencies_without_session_are_device_unavailable():
    result = resolve_custom_agent_capabilities(
        selected_tool_refs=[SAVED_TOOL],
        selected_skill_refs=[],
        live_tool_refs=[],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=False,
    )

    assert result.status == "device_unavailable"
    assert result.resolved_client_tool_refs == []
    assert len(result.warnings) == 1


def test_same_logical_tool_on_non_request_device_is_not_a_candidate():
    result = resolve_custom_agent_capabilities(
        selected_tool_refs=[SAVED_TOOL],
        selected_skill_refs=[],
        live_tool_refs=[_live_tool(device_id="device-b")],
        live_skill_refs=[],
        request_device_id="device-a",
        device_available=True,
    )

    assert result.status == "degraded"
    assert result.resolved_client_tool_refs == []


def test_client_tool_logical_key_rejects_incomplete_or_server_refs():
    assert client_tool_logical_key(SAVED_TOOL) == (
        "desktop-commander",
        "desktop-commander::read_file",
    )
    assert client_tool_logical_key({"type": "client", "qualified_tool_id": "x::y"}) is None
    assert client_tool_logical_key({"type": "server_mcp", "server_name": "x", "qualified_tool_id": "x::y"}) is None


def test_binding_key_covers_every_submitted_live_identity_field():
    assert client_tool_binding_key(SAVED_TOOL) == (
        "desktop-commander",
        "desktop-commander::read_file",
        "device-a",
        "session-a",
        "1",
        "instance-a",
        "read_file",
    )


def test_duplicate_missing_refs_emit_one_missing_entry_and_warning():
    result = resolve_custom_agent_capabilities(
        selected_tool_refs=[SAVED_TOOL, dict(SAVED_TOOL)],
        selected_skill_refs=[],
        live_tool_refs=[],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=True,
    )

    assert len(result.missing_tools) == 1
    assert len(result.warnings) == 1


def test_legacy_skill_ref_matches_current_client_ref_but_not_another_name():
    legacy = {"source": "server", "lookup_name": "kobo-library", "name": "kobo-library"}
    current = {"source": "client", "lookup_name": "kobo-library", "name": "kobo-library"}
    other = {"source": "client", "lookup_name": "calendar", "name": "calendar"}

    assert skill_refs_match(legacy, current) is True
    assert skill_refs_match(legacy, other) is False
```

- [ ] **Step 2: Run resolver tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_custom_agent_capability_resolver.py -q
```

Expected: collection fails because the resolver module does not exist.

- [ ] **Step 3: Implement the pure resolver**

Create `app/services/custom_agent_capability_resolver.py` with these public types and functions:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CapabilityStatus = Literal["ready", "degraded", "device_unavailable"]
_LIVE_IDENTITY_FIELDS = (
    "device_id",
    "session_id",
    "catalog_version",
    "tool_instance_id",
    "server_name",
    "qualified_tool_id",
    "tool_name",
)


@dataclass(frozen=True)
class CapabilityResolution:
    status: CapabilityStatus
    resolved_client_tool_refs: list[dict[str, Any]]
    missing_tools: list[dict[str, Any]]
    missing_skills: list[dict[str, Any]]
    warnings: list[str]


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, dict):
        merged = dict(metadata)
        merged.setdefault("tool_name", metadata.get("source_tool_name"))
        return merged
    return {
        "source": getattr(value, "source", None),
        "lookup_name": getattr(value, "lookup_name", None),
        "name": getattr(value, "name", None),
        "device_id": getattr(value, "bound_device_id", None),
        "session_id": getattr(value, "bound_session_id", None),
    }


def client_tool_logical_key(value: Any) -> tuple[str, str] | None:
    item = _mapping(value)
    explicit_type = str(item.get("type") or "").strip()
    if explicit_type and explicit_type != "client":
        return None
    if not explicit_type and not (
        bool(item.get("is_client_tool"))
        or "client" in str(item.get("tool_origin") or item.get("origin") or "")
    ):
        return None
    server_name = str(item.get("server_name") or "").strip()
    qualified_id = str(item.get("qualified_tool_id") or "").strip()
    if not server_name or not qualified_id:
        return None
    return server_name, qualified_id


def client_tool_binding_key(value: Any) -> tuple[str, ...] | None:
    """Full identity required when a caller submits a new current option."""
    item = _mapping(value)
    logical = client_tool_logical_key(item)
    fields = (
        item.get("device_id"),
        item.get("session_id"),
        item.get("catalog_version"),
        item.get("tool_instance_id"),
        item.get("tool_name") or item.get("source_tool_name"),
    )
    if logical is None or any(not str(field or "").strip() for field in fields):
        return None
    return (*logical, *(str(field) for field in fields))


def _skill_aliases(value: Any) -> tuple[str, set[str]]:
    item = _mapping(value)
    source = str(item.get("source") or "client").strip().lower()
    if source == "server":
        source = "client"
    aliases = {
        str(candidate).strip()
        for candidate in (item.get("lookup_name"), item.get("name"))
        if str(candidate or "").strip()
    }
    return source, aliases


def skill_logical_key(value: Any) -> tuple[str, str] | None:
    item = _mapping(value)
    source, aliases = _skill_aliases(item)
    lookup_name = str(item.get("lookup_name") or item.get("name") or "").strip()
    if not source or not lookup_name or not aliases:
        return None
    return source, lookup_name


def skill_refs_match(left: Any, right: Any) -> bool:
    left_source, left_aliases = _skill_aliases(left)
    right_source, right_aliases = _skill_aliases(right)
    return bool(
        left_source
        and left_source == right_source
        and left_aliases
        and left_aliases & right_aliases
    )


def resolve_custom_agent_capabilities(
    *,
    selected_tool_refs: list[dict[str, Any]],
    selected_skill_refs: list[dict[str, Any]],
    live_tool_refs: list[Any],
    live_skill_refs: list[Any],
    request_device_id: str | None,
    device_available: bool,
) -> CapabilityResolution:
    selected_clients = [
        ref for ref in selected_tool_refs if str(ref.get("type") or "") == "client"
    ]
    live_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for tool in live_tool_refs if device_available and request_device_id else []:
        item = _mapping(tool)
        if request_device_id and str(item.get("device_id") or "") != str(request_device_id):
            continue
        key = client_tool_logical_key(item)
        if key is not None:
            live_by_key[key] = item
    resolved: list[dict[str, Any]] = []
    missing_tools: list[dict[str, Any]] = []
    warnings: list[str] = []
    missing_tool_keys: set[tuple[Any, ...]] = set()

    for ref in selected_clients:
        live = live_by_key.get(client_tool_logical_key(ref))
        if live is None:
            missing_key = client_tool_logical_key(ref) or ("invalid", id(ref))
            if missing_key in missing_tool_keys:
                continue
            missing_tool_keys.add(missing_key)
            missing = {
                "server_name": str(ref.get("server_name") or ""),
                "qualified_tool_id": str(ref.get("qualified_tool_id") or ""),
                "tool_name": ref.get("tool_name"),
            }
            missing_tools.append(missing)
            warnings.append(
                f"Selected MCP tool '{missing['qualified_tool_id']}' is not available on this device."
            )
            continue
        rebound = dict(ref)
        for field in _LIVE_IDENTITY_FIELDS:
            if live.get(field) is not None:
                rebound[field] = live[field]
        resolved.append(rebound)

    live_skills = list(live_skill_refs) if device_available and request_device_id else []
    missing_skills: list[dict[str, Any]] = []
    missing_skill_keys: set[tuple[Any, ...]] = set()
    for ref in selected_skill_refs:
        if any(skill_refs_match(ref, live) for live in live_skills):
            continue
        missing_key = skill_logical_key(ref) or ("invalid", id(ref))
        if missing_key in missing_skill_keys:
            continue
        missing_skill_keys.add(missing_key)
        lookup_name = str(ref.get("lookup_name") or ref.get("name") or "")
        name = str(ref.get("name") or lookup_name)
        missing_skills.append({"lookup_name": lookup_name, "name": name})
        warnings.append(f"Selected skill '{lookup_name}' is not available on this device.")

    has_local_dependencies = bool(selected_clients or selected_skill_refs)
    if has_local_dependencies and not device_available:
        status: CapabilityStatus = "device_unavailable"
    elif missing_tools or missing_skills:
        status = "degraded"
    else:
        status = "ready"

    return CapabilityResolution(
        status=status,
        resolved_client_tool_refs=resolved,
        missing_tools=missing_tools,
        missing_skills=missing_skills,
        warnings=list(dict.fromkeys(warnings)),
    )
```

During implementation, keep the shown API and assertions exact. Formatting may wrap long expressions without changing behavior.

- [ ] **Step 4: Run resolver tests and verify GREEN**

Run the Step 2 command. Expected: nine tests pass.

- [ ] **Step 5: Run Ruff on the new unit**

```powershell
.\.venv\Scripts\python.exe -m ruff check app/services/custom_agent_capability_resolver.py tests/test_custom_agent_capability_resolver.py
.\.venv\Scripts\python.exe -m ruff format --check app/services/custom_agent_capability_resolver.py tests/test_custom_agent_capability_resolver.py
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit**

```powershell
git add app/services/custom_agent_capability_resolver.py tests/test_custom_agent_capability_resolver.py
git commit -m "feat: resolve custom agent capabilities by logical identity"
```

### Task 4 (T004): Expose device snapshots and contextual agent availability

**Files:**
- Modify: `tests/test_custom_agents_service.py:145-676`
- Modify: `app/services/custom_agent_service.py`
- Modify: `app/api/custom_agents.py`
- Modify: `tests/test_custom_agents_api.py`

- [ ] **Step 1: Extend service fakes and write failing availability tests**

In `tests/test_custom_agents_service.py`, change `_fake_skills` to return client skills only and add a snapshot fake:

```python
def _fake_skills(user_id, device_id):
    if device_id != "desktop-1":
        return []
    return [{"source": "client", "lookup_name": "data-analysis", "name": "data-analysis"}]


def _fake_device_snapshot(user_id, device_id):
    if device_id == "syncing-desktop":
        return {
            "device_id": device_id,
            "session_id": "syncing-session",
            "tool_catalog_version": 0,
            "skill_catalog_version": 0,
            "status": "unavailable",
        }
    if device_id != "desktop-1":
        return {
            "device_id": device_id,
            "session_id": None,
            "tool_catalog_version": None,
            "skill_catalog_version": None,
            "status": "unavailable",
        }
    return {
        "device_id": "desktop-1",
        "session_id": "session-1",
        "tool_catalog_version": 1,
        "skill_catalog_version": 2,
        "status": "ready",
    }
```

Pass `get_device_snapshot=_fake_device_snapshot` into `_Env.service`, then add:

```python
def test_contextual_read_reports_missing_local_capabilities(env):
    selected = {
        "type": "client",
        "device_id": "desktop-1",
        "session_id": "session-1",
        "catalog_version": "v1",
        "tool_instance_id": "csv-profile-instance",
        "server_name": "csv",
        "qualified_tool_id": "csv::profile",
        "tool_name": "profile",
    }
    created = env.service.create_agent(
        env.owner_id,
        _payload(tool_refs=[selected]),
        device_id="desktop-1",
    )
    env.service._list_client_tool_refs = lambda _user_id, _device_id: []

    read = env.service.get_agent(env.owner_id, created.id, device_id="desktop-1")

    assert read.availability is not None
    assert read.availability.status == "degraded"
    assert read.availability.missing_tools[0].qualified_tool_id == (
        "csv::profile"
    )


@pytest.mark.asyncio
async def test_options_echo_complete_device_snapshot(env):
    options = await env.service.get_options(env.owner_id, device_id="desktop-1")

    assert options.device_snapshot.status == "ready"
    assert options.device_snapshot.device_id == "desktop-1"
    assert options.device_snapshot.session_id == "session-1"
    assert options.device_snapshot.tool_catalog_version == 1
    assert options.device_snapshot.skill_catalog_version == 2


@pytest.mark.asyncio
async def test_options_never_label_unsynced_empty_catalogs_ready(env):
    options = await env.service.get_options(env.owner_id, device_id="syncing-desktop")

    assert options.device_snapshot.status == "unavailable"
    assert options.client_tools == []
    assert options.skills == []


def test_context_retries_when_snapshot_rotates_during_catalog_read(env):
    snapshots = iter([
        {"device_id": "desktop-1", "session_id": "s1", "tool_catalog_version": 1,
         "skill_catalog_version": 1, "status": "ready"},
        {"device_id": "desktop-1", "session_id": "s2", "tool_catalog_version": 1,
         "skill_catalog_version": 1, "status": "ready"},
        {"device_id": "desktop-1", "session_id": "s2", "tool_catalog_version": 1,
         "skill_catalog_version": 1, "status": "ready"},
    ])
    tool_reads = iter([[{"session_id": "s1"}], [{"session_id": "s2"}]])
    env.service._get_device_snapshot = lambda _u, _d: next(snapshots)
    env.service._list_client_tool_refs = lambda _u, _d: next(tool_reads)
    env.service._list_skill_refs = lambda _u, _d: []

    snapshot, tools, _skills = env.service._load_device_context(
        env.owner_id, "desktop-1"
    )

    assert snapshot["session_id"] == "s2"
    assert tools == [{"session_id": "s2"}]
```

- [ ] **Step 2: Run service tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_custom_agents_service.py -k "contextual_read or complete_device_snapshot or unsynced_empty or snapshot_rotates" -q
```

Expected: constructor/signature/field failures because snapshot lookup and contextual availability are absent.

- [ ] **Step 3: Add production snapshot lookup**

Add the constructor dependency:

```python
get_device_snapshot: Callable[[str | None, str | None], dict[str, Any]] | None = None,
```

Assign it with:

```python
self._get_device_snapshot = get_device_snapshot or _default_get_device_snapshot
```

Add the default lookup near the existing production lookup helpers:

```python
def _default_get_device_snapshot(user_id: str | None, device_id: str | None) -> dict[str, Any]:
    unavailable = {
        "device_id": device_id,
        "session_id": None,
        "tool_catalog_version": None,
        "skill_catalog_version": None,
        "status": "unavailable",
    }
    if not user_id or not device_id:
        return unavailable
    try:
        device_uuid = UUID(str(device_id))
    except (TypeError, ValueError, AttributeError):
        return unavailable
    session = ClientDeviceService.lookup_active_session(device_uuid)
    if session is None or str(session.user_id) != str(user_id):
        return unavailable
    if session.tool_catalog_version <= 0 or session.skill_catalog_version <= 0:
        return {
            "device_id": str(session.device_id),
            "session_id": session.session_id,
            "tool_catalog_version": session.tool_catalog_version,
            "skill_catalog_version": session.skill_catalog_version,
            "status": "unavailable",
        }
    return {
        "device_id": str(session.device_id),
        "session_id": session.session_id,
        "tool_catalog_version": session.tool_catalog_version,
        "skill_catalog_version": session.skill_catalog_version,
        "status": "ready",
    }
```

Import `ClientDeviceService`, the new resolver, and schema types.

- [ ] **Step 4: Build one contextual read path**

Add a consistency helper. It retries once if a reconnect/resync changes identity
while catalogs are being materialized, then fails closed as unavailable rather than
pairing one session's options with another session's cache key:

```python
@staticmethod
def _snapshot_identity(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    return (
        snapshot.get("device_id"),
        snapshot.get("session_id"),
        snapshot.get("tool_catalog_version"),
        snapshot.get("skill_catalog_version"),
        snapshot.get("status"),
    )

def _load_device_context(
    self, owner_id: UUID, device_id: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    user_id = str(owner_id)
    snapshot = self._get_device_snapshot(user_id, device_id)
    if snapshot.get("status") != "ready":
        return snapshot, [], []

    for _attempt in range(2):
        before = snapshot
        live_tools = self._list_client_tool_refs(user_id, device_id)
        live_skills = self._list_skill_refs(user_id, device_id)
        after = self._get_device_snapshot(user_id, device_id)
        if (
            after.get("status") == "ready"
            and self._snapshot_identity(before) == self._snapshot_identity(after)
        ):
            return after, live_tools, live_skills
        snapshot = after
        if snapshot.get("status") != "ready":
            return snapshot, [], []

    unstable = dict(snapshot)
    unstable["status"] = "unavailable"
    return unstable, [], []
```

Add the complete availability adapter:

```python
@staticmethod
def _availability_for(
    agent: CustomAgent,
    *,
    snapshot: dict[str, Any],
    live_tools: list[dict[str, Any]],
    live_skills: list[dict[str, Any]],
) -> CustomAgentAvailability:
    resolution = resolve_custom_agent_capabilities(
        selected_tool_refs=list(agent.tool_refs or []),
        selected_skill_refs=list(agent.skill_refs or []),
        live_tool_refs=live_tools,
        live_skill_refs=live_skills,
        request_device_id=snapshot.get("device_id"),
        device_available=snapshot.get("status") == "ready",
    )
    return CustomAgentAvailability(
        status=resolution.status,
        device_id=snapshot.get("device_id"),
        session_id=snapshot.get("session_id"),
        missing_tools=resolution.missing_tools,
        missing_skills=resolution.missing_skills,
        warnings=resolution.warnings,
    )
```

Change service read signatures to use that single context:

```python
def list_agents(self, owner_id: UUID, *, device_id: str | None = None) -> list[CustomAgentRead]:
    snapshot, live_tools, live_skills = self._load_device_context(owner_id, device_id)
    return [
        self._to_read(
            agent,
            availability=self._availability_for(
                agent, snapshot=snapshot, live_tools=live_tools, live_skills=live_skills
            ),
        )
        for agent in self.repository.list_by_owner(owner_id)
    ]


def get_agent(
    self, owner_id: UUID, custom_agent_id: UUID, *, device_id: str | None = None
) -> CustomAgentRead:
    agent = self._load_owned_or_raise(owner_id, custom_agent_id)
    snapshot, live_tools, live_skills = self._load_device_context(owner_id, device_id)
    return self._to_read(
        agent,
        availability=self._availability_for(
            agent,
            snapshot=snapshot,
            live_tools=live_tools,
            live_skills=live_skills,
        ),
    )
```

Change `_to_read()` explicitly:

```python
@staticmethod
def _to_read(
    agent: CustomAgent,
    availability: CustomAgentAvailability | None = None,
) -> CustomAgentRead:
    value = CustomAgentRead.model_validate(agent)
    return value.model_copy(update={"availability": availability})
```

In `get_options()`, replace the independent catalog reads with:

```python
snapshot, client_tools, skills = self._load_device_context(owner_id, device_id)
return CustomAgentOptions(
    providers=await self._list_providers(owner_id),
    server_default_tools=list(self._list_server_tool_refs()),
    server_tools=[],
    client_tools=client_tools,
    client_servers=self._group_client_servers(client_tools),
    skills=skills,
    device_snapshot=DeviceCatalogSnapshot.model_validate(snapshot),
)
```

- [ ] **Step 5: Pass device context through list and single-read routes**

Change `app/api/custom_agents.py` handlers:

```python
async def list_custom_agents(
    custom_agent_service: CustomAgentService,
    user_id: UUID,
    device_id: str | None = _DEVICE_QUERY,
) -> ApiResponse[list[CustomAgentRead]]:
    result = custom_agent_service.list_agents(user_id, device_id=device_id)
```

```python
async def get_custom_agent(
    custom_agent_id: UUID,
    custom_agent_service: CustomAgentService,
    user_id: UUID,
    device_id: str | None = _DEVICE_QUERY,
) -> ApiResponse[CustomAgentRead]:
    result = custom_agent_service.get_agent(
        user_id, custom_agent_id, device_id=device_id
    )
```

Add this API serialization test (the fake device ID intentionally has no active
session, so it also locks the unavailable snapshot shape):

```python
def test_contextual_reads_and_options_serialize_camel_case_contract(api):
    owner = api[0]
    agent_id = owner.post("/custom-agents", json=_create_body()).json()["data"]["id"]

    listed = owner.get("/ai/custom-agents?deviceId=desktop-1").json()["data"][0]
    fetched = owner.get(
        f"/ai/custom-agents/{agent_id}?deviceId=desktop-1"
    ).json()["data"]
    options = owner.get(
        "/ai/custom-agents/options?deviceId=desktop-1"
    ).json()["data"]

    assert listed["availability"]["status"] == "ready"
    assert fetched["availability"]["missingTools"] == []
    assert options["deviceSnapshot"]["status"] == "unavailable"
    assert "toolCatalogVersion" in options["deviceSnapshot"]
```

- [ ] **Step 6: Run service and API tests and verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_custom_agents_service.py tests/test_custom_agents_api.py -q
```

Expected: all Custom Agent service/API tests pass against configured PostgreSQL.

- [ ] **Step 7: Commit**

```powershell
git add app/services/custom_agent_service.py app/api/custom_agents.py tests/test_custom_agents_service.py tests/test_custom_agents_api.py
git commit -m "feat: expose custom agent device availability"
```

### Task 5 (T005): Preserve unavailable saved intent and deduplicate portable selections

**Files:**
- Modify: `tests/test_custom_agents_service.py:323-429,639-676`
- Modify: `app/services/custom_agent_service.py:339-430`

- [ ] **Step 1: Write failing portable update and dedupe tests**

First update `_fake_client_tools()` and its affected assertions so the fake uses
`qualified_tool_id="csv::profile"`; keep `tool_name="profile"` and use
`client__csv__profile` only where a model-callable tool name is expected. Then add:

```python
def test_update_preserves_existing_unavailable_ref_but_rejects_new_fabricated_ref(env):
    selected = {
        "type": "client",
        "device_id": "desktop-1",
        "session_id": "session-1",
        "catalog_version": "v1",
        "tool_instance_id": "csv-profile-instance",
        "server_name": "csv",
        "qualified_tool_id": "csv::profile",
        "tool_name": "profile",
    }
    created = env.service.create_agent(
        env.owner_id,
        _payload(tool_refs=[selected]),
        device_id="desktop-1",
    )

    preserved = env.service.update_agent(
        env.owner_id,
        created.id,
        CustomAgentUpdate(prompt="Changed", tool_refs=[selected]),
        device_id="not-connected",
    )
    assert preserved.tool_refs == [selected]

    fabricated = dict(selected, qualified_tool_id="csv::fabricated", tool_name="fabricated")
    with pytest.raises(CustomAgentValidationError):
        env.service.update_agent(
            env.owner_id,
            created.id,
            CustomAgentUpdate(tool_refs=[selected, fabricated]),
            device_id="not-connected",
        )


def test_new_ref_requires_full_current_binding_not_only_live_logical_key(env):
    forged = {
        "type": "client",
        "device_id": "desktop-1",
        "session_id": "forged-session",
        "catalog_version": "v1",
        "tool_instance_id": "forged-instance",
        "server_name": "csv",
        "qualified_tool_id": "csv::profile",
        "tool_name": "profile",
    }

    with pytest.raises(CustomAgentValidationError):
        env.service.create_agent(
            env.owner_id,
            _payload(name="Forged", tool_refs=[forged]),
            device_id="desktop-1",
        )


def test_client_ref_dedupe_uses_server_and_qualified_id_not_device_instance():
    a = {
        "type": "client",
        "server_name": "desktop-commander",
        "qualified_tool_id": "desktop-commander::read_file",
        "device_id": "device-a",
        "session_id": "session-a",
        "tool_instance_id": "instance-a",
    }
    b = dict(
        a,
        device_id="device-b",
        session_id="session-b",
        tool_instance_id="instance-b",
    )

    assert CustomAgentService._dedupe_tool_refs([a, b]) == [a]


def test_existing_skill_survives_absence_but_new_and_legacy_source_do_not(env):
    selected = {
        "source": "client",
        "lookup_name": "data-analysis",
        "name": "data-analysis",
    }
    created = env.service.create_agent(
        env.owner_id,
        _payload(name="Skill Agent", skill_refs=[selected]),
        device_id="desktop-1",
    )
    env.service._list_skill_refs = lambda _user_id, _device_id: []

    preserved = env.service.update_agent(
        env.owner_id,
        created.id,
        CustomAgentUpdate(prompt="Changed", skill_refs=[selected]),
        device_id="not-connected",
    )
    assert preserved.skill_refs == [selected]

    new_ref = {"source": "client", "lookup_name": "never-installed", "name": "never-installed"}
    with pytest.raises(CustomAgentValidationError):
        env.service.update_agent(
            env.owner_id,
            created.id,
            CustomAgentUpdate(skill_refs=[selected, new_ref]),
            device_id="not-connected",
        )

    env.service._list_skill_refs = _fake_skills
    legacy_new = dict(selected, source="server")
    with pytest.raises(CustomAgentValidationError):
        env.service.create_agent(
            env.owner_id,
            _payload(name="Legacy New", skill_refs=[legacy_new]),
            device_id="desktop-1",
        )


def test_skill_ref_dedupe_normalizes_legacy_server_source():
    legacy = {"source": "server", "lookup_name": "kobo-library", "name": "kobo-library"}
    current = dict(legacy, source="client")

    assert CustomAgentService._dedupe_skill_refs([legacy, current]) == [legacy]
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_custom_agents_service.py -k "preserves_existing_unavailable or requires_full_current_binding or dedupe_uses_server or existing_skill_survives or skill_ref_dedupe" -q
```

Expected: the current validator rejects the preserved ref and current dedupe retains both device instances.

- [ ] **Step 3: Validate new selections separately from existing intent**

Change update validation calls to pass the existing refs:

```python
self._validate_tool_refs(
    owner_id,
    tool_refs,
    device_id,
    existing_refs=list(existing.tool_refs or []),
)
self._validate_skill_refs(
    owner_id,
    skill_refs,
    device_id,
    existing_refs=list(existing.skill_refs or []),
)
```

Implement client validation with full live binding identity for new logical keys:

```python
available_client_bindings = {
    client_tool_binding_key(tool)
    for tool in self._list_client_tool_refs(str(owner_id), device_id)
    if client_tool_binding_key(tool) is not None
}
existing_client_keys = {
    client_tool_logical_key(ref)
    for ref in (existing_refs or [])
    if client_tool_logical_key(ref) is not None
}

# Inside the client-ref branch:
logical_key = client_tool_logical_key(ref)
binding_key = client_tool_binding_key(ref)
if logical_key is None or (
    logical_key not in existing_client_keys
    and binding_key not in available_client_bindings
):
    raise CustomAgentValidationError(
        detail=(
            f"Client tool '{ref.get('qualified_tool_id')}' is not available "
            "in the active device catalog"
        ),
    )
```

Server MCP validation remains unchanged. For skills, allow compatibility matching
only against existing refs; a new ref must be an exact current option including its
source, lookup name, and name:

```python
available_skills = self._list_skill_refs(str(owner_id), device_id)
existing_skills = list(existing_refs or [])
for ref in skill_refs:
    retains_existing = any(skill_refs_match(ref, old) for old in existing_skills)
    is_exact_live_option = any(
        str(ref.get("source") or "") == str(live.get("source") or "")
        and str(ref.get("lookup_name") or "") == str(live.get("lookup_name") or "")
        and str(ref.get("name") or "") == str(live.get("name") or "")
        for live in available_skills
    )
    if not retains_existing and not is_exact_live_option:
        raise CustomAgentValidationError(
            detail=f"Skill '{ref.get('lookup_name')}' is not available",
        )
```

Creation passes no existing refs, so both fabricated live bindings and newly
submitted legacy `source="server"` refs fail.

Use these complete final method bodies (they supersede the partial fragments above):

```python
def _validate_tool_refs(
    self,
    owner_id: UUID,
    tool_refs: list[dict[str, Any]],
    device_id: str | None,
    *,
    existing_refs: list[dict[str, Any]] | None = None,
) -> None:
    if not tool_refs:
        return
    available_client_bindings = {
        key
        for tool in self._list_client_tool_refs(str(owner_id), device_id)
        if (key := client_tool_binding_key(tool)) is not None
    }
    existing_client_keys = {
        key
        for ref in (existing_refs or [])
        if (key := client_tool_logical_key(ref)) is not None
    }
    available_server_ids = {
        str(tool.get("qualified_tool_id"))
        for tool in self._list_server_tool_refs()
        if tool.get("qualified_tool_id")
    }
    for ref in tool_refs:
        ref_type = ref.get("type")
        if ref_type in {"server_mcp", "server_default"}:
            if str(ref.get("qualified_tool_id")) not in available_server_ids:
                raise CustomAgentValidationError(
                    detail=f"Server MCP tool '{ref.get('qualified_tool_id')}' is not available",
                )
            continue
        if ref_type != "client":
            raise CustomAgentValidationError(detail=f"Invalid tool ref type: {ref_type}")
        logical_key = client_tool_logical_key(ref)
        binding_key = client_tool_binding_key(ref)
        if logical_key is None or (
            logical_key not in existing_client_keys
            and binding_key not in available_client_bindings
        ):
            raise CustomAgentValidationError(
                detail=(
                    f"Client tool '{ref.get('qualified_tool_id')}' is not available "
                    "in the active device catalog"
                ),
            )


def _validate_skill_refs(
    self,
    owner_id: UUID,
    skill_refs: list[dict[str, Any]],
    device_id: str | None,
    *,
    existing_refs: list[dict[str, Any]] | None = None,
) -> None:
    if not skill_refs:
        return
    available_skills = self._list_skill_refs(str(owner_id), device_id)
    existing_skills = list(existing_refs or [])
    for ref in skill_refs:
        retains_existing = any(skill_refs_match(ref, old) for old in existing_skills)
        is_exact_live_option = any(
            str(ref.get("source") or "") == str(live.get("source") or "")
            and str(ref.get("lookup_name") or "")
            == str(live.get("lookup_name") or "")
            and str(ref.get("name") or "") == str(live.get("name") or "")
            for live in available_skills
        )
        if not retains_existing and not is_exact_live_option:
            raise CustomAgentValidationError(
                detail=f"Skill '{ref.get('lookup_name')}' is not available",
            )
```

- [ ] **Step 4: Change client and skill dedupe identity**

Replace the client branch in `_dedupe_tool_refs()` with:

```python
logical_key = client_tool_logical_key(ref)
key = ("client", *logical_key) if logical_key is not None else (
    "client-invalid",
    ref.get("device_id"),
    ref.get("tool_instance_id"),
)
```

Keep first-seen order. Do not deduplicate server and client refs together even when their qualified IDs match.

Add the parallel skill helper and use it in both create and update before validation:

```python
@staticmethod
def _dedupe_skill_refs(skill_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for ref in skill_refs:
        logical_key = skill_logical_key(ref)
        key = (
            ("skill", *logical_key)
            if logical_key is not None
            else (
                "skill-invalid",
                ref.get("source"),
                ref.get("lookup_name"),
                ref.get("name"),
            )
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped
```

Replace both raw skill list comprehensions with
`self._dedupe_skill_refs([...])`.

- [ ] **Step 5: Run full service tests and verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_custom_agents_service.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add app/services/custom_agent_service.py tests/test_custom_agents_service.py
git commit -m "fix: preserve portable custom agent selections"
```

### Task 6 (T006): Use portable resolution at runtime and warn the agent

**Files:**
- Modify: `tests/test_custom_agent_client_tool_resync.py`
- Modify: `tests/test_custom_agents_tools.py`
- Modify: `tests/test_custom_agents_graph.py`
- Modify: `app/ai/custom_agent_runtime.py`
- Modify: `app/ai/agents/custom_agent.py`

- [ ] **Step 1: Write the failing machine-B rebinding test**

First change `_QID` in `tests/test_custom_agent_client_tool_resync.py` to
`"csv::profile"` and change the other-tool qualified ID to `"other::thing"`.
Keep each fake tool's callable `name` as `client__csv__profile` or
`client__other__thing`. Then append:

```python
def test_saved_machine_a_ref_rebases_to_same_logical_tool_on_machine_b():
    spec = _spec([PERSISTED_CLIENT_REF])
    live_b = _live_tool(
        device_id="desktop-2",
        session_id="session-b",
        instance="csv-profile-instance-b",
    )

    rebased = rebase_client_tool_refs(
        spec.allowed_client_tool_refs,
        [live_b],
        request_device_id="desktop-2",
    )
    rebased_spec = spec.model_copy(update={"allowed_client_tool_refs": rebased})
    allowed, warnings = filter_tools_for_custom_agent(
        [live_b], rebased_spec, request_device_id="desktop-2"
    )

    assert rebased[0]["device_id"] == "desktop-2"
    assert [tool.name for tool in allowed] == ["client__csv__profile"]
    assert warnings == []
```

Keep `test_rebase_does_not_authorize_foreign_device_tool`: it requests desktop-1 while only desktop-2 is live, so it must still fail closed.

- [ ] **Step 2: Write failing missing-skill prompt/metadata tests**

In `tests/test_custom_agents_graph.py`, create a Custom Agent with warnings and assert:

```python
def test_custom_agent_prompt_mentions_missing_device_capabilities():
    agent = _custom_agent()
    agent.set_runtime_warnings([
        "Selected skill 'kobo-library' is not available on this device."
    ])

    prompt = agent._build_system_prompt(None, False)

    assert "DEVICE CAPABILITY NOTICE" in prompt
    assert "kobo-library" in prompt
    assert "continue with available capabilities" in prompt
```

In `tests/test_custom_agents_tools.py`, add a test that patches
`app.ai.agents.custom_agent.list_resolved_skills` to return `[]`, patches
`get_active_client_runtime_session` to a session with both catalog versions set to
`1`, and creates a spec selecting `kobo-library`. After
`_get_tools_for_binding(user_id="user-1", device_id="desktop-2")`, assert:

```python
assert agent._runtime_warnings == [
    "Selected skill 'kobo-library' is not available on this device."
]
assert "kobo-library" not in agent._build_skills_suffix(
    user_id="user-1", device_id="desktop-2"
)
```

- [ ] **Step 3: Run runtime tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_custom_agent_client_tool_resync.py tests/test_custom_agents_tools.py tests/test_custom_agents_graph.py -k "machine_a_ref or missing_device_capabilities or missing_selected_skill" -q
```

Expected: cross-device rebase stays stale and missing skills do not populate runtime warnings/prompt context.

- [ ] **Step 4: Delegate rebase to the shared resolver**

In `app/ai/custom_agent_runtime.py`, import `client_tool_logical_key` alongside the
resolver, retain the public `rebase_client_tool_refs()` function for compatibility,
and implement it with:

```python
resolution = resolve_custom_agent_capabilities(
    selected_tool_refs=refs,
    selected_skill_refs=[],
    live_tool_refs=live_tools,
    live_skill_refs=[],
    request_device_id=request_device_id,
    device_available=bool(request_device_id),
)
resolved_by_key = {
    client_tool_logical_key(ref): ref
    for ref in resolution.resolved_client_tool_refs
    if client_tool_logical_key(ref) is not None
}
rebased = [
    resolved_by_key.get(client_tool_logical_key(ref), ref)
    for ref in refs
]
return rebased if rebased != refs else refs
```

This compatibility wrapper deliberately retains an unresolved stale ref because its
existing callers/tests use identity-preserving no-op semantics. The Custom Agent
binding path below uses `resolution.resolved_client_tool_refs` directly and therefore
does **not** put missing refs into its allowlist. The caller supplies tools from
`request_device_id`, and the resolver independently enforces that device boundary.
Do not add any all-user device lookup. The existing exact matcher remains unchanged.

- [ ] **Step 5: Resolve MCP and skill availability together in CustomAgent**

Import `list_resolved_skills` and `resolve_custom_agent_capabilities`. Change `_request_spec()` to accept `user_id` and return `(spec, warnings)`:

```python
def _request_spec(
    self,
    *,
    user_id: str | None,
    device_id: str | None,
    live_tools: list[BaseTool],
) -> tuple[AgentRuntimeSpec, list[str]]:
    active_session = get_active_client_runtime_session(
        user_id=user_id,
        device_id=device_id,
    )
    device_ready = bool(
        active_session is not None
        and getattr(active_session, "tool_catalog_version", 0) > 0
        and getattr(active_session, "skill_catalog_version", 0) > 0
    )
    resolution = resolve_custom_agent_capabilities(
        selected_tool_refs=self._spec.allowed_client_tool_refs,
        selected_skill_refs=self._spec.allowed_skill_refs,
        live_tool_refs=live_tools,
        live_skill_refs=list_resolved_skills(user_id=user_id, device_id=device_id),
        request_device_id=str(device_id) if device_id else None,
        device_available=device_ready,
    )
    spec = self._spec.model_copy(
        update={"allowed_client_tool_refs": resolution.resolved_client_tool_refs}
    )
    return spec, resolution.warnings
```

Change the call site and warning merge explicitly:

```python
spec, resolution_warnings = self._request_spec(
    user_id=user_id,
    device_id=device_id,
    live_tools=remote_tools,
)

# Keep the existing availability/filter calls. At the final assignment:
self.set_runtime_warnings(
    list(dict.fromkeys([*resolution_warnings, *warnings]))
)
```

Because the spec contains only resolved client refs, the strict filter does not emit
a second warning for an already-known missing ref. Its warnings now represent only
an unexpected exact-match loss after resolution.

- [ ] **Step 6: Add the warning prompt suffix**

Override `_build_system_prompt()` in `CustomAgent`:

```python
def _build_system_prompt(
    self,
    persona: str | None,
    has_tool_context: bool,
    **kwargs: Any,
) -> str:
    prompt = super()._build_system_prompt(
        persona,
        has_tool_context,
        **kwargs,
    )
    if not self._runtime_warnings:
        return prompt
    warnings = "\n".join(f"- {warning}" for warning in self._runtime_warnings)
    return (
        f"{prompt}\n\nDEVICE CAPABILITY NOTICE:\n{warnings}\n"
        "Continue with available capabilities. Do not claim a missing tool or skill was used."
    )
```

The normal invocation order binds tools before building the system prompt, so the warning list is populated for that invocation.

- [ ] **Step 7: Run runtime tests and verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_custom_agent_client_tool_resync.py tests/test_custom_agents_tools.py tests/test_custom_agents_graph.py tests/test_agent_metadata.py -q
```

Expected: all tests pass, including strict stale-session/foreign-device rejection.

- [ ] **Step 8: Commit**

```powershell
git add app/ai/custom_agent_runtime.py app/ai/agents/custom_agent.py tests/test_custom_agent_client_tool_resync.py tests/test_custom_agents_tools.py tests/test_custom_agents_graph.py
git commit -m "feat: rebind portable custom agent capabilities"
```

### Task 7 (T007): Align proxy, Streamlit, and frontend cache behavior

**Files:**
- Modify: `tests/client_backend/test_custom_agents_proxy.py`
- Verify: `client_backend/api/proxy.py`
- Modify: `tests/test_demo_custom_agents.py`
- Modify: `demo.py:1211-1248,1310-1444,1736-1965`
- Modify: `plans/CUSTOM_AGENTS_FE_CONTRACT.md`
- Modify: `README.md`

- [ ] **Step 1: Lock sidecar stamping for every contextual read**

Extend `test_proxy_forwards_active_device_id_for_options_and_mutations` to request list and single-read routes:

```python
with client:
    assert client.get("/custom-agents").status_code == 200
    assert client.get("/custom-agents/abc").status_code == 200
    assert client.get("/custom-agents/options").status_code == 200

params_by_call = {(method, path): kwargs["params"] for method, path, kwargs in server.calls}
for key in (
    ("GET", "/custom-agents"),
    ("GET", "/custom-agents/abc"),
    ("GET", "/custom-agents/options"),
):
    assert ("deviceId", "device-123") in params_by_call[key]
```

Add the same assertions for `/ai/custom-agents` aliases. The current proxy is expected to pass; if it does, this is a contract-locking test rather than a production edit.

- [ ] **Step 2: Run proxy tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/client_backend/test_custom_agents_proxy.py -q
```

Expected: all pass. If a route omits stamping, change only that route to use `_params_with_active_device(request)` and rerun.

- [ ] **Step 3: Write failing Streamlit stable-identity and warning tests**

Update the current reconnect test so the saved ref is device A and the current option is device B with the same server/qid. Add:

```python
def test_custom_agent_edit_matches_same_logical_tool_on_another_device(monkeypatch):
    saved = {
        "type": "client",
        "device_id": "device-a",
        "session_id": "session-a",
        "catalog_version": "1",
        "tool_instance_id": "instance-a",
        "server_name": "desktop_commander",
        "qualified_tool_id": "desktop_commander::read_file",
        "tool_name": "read_file",
    }
    current = dict(
        saved,
        device_id="device-b",
        session_id="session-b",
        catalog_version="2",
        tool_instance_id="instance-b",
    )

    assert demo._custom_agent_tool_refs_available([saved], [], [current])
    assert demo._custom_agent_selected_tool_keys([saved], [], [current]) == [
        demo._custom_agent_tool_option_key(current)
    ]


def test_missing_ref_preservation_keeps_unavailable_account_wide_intent():
    missing = {
        "type": "client",
        "server_name": "desktop_commander",
        "qualified_tool_id": "desktop_commander::read_file",
    }
    available = [{"type": "server_mcp", "qualified_tool_id": "calc::add"}]

    assert demo._preserve_missing_custom_agent_tool_refs(
        existing_refs=[missing],
        rebuilt_refs=available,
        current_client_tools=[],
    ) == [missing, available[0]]


def test_missing_skill_preservation_normalizes_legacy_source():
    legacy = {"source": "server", "lookup_name": "kobo-library", "name": "kobo-library"}
    current = {"source": "client", "lookup_name": "kobo-library", "name": "kobo-library"}

    # Available on this device: rebuilt current selection replaces the legacy ref.
    assert demo._preserve_missing_custom_agent_skill_refs(
        existing_refs=[legacy],
        rebuilt_refs=[current],
        current_skills=[current],
    ) == [current]

    # Missing on this device: retain the account-wide saved intent.
    assert demo._preserve_missing_custom_agent_skill_refs(
        existing_refs=[legacy],
        rebuilt_refs=[],
        current_skills=[],
    ) == [legacy]
```

- [ ] **Step 4: Run demo tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_demo_custom_agents.py -k "another_device or missing_ref_preservation or missing_skill_preservation" -q
```

Expected: stable key still includes device ID and the preservation helper is absent.

- [ ] **Step 5: Implement stable matching and non-blocking degraded hints**

Change `_custom_agent_client_tool_stable_key()` to:

```python
def _custom_agent_client_tool_stable_key(
    tool: dict[str, Any],
) -> tuple[str, str] | None:
    if str(_custom_agent_value(tool, "type") or "") != "client":
        return None
    server_name = str(
        _custom_agent_value(tool, "server_name", "serverName") or ""
    ).strip()
    qualified_id = str(
        _custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId") or ""
    ).strip()
    if not server_name or not qualified_id:
        return None
    return server_name, qualified_id
```

Add explicit preservation helpers:

```python
def _preserve_missing_custom_agent_tool_refs(
    *,
    existing_refs: list[dict[str, Any]],
    rebuilt_refs: list[dict[str, Any]],
    current_client_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_keys = {
        key
        for tool in current_client_tools
        if (key := _custom_agent_client_tool_stable_key(tool)) is not None
    }
    missing = [
        ref
        for ref in existing_refs
        if str(_custom_agent_value(ref, "type") or "") == "client"
        and _custom_agent_client_tool_stable_key(ref) not in current_keys
    ]
    merged = [*missing, *rebuilt_refs]
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for ref in merged:
        if str(_custom_agent_value(ref, "type") or "") == "client":
            key = ("client", _custom_agent_client_tool_stable_key(ref))
        else:
            key = (
                "server",
                str(_custom_agent_value(ref, "qualified_tool_id", "qualifiedToolId") or ""),
            )
        if key in seen:
            continue
        seen.add(key)
        result.append(ref)
    return result


def _custom_agent_normalized_skill_key(ref: dict[str, Any]) -> tuple[str, str] | None:
    key = _custom_agent_skill_key(ref)
    if key is None:
        return None
    source, lookup_name = key
    return ("client" if source == "server" else source, lookup_name)


def _preserve_missing_custom_agent_skill_refs(
    *,
    existing_refs: list[dict[str, Any]],
    rebuilt_refs: list[dict[str, Any]],
    current_skills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = [
        ref
        for ref in existing_refs
        if not _custom_agent_skill_refs_available([ref], current_skills)
    ]
    merged = [*missing, *rebuilt_refs]
    seen: set[tuple[str, str] | tuple[str, int]] = set()
    result: list[dict[str, Any]] = []
    for ref in merged:
        key = _custom_agent_normalized_skill_key(ref) or ("invalid", id(ref))
        if key in seen:
            continue
        seen.add(key)
        result.append(ref)
    return result
```

Read `deviceSnapshot` before building client pickers. When its status is not
`ready`, force `client_tools=[]` and `skills=[]`, render a short syncing/offline
notice, and continue rendering provider/server options and editable prompt/model
fields. Never treat an unavailable snapshot as cacheable ready data.

For each existing agent, render `availability.warnings` with `st.warning()` and a
`Keep unavailable device selections` checkbox (default `True`). Remove the
all-or-nothing disabled state and the “Reconnect the original device” captions.
Always rebuild available selections on save. When the checkbox is checked, pass
them through both preservation helpers; when unchecked, submit the rebuilt lists
directly so the user can intentionally clear every missing local selection. This
preserves missing refs on unrelated edits without making them impossible to remove.

Change `list_custom_agents()` to request:

```python
status, payload = _custom_agent_request(
    "GET", f"/custom-agents{_custom_agent_device_query()}"
)
```

- [ ] **Step 6: Update the frontend contract**

Document these exact rules in `plans/CUSTOM_AGENTS_FE_CONTRACT.md`:

```text
- Cache options by returned deviceSnapshot, not user ID alone.
- Cache client options only when deviceSnapshot.status=ready; unavailable includes offline and not-yet-synced sessions.
- Match saved client MCP selections by (server_name, qualified_tool_id).
- Require the full current device/session/catalog/tool-instance/tool-name identity for newly submitted client refs.
- Show availability.status=degraded/device_unavailable as non-blocking.
- Preserve missing saved refs on unrelated edits; do not silently clear them.
- Provide an explicit control to remove retained unavailable refs.
- Never merge catalogs or HITL settings from two devices.
```

Add a concise README paragraph under Custom Agents stating account-wide desired capability/device-local execution semantics.

- [ ] **Step 7: Run demo, proxy, and compile checks**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_demo_custom_agents.py tests/client_backend/test_custom_agents_proxy.py -q
.\.venv\Scripts\python.exe -m py_compile demo.py client_backend/api/proxy.py
```

Expected: tests pass and both modules compile.

- [ ] **Step 8: Commit**

```powershell
git add demo.py tests/test_demo_custom_agents.py tests/client_backend/test_custom_agents_proxy.py plans/CUSTOM_AGENTS_FE_CONTRACT.md README.md
git commit -m "feat: show degraded custom agent capabilities"
```

### Task 8 (T008): Prove skills and HITL remain device-local

**Files:**
- Create: `tests/test_skill_device_isolation.py`
- Modify: `tests/test_hitl_settings_device_isolation.py`

- [ ] **Step 1: Add same-user two-device skill isolation tests**

Create `tests/test_skill_device_isolation.py`:

```python
from types import SimpleNamespace
from uuid import uuid4

from app.ai.skill_resolver import list_resolved_skills


def test_client_skills_are_read_only_from_requested_device(monkeypatch):
    user_id = uuid4()
    device_a = uuid4()
    device_b = uuid4()
    sessions = {
        device_a: SimpleNamespace(
            user_id=user_id,
            session_id="session-a",
            skill_catalog={
                "skills": [
                    {"name": "cli-anything-google-calendar", "enabled": True},
                    {"name": "kobo-library", "enabled": True},
                ]
            },
        ),
        device_b: SimpleNamespace(
            user_id=user_id,
            session_id="session-b",
            skill_catalog={"skills": []},
        ),
    }
    monkeypatch.setattr(
        "app.services.client_device_service.ClientDeviceService.lookup_active_session",
        lambda device_id: sessions.get(device_id),
    )

    skills_a = list_resolved_skills(user_id=str(user_id), device_id=str(device_a))
    skills_b = list_resolved_skills(user_id=str(user_id), device_id=str(device_b))

    assert {skill.name for skill in skills_a} == {
        "cli-anything-google-calendar",
        "kobo-library",
    }
    assert skills_b == []


def test_foreign_user_cannot_read_device_skills(monkeypatch):
    owner_id = uuid4()
    other_id = uuid4()
    device_id = uuid4()
    monkeypatch.setattr(
        "app.services.client_device_service.ClientDeviceService.lookup_active_session",
        lambda _device_id: SimpleNamespace(
            user_id=owner_id,
            session_id="session-a",
            skill_catalog={"skills": [{"name": "kobo-library", "enabled": True}]},
        ),
    )

    assert list_resolved_skills(user_id=str(other_id), device_id=str(device_id)) == []
```

- [ ] **Step 2: Add same-name MCP HITL isolation test**

Add this method to that file's `_MemoryRepository` so the test exercises the
actual turn-policy shape as well as settings reads:

```python
def build_policy(self, user_id, device_id):
    grouped = {
        "client_mcp": {"servers": {}, "tools": {}},
        "client_skill": {"servers": {}, "tools": {}},
    }
    for row in self.list_by_device(user_id, device_id):
        target = "servers" if row.scope_type == "server" else "tools"
        grouped[row.tool_origin][target][row.scope_value] = bool(row.require_approval)
    return grouped
```

Append:

```python
def test_same_mcp_server_name_keeps_independent_device_rules(owned_devices):
    user_id, device_a, device_b = owned_devices
    repo = _MemoryRepository()
    repo.rows.extend([
        _row(user_id, device_a, "client_mcp", "server", "desktop-commander", True),
        _row(user_id, device_b, "client_mcp", "server", "desktop-commander", False),
    ])
    service = HitlSettingsService(repo)

    settings_a = service.get_settings(user_id, str(device_a))
    settings_b = service.get_settings(user_id, str(device_b))
    policy_a = service.build_turn_policy(user_id, device_a)
    policy_b = service.build_turn_policy(user_id, device_b)

    assert settings_a["servers"][0]["require_approval"] is True
    assert settings_b["servers"][0]["require_approval"] is False
    assert policy_a["client_rules"]["client_mcp"]["servers"] == {
        "desktop-commander": True
    }
    assert policy_b["client_rules"]["client_mcp"]["servers"] == {
        "desktop-commander": False
    }
```

- [ ] **Step 3: Run isolation tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_skill_device_isolation.py tests/test_hitl_settings_device_isolation.py -q
```

Expected: all tests pass against the already device-scoped skill/HITL implementation. A failure indicates regression in the prior fix and must be investigated before continuing; do not weaken these assertions.

- [ ] **Step 4: Run adjacent policy suites**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hitl_api.py tests/test_hitl_turn_policy_injection.py tests/test_hitl_gate_policy.py tests/test_tool_approval_setting_repository.py tests/test_client_tool_isolation.py tests/test_multi_sidecar_hardening.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit regression evidence**

```powershell
git add tests/test_skill_device_isolation.py tests/test_hitl_settings_device_isolation.py
git commit -m "test: lock skill and HITL device isolation"
```

### Task 9 (T009): Final contract and regression verification

**Files:**
- Verify all files listed in the Source Map
- Update only if verification reveals stale wording: `plans/CUSTOM_AGENTS_FE_CONTRACT.md`, `README.md`

- [ ] **Step 1: Run the complete feature matrix**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/client_backend/test_runtime_bridge.py tests/client_backend/test_custom_agents_proxy.py tests/test_custom_agent_capability_resolver.py tests/test_custom_agent_client_tool_resync.py tests/test_custom_agents_service.py tests/test_custom_agents_api.py tests/test_custom_agents_tools.py tests/test_custom_agents_graph.py tests/test_demo_custom_agents.py tests/test_skill_device_isolation.py tests/test_hitl_settings_device_isolation.py tests/test_hitl_turn_policy_injection.py tests/test_hitl_gate_policy.py tests/test_hitl_api.py tests/test_tool_approval_setting_repository.py tests/test_client_tool_isolation.py tests/test_multi_sidecar_hardening.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the broader non-live suite**

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/client_backend/test_live_server_integration.py
```

Expected: all runnable tests pass; environment-dependent skips remain explicitly reported as skips.

- [ ] **Step 3: Run format, lint, and whitespace checks**

```powershell
.\.venv\Scripts\python.exe -m ruff check app/services/custom_agent_capability_resolver.py app/services/custom_agent_service.py app/schemas/custom_agent.py app/api/custom_agents.py app/ai/custom_agent_runtime.py app/ai/agents/custom_agent.py client_backend/services/runtime_bridge.py tests/test_custom_agent_capability_resolver.py tests/test_custom_agent_client_tool_resync.py tests/test_custom_agents_service.py tests/test_custom_agents_api.py tests/test_custom_agents_tools.py tests/test_custom_agents_graph.py tests/test_skill_device_isolation.py tests/test_hitl_settings_device_isolation.py
.\.venv\Scripts\python.exe -m ruff format --check app/services/custom_agent_capability_resolver.py app/services/custom_agent_service.py app/schemas/custom_agent.py app/api/custom_agents.py app/ai/custom_agent_runtime.py app/ai/agents/custom_agent.py client_backend/services/runtime_bridge.py tests/test_custom_agent_capability_resolver.py tests/test_custom_agent_client_tool_resync.py tests/test_custom_agents_service.py tests/test_custom_agents_api.py tests/test_custom_agents_tools.py tests/test_custom_agents_graph.py tests/test_skill_device_isolation.py tests/test_hitl_settings_device_isolation.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 4: Perform a contract consistency scan**

```powershell
rg -n "deviceSnapshot|availability|degraded|device_unavailable|server_name, qualified_tool_id|serverName, qualifiedToolId|custom_agent_warnings" app client_backend demo.py plans/CUSTOM_AGENTS_FE_CONTRACT.md README.md tests
```

Confirm all of the following from the output:

- public responses use camel-case aliases;
- internal refs remain snake_case;
- frontend matching uses server plus qualified ID;
- no implementation scans all devices for a portable match;
- warnings cover tools and skills;
- HITL language remains device-local.

- [ ] **Step 5: Record final verification in the plan progress log or commit message**

If documentation changed during verification:

```powershell
git add plans/CUSTOM_AGENTS_FE_CONTRACT.md README.md
git commit -m "docs: finalize custom agent device capability contract"
```

If no file changed, record the exact passing counts in the implementation handoff without creating an empty commit.

## Acceptance Matrix

| Requirement | Primary evidence |
|---|---|
| No false-ready empty snapshot | T001 bridge event tests + T004 zero-version and rotation guards |
| Same MCP on A/B works independently | T003 resolver + T006 runtime rebind |
| Missing MCP produces hint, agent still runs | T004 availability + T006 prompt/metadata |
| Missing skill produces hint, no leakage | T003/T006 + T008 skill isolation |
| Another device is never used for execution | Existing foreign-device tests + T006 |
| Existing missing refs survive edits; fabricated new bindings fail | T005 exact validation/dedupe + T007 UI preservation/removal control |
| Frontend cache cannot mix snapshots | T002/T004 response + T007 contract |
| HITL settings remain local | T008 same-name server regression |
| No schema migration required | Source review + unchanged Alembic heads |

## Execution Notes

- Preserve unrelated working-tree changes. Stage only the paths named by each task.
- Do not run the live sidecar integration test unless the canonical server and sidecar prerequisites are intentionally available.
- Stop after any RED test that fails for a reason other than the missing behavior described in that step.
- Keep runtime resolution pure and deterministic; do not add network calls or database writes to the resolver.
- Do not weaken the exact five-field matcher or dispatch-time session/catalog/tool-instance validation.
- Do not add server-side skills or account-wide HITL rules while implementing portable Custom Agent intent.
