# MCP Configuration V2 and Device Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace legacy MCP configuration handling with a strict schema-v2 registry/profile model that is encrypted, atomic, relocatable, and isolated per device for users signed into multiple devices.

**Architecture:** Repository-owned bundled definitions remain in one schema-v2 registry. A device-scoped profile stores only bundled enabled-state overrides and custom definitions; a separate encrypted store holds environment/header values. An isolated idempotent migrator is the only code allowed to parse old formats, while the manager and API consume a typed `MCPConfigStore`.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI, LangChain MCP adapters, pytest, Windows DPAPI/Fernet through existing security helpers

---

## File Map

- Create `client_backend/schemas/mcp_config.py`: strict registry/profile and transport models.
- Create `client_backend/services/mcp_secret_store.py`: encrypted device-scoped MCP credentials.
- Create `client_backend/services/mcp_config_store.py`: typed merge, validation, paths, and atomic mutations.
- Create `client_backend/services/mcp_config_migration.py`: isolated legacy-to-v2 migration.
- Create `tests/client_backend/test_mcp_config_v2.py`: schema, storage, credentials, migration, and device-isolation tests without FastAPI/httpx dependencies.
- Modify `client_backend/core/paths.py`: device-scoped profile helper.
- Modify `app/ai/mcp_config.json`: canonical schema-v2 bundled registry.
- Modify `client_backend/services/local_mcp_manager.py`: consume effective typed configurations only.
- Modify `client_backend/services/runtime_bridge.py`: bind the manager to the authenticated installation scope.
- Modify `client_backend/api/mcp.py`: use scoped store/manager mutations and expose source metadata.
- Modify `client_backend/main.py`, `client_backend/cli.py`: construct an explicit scope.
- Replace legacy expectations in `tests/client_backend/test_local_mcp_manager.py`, `test_runtime_bridge.py`, and `test_mcp_tool_execution_api.py`.

### Task 1: Strict V2 Models and Device Paths

- [ ] Add failing tests proving:
  - `schemaVersion != 2` is rejected;
  - unknown fields are rejected;
  - stdio requires `command`, HTTP requires `url`;
  - two device identifiers for one user resolve different MCP directories.

Run:

```powershell
conda run -n agents python -m pytest tests/client_backend/test_mcp_config_v2.py -k "schema or device_path" -v
```

Expected: collection fails because `client_backend.schemas.mcp_config` and the
device-scoped path helper do not exist.

- [ ] Implement strict models using `ConfigDict(extra="forbid", populate_by_name=True)`:

```python
class MCPProfileScope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    user_id: str = Field(min_length=1)
    device_identifier: str = Field(min_length=1)

class BundledOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool

class MCPProfileDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_version: Literal[2] = Field(alias="schemaVersion")
    bundled_overrides: dict[str, BundledOverride] = Field(
        default_factory=dict, alias="bundledOverrides"
    )
    custom_servers: dict[str, CustomServerDefinition] = Field(
        default_factory=dict, alias="customServers"
    )
```

Use a discriminated union for stdio and HTTP definitions. Add:

```python
def get_device_profile_subdir(
    user_id: str, device_identifier: str, subdir: str
) -> Path:
    return get_profile_subdir(user_id, "devices") / device_identifier / subdir
```

Validate both identifiers as single safe path components before joining.

- [ ] Convert `app/ai/mcp_config.json` to:

```json
{"schemaVersion": 2, "servers": {...}}
```

Keep the current bundled definitions and replace `enabled` with
`enabledByDefault`.

- [ ] Run the focused tests and Ruff; commit:

```powershell
git add -- client_backend/schemas/mcp_config.py client_backend/core/paths.py app/ai/mcp_config.json tests/client_backend/test_mcp_config_v2.py
git commit -m "feat: define device-scoped MCP config v2"
```

### Task 2: Encrypted Device-Scoped Credentials

- [ ] Add failing tests proving two devices under one user cannot read each
  other's credentials, the file contains no plaintext token, listing returns
  names only, and deleting a server removes its binding.

- [ ] Implement `MCPSecretStore(scope, root_override=None)` with:

```python
def get_for_server(self, server_name: str) -> MCPServerCredentials: ...
def set_for_server(
    self, server_name: str, *, env: dict[str, str], headers: dict[str, str]
) -> None: ...
def delete_server(self, server_name: str) -> bool: ...
```

Encrypt one versioned payload with `encrypt_local_secret`, decrypt with
`decrypt_local_secret`, validate environment/header names, write through a
same-directory temporary file plus `os.replace`, and never log values.

- [ ] Run:

```powershell
conda run -n agents python -m pytest tests/client_backend/test_mcp_config_v2.py -k credentials -v
conda run -n agents python -m ruff check client_backend/services/mcp_secret_store.py tests/client_backend/test_mcp_config_v2.py
```

- [ ] Commit:

```powershell
git add -- client_backend/services/mcp_secret_store.py tests/client_backend/test_mcp_config_v2.py
git commit -m "feat: encrypt device-scoped MCP credentials"
```

### Task 3: Canonical Configuration Store

- [ ] Add failing tests for registry/profile merging, bundled-name collision,
  bundled path resolution, custom profile-relative paths, atomic-write failure,
  independent device mutations, and credential redaction.

- [ ] Implement `MCPConfigStore`:

```python
class MCPConfigStore:
    def __init__(
        self,
        scope: MCPProfileScope,
        *,
        registry_path: Path | None = None,
        profile_root: Path | None = None,
    ): ...

    def load_profile(self) -> MCPProfileDocument: ...
    def list_effective_servers(self) -> list[EffectiveMCPServer]: ...
    def save_custom_server(
        self, name: str, definition: CustomServerDefinition, credentials: ...
    ) -> None: ...
    def delete_server(self, name: str) -> Literal["disabled_bundled", "deleted_custom"]: ...
    def set_enabled(self, name: str, enabled: bool) -> None: ...
```

All profile writes validate a complete candidate document before atomic replace.
Bundled Python commands become `sys.executable`; bundled relative paths and cwd
resolve against the application root. Custom paths resolve against the
device-scoped MCP directory. Custom names colliding with bundled names raise
`MCPConfigConflictError`.

- [ ] Run store tests, Ruff, and commit:

```powershell
git add -- client_backend/services/mcp_config_store.py tests/client_backend/test_mcp_config_v2.py
git commit -m "feat: centralize canonical MCP config storage"
```

### Task 4: Isolated One-Time Migrator

- [ ] Add failing migration fixtures for snake-case, camel-case, dual-key,
  exact bundled matches, custom servers, credentials, modified bundled-name
  conflicts, backups, receipts, idempotency, and simulated write failure.

- [ ] Implement only in `mcp_config_migration.py`:

```python
def migrate_legacy_mcp_profile(
    scope: MCPProfileScope,
    *,
    legacy_path: Path,
    store: MCPConfigStore,
) -> MigrationResult: ...
```

The migrator may parse `mcp_servers`, `mcpServers`, and `_sample_chatbot_*`.
No other module may reference those keys. It must compare normalized legacy
entries against registry entries while allowing only enabled-state differences,
extract credentials, back up the source, write a secret-free receipt, and stop
unchanged on reserved-name conflicts.

- [ ] Run migration tests and assert the legacy-key inventory:

```powershell
conda run -n agents python -m pytest tests/client_backend/test_mcp_config_v2.py -k migration -v
rg -n 'mcp_servers|_sample_chatbot_seed|_sample_chatbot_bundled_servers' client_backend app tests/client_backend
```

Expected after later cleanup: production matches exist only in the migration
module; test matches exist only in migration fixtures.

- [ ] Commit:

```powershell
git add -- client_backend/services/mcp_config_migration.py tests/client_backend/test_mcp_config_v2.py
git commit -m "feat: migrate legacy MCP profiles to v2"
```

### Task 5: Manager and Runtime Bridge Conversion

- [ ] Add failing tests proving `LocalMCPManager` requires a scope/store, starts
  real bundled `time`, reports useful nested exception causes, and two scoped
  managers publish independent catalogs.

- [ ] Remove config parsing, seeding, canonicalization, provenance, and path
  guessing from `LocalMCPManager`. Build `MultiServerMCPClient` exclusively from
  `store.list_effective_servers()`.

- [ ] Replace the single global with a scope-keyed manager:

```python
def get_mcp_manager(scope: MCPProfileScope | None = None) -> LocalMCPManager: ...
async def shutdown_mcp_manager(scope: MCPProfileScope | None = None) -> None: ...
```

Calls without an explicit scope may resolve the current authenticated user and
installation identifier, but must fail when either is unavailable.

- [ ] Bind `RuntimeBridgeService` once to its `MCPProfileScope` and reuse that
  scope for initialize, call, and catalog generation.

- [ ] Extend `client_backend/cli.py` with non-interactive maintenance commands:

```python
def migrate_mcp_config() -> int:
    """Migrate the active authenticated installation profile and print no secrets."""

def doctor_mcp_servers(server_names: list[str]) -> int:
    """Initialize selected effective servers and return nonzero when any fail."""
```

Expose them as `mcp migrate` and `mcp doctor --servers <csv>`. Migration prints
only status, backup/receipt paths, and server names. Doctor prints server status,
tool count, and a redacted terminal error.

- [ ] Run manager/runtime tests and commit:

```powershell
git add -- client_backend/services/local_mcp_manager.py client_backend/services/runtime_bridge.py client_backend/main.py client_backend/cli.py tests/client_backend/test_local_mcp_manager.py tests/client_backend/test_runtime_bridge.py
git commit -m "refactor: bind MCP runtime to device-scoped v2 config"
```

### Task 6: API Conversion and Device Isolation

- [ ] Add failing API service tests using direct endpoint calls with two
  `LocalSessionPayload` values. Verify:
  - reads show `source`;
  - custom add/update/delete stays in the authenticated device scope;
  - bundled delete writes a disabled override;
  - reserved collisions return 409;
  - credential values never appear;
  - refresh targets only the authenticated device bridge.

- [ ] Replace direct JSON helpers in `client_backend/api/mcp.py` with:

```python
def _scope(session: LocalSessionPayload) -> MCPProfileScope:
    return MCPProfileScope(
        user_id=session.user_id,
        device_identifier=session.device_identifier,
    )
```

Every endpoint receives the session dependency, opens its scoped store/manager,
and uses typed mutations. Delete `canonicalize_mcp_config_document`,
`_remove_bundled_provenance`, and all direct file writes.

- [ ] Run API and isolation tests; commit:

```powershell
git add -- client_backend/api/mcp.py tests/client_backend/test_mcp_config_v2.py tests/client_backend/test_mcp_tool_execution_api.py
git commit -m "refactor: isolate MCP APIs by authenticated device"
```

### Task 7: Legacy Inventory Removal and Production Verification

- [ ] Delete superseded legacy tests and replace their assertions with v2
  equivalents. Update imports and singleton fixtures.

- [ ] Verify legacy-key isolation:

```powershell
rg -n 'canonicalize_mcp_config_document|_sample_chatbot_seed|_sample_chatbot_bundled_servers|\"mcp_servers\"' client_backend app tests/client_backend
```

Expected: legacy document keys appear only in
`mcp_config_migration.py` and its migration fixtures.

- [ ] Run static and focused suites:

```powershell
conda run -n agents python -m ruff check client_backend tests/client_backend
conda run -n agents python -m pytest tests/client_backend/test_mcp_config_v2.py tests/client_backend/test_runtime_bridge.py tests/test_client_tool_isolation.py tests/test_client_invocation_isolation.py tests/test_skill_device_isolation.py -q
```

- [ ] Back up and migrate the current profile using the production migrator.
Do not print configuration or credentials. Verify the receipt contains names and
hashes only, and verify the three reported servers discover tools:

```powershell
conda run -n agents python -m client_backend.cli mcp migrate
conda run -n agents python -m client_backend.cli mcp doctor --servers widgets,tavily,time
```

Expected: all three sessions initialize and return at least one tool; no
`Connection closed`.

- [ ] Run the broad suite:

```powershell
conda run -n agents python -m pytest -q
```

Record unrelated environment-contract failures separately, including the known
`httpx2`/LangChain environment mismatch, and require every MCP/device-isolation
test to pass.

- [ ] Commit final cleanup if required:

```powershell
git add -- client_backend app/ai/mcp_config.json tests/client_backend docs
git commit -m "test: verify MCP v2 migration and device isolation"
```
