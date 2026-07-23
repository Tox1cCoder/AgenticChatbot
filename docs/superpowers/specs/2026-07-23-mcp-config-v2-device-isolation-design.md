# MCP Configuration V2 and Device Isolation Design

## Problem

The local MCP subsystem currently mixes three concerns in one unversioned JSON
document:

- repository-owned bundled server definitions;
- user-owned custom server definitions and credentials;
- legacy provenance fields used to guess which entries are bundled.

Older profile files have no provenance fields. Their repository-relative Python
scripts are therefore resolved relative to the profile directory and terminate
with `No such file or directory`; the MCP client surfaces only `Connection
closed`.

The runtime and API also contain duplicate legacy normalization and mutation
logic. Local profile paths are scoped by server and user, but not explicitly by
device. Separate physical filesystems normally hide that gap, while the
server-side runtime correctly scopes catalogs and dispatch by `device_id`.

## Goals

- Replace all normal-path legacy parsing with one canonical schema version.
- Keep bundled definitions relocatable and owned by the repository.
- Preserve user-defined servers during a one-time migration.
- Isolate MCP configuration, credentials, catalogs, and mutations by device for
  two devices signed into the same user account.
- Centralize validated, atomic configuration persistence.
- Encrypt MCP environment and HTTP-header values at rest.
- Produce actionable configuration errors without logging secret values.

## Non-Goals

- Synchronizing local MCP configuration between devices.
- Sharing local credentials across devices.
- Changing the canonical server's already device-scoped runtime protocol.
- Changing MCP tool IDs or tool execution semantics.
- Supporting arbitrary overrides of bundled server commands or script paths.

## Canonical Sources

### Bundled registry

`app/ai/mcp_config.json` becomes the repository-owned schema-v2 registry:

```json
{
  "schemaVersion": 2,
  "servers": {
    "time": {
      "transport": "stdio",
      "command": "python",
      "args": ["app/ai/mcp_servers/time_server.py"],
      "enabledByDefault": true,
      "description": "Current time lookup utility with timezone support"
    }
  }
}
```

Only application code and releases modify this file. Bundled names are reserved.
Bundled Python commands use the active sidecar interpreter. Relative bundled
paths and working directories resolve against the repository/application root,
never the profile directory or current workspace.

### Device profile

Each device stores only local state and custom definitions:

```text
{profile_root}/{server_hash}/{user_id}/devices/{device_identifier}/mcp/
  config.v2.json
  credentials.json
  migration/
```

`config.v2.json` has one accepted shape:

```json
{
  "schemaVersion": 2,
  "bundledOverrides": {
    "time": {"enabled": false}
  },
  "customServers": {
    "desktop-commander": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@wonderwhy-er/desktop-commander@latest"],
      "cwd": null,
      "envKeys": [],
      "enabled": true,
      "description": ""
    }
  }
}
```

Bundled overrides permit only `enabled`. Custom stdio definitions permit
`command`, `args`, `cwd`, `envKeys`, `enabled`, and `description`. Custom HTTP
definitions permit `url`, `headerKeys`, `enabled`, and `description`.

Unknown fields and unknown schema versions are rejected.

## Device Scope

The stable installation `device_identifier` is the local persistence key because
it exists before server registration and remains stable if the server-assigned
device UUID is refreshed. MCP API handlers obtain `user_id` and
`device_identifier` from the verified `LocalSessionPayload`; request bodies and
query parameters cannot choose another device's scope.

`LocalMCPManager` and the configuration store receive an explicit immutable
scope object:

```python
@dataclass(frozen=True)
class MCPProfileScope:
    user_id: str
    device_identifier: str
```

The manager singleton is keyed by this scope. A scope change shuts down the old
manager before constructing another. Runtime-bridge initialization uses its own
stable installation identifier and authenticated user, producing the same scope
as local API calls.

The canonical server already isolates live sessions, catalogs, versions, tool
instance IDs, dispatch queues, and results by server-assigned `device_id`. Tests
will verify both sides together for one user with two device identities.

## Credentials

Environment-variable values and HTTP-header values are not stored in
`config.v2.json`. A device-scoped `MCPSecretStore` stores:

```json
{
  "version": 1,
  "servers": {
    "notionApi": {
      "env": {"NOTION_TOKEN": "..."},
      "headers": {}
    }
  }
}
```

The entire payload is encrypted with the existing `encrypt_local_secret`
primitive (Windows DPAPI or the existing Fernet fallback) and written to
`credentials.json` with best-effort restrictive permissions.

The API accepts credential values on create/update but returns only configured
key names and redacted placeholders. Logs and migration reports include server
and key names only. Removing a custom server removes its credential entry.

## Configuration Store

Introduce a single `MCPConfigStore` responsible for:

- resolving the device-scoped profile path;
- validating registry and profile documents with strict Pydantic models;
- combining bundled definitions, allowed overrides, custom definitions, and
  decrypted credentials into runtime server configurations;
- rejecting custom names that collide with bundled names;
- performing atomic writes via a temporary file in the target directory followed
  by `os.replace`;
- serializing writes with a process-local async lock;
- exposing typed add, update, delete, and toggle operations.

The local MCP manager and MCP API must use this store. Neither component reads or
writes JSON directly.

## One-Time Migration

Legacy support is isolated in
`client_backend/services/mcp_config_migration.py`. Normal runtime modules do not
import legacy normalization helpers.

Before the v2 store initializes, the migrator checks the old user-scoped profile
path:

1. If a valid device-scoped v2 file exists, do nothing.
2. If no legacy file exists, create an empty v2 profile.
3. Read the legacy file and reject malformed documents without changing them.
4. Canonically merge `mcp_servers` and `mcpServers` inside the migrator only;
   canonical entries win on duplicate names.
5. Compare each legacy server with the current bundled registry after ignoring
   the allowed enabled-state difference.
6. Convert matching bundled entries to `bundledOverrides`.
7. Preserve nonmatching, noncolliding entries as `customServers`.
8. Extract `env` and `headers` values into the encrypted credential store.
9. If a nonmatching custom entry uses a reserved bundled name, stop with a
   conflict report; do not guess or overwrite.
10. Write timestamped backups beneath the device migration directory.
11. Atomically write credentials and `config.v2.json`.
12. Write a migration receipt containing source hash, backup path, timestamp,
    and migrated server names but no credentials.

The migration is idempotent. Failure leaves the legacy source untouched and does
not produce a partial v2 configuration. After the supported upgrade window, the
isolated migration module can be removed without changing runtime code.

The current installation's legacy profile is migrated only to the current
`device_identifier`. Other devices migrate their own local profiles.

## API Behavior

The existing endpoint paths remain stable.

- List and detail responses add `source: "bundled" | "custom"` and return the
  effective enabled state.
- Adding or updating a custom server whose name is bundled returns `409`.
- Toggling a bundled server writes only an enabled override.
- Deleting a bundled server disables it; its trusted registry definition remains.
- Deleting a custom server removes its definition and credentials.
- API writes reload only the manager for the authenticated local scope and then
  refresh that device's runtime-bridge catalogs.
- Tool execution continues resolving by qualified tool ID and current manager
  scope.

## Legacy Removal

Delete from normal runtime and API code:

- `canonicalize_mcp_config_document`;
- dual-key reads and writes;
- `_sample_chatbot_seed`;
- `_sample_chatbot_bundled_servers`;
- provenance-removal mutation helpers;
- repository-default copying into profile files;
- direct JSON mutation in API handlers;
- user-only MCP manager/config singleton resolution.

Replace legacy compatibility tests with schema-v2, migration-boundary, and
device-isolation tests. Legacy fixtures exist only in migration tests.

## Failure Handling

- Invalid v2 documents fail initialization with a concise schema error and file
  path; the system does not silently start with an empty catalog.
- One failing MCP server is marked unavailable without blocking other configured
  servers.
- Startup errors unwrap exception groups to log the useful terminal cause, such
  as a missing executable or script, while retaining the full traceback at debug
  level.
- Credential decryption failure disables only the affected server and never logs
  secret contents.
- Migration conflicts and failures leave the source and any prior valid v2 file
  unchanged.

## Test Strategy

### Schema and store

- Accept exactly schema v2 and reject unknown versions/fields.
- Validate transport-specific required fields.
- Resolve bundled scripts against the repository root with the active
  interpreter.
- Resolve custom relative paths against the device profile.
- Verify atomic replacement and unchanged valid state after a simulated failure.
- Verify credential encryption, redacted API reads, and credential deletion.

### Migration

- Migrate snake-case, camel-case, and dual-key legacy fixtures.
- Recognize exact bundled definitions without copying their paths.
- Preserve custom servers and credentials.
- Stop safely on modified bundled-name collisions.
- Verify backup, receipt, idempotency, and no secret values in logs/receipts.

### Device isolation

For the same `user_id` and two distinct `device_identifier` values:

- resolve different config and credential paths;
- create different custom servers and enabled overrides;
- prove API reads and mutations cannot cross scopes;
- produce independent local tool catalogs and catalog versions;
- refresh only the authenticated device's runtime catalog;
- maintain server-side sessions, queues, and dispatch by different `device_id`
  values;
- reject a tool instance from device A when invoked through device B.

### End-to-end

- Migrate a legacy profile matching the reported failure.
- Initialize real stdio sessions for `widgets`, `tavily`, and `time`.
- Confirm their tools are discovered without `Connection closed`.
- Run existing local MCP API, runtime bridge, client-tool isolation, and
  multi-sidecar regression suites.

## Rollout

1. Ship schema-v2 registry, models, store, credential store, and migrator.
2. Migrate the current device profile before manager initialization.
3. Verify the migration receipt and device-scoped catalog publication.
4. Retain legacy files as backups during the upgrade window.
5. Remove the isolated migrator in a later release after the upgrade window;
   runtime code is already legacy-free.

## Acceptance Criteria

- The reported bundled Python servers start and publish tools.
- Runtime and API code accept only schema v2.
- Existing custom servers and credential values survive migration.
- No bundled command/path definition is copied into a device profile.
- Two devices on one account have independent configurations, credentials,
  catalogs, versions, and dispatch.
- No secret value appears in configuration responses, logs, receipts, or tests.
- Focused and related automated tests pass.
