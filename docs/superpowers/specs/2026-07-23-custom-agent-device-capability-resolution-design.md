# Custom-Agent Device Capability Resolution Design

**Date:** 2026-07-23
**Status:** Approved for implementation planning

## Goal

Keep a custom agent's requested MCP tools and skills synchronized with its
account-wide definition while resolving and executing those capabilities only
from the active sidecar device. A custom agent remains usable when a requested
local capability is missing, but the API, UI, and completed response clearly
report that its behavior may be degraded.

## Problem

Custom agents are account-owned records, but their saved client-tool references
currently contain device-, session-, catalog-, and instance-specific fields.
Runtime rebasing handles a reconnect only when the saved and active device IDs
are identical. Consequently, a tool selected on machine A is treated as stale
on machine B even when machine B publishes the same MCP server and qualified
tool ID.

There is also a catalog-readiness race. The sidecar sets its connected event
immediately after the WebSocket handshake and only then uploads its MCP and
skill catalogs. Login/session restoration may therefore return before catalog
sync completes. A frontend options request in that window receives a valid
device identity with empty client catalogs and can cache the empty snapshot.
This produces the reported combination of no active device tools plus saved
selections that are not in the current snapshot.

Earlier work has already moved skills and HITL rules onto device-scoped lookup
paths. Those paths require regression verification, not a return to
account-wide runtime state.

## Product Semantics

The custom-agent definition is account-wide. Its prompt, model, server-side
tool preferences, requested client MCP tools, and requested skills follow the
user between devices.

Client capability availability is device-local:

- A saved MCP selection expresses the stable intent to use one exact logical
  tool, identified by `(server_name, qualified_tool_id)`.
- A saved skill selection expresses the stable intent to use one exact skill,
  identified by its client source and lookup name/name compatibility identity.
- Volatile client-tool fields (`device_id`, `session_id`, `catalog_version`, and
  `tool_instance_id`) remain last-bound metadata and authorization inputs. They
  do not define the account-wide logical selection.
- At request time, a saved selection may rebind only to an exact logical match
  in the requesting device's active catalog. Resolution is deterministic; it
  never uses fuzzy names, another connected device, or a server-side fallback.
- The live match supplies the requesting device, session, catalog version, and
  tool instance used by the existing strict binding and dispatch checks.
- A missing capability is skipped. The custom agent continues with its other
  available skills, client tools, server tools, and normal language ability.

Installing the same MCP server on machines A and B therefore gives both devices
the same requested logical capability, but each invocation uses only that
machine's live tool instance. If only A has the MCP server, B sees a degraded
agent and cannot execute A's tool.

## Architecture

### 1. Catalog readiness boundary

`RuntimeBridgeService` will not signal initial readiness until both the MCP tool
catalog and skill catalog have synchronized successfully for the newly
registered WebSocket session. Authentication may still start the bridge in the
background, but its bounded readiness wait now covers registration, handshake,
and both catalog uploads.

The runtime state may internally enter `CONNECTED` after the handshake so the
catalog upload can use the registered connection. The public connected event is
the readiness boundary and is set only after `refresh_catalogs()` returns.
Catalog-sync failure must not publish a ready event; the existing reconnect loop
handles the failure. This removes the deterministic login/options race without
introducing sleeps.

Every successful custom-agent options response will include additive snapshot
metadata:

```json
{
  "deviceSnapshot": {
    "deviceId": "device-uuid",
    "sessionId": "session-id",
    "toolCatalogVersion": 3,
    "skillCatalogVersion": 2,
    "status": "ready"
  }
}
```

When no caller-owned active session exists, `deviceSnapshot.status` is
`"unavailable"`, session/version fields are null, and client options are empty.
The frontend keys its options cache by the complete snapshot identity and
invalidates it when that identity changes. An unavailable snapshot is a runtime
state, not proof that the account-wide selections should be deleted.

### 2. Shared exact capability resolver

A focused resolver will compare saved requested capabilities with one
caller-owned active device session. It returns:

- current exact client-tool refs with live volatile identity fields;
- resolved client skills;
- structured missing tool and skill entries; and
- safe human-readable warnings derived from those missing entries.

Client MCP resolution requires both `server_name` and `qualified_tool_id` to
match. Requiring both prevents accidental substitution when two servers expose
similar tool names. Resolution first filters the live catalog to the request's
device and current session; a catalog belonging to the same user but another
device is never a candidate.

Skill resolution continues through `skill_resolver`, which already reads only
the caller-owned active device's skill catalog. The shared resolver adds the
missing-set calculation so selected skills that did not resolve become visible
warnings instead of disappearing silently.

Runtime filtering and dispatch remain strict. Portable resolution happens
before the existing exact metadata matcher, then the matcher and sidecar
dispatch validation still require current device/session/catalog/instance
identity. The portable account-wide selection is not itself an execution
credential.

### 3. Availability in custom-agent reads

Custom-agent list and single-read routes will accept the same optional
`deviceId` context already stamped by the sidecar. Each returned custom agent
will contain an additive `availability` object:

```json
{
  "status": "degraded",
  "deviceId": "device-uuid",
  "sessionId": "session-id",
  "missingTools": [
    {
      "serverName": "desktop-commander",
      "qualifiedToolId": "desktop-commander::read_file",
      "toolName": "read_file"
    }
  ],
  "missingSkills": [
    {
      "lookupName": "kobo-library",
      "name": "kobo-library"
    }
  ],
  "warnings": [
    "Selected MCP tool 'desktop-commander::read_file' is not available on this device.",
    "Selected skill 'kobo-library' is not available on this device."
  ]
}
```

Status rules are:

- `ready`: all requested local capabilities resolve, or the agent requests no
  local capabilities;
- `degraded`: an active device snapshot exists but at least one requested local
  capability is missing;
- `device_unavailable`: the agent requests local capabilities but no
  caller-owned active device snapshot can be read.

The availability response does not expose another device's live catalog,
installation list, paths, secrets, session IDs, or tool instance IDs. Saved
refs—including their last-bound opaque identity fields—remain present in
`toolRefs`/`skillRefs` for wire compatibility, but they grant no access to that
prior device and `availability` is authoritative for current UI status. The
frontend matches current selectable MCP entries by the same stable
`(server_name, qualified_tool_id)` identity rather than by saved device or
session fields.

The UI should show a non-blocking warning badge on degraded agents, list the
missing capabilities, and explain that the agent can still run but may behave
differently. It should offer refresh and configuration actions, not force the
user to clear saved selections.

### 4. Create and update validation

New client capability selections must come from the active device snapshot.
This prevents callers from fabricating logical refs that have never been
observed. Updating unrelated custom-agent fields must not fail because an
existing requested capability is currently unavailable.

When a complete tool/skill list is submitted during update:

- a currently available ref is accepted;
- an unavailable ref already present on that custom agent may be retained;
- a new unavailable ref is rejected with the existing custom-agent validation
  error family; and
- client MCP refs are deduplicated by stable logical identity, not volatile
  device/session/instance identity.

This lets a frontend preserve account-wide intent while editing on a device
that lacks some capabilities. Selecting the same logical MCP tool on machine B
does not create a second account-wide selection merely because its runtime
instance differs.

No database migration is required. Existing refs already carry the stable
server and qualified-tool fields. Their device/session/instance fields become
last-bound compatibility metadata, and all new resolution uses the stable
identity before creating an in-memory exact runtime binding.

### 5. Runtime warnings

The runtime will use the shared resolver before building the custom agent's
restricted tools and skill prompt suffix. Missing MCP tools and missing skills
produce one deduplicated warning per logical capability. These warnings are:

- available before execution through the management API's `availability`;
- available to the custom agent as concise context so it does not assume the
  missing capability exists; and
- preserved in final assistant message metadata under the existing
  `custom_agent_warnings` field.

Warnings must not contain local paths, secrets, arguments, tool results, or
catalog contents from another device. A missing capability never causes the
runtime to broaden tool search or borrow a same-named capability outside the
requesting device.

### 6. Skills and HITL isolation

Skills remain client-owned only. Server-side skill enumeration is not restored.
All skill options, prompt summaries, activation, missing-skill calculation, and
execution resolve through the requesting device's active skill catalog. A skill
installed only on machine A is absent and reported missing on B; the server
never supplies A's skill content to B.

Editable HITL rules remain keyed by
`(user_id, device_id, tool_origin, scope_type, scope_value)`. Settings reads
return only the requested caller-owned device's rules. Writes still require an
active catalog target for that device. The custom-agent portable selection does
not make HITL policy portable: when the same logical MCP tool is available on B,
B's own HITL rule (or inherited default) governs its execution.

## Error Handling

- Initial catalog synchronization failure leaves the sidecar not ready and
  follows the existing reconnect/error path; it must not return a false-ready
  empty snapshot.
- A catalog that disappears after options were loaded resolves as degraded at
  read/invocation time and cannot pass strict dispatch validation.
- A session or catalog rotation between binding and dispatch continues to fail
  closed through the existing session/catalog/tool-instance checks.
- Missing client capabilities never make an otherwise valid custom agent
  unusable and do not turn normal prompt/model edits into validation failures.
- Requests naming an unowned device receive no client capabilities, matching
  the existing owner checks.

## Testing Strategy

Tests will reproduce and lock down the following boundaries:

1. The bridge readiness event remains unset during catalog uploads and is set
   only after both uploads succeed; a failed upload never publishes readiness.
2. A custom agent saved with machine A's exact `desktop-commander` ref resolves
   to machine B's live ref when B exposes the exact same server and qualified
   tool ID.
3. The same tool name under a different server or qualified ID does not resolve.
4. Resolution considers only the request device even when A and B are connected
   simultaneously under the same account.
5. A missing MCP tool produces degraded management availability and final
   runtime warning metadata while the agent remains executable.
6. A missing selected skill produces the same degraded behavior and never
   exposes a skill installed only on another device.
7. Create rejects fabricated unavailable refs; update preserves an already
   saved unavailable ref and deduplicates the same logical MCP selected from a
   second device.
8. Options and agent reads expose snapshot/device identity suitable for frontend
   cache boundaries.
9. Existing two-device skill isolation tests prove `cli-anything-google-calendar`
   and `kobo-library` cannot leak from A to B.
10. Existing HITL repository/API/gate tests prove rules from A are absent on B,
    including when both devices publish the same MCP server name.
11. Existing strict runtime dispatch tests continue to reject stale session,
    catalog, and tool-instance identifiers.

Focused tests will cover the resolver and sidecar bridge directly. The broader
custom-agent, client-runtime isolation, skills, HITL, proxy, and API suites will
be run before completion.

## Frontend Contract Changes

The API additions are backward-compatible:

- `CustomAgentOptions.deviceSnapshot` is additive.
- `CustomAgentRead.availability` is additive and populated when device context
  is available or the agent requests local capabilities.
- list/read requests through the sidecar are stamped with the active device ID;
  direct canonical API clients should pass `deviceId` to receive contextual
  availability.

Frontend selection matching must use stable logical identities. Device,
session, catalog, and tool-instance values remain necessary when submitting a
new current option, but they must not be used as the identity of an
account-wide saved selection. Cache keys must include the returned snapshot
identity. A `degraded` or `device_unavailable` state is displayed as a warning,
not as a disabled agent.

## Non-Goals

- Sharing a sidecar's live catalog, skill content, credentials, or HITL policy
  with another device.
- Fuzzy substitution of similarly named MCP servers, tools, or skills.
- Making server-side MCP selection a new permission boundary.
- Changing custom-agent ownership, conversation attachment, or model-selection
  semantics.
- Persisting volatile rebased session/catalog/tool-instance values after every
  invocation.

## Acceptance Criteria

- Opening Custom Agent management immediately after login cannot observe a
  false-ready empty catalog.
- The same account-wide requested MCP tool works on A and B when each active
  device independently publishes the exact logical capability.
- A capability installed only on A is never executable or disclosed as live on
  B; B reports the custom agent as degraded instead.
- Missing MCP tools and skills are visible before execution and in completed
  assistant metadata, while the custom agent remains usable.
- Skills and HITL settings remain strictly isolated by active device.
- Existing strict client-runtime authorization checks remain intact.
