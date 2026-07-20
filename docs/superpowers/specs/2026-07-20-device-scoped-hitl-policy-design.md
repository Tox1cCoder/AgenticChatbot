# Device-Scoped HITL Policy Design

## Goal

Prevent HITL settings for client-side MCP servers and skills on one device
from appearing or taking effect on another device owned by the same user.
Keep global-tool policy read-only and controlled by server configuration.

## Root Cause

Editable HITL rules are currently keyed only by user, scope type, and scope
value. The sidecar proxies HITL settings requests without binding its device,
and runtime policy is loaded before the chat request's device is validated.
Consequently, a skill command or local MCP rule saved on machine A is returned
on machine B and can affect a same-named tool there.

The skill bundles and local MCP configurations are already device-local. The
leak is limited to their account-wide approval-policy records.

## Policy Boundary

There are two policy owners:

- Global and server-hosted tools use read-only policy from server
  configuration. Safe built-ins such as search and time are available by
  default and normally require no approval. Users cannot edit this policy.
- Client MCP and client skill tools use editable rules scoped to the
  authenticated user and the originating device.

Editable rows identify their target with:

```text
(user_id, device_id, tool_origin, scope_type, scope_value)
```

`tool_origin` is restricted to `client_mcp` and `client_skill`.
`scope_type` remains `server` or `tool`, preserving the runtime precedence of
exact tool override, then server default, then configured/global fallback.

## Data Model and Migration

`tool_approval_settings` gains:

- `device_id`, a non-null foreign key to `client_devices.id`;
- `tool_origin`, a non-null constrained string containing `client_mcp` or
  `client_skill`.

The uniqueness constraint becomes the five-column identity above. Repository
list, set, delete, and policy-building operations require user and device.

Existing account-wide rows cannot be mapped safely to a device. The migration
performs a documented one-time reset of those rows before making the new
columns non-null. It does not infer ownership from names or transient catalogs.

## API Contract

The sidecar stamps its registered device ID on `GET`, `POST`, and `DELETE
/hitl/settings`. It overrides a caller-supplied device ID rather than trusting
stale frontend state.

The canonical server:

1. requires `deviceId` (with `device_id` accepted as a compatibility alias);
2. verifies that the device belongs to the authenticated user;
3. returns or mutates only that device's editable rules;
4. accepts `toolOrigin` on writes and deletes;
5. accepts only `client_mcp` and `client_skill` origins;
6. validates new rules against the active device catalog;
7. permits deletion of an existing device rule after its target is removed.

The response echoes `deviceId`. Skill command rules remain in `tools` because
HITL approves executable tool calls, but every rule includes `toolOrigin`, so
the frontend can group `client_skill` rules under a Skills heading.

```json
{
  "masterEnabled": true,
  "deviceId": "device-uuid",
  "globalTools": [],
  "servers": [
    {
      "scopeType": "server",
      "scopeValue": "desktop-commander",
      "toolOrigin": "client_mcp",
      "requireApproval": false
    }
  ],
  "tools": [
    {
      "scopeType": "tool",
      "scopeValue": "skill::kobo-library::run_skill_command",
      "toolOrigin": "client_skill",
      "requireApproval": false
    }
  ]
}
```

Machine B receives empty `servers` and `tools` arrays when it has no local
rules, regardless of rules configured on machine A.

Stable client errors are:

| Status | Code | Meaning |
|---|---|---|
| 422 | `HITL_DEVICE_REQUIRED` | No sidecar device context was supplied. |
| 404 | `HITL_DEVICE_NOT_FOUND` | The device is unknown or not owned by the caller. |
| 409 | `HITL_DEVICE_RUNTIME_UNAVAILABLE` | A write cannot validate against an active device catalog. |
| 422 | `HITL_TOOL_ORIGIN_INVALID` | The origin is not `client_mcp` or `client_skill`. |
| 409 | `HITL_TARGET_UNAVAILABLE` | A new or updated rule does not identify a target in the active device catalog. |

Unknown and foreign devices share one response to avoid exposing another
user's device identifiers. Writes fail closed; they never fall back to
account-wide storage.

## Runtime Enforcement

The message service validates the request device before resolving HITL
policy. It then builds policy using `(user_id, device_id)`. A turn without a
valid active device receives no editable client policy and binds no client
tools.

Policy matching uses tool provenance as part of identity. Device rules apply
only when the call origin is `client_mcp` or `client_skill`; they cannot change
approval behavior for internal or server MCP tools with a matching name.
Server-configured global policy remains the only editable-policy-independent
source for global tools. Existing mutation fallback behavior remains intact.

## Sidecar and Frontend Behavior

The frontend continues to call the local sidecar. It does not select or trust
a device ID itself. The sidecar supplies the currently registered ID for all
HITL settings verbs.

The frontend must:

- send `toolOrigin` when setting or clearing a rule;
- use response `deviceId` as the cache boundary;
- treat `globalTools` and `masterEnabled` as read-only;
- render `client_skill` tool rules in the Skills UI if desired;
- discard legacy account-wide cached HITL data after deployment.

The self-contained frontend handoff is written to
`plans/HITL_DEVICE_SCOPING_FE_CHANGELOG.md`.

## Testing

Regression coverage will prove:

- sidecar GET/POST/DELETE always stamp the active device and override stale
  incoming IDs;
- a user with rules on device A receives empty editable arrays on device B;
- repository uniqueness and CRUD include device and origin;
- invalid or foreign devices and invalid origins fail closed;
- writes reject targets missing from the active device catalog;
- deletes still clear stale rules after uninstall;
- response serialization echoes `deviceId` and `toolOrigin`;
- runtime policy is resolved after device validation and only for that device;
- client rules never match internal or server MCP provenance;
- the migration resets legacy rows and installs the required constraints;
- existing HITL interrupt, skill execution, MCP, and client isolation suites
  remain green.

Verification includes focused tests, the full relevant regression matrix,
formatting/lint checks, and migration head validation.

## Out of Scope

- User editing of global/server-hosted tool policy.
- Synchronizing local MCP or skill rules between devices.
- Heuristic migration of ambiguous account-wide rules.
- Moving skill commands out of tool-call policy semantics.
