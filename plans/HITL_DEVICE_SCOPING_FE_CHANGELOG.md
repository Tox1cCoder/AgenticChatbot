# HITL Device Scoping — FE Changelog

## Summary

HITL settings for local MCP and skill tools are now device-scoped. A rule
created on machine A must not appear or apply on machine B. Global tool policy
remains read-only and server-controlled.

This is a breaking management-contract change. Existing editable HITL rules
are reset once during migration because legacy rows have no reliable device
owner.

## Endpoints

Call these routes through the local sidecar:

| Endpoint | Change |
|---|---|
| `GET /hitl/settings` | Sidecar adds its `deviceId`; response contains only this device's rules. |
| `POST /hitl/settings` | Each item now requires `toolOrigin`. |
| `DELETE /hitl/settings` | Requires `toolOrigin` in addition to scope type/value. |

Do not persist or forward a device ID from another installation. The sidecar
owns device binding and overrides stale IDs.

## Response

```json
{
  "success": true,
  "message": "HITL settings retrieved",
  "data": {
    "deviceId": "device-uuid",
    "masterEnabled": true,
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
  },
  "error": null
}
```

Skill commands intentionally remain in `tools`: HITL approves their executable
tool calls. Use `toolOrigin: "client_skill"` to group them under Skills in the
UI. Local MCP entries use `toolOrigin: "client_mcp"`.

`masterEnabled` and `globalTools` are read-only. Do not render editing controls
for global policy.

## Write Examples

Set a client skill command rule:

```json
{
  "items": [
    {
      "scopeType": "tool",
      "scopeValue": "skill::kobo-library::run_skill_command",
      "toolOrigin": "client_skill",
      "requireApproval": true
    }
  ]
}
```

Clear it:

```http
DELETE /hitl/settings?scopeType=tool&scopeValue=skill%3A%3Akobo-library%3A%3Arun_skill_command&toolOrigin=client_skill
```

## FE Actions

1. Add `toolOrigin` to all HITL set/clear operations.
2. Key HITL query caches by response `deviceId` and clear legacy cached data.
3. Treat empty `servers` and `tools` as the correct state for a blank device.
4. Group `client_skill` entries under Skills and `client_mcp` entries under
   local MCP controls.
5. Never expose controls for `globalTools` or `masterEnabled`.
6. Handle stable 4xx responses for missing/invalid device context, invalid
   origin, inactive device runtime, and a target absent from the local catalog.

## Errors

| Status | Code | FE behavior |
|---|---|---|
| 422 | `HITL_DEVICE_REQUIRED` | Re-establish the local sidecar session. |
| 404 | `HITL_DEVICE_NOT_FOUND` | Re-register or re-authenticate this installation. |
| 409 | `HITL_DEVICE_RUNTIME_UNAVAILABLE` | Wait for the sidecar runtime bridge to reconnect, then retry. |
| 422 | `HITL_TOOL_ORIGIN_INVALID` | Fix the request; send `client_mcp` or `client_skill`. |
| 409 | `HITL_TARGET_UNAVAILABLE` | Refresh the local MCP/skill catalog before retrying. |

## Compatibility Notes

- `device_id`, `scope_type`, `scope_value`, and `tool_origin` snake-case query
  aliases remain accepted where query parameters are used.
- A chat request without a valid sidecar device binds no local tools and loads
  no editable client HITL rules.
- Local skill bundles, secrets, and MCP configuration remain device-only; this
  change fixes approval-policy isolation for those resources.
