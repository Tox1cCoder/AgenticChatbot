# Skills, MCP Servers, and HITL Settings — Frontend Contract

Companion to [AI_SDK_FE_CONTRACT.md](AI_SDK_FE_CONTRACT.md), which owns the
chat stream, interrupts-during-chat, and message history. This document owns
the management surfaces: skills, MCP servers, and HITL approval settings.

## The Two Backends

The FE talks to exactly two HTTP surfaces. Knowing which one owns a resource
answers most questions in this document.

| Surface | Base | Owns |
|---|---|---|
| **Device sidecar** (local `client_backend`) | the FE's local sidecar URL | Skills (install, readiness, secrets, toggle), device-local MCP servers, auth/local session, chat proxying |
| **Canonical server** (via sidecar proxy or direct) | server URL | Conversations, chat stream, HITL settings, server-side MCP servers, device registry |

Rules that follow from this split:

- **Skills are device-local.** A skill exists only on the device whose sidecar
  installed or scanned it. There is **no** canonical endpoint that lists skill
  bundles across devices. Two machines each see only their own `GET /skills`.
- The sidecar **proxies** selected canonical routes (`/hitl/settings`,
  `/hitl/interrupts/{id}`, `/providers`, `/model-config`, `/users/{id}`) so the
  FE can use one base URL. Proxied responses are byte-identical to the
  canonical response; query parameters pass through unchanged.
- The canonical server knows each device's **catalog** (tool/skill names and
  descriptions synced by that device's sidecar) but never bundle contents,
  secrets, or file paths.

## Response Envelopes

Both backends use the same envelope:

```json
{ "success": true, "code": "OPTIONAL_ERROR_CODE", "message": "...", "data": {}, "error": null }
```

| Rule | Detail |
|---|---|
| `success` / `message` / `data` / `error` | Always present. |
| `code` | Canonical server only, and only when a stable machine-readable code exists (errors). Absent otherwise — never `null` on the wire from the canonical server; the sidecar never sends it. |
| Casing | `data` payload keys are camelCase on FE-facing endpoints (`masterEnabled`, `scopeValue`, `totalCount`, `sourceHash`). Snake-case request equivalents are accepted where documented. |
| Sidecar install/setup errors | HTTP 4xx with `detail: {"code": "<SKILL_*>", "message": "..."}` (FastAPI detail object), not the envelope. Codes are listed in [docs/skill-runtime.md](../docs/skill-runtime.md#errors). |

Endpoints that are **not** FE-facing (sidecar↔server protocol; do not build UI
on them): `POST /client-devices/register`, `POST /client-devices/heartbeat`,
`PUT /client-devices/{id}/tool-catalog`, `PUT /client-devices/{id}/skill-catalog`,
`WS /device-runtime/...`.

## Skills (Device Sidecar)

All routes require the sidecar local session (`Authorization: Bearer
<localSessionToken>` from the sidecar login response).

| Endpoint | Purpose |
|---|---|
| `GET /skills` | List this device's skills with readiness. |
| `GET /skills/{name}` | One skill, including its `content` (SKILL.md body). |
| `PATCH /skills/{name}/toggle?enabled=` | Enable/disable a skill on this device. |
| `POST /skills/reload` | Rescan configured skill roots. |
| `POST /skills/install/preview` | Inspect a bundle without copying/executing. Returns `sourceHash`. |
| `POST /skills/install` | Install a bundle (`sourcePath`, optional `expectedSourceHash`, `approveSetup`). |
| `POST /skills/{name}/setup` | Prepare the Python runtime. Requires `approveSetup: true` + exact `expectedSourceHash`. |
| `POST /skills/uninstall` | Remove an installed bundle. |
| `GET /skills/installed` | Installed bundles (device-local paths included). |
| `POST /skills/{name}/secrets` | Bind a secret. Body `{"name": "ENV_NAME", "value": "..."}`. |
| `GET /skills/{name}/secrets` | Configured secret **names** only. |
| `DELETE /skills/{name}/secrets/{secretName}` | Remove one binding. |

### Skill summary shape

`GET /skills` → `data`:

```json
{
  "skills": [
    {
      "name": "cli-anything-google-calendar",
      "description": "…",
      "enabled": true,
      "folderPath": "C:\\…\\skills\\installed",
      "sourceHash": "abc…64 hex chars",
      "commandCapable": true,
      "runtimeStatus": "ready",
      "setupStatus": "not_applicable"
    }
  ],
  "totalCount": 1,
  "enabledCount": 1
}
```

| Field | Rules |
|---|---|
| `runtimeStatus` | `ready`, `not_ready`, or `instruction_only`. Only `ready` skills publish a command tool. |
| `setupStatus` | `not_applicable`, `ready`, `setup_required`, `failed`, or `stale`. Drive the "Set up" button from `setup_required` / `stale`. |
| `commandCapable` | Convenience: `runtimeStatus === "ready"`. |
| `sourceHash` | Echo as `expectedSourceHash` on `POST /skills/{name}/setup`. |
| `folderPath` | Device-local path. Display only; never send to the canonical server. |

`GET /skills/{name}` adds `content` (the SKILL.md body).

### Secrets

- Values are **write-only**: stored encrypted, injected into the skill's child
  process as environment variables at execution, redacted from command output,
  and never returned by any endpoint. `GET` returns names only:
  `{"secrets": [{"name": "GOOGLE_CALENDAR_ACCESS_TOKEN", "configured": true}]}`.
- The store strips surrounding whitespace on save and rejects blank values
  with HTTP 400 (`secret value must not be empty or whitespace-only`).
- Secret names must be valid environment-variable identifiers
  (`[A-Za-z_][A-Za-z0-9_]*`); anything else is HTTP 400.
- Secrets are **per skill, per device, per user profile**. Configuring a
  secret on one machine does nothing on another machine.
- UX guidance: never render a secret value after submission; treat an expired
  upstream credential (e.g. an OAuth 401 in a skill's command error) as
  "reconfigure this secret", not as a runtime failure.

### How skills reach the chat

FE does not call skill commands directly. The flow is: the sidecar syncs a
per-device catalog → the chat turn (with the request's `deviceId`) binds that
device's tools → the model calls `activate_skill`, then the published command
tool (`client__skill_<name>__run_skill_command`). Approval is gated through
the HITL interrupt flow in the AI SDK contract. A chat turn without a
`deviceId` binds **no** client tools or skills.

## MCP Servers

Two independent inventories with the same route shapes:

| Inventory | Base | Visibility |
|---|---|---|
| Server-side MCP | canonical `/mcp/*` | Shared by every client of the account/server. |
| Device-local MCP | sidecar `/mcp/*` | This device only; reaches chat via the device catalog like skills. |

Routes (both): `GET /servers`, `GET /servers/{name}`, `POST /servers`,
`POST /servers/from-url`, `DELETE /servers/{name}`,
`PATCH /servers/{name}/toggle?enabled=`, `GET /tools?serverName=`,
`GET /tools/{toolName}`, `POST /tools/{toolName}/execute`.

`GET /mcp/servers` → `data`:

```json
{
  "servers": [
    { "name": "desktop_commander", "transport": "stdio", "enabled": true,
      "description": "…", "toolCount": 12, "config": {} }
  ],
  "totalCount": 1,
  "enabledCount": 1
}
```

`GET /mcp/tools` → `data`:

```json
{
  "tools": [
    { "name": "start_process", "description": "…", "argsSchema": {},
      "serverName": "desktop_commander" }
  ],
  "totalCount": 12,
  "serversCount": 1
}
```

In chat, client-side tools appear to the model as
`client__<server>__<tool>`; skills' command tools use the reserved server name
`skill_<skill-name-with-underscores>`. Those `skill_*` server names also
appear as HITL scope values (below).

## HITL Settings (Canonical, proxied by the sidecar)

Per-user approval policy for tool execution. **Rules are account-wide**: they
follow the user across devices. Whether the rule's *target* exists is
per-device — that asymmetry is exposed explicitly (see `available`).

| Endpoint | Purpose |
|---|---|
| `GET /hitl/settings?deviceId=` | Read policy. Optional `deviceId` adds per-rule availability. |
| `POST /hitl/settings` | Upsert rules: `{"items": [{"scopeType", "scopeValue", "requireApproval"}]}`. |
| `DELETE /hitl/settings?scopeType=&scopeValue=` | Remove one rule. Snake-case params also accepted. |
| `GET /hitl/interrupts/{interruptId}` | Interrupt lifecycle — documented in the AI SDK contract. |

`GET /hitl/settings?deviceId=<uuid>` → `data`:

```json
{
  "masterEnabled": true,
  "globalTools": ["dangerous_tool"],
  "servers": [
    { "scopeType": "server", "scopeValue": "skill_cli_anything_google_calendar",
      "requireApproval": true, "available": false }
  ],
  "tools": [
    { "scopeType": "tool",
      "scopeValue": "skill::cli-anything-google-calendar::run_skill_command",
      "requireApproval": true, "available": false }
  ]
}
```

| Field | Rules |
|---|---|
| `masterEnabled` | Server-level HITL kill switch (read-only here). When false, no approvals are enforced regardless of rules. |
| `globalTools` | Server-configured always-require-approval tool names (read-only here). |
| `scopeType` | `server` (rule covers every tool of that server name) or `tool` (one tool; overrides its server rule). |
| `scopeValue` | Server scope: an MCP server name or a skill's `skill_<name>` catalog server name. Tool scope: a qualified id (`server::tool`, `skill::<name>::run_skill_command`) or bare tool name. |
| `requireApproval` | The explicit decision for that scope. |
| `available` | **Present only when the request passed `deviceId`.** `true` when the scope target currently exists in that device's live catalog or the server-side MCP registry. |

FE behavior:

- **Always pass your own `deviceId`** (from the sidecar login response) when
  rendering a settings screen. Render `available: false` rules as
  "Not on this device" — do not hide them (they are real account policy and
  take effect wherever the target exists), and do not treat them as errors.
- A rule referencing a skill installed on another of the user's machines is
  the expected way you'll encounter `available: false`. It does **not** mean
  the skill leaked to this device: the skill cannot be listed, activated, or
  executed here.
- Without `deviceId`, the response is exactly the legacy shape (no `available`
  key anywhere). Presence-check the field; never expect `null`.
- `POST` echoes the full updated settings (same shape as `GET`, without
  `available`). Send only changed rules; upsert is per `(scopeType, scopeValue)`.
- Approval-at-runtime (the `data-interrupt` flow, decisions, resume) is owned
  by [AI_SDK_FE_CONTRACT.md](AI_SDK_FE_CONTRACT.md#human-in-the-loop).

## Cross-Device Semantics (read this before filing "leak" bugs)

| Data | Scope | Where another device sees it |
|---|---|---|
| Skill bundles, readiness, secrets | Device | Never. |
| Device tool/skill catalog | Device (synced to server per device) | Only its names/descriptions, and only in that device's own chat turns. |
| HITL rules | **User (account-wide)** | `GET /hitl/settings` everywhere, flagged `available: false` where the target is absent. |
| Custom agents and their saved tool/skill refs | **User (account-wide)** | Agent definitions everywhere; client tool refs bound to another device are skipped at run time and reported via `custom_agent_warnings`. |
| Conversations, messages | User | Everywhere. |

## Error Codes Quick Reference

| Code | Endpoint family | Meaning |
|---|---|---|
| `HITL_SCOPE_PARAMS_REQUIRED` | DELETE /hitl/settings | Missing `scopeType`/`scopeValue`. |
| `INTERRUPT_NOT_FOUND` | GET /hitl/interrupts/{id} | Unknown or foreign interrupt. |
| `SKILL_INSTALL_INVALID` | sidecar /skills | Bad bundle source/structure/hash (HTTP 400; 404 on uninstall). |
| `SKILL_INSTALL_CONFLICT` | sidecar /skills/install | Bundle already installed (HTTP 409). |
| `SKILL_SETUP_REQUIRED` / `SKILL_SETUP_FAILED` / `SKILL_RUNTIME_STALE` | sidecar /skills | Setup lifecycle; re-preview to obtain a fresh `sourceHash`. |
