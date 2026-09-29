# Sandbox Frontend Contract

**For:** frontend developers building the Desktop Commander sandbox switch. **Backend as of:** 2026-09-28. This is the device-local portion of [Skills, MCP, and HITL Frontend Contract](SKILLS_MCP_HITL_FE_CONTRACT.md).

## Where to call

Call the **local sidecar** on the current Windows device. These routes are not on the canonical server. Send `Authorization: Bearer <token>` with either the local session token or this sidecar's active upstream access token. Do not send a device ID: the sidecar selects its own installation and profile.

| Method | Path | Body | Success |
|---|---|---|---|
| `GET` | `/mcp/sandbox` | none | `200` |
| `PUT` | `/mcp/sandbox` | `{ "mode": "off" }` or `{ "mode": "workspace" }` | `200` |

Both success responses use `{ "success": true, "message": string, "data": SandboxState, "error": null }`:

```ts
type SandboxState = {
  mode: "off" | "workspace";
  accountReady: boolean;
};
```

`mode` is the effective mode on **this sidecar profile**. The saved mode survives a sidecar restart; if no saved mode exists, `CLIENT_SANDBOX_MODE` supplies the default. `off` runs Desktop Commander as the signed-in Windows user. `workspace` launches it as a restricted local Windows account. The sidecar grants that account write access to configured workspace roots; ordinary Windows permissions still apply elsewhere. If no roots are configured, it uses `<CLIENT_PROFILE_ROOT>/workspace`. Neither the mode nor the managed workspace is synced to the user's other devices. Another Windows user normally has a separate sidecar profile, mode, credentials, and sandbox account.

`accountReady` means this Windows profile has stored sandbox credentials. It does **not** guarantee Desktop Commander started successfully. The operator sets up the account with `python -m client_backend sandbox setup` and can check it with `python -m client_backend sandbox status`. The FE only reads or changes `mode`.

Profiles set up before this change may still use the legacy shared `KaniSandbox` account. Their existing setup continues to run; the operator should run `sandbox setup` once per Windows profile to move each to its own account. The switch does not perform this privileged migration.

## What a toggle does

`PUT` persists the mode, restarts the local MCP manager so Desktop Commander launches under the selected Windows identity, then republishes the device tool and skill catalogs to the canonical server if the runtime bridge is connected. If the server has lost that runtime session, the sidecar reconnects and republishes the catalogs before returning success. Tool availability is tied to the current device's runtime session.

After `PUT`, refetch `GET /mcp/servers` (and `/mcp/tools` if shown). The `desktop-commander` server row reports `enabled`, `running`, `toolCount`, and `error`:

- `mode: "workspace"`, `accountReady: false`: show setup guidance. Desktop Commander stays off even if its MCP configuration is enabled.
- `accountReady: true`, `running: false`: show the server row's `error`; runtime installation, permissions, or launch can still fail.
- `running: true`: the live `toolCount` and tool list reflect the new mode.

The other MCP servers are not sandboxed by this switch. Chat requests through the sidecar use its own registered device; the FE must not copy a device ID from another client. Two devices signed into the same login can have different modes and tool catalogs.

## Errors and recovery

The sidecar returns FastAPI errors as `{ "detail": string | object }`, not the success envelope. Handle the HTTP status first.

| Status | Cause | FE action |
|---|---|---|
| `401` | Missing, expired, or mismatched local bearer session | Restore or renew this sidecar's login. |
| `422` | `mode` is missing or is not `off` or `workspace` | Keep the previous selection and report invalid input. |
| `503` | A lost runtime session could not reconnect during catalog publication | Refetch `/mcp/sandbox`, `/runtime/status`, and `/mcp/servers`; show that device tools are unavailable until reconnect. The mode may already be saved. |

Do not treat a failed `PUT` as a rollback: read `GET /mcp/sandbox` again to display the persisted mode. A successful `PUT` confirms catalog publication when connected; `GET /mcp/servers` is the source of truth for whether Desktop Commander actually runs.
