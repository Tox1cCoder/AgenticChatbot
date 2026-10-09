# Sandbox Frontend Contract

**For:** frontend developers building the Desktop Commander sandbox switch. **Backend as of:** 2026-10-06. This is the device-local portion of [Skills, MCP, and HITL Frontend Contract](SKILLS_MCP_HITL_FE_CONTRACT.md).

## Where to call

Call the **local sidecar** on the current Windows device (default `http://127.0.0.1:8100`). These routes are not on the canonical server. Every route is also mounted under `/api` (`/api/mcp/sandbox`). Do not send a device ID: the sidecar selects its own installation and profile.

| Method | Path | Body | Success |
|---|---|---|---|
| `GET` | `/mcp/sandbox` | none | `200` |
| `PUT` | `/mcp/sandbox` | `{ "mode": "off" }` or `{ "mode": "workspace" }` | `200` |

### Request requirements

Every request passes two layers of checks. The sidecar checks the transport (Host, Origin, launch token) before any route runs, and the route then checks the session.

1. **Host** must be `localhost`, `127.0.0.1`, `[::1]`, or the configured `CLIENT_BACKEND_HOST`. Call the sidecar by one of those names, never through a custom hostname.
2. **Caller identity**, one of:
   - **Browser code:** the page's `Origin` must be in `CLIENT_ALLOWED_ORIGINS`. The defaults are `http://localhost:3000`, `http://127.0.0.1:3000`, `http://[::1]:3000`, `http://localhost:8501`, `http://127.0.0.1:8501`. A UI served from any other origin is refused until the operator adds that origin. On `PUT`, a disallowed `Origin` is refused even when the launch token is present.
   - **Non-browser code** (desktop shell, Node, Python): send `X-Kani-Client: <launch token>`. The sidecar writes a fresh token to `<CLIENT_PROFILE_ROOT>/.client_launch_token` on every start, readable only by the signed-in Windows user. After a sidecar restart the old token stops working, so re-read the file once on `401 "Invalid launch token"` and retry.
3. **Session:** `Authorization: Bearer <token>`, either the local session token or this sidecar's active upstream access token.

Both success responses use `{ "success": true, "message": string, "data": SandboxState, "error": null }`:

```ts
type SandboxState = {
  mode: "off" | "workspace";
  accountReady: boolean;
};
```

`mode` is the effective mode on **this sidecar profile**. The saved mode survives a sidecar restart; if no saved mode exists, `CLIENT_SANDBOX_MODE` supplies the default. `off` runs Desktop Commander as the signed-in Windows user. `workspace` launches it as a restricted local Windows account. The sidecar grants that account write access to configured workspace roots; ordinary Windows permissions still apply elsewhere. If no roots are configured, it uses `<CLIENT_PROFILE_ROOT>/workspace`. Neither the mode nor the managed workspace is synced to the user's other devices. Another Windows user normally has a separate sidecar profile, mode, credentials, and sandbox account.

`accountReady` means this Windows profile has stored sandbox credentials. It does **not** guarantee Desktop Commander started successfully. The operator sets up the account with `python -m client_backend sandbox setup` and can check it with `python -m client_backend sandbox status`. The FE only reads or changes `mode`. The FE can still switch to `workspace` while `accountReady` is `false`; the `PUT` succeeds, but Desktop Commander stays off until setup runs.

Profiles set up before this change may still use the legacy shared `KaniSandbox` account. Their existing setup continues to run; the operator should run `sandbox setup` once per Windows profile to move each to its own account. The switch does not perform this privileged migration.

## What a toggle does

`PUT` persists the mode, restarts the local MCP manager so Desktop Commander launches under the selected Windows identity, then republishes the device tool and skill catalogs to the canonical server if the runtime bridge is connected. If the server has lost that runtime session, the sidecar reconnects and republishes the catalogs before returning success. Tool availability is tied to the current device's runtime session.

Side effects the UI must account for:

- **Every local MCP server restarts**, not only Desktop Commander. Any in-flight local tool call on this device is interrupted. Send `PUT` only when the selection actually changes; re-sending the current mode still restarts everything.
- **`PUT` can be slow.** The first switch to `workspace` may install the pinned Desktop Commander runtime, and granting a large workspace folder updates permissions on every file inside it. Expect seconds to minutes, not milliseconds. Do not apply a short client timeout; show a pending state and disable the switch until the response arrives.
- **Send one `PUT` at a time.** The sidecar does not serialize overlapping toggles. Queue or drop a second click while one is pending.

After `PUT`, refetch `GET /mcp/servers` (and `/mcp/tools` if shown). The `desktop-commander` server row reports `enabled`, `running`, `toolCount`, and `error`:

- `mode: "workspace"`, `accountReady: false`: show setup guidance. Desktop Commander stays off even if its MCP configuration is enabled.
- `accountReady: true`, `running: false`: show the server row's `error` verbatim; it names the cause and the operator command to run. Runtime installation, permissions, or launch can still fail, and a configured workspace root that contains a `.git` folder or a Python environment is refused.
- `running: true`: the live `toolCount` and tool list reflect the new mode.

The other MCP servers are not sandboxed by this switch. Chat requests through the sidecar use its own registered device; the FE must not copy a device ID from another client. Two devices signed into the same login can have different modes and tool catalogs.

## Errors and recovery

The sidecar returns errors as JSON `{ "detail": string | object[] }` (except `500`), not the success envelope. Handle the HTTP status first, then use `detail` to pick between causes that share a status.

| Status | `detail` | Cause | FE action |
|---|---|---|---|
| `400` | `"Host is not allowed"` | Request addressed to a non-loopback hostname | Fix the sidecar base URL. Not recoverable at runtime. |
| `403` | `"Origin is not allowed"` | `PUT` from a page whose origin is not in `CLIENT_ALLOWED_ORIGINS` | Configuration error; the operator must add the origin. |
| `401` | `"Missing launch token"` | Non-browser caller without `X-Kani-Client`, or a browser origin not in the allowlist | Send the launch token, or serve the UI from an allowed origin. |
| `401` | `"Invalid launch token"` | The sidecar restarted and rotated its token | Re-read the token file once and retry. |
| `401` | any other string | Missing, expired, or mismatched bearer session | Restore or renew this sidecar's login. |
| `422` | string | `mode` is missing or is not `off` or `workspace` | Keep the previous selection and report invalid input. |
| `422` | array | Body is not a JSON object | Client bug; send `{ "mode": ... }` with `Content-Type: application/json`. |
| `503` | `"The sidecar has no launch token; restart it"` | The sidecar could not write its launch token at startup | Tell the user to restart the sidecar. |
| `503` | `"Device runtime did not reconnect after its session expired"` | A lost runtime session could not reconnect during catalog publication | Refetch `/mcp/sandbox`, `/runtime/status`, and `/mcp/servers`; show that device tools are unavailable until reconnect. The mode is already saved. |
| `500` | none (plain-text body) | Unexpected failure while saving, restarting, or publishing | Same as the reconnect `503`: refetch and display the persisted state. |

Do not treat a failed `PUT` as a rollback: read `GET /mcp/sandbox` again to display the persisted mode, which may be either the old or the new value. A successful `PUT` confirms catalog publication when connected; `GET /mcp/servers` is the source of truth for whether Desktop Commander actually runs.
