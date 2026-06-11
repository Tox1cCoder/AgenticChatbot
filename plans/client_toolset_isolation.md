# Per-Client Toolset Isolation v2 — Skills, Global Tools, Streamlit Client

**Branch:** `Thai-Postgre-FastAPI`
**Status:** Spec approved by product owner 2026-06-11; implementation plan pending
**Date:** 2026-06-11
**Predecessor:** `plans/client_invocation.md` (v1 — device-scoped client *runtime tool* isolation, completed 2026-06-11)

---

## 1. Specification

### 1.1 Problem Statement

v1 closed cross-client leakage for client runtime (MCP) tools, but a sidecar
client on a different computer can still see and invoke **skills defined in
this repo's `skills/` directory**. Root cause: skills have a second,
device-unscoped exposure path — `_list_server_skills()`
(`app/ai/skill_resolver.py:63-81`, called unconditionally at line 164) injects
every enabled server skill into every chat turn's system prompt, and
`activate_skill` loads them from the global server registry
(`app/ai/skills_tool.py:142-153`). Client-synced skills are correctly
device-scoped; server skills are global by construction.

Separately, the server's own MCP config (`app/ai/mcp_config.json`) runs
machine-specific tools (`desktop-commander`, `excel`) globally — every client
can invoke tools that execute on the server box.

Finally, the streamlit demo (`demo.py`) talks directly to the FastAPI server
with no client identity at all, so it cannot have a toolset of its own.

### 1.2 Product Model (decided 2026-06-11)

The product is a Claude-Desktop-style chat backend serving many clients
simultaneously (smaller scale):

- **A client = one sidecar installation** (`client_backend`) with its own
  persisted per-installation device identity (v1 D4 fix). Any UI pointing at
  that sidecar shares that client's toolset. A second client on the same
  machine = a second sidecar installation (own config dir, own port).
- **Every chat turn binds exactly three tool groups, nothing else:**

| Group | Source | Scope |
|---|---|---|
| Global defaults | Server `mcp_config.json` enabled servers: **time, tavily, widgets** | All clients, all users |
| Client MCP tools | Originating sidecar's `LocalMCPManager` catalog | Strictly the turn's `device_id` (v1, done) |
| Client skills | Originating sidecar's synced skill catalog | Strictly the turn's `device_id` (**this spec**) |

- Internal tools (`tool_search`, `activate_skill`, `hand_off`, etc.) remain
  global infrastructure.
- **The repo's production frontend is developed by the FE team in another
  repo.** `demo.py` is this repo's development client UI and must keep full
  feature parity for development use.

### 1.3 Requirements

- **FR-1 Skills are client-owned only.** A chat turn lists and resolves only
  the skills of the client the message originated from. A turn with no
  connected client lists zero skills. The server process owns no skills.
- **FR-2 No cross-client knowledge.** The model cannot discover (via prompt
  summaries, `activate_skill`, `tool_search`, snapshots, or checkpoints) that
  another client's skills or tools exist. Error wording mentions only "this
  chat session's client" — never other devices (carried over from v1).
- **FR-3 Global toolset = enabled server MCP servers, kept minimal.** The
  global-default set is `time`, `tavily`, `widgets`. Machine-specific servers
  (`desktop-commander`, `mcp-server-for-revit`, `excel`) are removed from the
  server config; they return as device-scoped `client__` tools via each
  machine's sidecar MCP config.
- **FR-4 Streamlit demo is a sidecar UI.** `demo.py` targets the local
  `client_backend` instead of the server directly, authenticates through the
  sidecar's local-session flow, and never handles `device_id` itself (the
  proxy stamps it — v1 Layer 1). Its toolset = global tools + this machine's
  sidecar tools/skills.
- **FR-5 Demo feature parity.** Everything `demo.py` does today keeps working
  through the sidecar: auth, conversations, SSE streaming incl. subagent
  progress, HITL interrupt/resume, widgets, custom agents, documents,
  feedback, task plans. Missing proxy routes are added (additive only — must
  not break the FE team's existing sidecar contract).
- **FR-6 Legacy cleanup.** Remove the server-skills runtime and every
  fallback/compatibility shim made dead by this change. No flag-gated dead
  code paths; no orphaned config files; tests and README updated to the new
  contract. Cleanup is scoped to the areas this design touches.
- **FR-7 Graceful unavailability.** `activate_skill` on a name that doesn't
  resolve in this turn's client catalog returns a graceful tool error result
  (same family as v1's `_ERR_*` constants); the turn completes normally.
- **NFR-1** No regression to single-client flows, HITL interrupt/resume,
  server-side global MCP tools, or the FE team's sidecar/server API contract.
- **NFR-2** Old conversations checkpointed with server-skill state keep
  working: restore drops server-sourced entries; the turn proceeds.

### 1.4 Out of Scope

- Cross-client tool/skill sharing or per-user global skill packs.
- Cryptographic origin proof (unchanged from v1 — future hardening).
- Changes to the FE team's frontend repo.
- A new "global skills" feature (explicitly rejected: clients own all skills).

---

## 2. Design

### 2.1 Skills: hard removal of the server path

- Delete `_list_server_skills()` and its call in `list_resolved_skills()`
  (`app/ai/skill_resolver.py:63-81`, `:164`). Resolution becomes the client
  catalog of the turn's `(user_id, device_id)` only; no device → zero skills.
- Delete the `source="server"` branch of `activate_skill`
  (`app/ai/skills_tool.py:142-153`). Unresolvable skill → graceful tool error
  per FR-7.
- Remove the server skills runtime: `app/ai/skills_registry.py`,
  `skills_config.json`, the startup hook in `app/main.py:87-89`, and the
  server path in `app/ai/skills_snapshot.py`. Implementation note (2026-06-11): skills are never checkpointed —
  summaries are rebuilt from `list_resolved_skills()` every turn, so NFR-2
  holds with no restore-filtering needed. `skills_snapshot.py` was a demo-UI
  helper (not graph snapshotting) and is deleted with the demo's switch to
  the sidecar `/skills` API.
- `shared/skills/front_matter.py` stays — the sidecar's
  `LocalSkillsRegistry` uses it.
- The repo's `skills/` directory stays as content. On this machine it is
  served by adding its path to the local sidecar's `skills_roots`; it syncs
  through the normal client skill-catalog like any other client's skills.
- Custom agents whose `allowed_skill_refs` reference former server skills
  degrade gracefully: unresolvable refs are not listed (existing behavior).

### 2.2 Server MCP config = the global allowlist

No new enforcement code — binding already treats enabled server MCP as global
and client MCP as device-scoped. The policy change is config + migration:

- `app/ai/mcp_config.json`: delete the `desktop-commander`,
  `mcp-server-for-revit`, and `excel` entries. `time`, `tavily`, `widgets`
  stay enabled; repo demo servers (`calculator`, `form_filler`, `boring`)
  stay as disabled examples.
- Add `desktop-commander` and `excel` to the local sidecar's MCP config on
  this machine (deployment step, documented).
- README documents the contract: *enabled server MCP servers are by
  definition global-default tools; anything machine-specific belongs in a
  sidecar's local MCP config.*

### 2.3 Streamlit demo → sidecar UI

- `demo.py` base URL becomes the local `client_backend` address
  (env-configurable, e.g. `CLIENT_BACKEND_URL`); the direct-to-server mode is
  removed (FR-6 — no fallback paths).
- Auth switches to the sidecar's local-session flow
  (`client_backend/api/auth.py` + upstream auth). The demo performs no
  device handling — the proxy stamps `device_id` (v1 Layer 1).
- Endpoint-coverage audit: enumerate every endpoint `demo.py` calls; confirm
  the sidecar serves or proxies each (`/messages/stream` SSE already
  proxied); add missing proxy routes additively.
- Widget websockets keep connecting to the canonical server (existing
  phase-1 proxy design — `client_backend/api/proxy.py:327-343`); only the
  token mint goes through the proxy.
- Dev setup (README): run server + local sidecar + `demo.py`.

### 2.4 Legacy cleanup inventory (FR-6)

Known-dead after 2.1–2.3 (implementation must sweep for more in touched
areas):

- `app/ai/skills_registry.py`, `skills_config.json`,
  `get_server_skills_registry()` alias and all imports of it
  (`skill_resolver.py`, `skills_tool.py`, `skills_snapshot.py`,
  `app/main.py`).
- Server/client skills *parity* test (`tests/test_skills_parity.py`) —
  obsolete with a single registry; delete or repurpose for the client
  registry alone.
- Server-skill assertions in `tests/test_skills_tool.py`,
  `tests/test_skills_snapshot.py`, `tests/test_custom_agents_tools.py` —
  rewrite to the client-only contract.
- README references to a "shared registry spanning server, client, device"
  (README.md:55, 675-676) and any architecture docs describing server skills.
- `demo.py` direct-to-server code paths replaced by the sidecar flow.

### 2.5 Data Flow After Fix

```
UI (demo.py or FE app) → client_backend (sidecar)
        └─ local session auth; proxy stamps device_id := own registered id
     → server /messages/stream etc.
        └─ validate device_id (v1 Layer 2); state overwritten every turn
     → per-turn binding
        ├─ global server MCP (time, tavily, widgets)
        ├─ client__ tools for this device only (v1)
        └─ skill summaries for this device's catalog only (NEW)
     → activate_skill / tool call
        └─ resolves in this device's catalog or returns graceful error
     → dispatch to exactly this device's queue (v1 Layer 3)
```

---

## 3. Testing Strategy

Behavioral, not structural (same discipline as v1; bug-encoding tests must
fail against pre-fix code first):

- **Skill isolation:** devices A and B connected, same user. A turn from B
  lists only B's skills in the prompt suffix; `activate_skill` on a skill
  that exists only on A returns the graceful error and dispatches nothing.
- **No-client turn:** `device_id=None` → zero skills listed, zero client
  tools, global tools (time/tavily/widgets) still bound.
- **Global set:** desktop-commander/excel absent from the server tool
  catalog; time/tavily/widgets bound for every client.
- **NFR-2:** a conversation checkpointed before this change lists zero
  server skills on its next turn (prompt summaries are rebuilt per turn —
  covered by the summaries tests).
- **Custom agents:** `allowed_skill_refs` containing a former server skill →
  ref silently unlisted; no crash.
- **Demo parity:** endpoint-coverage check of `demo.py`'s calls vs. sidecar
  routes; manual/scripted smoke of auth + stream + HITL through the sidecar.
- **Existing suites** updated to the new contract and green:
  `test_skills_tool`, `test_skills_snapshot`, `test_skills_parity` (or
  removed), `test_custom_agents_tools`, `test_client_invocation_isolation`,
  `tests/client_backend/`.

## 4. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Old checkpoints resurrect server skills via snapshot restore. | Not an issue — skills are never checkpointed; summaries are rebuilt per turn (NFR-2). |
| Proxy gaps break a demo feature when switching base URL. | Endpoint-coverage audit is its own task before the switch; routes added additively. |
| Proxy additions disturb the FE team's contract. | Additive-only rule (FR-5); no existing route signatures change. |
| Removing desktop-commander/excel surprises other clients that relied on them globally. | Intended per product model; migration documented — each machine's sidecar adds what it needs. |
| Repo skills "disappear" for development until the sidecar serves them. | Deployment step documented in README dev setup; part of the demo-parity smoke test. |
