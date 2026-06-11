# Per-Client Toolset Isolation v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skills become client-owned only (server-skills runtime hard-removed), the server MCP config shrinks to the global-default allowlist (time, tavily, widgets), and `demo.py` becomes a UI for the local sidecar with full dev feature parity.

**Spec:** `plans/client_toolset_isolation.md` (approved 2026-06-11). Predecessor: `plans/client_invocation.md` (v1).

**Architecture:** Three independent workstreams. (A) Delete the server-skills exposure path (`skill_resolver._list_server_skills` → `skills_tool` server branch → `skills_registry`) so `list_resolved_skills` only ever reads the originating device's synced skill catalog. (B) Trim `app/ai/mcp_config.json` to global defaults and lock the policy with a test. (C) Point `demo.py` at the sidecar (`client_backend`, port 8100), authenticating with the sidecar's local session token; the proxy stamps device identity (v1 Layer 1), so the demo never handles tools/devices itself.

**Tech stack:** FastAPI server (`app/`), FastAPI sidecar (`client_backend/`), Streamlit (`demo.py`), LangChain `@tool`, pytest.

**Environment:** Run tests with `.conda\python.exe -m pytest` (Python 3.14, pytest 9.0.2 — the `.venv` runtime env has no pytest). Always exclude `tests/client_backend/test_live_server_integration.py` (needs a live server on :8000). Run commands from the repo root.

**Key research facts (verified 2026-06-11, trust these):**

- Skills are NOT checkpointed anywhere. `app/ai/skills_snapshot.py` is a demo-UI helper module (list/detail/reload of repo skills for `demo.py`'s settings panel), not graph-state snapshotting. The spec's NFR-2 ("old checkpoints keep working") is satisfied automatically because skill summaries are rebuilt fresh from `list_resolved_skills()` on every turn. Task 7 amends the spec text accordingly.
- `get_skills_generation()` (`app/ai/skills_registry.py:279-281`) has **zero callers** — no cache invalidation depends on the server registry.
- The sidecar already mirrors the server's API envelopes: `client_backend/api/skills.py` returns `{"success", "message", "data": {...}}` with the same `data` shapes demo.py already consumes; `client_backend/api/mcp.py` mirrors the server's `/mcp/*` paths.
- Sidecar auth contract (`client_backend/api/auth.py`): `POST /auth/login` proxies upstream login and returns `data.localSessionToken` (plus `accessToken`, `userId`, `deviceId`). All sidecar routes (`require_local_session`) require the **local session token** as Bearer — not the upstream access token.
- Sidecar default port: `8100` (`client_backend/core/config.py` `backend_port`, env `CLIENT_BACKEND_PORT`). Local skills roots: env `CLIENT_SKILLS_ROOTS` (comma-separated paths).
- Endpoint coverage audit (done 2026-06-11): every endpoint `demo.py` calls exists on the sidecar natively or via `client_backend/api/proxy.py`. No new proxy routes are needed.

---

### Task 1: Rewrite skill-visibility tests to encode the new contract (failing first)

**Files:**
- Rewrite: `tests/test_skills_tool.py` (replace entire file)

The current file asserts server+client merging — the old contract. Replace it with tests that encode FR-1/FR-2/FR-7. These MUST fail before Task 2's fix.

- [ ] **Step 1: Replace `tests/test_skills_tool.py` with exactly this content**

```python
"""Skill visibility is scoped to the originating client device (no server skills)."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai import skill_resolver, skills_tool


def _client_session(skills, *, user_id="user-1", session_id="session-1"):
    return SimpleNamespace(
        user_id=user_id,
        session_id=session_id,
        skill_catalog={"skills": skills},
    )


def test_summaries_list_only_the_bound_clients_skills(monkeypatch):
    device_id = str(uuid4())
    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda _device_uuid: _client_session(
            [{"name": "client-skill", "description": "client description", "enabled": True}]
        ),
    )

    summaries = skills_tool.get_available_skill_summaries(user_id="user-1", device_id=device_id)

    assert [(entry["name"], entry["source"]) for entry in summaries] == [
        ("client-skill", "client")
    ]


def test_summaries_are_empty_without_a_device():
    assert skills_tool.get_available_skill_summaries(user_id="user-1", device_id=None) == []


def test_summaries_scoped_to_the_originating_device(monkeypatch):
    device_a, device_b = str(uuid4()), str(uuid4())
    catalogs = {
        device_a: [{"name": "skill-a", "description": "a", "enabled": True}],
        device_b: [{"name": "skill-b", "description": "b", "enabled": True}],
    }

    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda device_uuid: (
            _client_session(catalogs[str(device_uuid)]) if str(device_uuid) in catalogs else None
        ),
    )

    summaries = skills_tool.get_available_skill_summaries(user_id="user-1", device_id=device_b)

    assert [entry["name"] for entry in summaries] == ["skill-b"]


@pytest.mark.asyncio
async def test_activate_skill_unknown_name_returns_graceful_error(monkeypatch):
    device_id = str(uuid4())
    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda _device_uuid: _client_session(
            [{"name": "client-skill", "description": "d", "enabled": True}]
        ),
    )

    dispatched = []

    async def _record_dispatch(**kwargs):
        dispatched.append(kwargs)
        return {"success": True, "result": ""}

    monkeypatch.setattr(skills_tool.ClientDeviceService, "dispatch_tool_call", _record_dispatch)

    tool = skills_tool.create_activate_skill_tool(user_id="user-1", device_id=device_id)
    # "find-skills" exists in this repo's skills/ folder; it must NOT resolve.
    result = await tool.ainvoke({"skill_name": "find-skills"})

    assert "not found" in result.lower()
    assert "client-skill" in result  # only this session's skills are offered
    assert dispatched == []
```

- [ ] **Step 2: Run and verify all four tests FAIL against current code**

Run: `.conda\python.exe -m pytest tests/test_skills_tool.py -q`
Expected failures (the bug, live): tests 1–3 fail because real repo skills (`browser`, `find-skills`, …) appear with `source="server"`; test 4 fails because `find-skills` resolves to the server registry and returns its content instead of "not found".

If any of them PASSES pre-fix, stop — the test doesn't encode the bug; fix the test before proceeding.

- [ ] **Step 3: Commit the failing tests**

```powershell
git add tests/test_skills_tool.py
git commit -m "test: encode client-only skill visibility contract (red)"
```

---

### Task 2: Remove the server-skills path from resolver, tool, and prompt

**Files:**
- Modify: `app/ai/skill_resolver.py` (delete lines 13, 63-81; edit line 164 and module docstring)
- Modify: `app/ai/skills_tool.py` (delete line 34 import and lines 141-153 server branch; edit factory docstring)
- Modify: `app/ai/agents/base_agent.py:1288-1302` (prompt wording + default source)
- Modify: `tests/test_custom_agents_tools.py:215-234` (mock client catalog instead of `_list_server_skills`)

- [ ] **Step 1: `app/ai/skill_resolver.py` — delete the server source**

Delete the import (line 13):

```python
from .skills_registry import get_server_skills_registry
```

Delete the whole `_list_server_skills()` function (lines 63-81). Change `list_resolved_skills()` (line 164) from:

```python
    combined = _list_server_skills()
    combined.extend(_list_client_skills(user_id=user_id, device_id=device_id))
```

to:

```python
    combined = _list_client_skills(user_id=user_id, device_id=device_id)
```

Change the module docstring (line 1) to:

```python
"""Client skill resolution for prompt binding and activation (skills are client-owned)."""
```

- [ ] **Step 2: `app/ai/skills_tool.py` — delete the server activation branch**

Delete the import (line 34):

```python
from .skills_registry import get_server_skills_registry
```

Delete this entire block inside `activate_skill` (lines 141-153):

```python
        if resolved_skill.source == "server":
            try:
                skill = get_server_skills_registry().get_skill(resolved_skill.name)
            except KeyError:
                return f"Error: skill '{resolved_skill.name}' is no longer available on the server."

            if not skill.enabled:
                return (
                    f"Error: skill '{resolved_skill.name}' exists but is currently disabled. "
                    "Only enabled skills can be activated."
                )

            return f"── Skill: {skill.name} ──\n\n{skill.content}\n\n── End Skill: {skill.name} ──"
```

In `create_activate_skill_tool`'s docstring, delete the sentence "Server-local skills continue to load directly from the canonical backend registry."

- [ ] **Step 3: `app/ai/agents/base_agent.py` — client-only prompt wording**

In `_build_skills_suffix` (~line 1288), replace the intro text:

```python
    parts = [
        "\n\n── Available Skills ──",
        "You have access to the following skills, provided by the client "
        "device connected to this chat session. Each skill contains "
        "detailed instructions that you can load on demand using the "
        "`activate_skill` tool. When a user's request seems related to "
        "a skill below, call `activate_skill` with the skill name to "
        "load its full instructions before responding.\n",
    ]
```

and at ~line 1302 change the default source:

```python
        source = str(skill.get("source") or "client").strip().lower()
```

- [ ] **Step 4: Fix `tests/test_custom_agents_tools.py` (it monkeypatches the deleted `_list_server_skills`)**

Replace `test_custom_agent_cannot_activate_unselected_skill` (lines 215-234) with:

```python
@pytest.mark.asyncio
async def test_custom_agent_cannot_activate_unselected_skill(monkeypatch):
    device_id = str(uuid4())
    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda _device_uuid: SimpleNamespace(
            user_id="user-1",
            session_id="session-1",
            skill_catalog={
                "skills": [
                    {"name": "data-analysis", "description": "d", "enabled": True},
                    {"name": "browser", "description": "b", "enabled": True},
                ]
            },
        ),
    )

    allowed_refs = [{"source": "client", "lookup_name": "data-analysis", "name": "data-analysis"}]
    tool = create_activate_skill_tool(
        user_id="user-1", device_id=device_id, allowed_skill_refs=allowed_refs
    )

    # "browser" is real but not in the agent's allowlist -> rejected as unavailable.
    result = await tool.ainvoke({"skill_name": "browser"})
    assert "not found" in result.lower()
    assert "data-analysis" in result
```

Add `from uuid import uuid4` and `from types import SimpleNamespace` to the file's imports if not already present; remove the now-unused `ResolvedSkill` import if nothing else in the file uses it.

- [ ] **Step 5: Run the focused suites**

Run: `.conda\python.exe -m pytest tests/test_skills_tool.py tests/test_custom_agents_tools.py tests/test_custom_agents_service.py tests/test_demo_custom_agents.py -q`
Expected: PASS. If `test_custom_agents_service.py` / `test_demo_custom_agents.py` fail on skill refs with `source="server"`: those refs now match nothing (by design — `_ref_matches_skill` requires exact source match). Update those specific assertions to use `source="client"` refs with a mocked client catalog, mirroring Step 4's pattern. Do not weaken the source-match logic itself.

- [ ] **Step 6: Commit**

```powershell
git add app/ai/skill_resolver.py app/ai/skills_tool.py app/ai/agents/base_agent.py tests/test_custom_agents_tools.py
git commit -m "feat: skills are client-owned only; remove server-skill exposure (FR-1/FR-2)"
```

---

### Task 3: Demo skills panel uses the sidecar API; delete `skills_snapshot`

**Files:**
- Modify: `demo.py:23-27` (delete import block), `demo.py:3273-3285` (rewrite three helpers)
- Delete: `app/ai/skills_snapshot.py`, `tests/test_skills_snapshot.py`
- Modify: `tests/test_skills_architecture.py` (assert the NEW architecture)

- [ ] **Step 1: Update `tests/test_skills_architecture.py` first (red)**

Replace `test_demo_uses_repo_skills_snapshot_instead_of_deleted_skills_api` (line ~18) with:

```python
def test_demo_manages_skills_through_the_sidecar_api():
    demo_source = _read_demo_source()
    assert "skills_snapshot" not in demo_source
    assert 'make_api_request("GET", "/skills")' in demo_source
    assert 'make_api_request("POST", "/skills/reload")' in demo_source
```

Reuse whatever mechanism the old test used to read `demo.py` source (check the top of the file; if it inlines the read, inline the same here as `demo_source = Path("demo.py").read_text(encoding="utf-8")` with the repo-root-relative path handling the file already uses).

Run: `.conda\python.exe -m pytest tests/test_skills_architecture.py -q`
Expected: FAIL (demo still imports skills_snapshot).

- [ ] **Step 2: Rewrite the demo helpers (`demo.py:3273-3285`)**

```python
def get_skills_list() -> dict[str, Any] | None:
    """Fetch the local sidecar's skills."""
    response = make_api_request("GET", "/skills")
    return response.get("data") if response else None


def get_skill_detail(name: str) -> dict[str, Any] | None:
    """Fetch full detail for one local sidecar skill."""
    response = make_api_request("GET", f"/skills/{name}")
    return response.get("data") if response else None


def reload_skills() -> dict[str, Any] | None:
    """Rescan the local sidecar's skill roots."""
    response = make_api_request("POST", "/skills/reload")
    return response.get("data") if response else None
```

The sidecar returns the **same `data` shapes** the panel already renders (`{"skills": [...], "totalCount", "enabledCount"}`; detail adds `content`; reload returns `{"message"}`) — see `client_backend/api/skills.py:30-102`. No panel rendering changes needed.

- [ ] **Step 3: Delete the demo's server import (`demo.py:23-27`)**

Delete:

```python
from app.ai.skills_snapshot import (
    get_repo_skill_detail_for_demo,
    list_repo_skills_for_demo,
    reload_repo_skills_for_demo,
)
```

- [ ] **Step 4: Delete the dead module and its tests**

```powershell
git rm app/ai/skills_snapshot.py tests/test_skills_snapshot.py
```

- [ ] **Step 5: Run**

Run: `.conda\python.exe -m pytest tests/test_skills_architecture.py tests/test_demo_custom_agents.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add demo.py tests/test_skills_architecture.py
git commit -m "feat: demo skills panel manages the sidecar's local skills"
```

---

### Task 4: Delete the server skills registry runtime

**Files:**
- Delete: `app/ai/skills_registry.py`, `app/ai/skills_config.json`, `tests/test_skills_parity.py`
- Modify: `app/main.py` (delete `init_skills()` at lines 84-93 and its call at line ~145)

- [ ] **Step 1: Delete the startup hook in `app/main.py`**

Delete the whole function:

```python
async def init_skills():
    """Pre-scan skills folder at startup."""
    try:
        from app.ai.skills_registry import get_skills_registry

        registry = get_skills_registry()
        skills = registry.get_all_skills()
        logger.info(f"Loaded {len(skills)} skills ({sum(s.enabled for s in skills)} enabled)")
    except Exception as e:
        logger.warning(f"Skills init failed (non-fatal): {e}")
```

and the `await init_skills()` line inside `lifespan()` (~line 145).

- [ ] **Step 2: Delete files**

```powershell
git rm app/ai/skills_registry.py app/ai/skills_config.json tests/test_skills_parity.py
```

`tests/test_skills_parity.py` is deleted wholesale: it compares the server parser against the client parser, and the client parser keeps its own coverage in `tests/client_backend/test_skills_registry.py`. (`shared/skills/front_matter.py` stays — the sidecar uses it.)

- [ ] **Step 3: Verify nothing references the deleted symbols**

Run: `rg -n "skills_registry|skills_config|get_server_skills_registry|skills_snapshot|_list_server_skills" app tests demo.py`
Expected: matches ONLY for `client_backend.services.local_skills_registry` (the client registry — allowed, e.g. `tests/test_client_backend_bundle.py`). Zero matches for the server module. If anything else matches, fix it before proceeding.

- [ ] **Step 4: Boot-import sanity check**

Run: `.conda\python.exe -c "import app.main"` and `.conda\python.exe -c "import demo"`
Expected: both import without ImportError (streamlit bare-mode warnings are fine).

- [ ] **Step 5: Run the skills + agents suites**

Run: `.conda\python.exe -m pytest tests/test_skills_tool.py tests/test_skills_architecture.py tests/test_custom_agents_tools.py tests/client_backend/test_skills_registry.py tests/client_backend/test_skills_api.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add app/main.py
git commit -m "chore: delete server skills registry runtime (FR-6)"
```

---

### Task 5: Server MCP config = global allowlist (time, tavily, widgets)

**Files:**
- Create: `tests/test_mcp_global_allowlist.py`
- Modify: `app/ai/mcp_config.json` (delete `desktop-commander`, `mcp-server-for-revit`, `excel` entries)

- [ ] **Step 1: Write the policy test (red)**

Create `tests/test_mcp_global_allowlist.py`:

```python
"""FR-3: enabled server MCP servers ARE the global-default toolset. Keep it minimal."""

import json
from pathlib import Path

GLOBAL_DEFAULT_SERVERS = {"widgets", "tavily", "time"}
MACHINE_SPECIFIC_SERVERS = ("desktop-commander", "mcp-server-for-revit", "excel")


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "app" / "ai" / "mcp_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def test_enabled_server_mcp_is_exactly_the_global_default_set():
    servers = _load_config()["mcp_servers"]
    enabled = {name for name, spec in servers.items() if spec.get("enabled")}
    assert enabled == GLOBAL_DEFAULT_SERVERS


def test_machine_specific_servers_are_not_in_the_server_config():
    servers = _load_config()["mcp_servers"]
    for name in MACHINE_SPECIFIC_SERVERS:
        assert name not in servers
```

Run: `.conda\python.exe -m pytest tests/test_mcp_global_allowlist.py -q`
Expected: FAIL (desktop-commander and excel are enabled today).

- [ ] **Step 2: Edit `app/ai/mcp_config.json`**

Delete the three whole entries `"desktop-commander"`, `"mcp-server-for-revit"`, `"excel"` (lines 64-97 in the current file). Keep `widgets`, `tavily`, `time` enabled and `calculator`, `form_filler`, `boring` as disabled examples. Mind the trailing comma after the now-last entry.

- [ ] **Step 3: Run + grep for stragglers**

Run: `.conda\python.exe -m pytest tests/test_mcp_global_allowlist.py -q` → PASS.
Run: `rg -n "desktop-commander|desktop_commander|excel-mcp|mcp-server-for-revit" app tests client_backend`
References in client-tool fixtures/tests (e.g. `client__desktop_commander__*` tool names in client-catalog tests) are CORRECT per the product model — leave them. Fix or delete only tests that assume these servers exist in the SERVER config.

- [ ] **Step 4: Commit**

```powershell
git add app/ai/mcp_config.json tests/test_mcp_global_allowlist.py
git commit -m "feat: server MCP config is the global-default allowlist (FR-3)"
```

---

### Task 6: `demo.py` becomes a sidecar UI

**Files:**
- Modify: `demo.py:45` (base URL), `demo.py:3306-3326` (login handler), `demo.py:3342-3375` (signup handler)

- [ ] **Step 1: Point the demo at the sidecar (`demo.py:45`)**

```python
# demo.py is a UI for the local client_backend sidecar (not the canonical server).
API_BASE_URL = os.environ.get("CHATBOT_API_BASE_URL", "http://127.0.0.1:8100")
```

- [ ] **Step 2: Login handler stores the LOCAL session token (`demo.py:3313-3322`)**

Sidecar routes authenticate with the local session token (`require_local_session`), not the upstream access token. Replace the success block of the Sign In form:

```python
                    if auth_response and "data" in auth_response:
                        data = auth_response["data"]
                        session_token = data.get("localSessionToken") or data["accessToken"]
                        st.session_state.auth_token = session_token
                        st.session_state.current_user_id = data["userId"]
                        st.session_state.device_id = data.get("deviceId")
                        st.session_state.current_user_profile = None
                        st.session_state.active_view = "chat"
                        st.session_state.show_login = False
                        st.session_state._ls_op = {
                            "token": session_token,
                            "uid": data["userId"],
                        }
                        st.toast("Welcome back!", icon=":material/check_circle:")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
```

(`localSessionToken` is always present after a successful sidecar login — `client_backend/api/auth.py:176-179`; the `accessToken` fallback only keeps the handler harmless if someone points `CHATBOT_API_BASE_URL` straight at the canonical server for debugging.)

- [ ] **Step 3: Same change in the signup handler (`demo.py:3361-3370`)**

Apply the identical `data = auth_response["data"]` / `session_token` / `device_id` pattern to the signup success block (it currently duplicates the login block verbatim).

- [ ] **Step 4: Static checks**

Run: `rg -n "accessToken" demo.py` → only the two `session_token` fallback lines remain.
Run: `.conda\python.exe -m pytest tests/test_skills_architecture.py tests/test_demo_custom_agents.py -q` → PASS.

Notes for the executor:
- Do NOT add any device handling to requests — the sidecar proxy stamps `device_id` on chat payloads (v1 Layer 1) and `deviceId` query params on custom-agent routes (`client_backend/api/proxy.py:16-23`). `st.session_state.device_id` is set for display/debug only; the existing `_custom_agent_device_query()` helper (demo.py:1171) now sends the real device id, which matches what the proxy would stamp anyway.
- `WIDGET_WS_BASE_URL` stays unchanged: the widget token mint goes through the sidecar proxy, and the returned `ws_url` intentionally points at the canonical server (phase-1 design, `client_backend/api/proxy.py:327-343`).

- [ ] **Step 5: Commit**

```powershell
git add demo.py
git commit -m "feat: demo.py authenticates and chats through the local sidecar (FR-4)"
```

---

### Task 7: Documentation + spec amendment

**Files:**
- Modify: `README.md` (Skills row ~line 55, Skills section ~lines 673-676, dev-run docs)
- Modify: `plans/client_toolset_isolation.md` (NFR-2 snapshot wording)

- [ ] **Step 1: README — Skills table row (~line 55)**

Replace the row's text with:

> **Skills** | Markdown-defined skills with YAML frontmatter, owned by each client device. The sidecar scans `CLIENT_SKILLS_ROOTS`, syncs a per-device catalog to the server, and serves skill content over the runtime bridge ([`client_backend/services/local_skills_registry.py`](client_backend/services/local_skills_registry.py)). The server has no skills of its own.

- [ ] **Step 2: README — Skills architecture section (~lines 673-676)**

Replace the Server/Client bullet list with:

> - **Client (only source of skills)** — [`LocalSkillsRegistry`](client_backend/services/local_skills_registry.py), scanning `CLIENT_SKILLS_ROOTS`; synced per-device to the server and resolved at chat time by [`skill_resolver.py`](app/ai/skill_resolver.py) strictly for the originating device.
> - To serve this repo's `skills/` folder during development, add its absolute path to the local sidecar's `CLIENT_SKILLS_ROOTS`.

- [ ] **Step 3: README — add the global-tools contract + dev run setup**

In the MCP section, add:

> **Global default tools.** Enabled servers in [`app/ai/mcp_config.json`](app/ai/mcp_config.json) are by definition global-default tools, visible to every client (currently `time`, `tavily`, `widgets` — enforced by `tests/test_mcp_global_allowlist.py`). Anything machine-specific (e.g. desktop-commander, excel) belongs in a sidecar's local MCP config (`<profile>/mcp/mcp_config.json`, same `mcpServers` JSON shape), where it becomes a device-scoped `client__` tool.

In the development/run docs (where the demo is described), document the three-process dev setup:

> 1. Canonical server: port 8000 (existing command).
> 2. Sidecar: port 8100 with `CLIENT_SKILLS_ROOTS` pointing at `<repo>/skills` — reuse the start command already documented in the README's Client Runtime Bridge section (verify it there; do not invent a new one).
> 3. Demo UI: `streamlit run demo.py` (talks to the sidecar via `CHATBOT_API_BASE_URL`, default `http://127.0.0.1:8100`).

- [ ] **Step 4: Spec amendment (`plans/client_toolset_isolation.md`)**

Implementation research corrected one spec assumption. In §2.1, replace:

> Snapshot **restore** drops any checkpointed `source="server"` entries (NFR-2).

with:

> Implementation note (2026-06-11): skills are never checkpointed — summaries are rebuilt from `list_resolved_skills()` every turn, so NFR-2 holds with no restore-filtering needed. `skills_snapshot.py` was a demo-UI helper (not graph snapshotting) and is deleted with the demo's switch to the sidecar `/skills` API.

In §3, replace the NFR-2 bullet ("restoring a checkpoint/snapshot containing server-skill entries drops them...") with:

> **NFR-2:** a conversation checkpointed before this change lists zero server skills on its next turn (prompt summaries are rebuilt per turn — covered by the summaries tests).

- [ ] **Step 5: Commit**

```powershell
git add README.md plans/client_toolset_isolation.md
git commit -m "docs: client-owned skills, global-tool contract, sidecar dev setup"
```

---

### Task 8: Full verification

- [ ] **Step 1: Full suite**

Run: `.conda\python.exe -m pytest tests --ignore=tests/client_backend/test_live_server_integration.py -q`
Expected: 0 failures. (Baseline before this work: 1065 passed; the count will shift slightly after deletions/additions.) Fix any failure before proceeding — do not skip tests to get green.

- [ ] **Step 2: Lint touched files**

Run: `.conda\python.exe -m ruff check app/ai/skill_resolver.py app/ai/skills_tool.py app/ai/agents/base_agent.py app/main.py demo.py tests/test_skills_tool.py tests/test_skills_architecture.py tests/test_custom_agents_tools.py tests/test_mcp_global_allowlist.py`
Expected: clean, or only findings already present on HEAD (verify against HEAD if unsure). Fix anything new.

- [ ] **Step 3: Manual smoke (dev parity, FR-5)**

1. Start server (port 8000), sidecar (port 8100, `CLIENT_SKILLS_ROOTS=<repo>\skills`), and `streamlit run demo.py` per the README section written in Task 7.
2. Sign in via the demo → expect chat view. (Verifies local-session auth.)
3. Send a message → SSE stream renders tokens and any subagent progress. (Verifies `/messages/stream` proxy.)
4. Open the skills panel → repo skills listed (served by the sidecar); toggle one off/on; reload. (Verifies `/skills` routes.)
5. Open the MCP panel → the sidecar's local servers listed. (Verifies `/mcp` routes.)
6. Ask the model "what time is it?" → time tool (global) works; ask it to use a repo skill → `activate_skill` round-trips through the runtime bridge.

Record the outcome of each step in an implementation log appended to this file — do not mark this task done on partial smoke results.

- [ ] **Step 4: Final commit + push**

```powershell
git add -A
git commit -m "chore: per-client toolset isolation v2 complete"
git push
```

---

## Execution notes

- **Order matters within Tasks 1→4** (tests red → fix → demo decouple → delete runtime); Tasks 5 and 6 are independent of each other and of 1–4, but run Task 8 only after everything else.
- **Keep each commit green** except Task 1 (intentional red, committed as such to prove the bug was encoded — same discipline as v1).
- **Do not touch** `client_backend/services/local_skills_registry.py`, `shared/skills/front_matter.py`, or any `client_runtime_tools`/`deferred_tool_state` code — v1 already hardened those; this plan must not regress them.
- If a step's expected result doesn't match reality, STOP and re-verify the research facts at the top before improvising.
