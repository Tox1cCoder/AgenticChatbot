# Device-Scoped HITL Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate editable HITL rules for local MCP and skill tools by user, device, and tool origin while keeping global policy read-only.

**Architecture:** The sidecar authoritatively stamps its registered device on every HITL settings request. The server validates device ownership and active catalog membership, persists rules under a device/origin composite identity, returns only that device's rules, and loads the same scoped policy after chat device validation. Runtime matching selects rules by explicit client provenance; global policy remains configuration-owned.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, dependency-injector, pytest, Streamlit.

---

## File Map

- `app/models/tool_approval_setting.py`: device/origin columns and constraints.
- `app/alembic/versions/z3a4b5c6d7e8_device_scope_tool_approval_settings.py`: one-time reset and schema migration.
- `app/repositories/tool_approval_setting.py`: device/origin-scoped CRUD and policy grouping.
- `app/schemas/hitl.py`: required tool origin and response device identity.
- `app/services/hitl_settings_service.py`: ownership, catalog validation, response assembly.
- `app/api/hitl.py`: required device context and origin-aware delete contract.
- `client_backend/api/proxy.py`: authoritative device stamping for all settings verbs.
- `app/services/message_service.py`: validate device before loading scoped policy.
- `app/ai/hitl_config.py`: origin-aware client policy matching.
- `demo.py`: send origin from MCP and skill controls.
- `plans/SKILLS_MCP_HITL_FE_CONTRACT.md`: replace account-wide semantics with the deployed contract.
- `plans/HITL_DEVICE_SCOPING_FE_CHANGELOG.md`: final FE handoff.

### Task 1: Lock the persistence contract with failing tests

**Files:**
- Modify: `tests/test_tool_approval_setting_model.py`
- Modify: `tests/test_tool_approval_setting_repository.py`
- Create: `tests/test_hitl_device_scope_migration.py`
- Modify: `app/models/tool_approval_setting.py`
- Modify: `app/repositories/tool_approval_setting.py`
- Create: `app/alembic/versions/z3a4b5c6d7e8_device_scope_tool_approval_settings.py`

- [ ] **Step 1: Write model and repository tests for the five-part identity**

Add assertions for `device_id`, `tool_origin`, the device foreign key, origin check, and unique tuple. Change repository examples to call:

```python
repo.set(user_id, device_id, "client_skill", "tool", qualified_id, True)
repo.list_by_device(user_id, device_id)
repo.build_policy(user_id, device_id)
```

Assert policy grouping is origin-aware:

```python
assert policy == {
    "client_mcp": {"servers": {"desktop_commander": True}, "tools": {}},
    "client_skill": {"servers": {}, "tools": {skill_qid: False}},
}
```

- [ ] **Step 2: Write migration tests**

Assert revision `z3a4b5c6d7e8` follows `y2z3a4b5c6d7`, deletes legacy rows before non-null alteration, adds both columns/constraints/indexes, and restores the legacy schema on downgrade without restoring deleted data.

- [ ] **Step 3: Run the tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tool_approval_setting_model.py tests/test_tool_approval_setting_repository.py tests/test_hitl_device_scope_migration.py -q
```

Expected: failures for missing columns, old method signatures, and absent migration.

- [ ] **Step 4: Implement the minimal model, repository, and migration**

Use constants:

```python
VALID_SCOPE_TYPES = {"server", "tool"}
VALID_TOOL_ORIGINS = {"client_mcp", "client_skill"}
```

All selectors must include `user_id`, `device_id`, and `tool_origin`. Add `ondelete="CASCADE"` to the device foreign key. Migration order is: drop old unique constraint, delete legacy rows, add nullable columns, add FK/check/indexes/new unique, alter both new columns non-null.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the Step 3 command. Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add app/models/tool_approval_setting.py app/repositories/tool_approval_setting.py app/alembic/versions/z3a4b5c6d7e8_device_scope_tool_approval_settings.py tests/test_tool_approval_setting_model.py tests/test_tool_approval_setting_repository.py tests/test_hitl_device_scope_migration.py
git commit -m "feat: scope HITL settings persistence by device"
```

### Task 2: Make the settings API device- and origin-aware

**Files:**
- Modify: `tests/test_hitl_api.py`
- Modify: `app/schemas/hitl.py`
- Modify: `app/services/hitl_settings_service.py`
- Modify: `app/api/hitl.py`

- [ ] **Step 1: Write cross-device API regression tests**

Create two `ClientDevice` records for one user and active `DeviceSessionRecord` catalogs. POST a machine-A skill rule with:

```json
{"items":[{"scopeType":"tool","scopeValue":"skill::kobo-library::run_skill_command","toolOrigin":"client_skill","requireApproval":false}]}
```

Assert GET for A returns it with `deviceId` and `toolOrigin`, while GET for B returns empty `servers` and `tools`. Cover missing device (422), foreign/unknown device (404), inactive catalog write (409), invalid origin (422), unavailable target (409), and deleting an uninstalled target.

- [ ] **Step 2: Run API tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hitl_api.py -q
```

Expected: current account-wide API returns A's rule for B and does not require origin/device.

- [ ] **Step 3: Implement schemas and service validation**

Add:

```python
ToolOrigin = Literal["client_mcp", "client_skill"]

class HitlScopeRule(_CamelModel):
    scope_type: Literal["server", "tool"]
    scope_value: str
    tool_origin: ToolOrigin
    require_approval: bool

class HitlSettingsResponse(_CamelModel):
    device_id: UUID
    master_enabled: bool
    global_tools: list[str]
    servers: list[HitlScopeRuleState]
    tools: list[HitlScopeRuleState]
```

Resolve UUID syntax first, query `ClientDevice` by `(id, user_id)`, and use the active session catalog for POST target validation. Match tool scope by `(origin, qualified_id)` and server scope by `(origin, server_name)`. Raise the exact approved error codes.

- [ ] **Step 4: Require device context on all API verbs**

Accept camel and snake aliases, pass device/origin through service methods, and return scoped data after POST/DELETE.

- [ ] **Step 5: Run API tests and verify GREEN**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add app/schemas/hitl.py app/services/hitl_settings_service.py app/api/hitl.py tests/test_hitl_api.py
git commit -m "feat: isolate HITL settings API by device"
```

### Task 3: Stamp device identity at the sidecar boundary

**Files:**
- Modify: `tests/client_backend/test_hitl_proxy.py`
- Modify: `client_backend/api/proxy.py`

- [ ] **Step 1: Replace the no-stamping regression test**

Mock `get_runtime_bridge().get_registered_device_id()` as `device-b`. Assert GET, POST, and DELETE pass `params_override` containing exactly one local `deviceId`; stale `deviceId=device-a` and `device_id=device-a` are both removed. Assert the interrupt lifecycle route remains unstamped.

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/client_backend/test_hitl_proxy.py -q
```

Expected: HITL settings proxy still passes no override.

- [ ] **Step 3: Implement authoritative query stamping**

Normalize both aliases:

```python
def _params_with_active_device(request: Request):
    params = [(k, v) for k, v in request.query_params.multi_items()
              if k not in {"deviceId", "device_id"}]
    device_id = get_runtime_bridge().get_registered_device_id()
    if device_id:
        params.append(("deviceId", device_id))
    return params
```

Pass it as `params_override` for the settings proxy only.

- [ ] **Step 4: Run and verify GREEN**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add client_backend/api/proxy.py tests/client_backend/test_hitl_proxy.py
git commit -m "fix: bind HITL settings proxy to local device"
```

### Task 4: Scope runtime enforcement by validated device and origin

**Files:**
- Modify: `tests/test_hitl_policy.py`
- Modify: `tests/test_hitl_turn_policy_injection.py`
- Modify: `app/ai/hitl_config.py`
- Modify: `app/services/message_service.py`

- [ ] **Step 1: Write runtime isolation tests**

Assert `_prepare_workflow_execution` validates device before calling `_resolve_hitl_policy(user_id, device_id)`. Assert device A and B repository calls differ. Add identity tests proving a `client_skill` rule affects only a `client_skill` identity and never same-named `client_mcp`, `server_mcp`, or `internal` identities.

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hitl_policy.py tests/test_hitl_turn_policy_injection.py -q
```

Expected: old top-level policy matches without origin and policy lookup lacks device.

- [ ] **Step 3: Implement origin-aware runtime policy**

Policy shape:

```python
{
    "master_enabled": is_hitl_enabled(),
    "client_rules": repository.build_policy(user_id, device_id) if device_id else {
        "client_mcp": {"servers": {}, "tools": {}},
        "client_skill": {"servers": {}, "tools": {}},
    },
    "global_tools": list(get_tools_requiring_approval()),
}
```

Only consult `client_rules[identity.origin]` for `client_mcp` and `client_skill`; global name configuration and mutation fallback remain unchanged. Validate device before policy resolution in message service.

- [ ] **Step 4: Run and verify GREEN**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add app/ai/hitl_config.py app/services/message_service.py tests/test_hitl_policy.py tests/test_hitl_turn_policy_injection.py
git commit -m "fix: enforce HITL policy for the active device only"
```

### Task 5: Update Streamlit controls and management contract

**Files:**
- Modify: `tests/test_hitl_demo_panel.py`
- Modify: `demo.py`
- Modify: `plans/SKILLS_MCP_HITL_FE_CONTRACT.md`
- Modify: `plans/HITL_DEVICE_SCOPING_FE_CHANGELOG.md`

- [ ] **Step 1: Write static UI contract tests**

Require `set_hitl_setting(tool_origin, ...)` and `clear_hitl_setting(tool_origin, ...)`. Assert skills use `client_skill`, MCP server/tool controls use `client_mcp`, response caching includes `deviceId`, and no global-policy edit control is rendered.

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hitl_demo_panel.py -q
```

Expected: helpers omit origin.

- [ ] **Step 3: Implement minimal UI request changes**

Send `toolOrigin` in POST items and DELETE query parameters. Keep skill command rules in `tools`; index by `(toolOrigin, scopeValue)` to avoid collisions. Do not add global editing controls.

- [ ] **Step 4: Update the long-form contract and changelog**

Remove statements that rules are account-wide, document device-only visibility, required origin, response `deviceId`, errors, one-time reset, and FE cache boundary. Ensure examples match the live schemas exactly.

- [ ] **Step 5: Run and verify GREEN**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add demo.py tests/test_hitl_demo_panel.py plans/SKILLS_MCP_HITL_FE_CONTRACT.md plans/HITL_DEVICE_SCOPING_FE_CHANGELOG.md
git commit -m "docs: publish device-scoped HITL frontend contract"
```

### Task 6: Regression and migration verification

**Files:**
- Modify only if a regression exposes a scoped defect in the files above.

- [ ] **Step 1: Run the focused isolation matrix**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hitl_api.py tests/test_hitl_policy.py tests/test_hitl_turn_policy_injection.py tests/test_hitl_config.py tests/test_tool_approval_setting_model.py tests/test_tool_approval_setting_repository.py tests/test_hitl_device_scope_migration.py tests/test_client_tool_isolation.py tests/test_client_invocation_isolation.py tests/test_skills_tool.py tests/client_backend/test_hitl_proxy.py -q
```

Expected: all pass.

- [ ] **Step 2: Run formatting and lint checks**

```powershell
.\.venv\Scripts\python.exe -m ruff check app/api/hitl.py app/schemas/hitl.py app/services/hitl_settings_service.py app/repositories/tool_approval_setting.py app/models/tool_approval_setting.py app/services/message_service.py app/ai/hitl_config.py client_backend/api/proxy.py tests/test_hitl_api.py tests/test_hitl_policy.py tests/test_hitl_turn_policy_injection.py tests/test_tool_approval_setting_model.py tests/test_tool_approval_setting_repository.py tests/test_hitl_device_scope_migration.py tests/client_backend/test_hitl_proxy.py
.\.venv\Scripts\python.exe -m ruff format --check app/api/hitl.py app/schemas/hitl.py app/services/hitl_settings_service.py app/repositories/tool_approval_setting.py app/models/tool_approval_setting.py app/services/message_service.py app/ai/hitl_config.py client_backend/api/proxy.py tests/test_hitl_api.py tests/test_hitl_policy.py tests/test_hitl_turn_policy_injection.py tests/test_tool_approval_setting_model.py tests/test_tool_approval_setting_repository.py tests/test_hitl_device_scope_migration.py tests/client_backend/test_hitl_proxy.py
```

Expected: exit 0 for both commands.

- [ ] **Step 3: Validate migration topology**

```powershell
.\.venv\Scripts\python.exe -m alembic heads
```

Expected: exactly `z3a4b5c6d7e8 (head)`.

- [ ] **Step 4: Run the broader HITL, skill, and client-backend suites**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hitl_*.py tests/test_skills_*.py tests/test_client_*.py tests/client_backend -q
```

Expected: all pass, with only documented environment skips.

- [ ] **Step 5: Inspect final diff and contract consistency**

```powershell
git diff --check
git status --short
git log --oneline -8
```

Expected: no whitespace errors or uncommitted implementation files; response examples and schema field names match.
