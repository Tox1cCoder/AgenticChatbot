# Tool Policy and Repository Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every reviewed tool-policy issue, restore a fully passing repository test suite, and reduce repository-wide Ruff findings from 95 to zero.

**Architecture:** Harden canonical identity and policy validation at construction boundaries, then enforce the client deadline around the complete sidecar operation and make terminal diagnostics consistent. Repair the seven unrelated suite failures at their demonstrated sources before applying behavior-preserving lint cleanup in isolated batches.

**Tech Stack:** Python 3.13, asyncio, Pydantic v2/pydantic-settings, FastAPI, LangChain MCP adapters, pytest/pytest-asyncio, Ruff.

---

### Task 1: Make all MCP policy identity application-owned

**Files:**
- Modify: `app/core/mcp_adapter_utils.py:230-248`
- Modify: `app/ai/tool_execution_policy.py:120-157`
- Test: `tests/test_mcp_adapter_utils.py`
- Test: `tests/test_tool_execution_policy.py`

- [ ] **Step 1: Write failing identity regressions**

Add tests proving that a server MCP tool cannot preserve a forged source name and that unknown origins fail closed:

```python
def test_clone_mcp_tool_overwrites_remote_source_tool_name():
    tool = SimpleNamespace(
        name="start_process",
        args_schema={"type": "object", "properties": {}},
        metadata={"source_tool_name": "trusted_read"},
    )
    cloned = clone_mcp_tool(tool, server_name="desktop")
    assert cloned.metadata["source_tool_name"] == cloned.name


def test_unknown_runtime_tool_origin_is_rejected():
    tool = SimpleNamespace(name="read", metadata={"tool_origin": "server"})
    with pytest.raises(ToolExecutionPolicyValidationError, match="tool_origin"):
        resolve_tool_execution_policy(
            tool,
            exposed_tool_name="read",
            invocation_kind="native_async",
        )
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_mcp_adapter_utils.py tests/test_tool_execution_policy.py -k "source_tool_name or unknown_runtime_tool_origin" -q
```

Expected: the forged source name remains and the unknown origin resolves instead of raising.

- [ ] **Step 3: Implement application-owned source identity and origin validation**

In `clone_mcp_tool()`, assign all canonical fields together:

```python
metadata["tool_origin"] = "server_mcp"
metadata["server_name"] = server_name
metadata["source_tool_name"] = cloned_tool.name
metadata["qualified_tool_id"] = f"{server_name}::{cloned_tool.name}"
```

In `resolve_tool_identity()`, reject any non-empty origin outside `_KNOWN_TOOL_ORIGINS`:

```python
if tool_origin not in _KNOWN_TOOL_ORIGINS:
    raise ToolExecutionPolicyValidationError(
        f"Unknown tool_origin {tool_origin!r}; expected one of {sorted(_KNOWN_TOOL_ORIGINS)}"
    )
```

- [ ] **Step 4: Run the focused identity suites and verify GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_mcp_adapter_utils.py tests/test_tool_execution_policy.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the identity fix**

```powershell
git add app/core/mcp_adapter_utils.py app/ai/tool_execution_policy.py tests/test_mcp_adapter_utils.py tests/test_tool_execution_policy.py
git commit -m "fix: harden tool policy identity ownership"
```

### Task 2: Strictly validate all policy inputs and authority

**Files:**
- Modify: `app/core/config.py:83-159,844-883,1429-1490`
- Modify: `app/ai/tool_execution_policy.py:80-100,266-278,380-446`
- Modify: `app/ai/planning_subagents.py:816-829`
- Test: `tests/test_tool_execution_policy.py`
- Test: `tests/test_planning_subagents.py`

- [ ] **Step 1: Write failing validation regressions**

Add focused tests for finite numbers, internal metadata bounds, duplicate selectors, global ordering, and config-only outer disable:

```python
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_policy_override_rejects_non_finite_timeout(value):
    with pytest.raises(ValidationError):
        ToolExecutionPolicyOverride(
            match={"tool_origin": "internal"},
            timeout_seconds=value,
        )


def test_internal_metadata_cannot_exceed_five_attempts():
    tool = SimpleNamespace(
        name="read",
        metadata={
            "application_execution_policy": {
                "max_attempts": 999,
                "retry_safe": True,
            }
        },
    )
    with pytest.raises(ToolExecutionPolicyValidationError, match="max_attempts"):
        resolve_tool_execution_policy(
            tool,
            exposed_tool_name="read",
            invocation_kind="native_async",
        )


def test_settings_reject_duplicate_policy_selectors():
    with pytest.raises(ValidationError, match="duplicate"):
        Settings(
            _env_file=None,
            secret_key="test-secret",
            tool_execution_policies={
                "a": {"match": {"tool_origin": "internal"}},
                "b": {"match": {"tool_origin": "internal"}},
            },
        )


def test_config_cannot_disable_dispatch_outer_timeout(monkeypatch):
    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "config-disable": ToolExecutionPolicyOverride(
                match={
                    "tool_origin": "internal",
                    "qualified_tool_id": "internal::dispatch_subagents",
                },
                disable_outer_timeout=True,
            )
        },
    )
    tool = SimpleNamespace(
        name="dispatch_subagents",
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::dispatch_subagents",
        },
    )
    with pytest.raises(ToolExecutionPolicyValidationError, match="trusted application"):
        resolve_tool_execution_policy(
            tool,
            exposed_tool_name="dispatch_subagents",
            invocation_kind="native_async",
        )
```

- [ ] **Step 2: Run validation tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_policy.py tests/test_planning_subagents.py -k "non_finite or five_attempts or duplicate_policy or config_cannot" -q
```

Expected: the current models accept non-finite values, raw internal metadata, duplicate selectors, or config-only disablement.

- [ ] **Step 3: Add strict policy models and Settings checks**

Use finite numeric fields in `ToolExecutionPolicyOverride`:

```python
timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
hard_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
total_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
max_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
```

Add an internal metadata model with the same constraints and `extra="forbid"`, validate `application_execution_policy` through `model_validate()`, and translate `ValidationError` into `ToolExecutionPolicyValidationError`.

Extend `Settings._cross_field_checks()` to:

```python
if self.tool_execution_max_interactive_timeout_seconds <= self.tool_execution_cancellation_grace_seconds:
    raise ValueError("tool execution maximum must exceed cancellation grace")
if self.tool_execution_client_execution_grace_seconds <= self.tool_execution_client_response_grace_seconds:
    raise ValueError("client execution grace must exceed client response grace")

selectors: dict[str, str] = {}
for key, override in self.tool_execution_policies.items():
    selector = override.match.model_dump_json(exclude_none=True)
    if selector in selectors:
        raise ValueError(
            f"duplicate tool execution policy selectors: {selectors[selector]!r}, {key!r}"
        )
    selectors[selector] = key
    if override.disable_outer_timeout:
        raise ValueError("deployment policy cannot disable the outer timeout")
```

Require `internal_trusted` as well as the allowlisted identity before honoring `disable_outer_timeout`.

- [ ] **Step 4: Run policy/configuration suites and verify GREEN**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_policy.py tests/test_planning_subagents.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit strict policy validation**

```powershell
git add app/core/config.py app/ai/tool_execution_policy.py app/ai/planning_subagents.py tests/test_tool_execution_policy.py tests/test_planning_subagents.py
git commit -m "fix: validate tool policy inputs at boundaries"
```

### Task 3: Enforce the full client execution deadline

**Files:**
- Modify: `client_backend/services/runtime_bridge.py:249-269,434-446,550-605`
- Modify: `client_backend/services/local_mcp_manager.py:527-598`
- Test: `tests/client_backend/test_runtime_bridge.py`
- Test: `tests/test_client_invocation_isolation.py`

- [ ] **Step 1: Write failing whole-operation timeout tests**

Add a bridge test whose execution ignores cancellation until released and assert that `_handle_tool_request()` sends a timeout response before release:

```python
@pytest.mark.asyncio
async def test_handle_tool_request_bounds_complete_client_execution(monkeypatch):
    bridge = _bridge()
    release = asyncio.Event()
    sent = []

    async def _hung_execution(_request):
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    async def _capture(payload):
        sent.append(payload)

    monkeypatch.setattr(bridge, "_execute_tool_request", _hung_execution)
    monkeypatch.setattr(bridge, "_send_runtime_message", _capture)
    request = ToolDispatchRequest(
        request_id="timeout-1",
        tool_name="read",
        qualified_tool_id="server::read",
        arguments={},
        timeout_seconds=0.01,
    )

    await asyncio.wait_for(bridge._handle_tool_request(request), timeout=0.1)
    assert sent[0].success is False
    assert sent[0].error_context.code == "TIMEOUT_CLIENT_EXECUTION"
    release.set()
    await asyncio.sleep(0)
```

- [ ] **Step 2: Run the timeout test and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests/client_backend/test_runtime_bridge.py -k "bounds_complete_client_execution" -q
```

Expected: `_handle_tool_request()` remains blocked until the outer test timeout.

- [ ] **Step 3: Add a bounded task runner around the complete operation**

Create one task for `_execute_tool_request()` and wait only for the request deadline:

```python
task = asyncio.create_task(self._execute_tool_request(request))
done, _ = await asyncio.wait({task}, timeout=float(request.timeout_seconds))
if not done:
    task.cancel()
    task.add_done_callback(_consume_task_exception)
    raise ClientExecutionTimeoutError(
        f"Client runtime execution exceeded {request.timeout_seconds}s"
    )
result = task.result()
```

Map `ClientExecutionTimeoutError` explicitly in `_build_runtime_error_context()`:

```python
return RuntimeErrorContext(
    message="Client runtime operation timed out.",
    code="TIMEOUT_CLIENT_EXECUTION",
    detail=detail or None,
)
```

Keep the MCP manager's inner timeout as defense in depth, but treat the bridge deadline as authoritative for initialization, discovery, invocation, and reload.

- [ ] **Step 4: Run runtime deadline suites and verify GREEN**

```powershell
.venv\Scripts\python.exe -m pytest tests/client_backend/test_runtime_bridge.py tests/test_client_invocation_isolation.py tests/test_skills_tool.py -q
```

Expected: all selected tests pass without leaked-task warnings.

- [ ] **Step 5: Commit the client deadline fix**

```powershell
git add client_backend/services/runtime_bridge.py client_backend/services/local_mcp_manager.py tests/client_backend/test_runtime_bridge.py tests/test_client_invocation_isolation.py
git commit -m "fix: bound complete client runtime execution"
```

### Task 4: Make policy errors and reconnect diagnostics consistent

**Files:**
- Modify: `app/ai/tool_execution.py:1262-1390,1434-1501,1706-1802`
- Modify: `app/ai/tool_error_policy.py:205-219`
- Test: `tests/test_tool_execution_recovery.py`
- Test: `tests/test_tool_error_policy.py`

- [ ] **Step 1: Write failing diagnostics and sanitization tests**

Add a reconnect failure test asserting the terminal history record and a policy ambiguity test asserting a fixed model message:

```python
assert artifact["attempt_history"][-1]["error_type"] == "timeout"
assert artifact["attempt_history"][-1]["auto_retry_allowed"] is False
assert "policy-a" not in output["content"]
assert json.loads(output["content"])["error_type"] == "configuration"
```

Remove tests importing or exercising `should_auto_retry_tool()` because the canonical resolver is now the only retry authority.

- [ ] **Step 2: Run diagnostics tests and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_recovery.py tests/test_tool_error_policy.py -k "reconnect_failure or policy_resolution" -q
```

Expected: reconnect history ends on the original session record and policy keys appear in model content.

- [ ] **Step 3: Record terminal reconnect failures and sanitize configuration errors**

When reconnect fails, append one terminal record with the reconnect outcome and `auto_retry_allowed=False` before building payloads. Catch `AmbiguousToolExecutionPolicyError` and `ToolExecutionPolicyValidationError` around policy resolution and return this fixed model payload:

```python
{"status": "error", "error_type": "configuration", "retryable": False,
 "message": "Tool execution is unavailable because its server policy is invalid.",
 "hint": "Use another available tool or report the configuration problem."}
```

Keep the raw exception only in the artifact diagnostic and server log. Delete `should_auto_retry_tool()` and its import/tests.

- [ ] **Step 4: Run execution/error suites and verify GREEN**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_recovery.py tests/test_tool_error_policy.py tests/test_tool_execution_rendering.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit diagnostics cleanup**

```powershell
git add app/ai/tool_execution.py app/ai/tool_error_policy.py tests/test_tool_execution_recovery.py tests/test_tool_error_policy.py
git commit -m "fix: align tool policy failure diagnostics"
```

### Task 5: Resolve the seven full-suite regressions

**Files:**
- Modify: `client_backend/services/server_api.py:618-693`
- Test: `tests/client_backend/test_server_api.py`
- Modify: `tests/test_brave_image_search_config.py:13-20`
- Modify: `tests/test_conversation_compaction_health.py:176-186`
- Modify: `app/ai/mcp_config.json`
- Modify: `tests/test_widget_runtime.py:514-522`

- [ ] **Step 1: Add a failing legacy single-upload contract test**

In `tests/client_backend/test_server_api.py`, use a stubbed `request_response()` and assert:

```python
assert captured["path"] == "/documents/upload"
assert captured["files"]["file"][0] == "report.txt"
assert result["data"]["document"]["id"] == "doc-1"
```

- [ ] **Step 2: Run the seven previously failing tests and verify the baseline failures**

```powershell
.venv\Scripts\python.exe -m pytest tests/client_backend/test_live_server_integration.py::test_live_document_upload_list_get_task_and_delete_flow tests/test_brave_image_search_config.py::test_brave_image_search_defaults tests/test_conversation_compaction_health.py tests/test_mcp_global_allowlist.py tests/test_widget_runtime.py::TestBuildToolArtifactWidgets::test_non_widget_artifact_still_respects_truncation_limit -q
```

Expected: the same seven failures reproduced in the review, plus the new single-upload unit regression.

- [ ] **Step 3: Restore the single-upload endpoint contract**

Change `upload_document_bytes()` to send one multipart field to `/documents/upload` and parse it with `_handle_response()`:

```python
response = await self.request_response(
    "POST",
    "/documents/upload",
    data={"conversation_id": conversation_id},
    files={"file": (filename, content, content_type)},
)
return await self._handle_response(response)
```

- [ ] **Step 4: Isolate environment defaults and use public route introspection**

Construct Brave settings with `_env_file=None`:

```python
return Settings(
    _env_file=None,
    secret_key="test-secret",
    environment="development",
    **overrides,
)
```

Use OpenAPI paths rather than FastAPI's private `app.routes` representation:

```python
paths = set(app.openapi()["paths"])
```

- [ ] **Step 5: Restore the declared global MCP configuration and artifact contract**

Remove `calculator` and `desktop-commander` from `app/ai/mcp_config.json`, leaving exactly `widgets`, `tavily`, `time`, and `brave_image_search` enabled. Update the stale artifact test to assert the current explicit default:

```python
def test_non_widget_artifact_preserves_full_output_by_default(self):
    artifact = build_tool_artifact(..., output_text="x" * 1200, error=None)
    assert len(artifact["output"]) == 1200
```

- [ ] **Step 6: Run focused regression tests and verify GREEN**

```powershell
.venv\Scripts\python.exe -m pytest tests/client_backend/test_server_api.py tests/client_backend/test_document_routes.py tests/client_backend/test_document_upload_proxy_guard.py tests/test_brave_image_search_config.py tests/test_conversation_compaction_health.py tests/test_mcp_global_allowlist.py tests/test_widget_runtime.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit full-suite regression fixes**

```powershell
git add client_backend/services/server_api.py tests/client_backend/test_server_api.py tests/test_brave_image_search_config.py tests/test_conversation_compaction_health.py app/ai/mcp_config.json tests/test_widget_runtime.py
git commit -m "fix: restore repository regression contracts"
```

### Task 6: Resolve Ruff semantic findings

**Files:**
- Modify: `app/ai/mcp_servers/widgets_server.py`
- Modify: `app/services/model_config_service.py`
- Modify: `client_backend/api/auth.py`
- Modify: `client_backend/services/server_api.py`
- Modify: `tests/test_hitl_policy.py`
- Modify: `tests/test_rich_response_contract.py`
- Modify: `tests/test_widgets_api.py`

- [ ] **Step 1: Capture the non-E501 lint baseline**

```powershell
.venv\Scripts\python.exe -m ruff check app client_backend tests --ignore E501
```

Expected: 16 findings across `E402`, `B904`, `B017`, `SIM102`, and `F401`.

- [ ] **Step 2: Apply the safe automatic import fix**

```powershell
.venv\Scripts\python.exe -m ruff check tests/test_hitl_policy.py --select F401 --fix
```

Expected: the unused `pytest` import is removed.

- [ ] **Step 3: Fix exception chaining and collapsible conditions**

Use `raise ServerConnectionError(...) from e` in all three `B904` sites. Combine each nested condition without changing its body:

```python
if invalid_key and fallback_config:
    ...

if target_username and current_username and (
    str(target_username).strip().lower() == str(current_username).strip().lower()
):
    return
```

- [ ] **Step 4: Make test exception assertions specific**

Import `pydantic.ValidationError` and replace the five `pytest.raises(Exception)` contexts in `tests/test_rich_response_contract.py` with `pytest.raises(ValidationError)`.

- [ ] **Step 5: Resolve intentional late imports explicitly**

Move imports above executable module setup where safe. Where import order is required by environment setup, append `# noqa: E402` to the exact intentional imports in `widgets_server.py` and `tests/test_widgets_api.py`.

- [ ] **Step 6: Verify semantic Ruff findings are zero**

```powershell
.venv\Scripts\python.exe -m ruff check app client_backend tests --ignore E501
```

Expected: exit 0 with no findings.

- [ ] **Step 7: Run tests covering semantic edits**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_rich_response_contract.py tests/test_widgets_api.py tests/test_hitl_policy.py tests/test_runtime_model_overrides.py tests/client_backend/test_server_api.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit semantic lint cleanup**

```powershell
git add app/ai/mcp_servers/widgets_server.py app/services/model_config_service.py client_backend/api/auth.py client_backend/services/server_api.py tests/test_hitl_policy.py tests/test_rich_response_contract.py tests/test_widgets_api.py
git commit -m "style: resolve semantic Ruff findings"
```

### Task 7: Resolve all E501 findings without behavior changes

**Files:**
- Modify: every file reported by `ruff check app client_backend tests --select E501`

- [ ] **Step 1: Record the exact E501 file list**

```powershell
.venv\Scripts\python.exe -m ruff check app client_backend tests --select E501 --output-format concise
```

Expected: 79 findings in the reviewed baseline.

- [ ] **Step 2: Wrap Python expressions and strings mechanically**

Use adjacent string literals inside parentheses, multiline calls, and multiline comprehensions. Preserve string contents. The required transformation pattern is:

```python
description=(
    "First unchanged portion of the text "
    "second unchanged portion of the text."
)
```

For SQL and JSON-like test strings, split only at whitespace already present so the resulting runtime value is unchanged:

```python
query = (
    "INSERT INTO table (first, second, third) "
    "SELECT first, second, third FROM source"
)
```

- [ ] **Step 3: Re-run E501 until the diagnostic list is empty**

```powershell
.venv\Scripts\python.exe -m ruff check app client_backend tests --select E501
```

Expected: exit 0 with no findings.

- [ ] **Step 4: Run the entire Ruff ruleset**

```powershell
.venv\Scripts\python.exe -m ruff check app client_backend tests
```

Expected: exit 0 with no findings.

- [ ] **Step 5: Run focused tests for files with executable expression wrapping**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_planning_agent_rubric.py tests/test_tool_approval_setting_repository.py tests/test_demo_meaningful_widgets.py tests/test_demo_stream_rendering.py tests/test_document_processing_service.py tests/test_runtime_model_overrides.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit E501 cleanup**

```powershell
git add app client_backend tests
git commit -m "style: clear repository line-length debt"
```

### Task 8: Align documentation and run final acceptance

**Files:**
- Modify: `docs/operations/tool-execution-policy.md:120-127`
- Modify: `plans/tool-execution-policy.md:1030-1221`
- Modify: `docs/superpowers/plans/2026-07-16-tool-policy-and-repository-remediation.md`

- [ ] **Step 1: Update the outer-timeout operations contract**

State that `internal::dispatch_subagents` requires exact identity, trusted application metadata, and the code-owned allowlist; deployment and remote metadata cannot grant the exception.

- [ ] **Step 2: Run the complete policy/runtime matrix**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_policy.py tests/test_tool_error_policy.py tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py tests/test_client_invocation_isolation.py tests/test_skills_tool.py tests/test_planning_subagents.py tests/test_mcp_adapter_utils.py tests/client_backend/test_runtime_bridge.py -q
```

Expected: all selected tests pass without leaked-task warnings.

- [ ] **Step 3: Run repository-wide Ruff and compilation**

```powershell
.venv\Scripts\python.exe -m ruff check app client_backend tests
.venv\Scripts\python.exe -m compileall -q app client_backend tests
```

Expected: both commands exit 0.

- [ ] **Step 4: Run the full test suite**

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expected: zero failures; environment-dependent tests may skip only through existing skip conditions.

- [ ] **Step 5: Check diff hygiene and status**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; status contains only the intended documentation updates before the final commit.

- [ ] **Step 6: Record fresh verification evidence and commit**

Update the original plan's Task 9 checkbox and progress log with the exact fresh test/lint counts, then commit:

```powershell
git add docs/operations/tool-execution-policy.md plans/tool-execution-policy.md docs/superpowers/plans/2026-07-16-tool-policy-and-repository-remediation.md
git commit -m "docs: record tool policy remediation verification"
```
