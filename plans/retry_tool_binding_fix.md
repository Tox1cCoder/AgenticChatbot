# Retry Tool Binding Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tool execution, retry, and deferred tool binding reliable enough for production agent loops while preserving LLM flexibility and keeping model-facing context compact.

**Architecture:** Centralize retry/error classification in the tool execution layer, expose only compact structured error summaries to the model, and keep full diagnostics in artifacts/logs. Graph, RAG, Planning, and subagent loops consume the same lightweight error signals for loop safety without hardcoding recovery decisions.

**Tech Stack:** Python, LangChain/LangGraph messages and tools, pytest/pytest-asyncio, existing MCP/client runtime bridge, existing deferred `tool_search` state.

---

## Spec

### Problem

The repository already has the right architectural pieces: deferred tool loading, tool execution artifacts, provider retry, planning circuit breakers, and RAG/tool budgets. The weak spots are that tool timeout/retry settings are declared but not enforced in the shared execution path, tool failures are flattened into free-form `Error: ...` strings, non-planning loops rely mostly on broad iteration budgets, graph execution helpers can return before resolving pending tool calls when binding produces an empty map, and prompt wording pushes `tool_search` too aggressively even when a direct bound tool is already available.

This creates production risks:

- A hung tool can stall a turn despite `tool_execution_timeout`.
- Transient transport failures are not consistently retried or summarized.
- If generic retry is added naively, side-effecting tools could be retried unsafely.
- The model may loop on the same failing call until the generic ReAct budget is consumed.
- Empty tool maps can leave unresolved tool-call messages in the graph.
- Prompt/test wording currently treats `tool_search` as the default first step for many real-environment tasks.

### Requirements

- Enforce `settings.tool_execution_timeout` around each individual tool invocation.
- Use `settings.tool_execution_max_retries` only for safe transient retries.
- Keep automatic retries conservative:
  - Always allow the existing MCP session reconnect retry for `ClosedResourceError` / `BrokenResourceError`.
  - Allow generic automatic retries only when the tool is marked retry-safe/idempotent through metadata, or when the tool is already in the existing internal loading-tool set such as `tool_search`.
  - For unknown side-effect tools, return a compact retryable error to the model instead of blindly retrying in code.
- Return compact model-facing error summaries with fields like `status`, `error_type`, `retryable`, `message`, and `hint`.
- Preserve full exception details, attempt count, and diagnostic category in artifacts/logs, not in ToolMessage context.
- Ensure every pending tool call gets a matching ToolMessage, including empty execution-map and missing-tool cases.
- Track repeated same-tool/error/argument failures across graph-level ReAct, RAG, and isolated subagent worker loops.
- Preserve Planning's existing consecutive-error breaker while feeding it compact artifact status from shared tool execution.
- Force a final no-tools synthesis after repeated tool errors, rather than ending on raw tool logs or continuing indefinitely.
- Update shared prompt guidance so directly bound tools are preferred when clearly suitable, while `tool_search` remains the discovery path for missing, ambiguous, named-integration, or unavailable tools.
- Keep prompt changes small. Do not inject tool inventories, long policies, or large action enums into the system prompt.
- Do not hardcode recovery behavior for individual tool names beyond existing tool classes such as `TOOL_LOADING_TOOLS`, pinned/deferred allowlists, and tool metadata.

### Repository Validation Notes

These plan items were checked against the current code before implementation:

- Confirmed: `tool_execution_timeout` and `tool_execution_max_retries` are declared in `app/core/config.py`, but `execute_tool_calls()` currently invokes tools directly and only has a special MCP session reconnect retry.
- Confirmed: missing tools are handled inside `execute_tool_calls()`, but `_tool_node()` and `_execute_agent_tool_calls()` can return early on an empty `tool_map`, leaving the preceding AI tool-call message unresolved.
- Confirmed: RAG already forces a final no-tools pass at its broad iteration limit, and Planning already has a consecutive-error breaker. The missing coverage is repeated same tool/error/argument handling before the broad budget is exhausted, plus isolated worker loop bounds.
- Confirmed: shared prompt and `tool_search` descriptions currently over-emphasize discovery for real-environment actions. The fix should keep runtime-scoped discovery for missing or ambiguous capabilities, but prefer clearly suitable bound tool schemas.
- Corrected from the original draft: implementation snippets must not import or instantiate `AgentGraph`; this codebase exposes `MultiAgentWorkflow`.

### Non-Goals

- Do not replace the LLM's recovery judgment with a rigid next-action state machine.
- Do not add a large `next_action` enum that tells the model exactly what to do after every failure.
- Do not dump stack traces, full exception reprs, or full artifacts into ToolMessages.
- Do not disable deferred tool loading.
- Do not make `tool_search` mandatory for every real-environment request.
- Do not refactor the full graph or agent hierarchy outside the retry/tool-binding surfaces.

### Compact Model Error Contract

ToolMessages should contain a short JSON string when a tool fails:

```json
{
  "status": "error",
  "error_type": "timeout",
  "retryable": true,
  "message": "Tool timed out after 30s.",
  "hint": "Retry only if the operation is likely safe; otherwise adjust inputs, use another available tool, discover a better tool, or ask the user."
}
```

This shape gives the model enough structure to reason dynamically without bloating context or scripting the next move.

Full diagnostics belong in tool artifacts:

```json
{
  "status": "error",
  "tool": "example_tool",
  "error_type": "timeout",
  "retryable": true,
  "attempts": 1,
  "exception_type": "TimeoutError",
  "diagnostic": "Tool timed out after 30s while awaiting ainvoke."
}
```

---

## File Map

- Create: `app/ai/tool_error_policy.py`
  - Owns compact error classification, retry safety checks, and model/artifact error payload creation.
- Modify: `app/ai/tool_execution.py`
  - Applies timeout, conservative retries, compact error summaries, and missing-tool output consistency.
- Modify: `app/ai/graph.py`
  - Removes empty-tool-map early exits in execution helpers, tracks repeated non-planning tool failures, and forces final synthesis consistently across graph and subagent loops.
- Modify: `app/ai/rag_tool_actions.py`
  - Converts special RAG document action errors into the same compact model-facing shape.
- Modify: `app/ai/prompts.py`
  - Rebalances prompt guidance toward bound-tool-first behavior without removing deferred discovery.
- Modify: `app/ai/tool_search_tool.py`
  - Aligns `tool_search` tool descriptions with the bound-tool-first policy.
- Modify: `app/core/config.py`
  - Adds a small generic repeated tool-error limit setting and validates it with existing positive integer config checks.
- Modify: `README.md`
  - Makes the documented retry/timeout behavior match the implemented behavior.
- Create/modify tests:
  - `tests/test_tool_error_policy.py`
  - `tests/test_tool_execution_recovery.py`
  - `tests/test_tool_execution_rendering.py`
  - `tests/test_graph_tool_budget.py`
  - `tests/test_rag_tool_loop_finalization.py`
  - `tests/test_graph_planning_subagents.py`
  - `tests/test_tool_search_prompt_guidance.py`

---

## Task 1: Add Compact Tool Error Policy

**Files:**
- Create: `app/ai/tool_error_policy.py`
- Test: `tests/test_tool_error_policy.py`

- [x] **Step 1: Write failing tests for compact error classification**

Create `tests/test_tool_error_policy.py`:

```python
from __future__ import annotations

from anyio import BrokenResourceError, ClosedResourceError

from app.ai.tool_error_policy import (
    ToolErrorKind,
    build_tool_error_payloads,
    classify_tool_error,
    should_auto_retry_tool,
)


class _Tool:
    name = "example_tool"
    metadata = {}


class _RetrySafeTool:
    name = "safe_reader"
    metadata = {"retry_safe": True}


class _StringMetadataTool:
    name = "unsafe_string_flag"
    metadata = {"retry_safe": "true"}


class _ToolSearchTool:
    name = "tool_search"
    metadata = {}


def test_timeout_error_is_compact_and_retryable():
    summary = classify_tool_error(
        TimeoutError("raw provider timeout with verbose internals"),
        tool_name="example_tool",
        timeout_seconds=30,
        attempts=1,
    )

    assert summary.error_type == ToolErrorKind.TIMEOUT.value
    assert summary.retryable is True
    assert "30s" in summary.message
    assert "raw provider" not in summary.message


def test_argument_error_is_not_retryable_by_code():
    summary = classify_tool_error(
        TypeError("unexpected keyword argument 'pathh'"),
        tool_name="read_file",
        timeout_seconds=30,
        attempts=1,
    )

    assert summary.error_type == ToolErrorKind.ARGUMENT.value
    assert summary.retryable is False
    assert "argument" in summary.hint.lower()


def test_session_errors_are_retryable_transport_failures():
    for exc in (ClosedResourceError(), BrokenResourceError()):
        summary = classify_tool_error(
            exc,
            tool_name="mcp_tool",
            timeout_seconds=30,
            attempts=1,
        )
        assert summary.error_type == ToolErrorKind.SESSION.value
        assert summary.retryable is True


def test_model_payload_stays_small_and_structured():
    summary = classify_tool_error(
        PermissionError("permission denied for /secret/token"),
        tool_name="read_file",
        timeout_seconds=30,
        attempts=1,
    )
    model_content, artifact_detail = build_tool_error_payloads(
        summary,
        tool_name="read_file",
        exception=PermissionError("permission denied for /secret/token"),
    )

    assert '"status":"error"' in model_content
    assert '"error_type":"permission"' in model_content
    assert len(model_content) < 500
    assert artifact_detail["exception_type"] == "PermissionError"
    assert artifact_detail["attempts"] == 1


def test_auto_retry_requires_retryable_error_and_safe_tool():
    retryable_summary = classify_tool_error(
        ConnectionError("connection reset"),
        tool_name="safe_reader",
        timeout_seconds=30,
        attempts=1,
    )
    argument_summary = classify_tool_error(
        ValueError("bad date"),
        tool_name="safe_reader",
        timeout_seconds=30,
        attempts=1,
    )

    assert (
        should_auto_retry_tool(
            _RetrySafeTool(),
            retryable_summary,
            tool_name="safe_reader",
        )
        is True
    )
    assert (
        should_auto_retry_tool(
            _Tool(),
            retryable_summary,
            tool_name="example_tool",
        )
        is False
    )
    assert (
        should_auto_retry_tool(
            _ToolSearchTool(),
            retryable_summary,
            tool_name="tool_search",
            retry_safe_tool_names={"tool_search"},
        )
        is True
    )
    assert (
        should_auto_retry_tool(
            _StringMetadataTool(),
            retryable_summary,
            tool_name="unsafe_string_flag",
        )
        is False
    )
    assert (
        should_auto_retry_tool(
            _RetrySafeTool(),
            argument_summary,
            tool_name="safe_reader",
        )
        is False
    )
```

- [x] **Step 2: Run the new tests and verify they fail**

Run:

```bash
python -m pytest tests/test_tool_error_policy.py -q
```

Expected: fails because `app.ai.tool_error_policy` does not exist.

- [x] **Step 3: Implement the error policy module**

Create `app/ai/tool_error_policy.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from anyio import BrokenResourceError, ClosedResourceError


class ToolErrorKind(str, Enum):
    TIMEOUT = "timeout"
    SESSION = "session"
    ARGUMENT = "argument"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    NETWORK = "network"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolErrorSummary:
    error_type: str
    retryable: bool
    message: str
    hint: str
    attempts: int

    def model_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error_type": self.error_type,
            "retryable": self.retryable,
            "message": self.message,
            "hint": self.hint,
        }


def _clean_text(value: Any, *, max_chars: int = 180) -> str:
    text = str(value or "").strip().replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def classify_tool_error(
    exc: BaseException,
    *,
    tool_name: str,
    timeout_seconds: int | float,
    attempts: int,
) -> ToolErrorSummary:
    raw = str(exc).lower()
    timeout_value = int(timeout_seconds) if float(timeout_seconds).is_integer() else timeout_seconds

    if isinstance(exc, TimeoutError):
        return ToolErrorSummary(
            error_type=ToolErrorKind.TIMEOUT.value,
            retryable=True,
            message=f"Tool timed out after {timeout_value}s.",
            hint=(
                "Retry only if the operation is likely safe; otherwise adjust inputs, "
                "use another available tool, discover a better tool, or ask the user."
            ),
            attempts=attempts,
        )

    if isinstance(exc, (ClosedResourceError, BrokenResourceError)):
        return ToolErrorSummary(
            error_type=ToolErrorKind.SESSION.value,
            retryable=True,
            message="Tool session was interrupted.",
            hint="The runtime may reconnect. If it still fails, use another available tool or ask the user.",
            attempts=attempts,
        )

    if isinstance(exc, TypeError):
        return ToolErrorSummary(
            error_type=ToolErrorKind.ARGUMENT.value,
            retryable=False,
            message="Tool arguments did not match the expected schema.",
            hint="Read the tool schema or prior error, then call the tool again only with corrected arguments.",
            attempts=attempts,
        )

    if isinstance(exc, (ValueError, KeyError)):
        return ToolErrorSummary(
            error_type=ToolErrorKind.VALIDATION.value,
            retryable=False,
            message=_clean_text(exc) or "Tool rejected the provided values.",
            hint="Correct the values before retrying. Do not repeat the same arguments.",
            attempts=attempts,
        )

    if isinstance(exc, (PermissionError,)):
        return ToolErrorSummary(
            error_type=ToolErrorKind.PERMISSION.value,
            retryable=False,
            message="Tool lacks permission for that operation.",
            hint="Ask the user for access, choose a permitted alternative, or explain the blocker.",
            attempts=attempts,
        )

    if isinstance(exc, FileNotFoundError) or "not found" in raw:
        return ToolErrorSummary(
            error_type=ToolErrorKind.NOT_FOUND.value,
            retryable=False,
            message=_clean_text(exc) or "Requested resource was not found.",
            hint="Verify the target exists, adjust the query/path, or ask the user for the correct target.",
            attempts=attempts,
        )

    if any(token in raw for token in ("connection", "network", "temporarily", "reset", "unavailable")):
        return ToolErrorSummary(
            error_type=ToolErrorKind.NETWORK.value,
            retryable=True,
            message="Tool failed because the connection or service was unavailable.",
            hint="Retry only if the tool is safe to repeat; otherwise use another tool or ask the user.",
            attempts=attempts,
        )

    return ToolErrorSummary(
        error_type=ToolErrorKind.UNKNOWN.value,
        retryable=False,
        message=_clean_text(exc) or "Tool failed with an unknown error.",
        hint="Do not repeat the same call without changing inputs or choosing a better tool.",
        attempts=attempts,
    )


def should_auto_retry_tool(
    tool: Any,
    summary: ToolErrorSummary,
    *,
    tool_name: str,
    retry_safe_tool_names: set[str] | None = None,
) -> bool:
    if not summary.retryable:
        return False
    if tool_name in (retry_safe_tool_names or set()):
        return True
    metadata = getattr(tool, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return False
    return metadata.get("retry_safe") is True or metadata.get("idempotent") is True


def build_tool_error_payloads(
    summary: ToolErrorSummary,
    *,
    tool_name: str,
    exception: BaseException,
) -> tuple[str, dict[str, Any]]:
    model_content = json.dumps(summary.model_dict(), ensure_ascii=False, separators=(",", ":"))
    artifact_detail = {
        "status": "error",
        "tool": tool_name,
        "error_type": summary.error_type,
        "retryable": summary.retryable,
        "attempts": summary.attempts,
        "exception_type": type(exception).__name__,
        "diagnostic": _clean_text(exception, max_chars=1000),
    }
    return model_content, artifact_detail
```

- [x] **Step 4: Run the focused tests and verify they pass**

Run:

```bash
python -m pytest tests/test_tool_error_policy.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/ai/tool_error_policy.py tests/test_tool_error_policy.py
git commit -m "feat: add compact tool error policy"
```

---

## Task 2: Enforce Timeout and Conservative Retries in Shared Tool Execution

**Files:**
- Modify: `app/ai/tool_execution.py`
- Test: `tests/test_tool_execution_recovery.py`
- Test: `tests/test_tool_execution_rendering.py`

- [ ] **Step 1: Add failing tests for timeout, safe retry, and unsafe retry**

Append to `tests/test_tool_execution_recovery.py`:

```python
import asyncio
import json

from app.core.config import settings


@pytest.mark.asyncio
async def test_execute_tool_calls_times_out_slow_tool(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 0.01)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 0)

    class _SlowTool:
        name = "slow_tool"
        metadata = {}

        async def ainvoke(self, args):
            await asyncio.sleep(1)
            return "too late"

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "slow_tool", "args": {}}],
        tool_map={"slow_tool": _SlowTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "timeout"
    assert payload["retryable"] is True
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error_type"] == "timeout"


@pytest.mark.asyncio
async def test_execute_tool_calls_retries_retry_safe_transient_failure(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 1)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 1)
    calls = 0

    class _RetrySafeTool:
        name = "safe_reader"
        metadata = {"retry_safe": True}

        async def ainvoke(self, args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("connection reset")
            return "ok"

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "safe_reader", "args": {}}],
        tool_map={"safe_reader": _RetrySafeTool()},
    )

    assert calls == 2
    assert outputs[0]["content"] == "ok"
    assert artifacts[0]["status"] == "success"


@pytest.mark.asyncio
async def test_execute_tool_calls_does_not_auto_retry_unknown_side_effect_tool(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 1)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 2)
    calls = 0

    class _MaybeSideEffectTool:
        name = "send_message"
        metadata = {}

        async def ainvoke(self, args):
            nonlocal calls
            calls += 1
            raise ConnectionError("connection reset")

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "send_message", "args": {"body": "hi"}}],
        tool_map={"send_message": _MaybeSideEffectTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert calls == 1
    assert payload["error_type"] == "network"
    assert payload["retryable"] is True
    assert artifacts[0]["attempts"] == 1
```

Append to `tests/test_tool_execution_rendering.py`:

```python
@pytest.mark.asyncio
async def test_execute_tool_calls_structured_error_render_stays_compact(monkeypatch):
    import json

    from app.core.config import settings

    monkeypatch.setattr(settings, "tool_execution_timeout", 1)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 0)

    class _BrokenTool:
        name = "broken_tool"
        metadata = {}

        async def ainvoke(self, args):
            raise PermissionError("permission denied for a very sensitive path")

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "broken_tool", "args": {}}],
        tool_map={"broken_tool": _BrokenTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert len(outputs[0]["content"]) < 500
    assert outputs[0]["render"]["type"] == "error"
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["render"]["type"] == "error"
```

Also modify the existing `test_execute_tool_calls_preserves_error_render_artifact` in the same file so it expects compact JSON instead of the old `Error: ...` string:

```python
@pytest.mark.asyncio
async def test_execute_tool_calls_preserves_error_render_artifact():
    import json

    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[{"id": "tool-call-err", "name": "failing_tool", "args": {}}],
        tool_map={"failing_tool": _FailingTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "validation"
    assert outputs[0]["render"]["type"] == "error"
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["output"] == outputs[0]["content"]
    assert artifacts[0]["render"]["type"] == "error"
    assert artifacts[0]["error_type"] == "validation"
    assert images == []
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py -q
```

Expected: new tests fail because timeout/max-retry policy is not enforced and structured error payloads are not present.

- [ ] **Step 3: Wrap tool invocation with timeout and conservative retry**

Modify `app/ai/tool_execution.py`:

```python
from .tool_error_policy import (
    build_tool_error_payloads,
    classify_tool_error,
    should_auto_retry_tool,
)
```

Add a helper near `invoke_tool`:

```python
def _tool_execution_timeout_seconds() -> float:
    value = getattr(settings, "tool_execution_timeout", 30) or 30
    try:
        parsed = float(value)
    except Exception:
        return 30.0
    return max(0.001, parsed)


def _tool_execution_max_retries() -> int:
    value = getattr(settings, "tool_execution_max_retries", 0) or 0
    try:
        parsed = int(value)
    except Exception:
        return 0
    return max(0, parsed)


async def invoke_tool_with_policy(
    tool: Any,
    tool_args: Any,
    *,
    tool_name: str,
) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    timeout_seconds = _tool_execution_timeout_seconds()
    max_retries = _tool_execution_max_retries()
    attempts = 0
    last_exc: BaseException | None = None

    while attempts <= max_retries:
        attempts += 1
        try:
            result = await asyncio.wait_for(
                invoke_tool(tool, tool_args),
                timeout=timeout_seconds,
            )
            return result, None, None
        except asyncio.TimeoutError as exc:
            last_exc = TimeoutError(f"Tool timed out after {timeout_seconds}s")
        except (ClosedResourceError, BrokenResourceError):
            raise
        except Exception as exc:
            last_exc = exc

        summary = classify_tool_error(
            last_exc,
            tool_name=tool_name,
            timeout_seconds=timeout_seconds,
            attempts=attempts,
        )
        if attempts > max_retries or not should_auto_retry_tool(
            tool,
            summary,
            tool_name=tool_name,
            retry_safe_tool_names=TOOL_LOADING_TOOLS,
        ):
            model_content, artifact_detail = build_tool_error_payloads(
                summary,
                tool_name=tool_name,
                exception=last_exc,
            )
            return None, artifact_detail, model_content

    fallback_exc = last_exc or RuntimeError("Tool failed")
    summary = classify_tool_error(
        fallback_exc,
        tool_name=tool_name,
        timeout_seconds=timeout_seconds,
        attempts=attempts,
    )
    model_content, artifact_detail = build_tool_error_payloads(
        summary,
        tool_name=tool_name,
        exception=fallback_exc,
    )
    return None, artifact_detail, model_content
```

In the normal execution branch of `execute_tool_calls`, replace:

```python
result = await invoke_tool(tool, tool_args)
normalized_result = normalize_tool_result_for_rendering(
    result,
    tool_name=tool_name,
)
```

with:

```python
result, error_detail, error_content = await invoke_tool_with_policy(
    tool,
    tool_args,
    tool_name=tool_name,
)
if error_detail is not None:
    normalized_result = normalize_tool_result_for_rendering(
        error_content,
        tool_name=tool_name,
    )
    render = dict(normalized_result.render)
    render["type"] = "error"
    render["error"] = str(
        error_detail.get("diagnostic")
        or error_detail.get("error_type")
        or "Tool execution failed"
    )
    outputs.append(
        {
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": error_content,
            "render": render,
        }
    )
    artifact = build_tool_artifact(
        tool_call_id=tool_id,
        tool_name=tool_name,
        tool_args=tool_args,
        output_text=error_content,
        error=str(error_detail.get("diagnostic") or error_detail.get("error_type")),
        max_output_chars=artifact_max_output_chars,
        render=render,
    )
    artifact.update(error_detail)
    artifacts.append(artifact)
    continue

normalized_result = normalize_tool_result_for_rendering(
    result,
    tool_name=tool_name,
)
```

Keep the existing `ClosedResourceError` / `BrokenResourceError` reconnect branch. The policy helper above re-raises those session errors so the reconnect code still runs. Ensure the retry call after reconnect also uses `asyncio.wait_for(..., timeout=_tool_execution_timeout_seconds())` so reconnect retries cannot hang. If reconnect or the post-reconnect retry fails, classify that final exception with `classify_tool_error(...)`, return compact JSON as the ToolMessage content, and attach the full diagnostic fields to the artifact.

- [ ] **Step 4: Run focused tool execution tests**

Run:

```bash
python -m pytest tests/test_tool_error_policy.py tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/ai/tool_execution.py tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py
git commit -m "feat: enforce compact tool execution retry policy"
```

---

## Task 3: Resolve Pending Tool Calls Even When Binding Fails

**Files:**
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/graph.py`
- Test: `tests/test_tool_execution_recovery.py`
- Test: `tests/test_graph_tool_budget.py`

- [ ] **Step 1: Add failing tests for empty tool maps**

Append to `tests/test_tool_execution_recovery.py`:

```python
@pytest.mark.asyncio
async def test_execute_tool_calls_empty_tool_map_returns_error_for_each_call(monkeypatch):
    import json

    monkeypatch.setattr(settings, "mcp_tool_search_enabled", False)

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[
            {"id": "call-1", "name": "missing_one", "args": {}},
            {"id": "call-2", "name": "missing_two", "args": {}},
        ],
        tool_map={},
    )

    assert [output["tool_call_id"] for output in outputs] == ["call-1", "call-2"]
    assert all(json.loads(output["content"])["status"] == "error" for output in outputs)
    assert all(artifact["status"] == "error" for artifact in artifacts)


@pytest.mark.asyncio
async def test_execute_tool_calls_missing_tool_name_returns_compact_argument_error():
    import json

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "args": {"query": "x"}}],
        tool_map={},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "argument"
    assert payload["retryable"] is False
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error_type"] == "argument"


@pytest.mark.asyncio
async def test_execute_tool_calls_client_device_mismatch_returns_compact_error(monkeypatch):
    import json

    class _ClientTool:
        name = "client__desktop__read_file"
        metadata = {}

        async def ainvoke(self, args):
            return "never"

    monkeypatch.setattr("app.ai.tool_execution.is_client_tool", lambda _tool: True)
    monkeypatch.setattr(
        "app.ai.tool_execution.get_client_tool_device_id",
        lambda _tool: "device-a",
    )

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "client__desktop__read_file", "args": {}}],
        tool_map={"client__desktop__read_file": _ClientTool()},
        device_id="device-b",
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "permission"
    assert payload["retryable"] is False
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error_type"] == "permission"
```

Append to `tests/test_graph_tool_budget.py`:

```python
@pytest.mark.asyncio
async def test_tool_node_resolves_pending_calls_when_tool_map_empty(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from app.ai.graph import MultiAgentWorkflow

    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    class _Agent:
        agent_config_key = "chat"
        tool_state_key = "chat"

    graph._resolve_runtime_agent = lambda state, selected: _Agent()

    async def _empty_tool_map(*args, **kwargs):
        return {}

    monkeypatch.setattr("app.ai.graph.ensure_agent_tool_map", _empty_tool_map)

    state = {
        "selected_agent": "chat_agent",
        "messages": [
            HumanMessage(content="run something"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "missing_tool",
                        "args": {},
                    }
                ],
            ),
        ],
    }

    updated = await graph._tool_node(state)

    assert isinstance(updated["messages"][-1], ToolMessage)
    assert updated["messages"][-1].tool_call_id == "call-1"
    assert "missing_tool" in updated["messages"][-1].content
```

Also update the existing `test_execute_tool_calls_missing_tool_does_not_double_prefix_error` in `tests/test_tool_execution_rendering.py` so it validates the compact missing-tool payload:

```python
@pytest.mark.asyncio
async def test_execute_tool_calls_missing_tool_uses_compact_error_payload():
    import json

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "tc-missing", "name": "nope_tool", "args": {}}],
        tool_map={},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "not_found"
    assert not outputs[0]["content"].startswith("Error:")
    assert outputs[0]["render"]["type"] == "error"
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error_type"] == "not_found"
```

- [ ] **Step 2: Run tests and verify the graph test fails**

Run:

```bash
python -m pytest \
  tests/test_tool_execution_recovery.py::test_execute_tool_calls_empty_tool_map_returns_error_for_each_call \
  tests/test_tool_execution_recovery.py::test_execute_tool_calls_missing_tool_name_returns_compact_argument_error \
  tests/test_tool_execution_recovery.py::test_execute_tool_calls_client_device_mismatch_returns_compact_error \
  tests/test_graph_tool_budget.py::test_tool_node_resolves_pending_calls_when_tool_map_empty \
  -q
```

Expected: these fail because `_tool_node` returns early on an empty tool map and pre-invocation errors are still free-form strings.

- [ ] **Step 3: Remove empty-map early exits before execution**

Modify `app/ai/graph.py`.

In `_tool_node`, remove:

```python
if not tool_map:
    return state
```

In `_execute_agent_tool_calls`, remove:

```python
if not tool_map:
    return [], [], []
```

Let `execute_tool_calls(..., tool_map={})` create one error ToolMessage per pending tool call.

- [ ] **Step 4: Make pre-invocation tool errors compact and guidance-neutral**

In `app/ai/tool_execution.py`, replace the non-client missing-tool message:

```python
f"Error: Tool {tool_name} not found. "
"Use tool_search first to discover and load the correct tool, "
"then call the discovered tool by name."
```

with compact JSON created through `ToolErrorSummary`:

```python
from .tool_error_policy import ToolErrorKind, ToolErrorSummary

summary = ToolErrorSummary(
    error_type=ToolErrorKind.NOT_FOUND.value,
    retryable=False,
    message=f"Tool {tool_name} is not currently bound.",
    hint=(
        "Use a currently bound suitable tool if one exists. If the needed capability "
        "is missing or ambiguous, use tool_search to discover it."
    ),
    attempts=1,
)
```

Use `build_tool_error_payloads(summary, tool_name=tool_name, exception=LookupError(...))` for model content and artifact details. Preserve the JSON model content directly in `outputs[*]["content"]`; do not pass it as the `error=` argument to `normalize_tool_result_for_rendering`, because that helper prefixes error text with `Error:`.

Apply the same compact-output path to these pre-invocation failures:

- Missing tool name: `error_type="argument"`, `retryable=False`, message `"Tool name is missing."`
- Missing client tool: `error_type="not_found"`, `retryable=False`, message `"Client tool {tool_name} is not available for the current device session."`
- Client device mismatch or absent required device context: `error_type="permission"`, `retryable=False`, message `"Client tool {tool_name} cannot execute from the current device session."`

Prefer a small local helper such as `_append_tool_error_output(...)` inside `execute_tool_calls()` to keep output/artifact/render construction identical for missing-name, missing-tool, device-binding, policy-timeout, and exception errors.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py tests/test_graph_tool_budget.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/ai/graph.py app/ai/tool_execution.py tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py tests/test_graph_tool_budget.py
git commit -m "fix: resolve pending tool calls on empty tool maps"
```

---

## Task 4: Add Generic Repeated Tool Failure Breakers

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/ai/graph.py`
- Test: `tests/test_graph_tool_budget.py`
- Test: `tests/test_rag_tool_loop_finalization.py`

- [ ] **Step 1: Add failing tests for repeated tool failure state**

Append to `tests/test_graph_tool_budget.py`:

```python
def test_apply_tool_outputs_tracks_same_error_streak(monkeypatch):
    from app.ai.graph import MultiAgentWorkflow
    from app.core.config import settings

    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {"messages": [], "context": {}}

    artifact = {
        "tool_call_id": "call-1",
        "tool": "read_file",
        "args": {"path": "missing.txt"},
        "status": "error",
        "error_type": "not_found",
        "output": "missing",
    }

    graph._apply_tool_outputs_to_state(
        state,
        tool_outputs=[{"tool_call_id": "call-1", "name": "read_file", "content": "missing"}],
        tool_artifacts=[artifact],
    )
    graph._apply_tool_outputs_to_state(
        state,
        tool_outputs=[{"tool_call_id": "call-2", "name": "read_file", "content": "missing"}],
        tool_artifacts=[{**artifact, "tool_call_id": "call-2"}],
    )

    streak = state["context"]["tool_error_streak"]
    assert streak["count"] == 2
    assert streak["limit"] == 2
    assert streak["signature"]["tool"] == "read_file"
    assert streak["signature"]["args"] == '{"path":"missing.txt"}'
```

Append to `tests/test_rag_tool_loop_finalization.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_rag_tools_node_tracks_document_tool_error_streak(monkeypatch):
    from langchain_core.messages import AIMessage

    from app.ai.graph import MultiAgentWorkflow
    from app.core.config import settings

    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    graph.rag_agent = object()
    graph.agents = {}

    async def fake_execute_search_documents_action(**kwargs):
        return "Error: Unknown action: nope", "nope", {}

    monkeypatch.setattr(
        "app.ai.graph.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.graph.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    state = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {},
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "rag-call-1",
                        "name": "search_documents",
                        "args": {"action": "nope"},
                    }
                ],
            )
        ],
    }

    await graph._rag_tools_node(state)

    streak = state["context"]["tool_error_streak"]
    assert streak["count"] == 1
    assert streak["signature"]["tool"] == "search_documents"


def test_rag_repeated_error_forces_final_response(monkeypatch):
    from app.ai.graph import MultiAgentWorkflow
    from app.core.config import settings

    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {
        "context": {
            "tool_error_streak": {
                "count": 2,
                "limit": 2,
                "signature": {
                    "tool": "search_documents",
                    "error_type": "validation",
                    "args": '{"action":"not_a_real_action"}',
                },
            }
        }
    }

    decision = graph._should_continue_rag(state)

    assert decision == "rag_agent"
    assert state["context"]["rag_force_final_response"] is True
    assert "repeated tool errors" in state["context"]["rag_tool_budget_notice"].lower()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest \
  tests/test_graph_tool_budget.py::test_apply_tool_outputs_tracks_same_error_streak \
  tests/test_rag_tool_loop_finalization.py::test_rag_tools_node_tracks_document_tool_error_streak \
  tests/test_rag_tool_loop_finalization.py::test_rag_repeated_error_forces_final_response \
  -q
```

Expected: the selected tests fail because generic streak tracking does not exist.

- [ ] **Step 3: Add config for generic tool-error breaker**

Modify `app/core/config.py` near tool execution settings:

```python
tool_execution_consecutive_errors_limit: int = Field(
    default=3,
    description="Maximum repeated same tool/error/argument outputs before forcing final no-tools synthesis.",
)
```

Add `"tool_execution_consecutive_errors_limit"` to the existing positive integer validation list near other runtime limits.

- [ ] **Step 4: Track compact repeated error signatures in graph state**

Add helpers to `app/ai/graph.py`:

```python
@staticmethod
def _tool_error_signature(artifact: dict[str, Any]) -> dict[str, str]:
    try:
        args_key = json.dumps(
            make_json_safe(artifact.get("args") or {}),
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        args_key = "{}"
    return {
        "tool": str(artifact.get("tool") or "unknown"),
        "error_type": str(artifact.get("error_type") or "unknown"),
        "args": args_key[:500],
    }


def _update_tool_error_streak(
    self,
    state: GraphState,
    tool_artifacts: list[dict[str, Any]] | None,
) -> None:
    context = GraphStateView(state).context_copy()
    errors = [
        artifact
        for artifact in tool_artifacts or []
        if isinstance(artifact, dict) and artifact.get("status") == "error"
    ]
    if not errors:
        context.pop("tool_error_streak", None)
        state["context"] = context
        return

    signature = self._tool_error_signature(errors[0])
    prior = context.get("tool_error_streak")
    prior_signature = prior.get("signature") if isinstance(prior, dict) else None
    try:
        prior_count = int(prior.get("count", 0)) if isinstance(prior, dict) else 0
    except Exception:
        prior_count = 0
    count = prior_count + 1 if prior_signature == signature else 1
    limit = max(1, int(getattr(settings, "tool_execution_consecutive_errors_limit", 3) or 3))
    context["tool_error_streak"] = {
        "count": count,
        "limit": limit,
        "signature": signature,
    }
    state["context"] = context
```

Call this helper at the end of `_apply_tool_outputs_to_state`, after the method has assigned the accumulated context back to `state["context"]`:

```python
self._update_tool_error_streak(state, tool_artifacts)
```

Also call the same helper at the end of `_rag_tools_node`, after RAG tool artifacts/images/render results have been merged into `context` and assigned back to `state["context"]`:

```python
state["context"] = context
self._update_tool_error_streak(state, tool_artifacts)
```

- [ ] **Step 5: Force final synthesis for repeated graph-level errors**

In `_route_tool_output`, before generic max-iteration checks, add:

```python
streak = (state_view.context() or {}).get("tool_error_streak")
if isinstance(streak, dict) and streak.get("count", 0) >= streak.get("limit", 3):
    if can_route_for_final_response:
        self._mark_force_final_response(
            state,
            reason="consecutive_tool_errors",
            scope="runtime",
            count=int(streak.get("count") or 0),
            limit=int(streak.get("limit") or 0),
        )
        return self._route_target_for(state, selected_agent)
    self._set_continuation_signal(
        state,
        should_continue=False,
        reason="consecutive_tool_errors",
        scope="runtime",
        count=int(streak.get("count") or 0),
        limit=int(streak.get("limit") or 0),
    )
    return "end"
```

- [ ] **Step 6: Force final synthesis for repeated RAG errors**

In `_should_continue_rag`, before the `agentic_iteration >= max_iterations` block, add:

```python
streak = context.get("tool_error_streak")
if isinstance(streak, dict) and streak.get("count", 0) >= streak.get("limit", 3):
    if context.get("rag_force_final_response"):
        return "end"
    context["rag_force_final_response"] = True
    context["rag_tool_budget_notice"] = (
        "Repeated tool errors occurred. Produce the best final answer from the "
        "document evidence already available, explain the blocker briefly, and do not call tools."
    )
    state["context"] = context
    return "rag_agent"
```

- [ ] **Step 7: Run focused graph/RAG tests**

Run:

```bash
python -m pytest tests/test_graph_tool_budget.py tests/test_rag_tool_loop_finalization.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add app/core/config.py app/ai/graph.py tests/test_graph_tool_budget.py tests/test_rag_tool_loop_finalization.py
git commit -m "feat: add generic repeated tool failure breaker"
```

---

## Task 5: Bring RAG Document Tool Errors Into the Same Compact Shape

**Files:**
- Modify: `app/ai/rag_tool_actions.py`
- Modify: `app/ai/graph.py`
- Test: `tests/test_rag_tool_loop_finalization.py`

- [ ] **Step 1: Add failing test for compact `search_documents` errors**

Append to `tests/test_rag_tool_loop_finalization.py`:

```python
@pytest.mark.asyncio
async def test_execute_search_documents_action_returns_compact_error_for_unknown_action():
    import json
    from types import SimpleNamespace

    from app.ai.rag_tool_actions import execute_search_documents_action

    result, action, evidence = await execute_search_documents_action(
        rag_agent=SimpleNamespace(),
        conversation_id="conv-1",
        user_id="user-1",
        tool_args={"action": "not_a_real_action"},
        context={},
        max_agentic_images=3,
    )

    payload = json.loads(result)
    assert action == "not_a_real_action"
    assert evidence == {}
    assert payload == {
        "status": "error",
        "error_type": "validation",
        "retryable": False,
        "message": "search_documents rejected the requested action.",
        "hint": "Use one of the supported document exploration actions from the tool schema.",
    }
```

This test fails before implementation because the current helper returns plain text like `Unknown action: not_a_real_action`.

- [ ] **Step 2: Add a compact RAG error helper**

In `app/ai/rag_tool_actions.py`, add:

```python
import json

from .tool_error_policy import ToolErrorKind, ToolErrorSummary


def compact_rag_tool_error(
    *,
    error_type: str,
    message: str,
    hint: str,
    retryable: bool = False,
) -> str:
    summary = ToolErrorSummary(
        error_type=error_type,
        retryable=retryable,
        message=message,
        hint=hint,
        attempts=1,
    )
    return json.dumps(summary.model_dict(), ensure_ascii=False, separators=(",", ":"))
```

Use it where `execute_search_documents_action` currently returns `"Error: ..."` for invalid actions, missing documents, invalid regex, or unsupported arguments. Keep success payloads unchanged.

For invalid actions, return:

```python
return compact_rag_tool_error(
    error_type=ToolErrorKind.VALIDATION.value,
    message="search_documents rejected the requested action.",
    hint="Use one of the supported document exploration actions from the tool schema.",
), action, evidence
```

- [ ] **Step 3: Preserve `error_type` and `retryable` on RAG artifacts**

In both `_rag_tools_node` and `_run_agent_in_isolated_context` RAG worker sections in `app/ai/graph.py`, after receiving a `result` from `execute_search_documents_action`, parse compact JSON errors:

```python
parsed_error: dict[str, Any] | None = None
if isinstance(result, str):
    with contextlib.suppress(Exception):
        candidate = json.loads(result)
        if isinstance(candidate, dict) and candidate.get("status") == "error":
            parsed_error = candidate
error = result if parsed_error or result.startswith("Error") else None
```

After `build_tool_artifact(...)`, add:

```python
if parsed_error:
    artifact["error_type"] = parsed_error.get("error_type")
    artifact["retryable"] = bool(parsed_error.get("retryable"))
```

Add `import contextlib` and `import json` to `app/ai/graph.py` if they are not already present.

- [ ] **Step 4: Run RAG-focused tests**

Run:

```bash
python -m pytest tests/test_rag_tool_loop_finalization.py tests/test_rag_agent.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/ai/rag_tool_actions.py app/ai/graph.py tests/test_rag_tool_loop_finalization.py
git commit -m "feat: normalize rag tool error summaries"
```

---

## Task 6: Bound Subagent Worker Tool Loops

**Files:**
- Modify: `app/ai/graph.py`
- Test: `tests/test_graph_planning_subagents.py`

- [ ] **Step 1: Add failing tests for worker loop limits**

Append to `tests/test_graph_planning_subagents.py`:

```python
def test_worker_loop_uses_tool_error_limit_from_settings(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 4, raising=False)
    assert settings.tool_execution_consecutive_errors_limit == 4
```

Add an integration-style worker test near existing `_run_agent_in_isolated_context` tests:

```python
@pytest.mark.asyncio
async def test_generic_worker_stops_after_repeated_tool_errors(monkeypatch):
    from app.ai.graph import MultiAgentWorkflow
    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
    from app.core.config import settings

    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)

    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    class _LoopingAgent:
        agent_config_key = "chat"
        tool_state_key = "chat"
        agent_type = AgentType.CHAT
        agent_id = "chat_agent"

        async def invoke_model_with_history(self, **kwargs):
            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[{"id": "call-1", "name": "missing_tool", "args": {}}],
                ),
                metadata={},
            )

    graph.agents = {"chat_agent": _LoopingAgent()}

    async def _empty_map(*args, **kwargs):
        return {}

    monkeypatch.setattr("app.ai.graph.ensure_agent_tool_map", _empty_map)

    response = await graph._run_agent_in_isolated_context(
        parent_state={"conversation_id": "conv-1", "user_id": "user-1"},
        agent_name="chat_agent",
        task_prompt="try missing tool",
    )

    assert response.metadata["pause_reason"] == "consecutive_tool_errors"
```

Also rename the existing worker tests whose names currently say `has_no_subagent_iteration_cap` so they reflect the new behavior without changing their assertions:

- Rename `test_run_agent_in_isolated_context_has_no_subagent_iteration_cap` to `test_run_agent_in_isolated_context_allows_generic_worker_under_iteration_cap`.
- Rename `test_run_agent_in_isolated_context_rag_has_no_subagent_iteration_cap` to `test_run_agent_in_isolated_context_allows_rag_worker_under_iteration_cap`.

Add one explicit max-iteration test for generic workers:

```python
@pytest.mark.asyncio
async def test_generic_worker_stops_after_iteration_limit(monkeypatch):
    from app.ai.graph import MultiAgentWorkflow
    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
    from app.core.config import settings

    monkeypatch.setattr(settings, "react_agent_max_iterations", 2)

    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    call_count = 0

    class _LoopingAgent:
        agent_config_key = "chat"
        tool_state_key = "chat"
        agent_type = AgentType.CHAT
        agent_id = "chat_agent"

        async def invoke_model_with_history(self, **kwargs):
            nonlocal call_count
            call_count += 1
            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[{"id": f"call-{call_count}", "name": "lookup", "args": {}}],
                ),
                metadata={},
            )

    graph.agents = {"chat_agent": _LoopingAgent()}

    class _Lookup:
        name = "lookup"

        async def ainvoke(self, args):
            return "ok"

    async def _tool_map(*args, **kwargs):
        return {"lookup": _Lookup()}

    monkeypatch.setattr("app.ai.graph.ensure_agent_tool_map", _tool_map)

    response = await graph._run_agent_in_isolated_context(
        parent_state={"conversation_id": "conv-1", "user_id": "user-1"},
        agent_name="chat_agent",
        task_prompt="loop until capped",
    )

    assert response.metadata["pause_reason"] == "worker_max_iterations"
    assert call_count == 2
```

Add one explicit max-iteration test for isolated RAG workers:

```python
@pytest.mark.asyncio
async def test_rag_worker_stops_after_iteration_limit(monkeypatch):
    from app.ai.graph import MultiAgentWorkflow
    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
    from app.core.config import settings

    monkeypatch.setattr(settings, "agentic_max_iterations", 2)

    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    call_count = 0

    async def fake_process_message(message, conversation_id):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return AgentResponse(
                agent_type=AgentType.RAG,
                agent_id="rag_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": f"rag-call-{call_count}",
                            "name": "search_documents",
                            "args": {"action": "scan_all"},
                        }
                    ],
                ),
                metadata={},
            )
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final answer"),
            metadata={},
        )

    rag_agent = type(
        "RagWorker",
        (),
        {
            "agent_config_key": "rag",
            "tool_state_key": "rag",
            "agent_type": AgentType.RAG,
            "agent_id": "rag_agent",
            "process_message": staticmethod(fake_process_message),
        },
    )()
    graph.rag_agent = rag_agent
    graph.agents = {"rag_agent": rag_agent}

    async def fake_execute_search_documents_action(**kwargs):
        return "SEARCH RESULT", "scan_all", {}

    monkeypatch.setattr(
        "app.ai.graph.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.graph.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    response = await graph._run_agent_in_isolated_context(
        parent_state={"conversation_id": "conv-1", "user_id": "user-1", "context": {}},
        agent_name="rag_agent",
        task_prompt="loop document search until capped",
    )

    assert response.metadata["pause_reason"] == "worker_max_iterations"
    assert call_count == 2
```

- [ ] **Step 2: Run the worker tests and verify failure**

Run:

```bash
python -m pytest \
  tests/test_graph_planning_subagents.py::test_generic_worker_stops_after_repeated_tool_errors \
  tests/test_graph_planning_subagents.py::test_generic_worker_stops_after_iteration_limit \
  tests/test_graph_planning_subagents.py::test_rag_worker_stops_after_iteration_limit \
  -q
```

Expected: all three fail because isolated worker loops have no local repeated-error breaker or iteration cap.

- [ ] **Step 3: Add local worker loop counters**

In `_run_agent_in_isolated_context`, before the generic worker `while True`, add:

```python
worker_iterations = 0
worker_error_signature: dict[str, str] | None = None
worker_error_count = 0
worker_error_limit = max(
    1,
    int(getattr(settings, "tool_execution_consecutive_errors_limit", 3) or 3),
)
worker_iteration_limit = max(1, int(getattr(settings, "react_agent_max_iterations", 50) or 50))
```

At the top of the generic worker `while True`, add:

```python
worker_iterations += 1
if worker_iterations > worker_iteration_limit:
    return AgentResponse(
        agent_type=getattr(agent, "agent_type", AgentType.CHAT),
        agent_id=agent_name,
        message=AgentMessage(
            role=MessageRole.ASSISTANT,
            content="Worker stopped after reaching its tool iteration limit.",
        ),
        metadata={"pause_reason": "worker_max_iterations"},
    )
```

After `accumulated_worker_artifacts.extend(artifacts)`, add:

```python
error_artifacts = [
    artifact
    for artifact in artifacts
    if isinstance(artifact, dict) and artifact.get("status") == "error"
]
if error_artifacts:
    signature = self._tool_error_signature(error_artifacts[0])
    if signature == worker_error_signature:
        worker_error_count += 1
    else:
        worker_error_signature = signature
        worker_error_count = 1
    if worker_error_count >= worker_error_limit:
        return AgentResponse(
            agent_type=getattr(agent, "agent_type", AgentType.CHAT),
            agent_id=agent_name,
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content=(
                    "Worker stopped after repeated tool errors. Use the available "
                    "tool results to explain the blocker."
                ),
            ),
            metadata={
                "pause_reason": "consecutive_tool_errors",
                "tool_error_streak": {
                    "count": worker_error_count,
                    "limit": worker_error_limit,
                    "signature": signature,
                },
            },
            tool_artifacts=list(accumulated_worker_artifacts),
        )
else:
    worker_error_signature = None
    worker_error_count = 0
```

Repeat the same iteration and repeated-error counter pattern for the RAG worker loop in `_run_agent_in_isolated_context`, using RAG artifacts accumulated in `accumulated_artifacts`. Use `settings.agentic_max_iterations` as the RAG worker iteration cap so document workers match the top-level RAG budget.

- [ ] **Step 4: Run subagent tests**

Run:

```bash
python -m pytest tests/test_graph_planning_subagents.py tests/test_event_streaming_subagents.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/ai/graph.py tests/test_graph_planning_subagents.py
git commit -m "fix: bound isolated worker tool loops"
```

---

## Task 7: Rebalance Tool Search Prompting Toward Bound Tools First

**Files:**
- Modify: `app/ai/prompts.py`
- Modify: `app/ai/tool_search_tool.py`
- Test: `tests/test_tool_search_prompt_guidance.py`

- [ ] **Step 1: Rewrite tests away from forced discovery**

Modify `tests/test_tool_search_prompt_guidance.py`.

Update the module docstring so it no longer says "`tool_search` makes discovery mandatory"; it should say that `tool_search` remains available for dynamic discovery while clear bound tools are preferred.

Replace the current test named `test_tool_exploration_suffix_mentions_tool_search` docstring and assertion message so it no longer implies mandatory discovery:

```python
def test_tool_exploration_suffix_mentions_tool_search_without_forcing_every_task():
    """TOOL_EXPLORATION_SUFFIX should describe discovery while preferring clear bound tools."""
    lower_suffix = TOOL_EXPLORATION_SUFFIX.lower()
    assert "tool_search" in lower_suffix
    assert "bound tool" in lower_suffix
    assert "clearly matches" in lower_suffix
```

Replace `test_tool_exploration_suffix_encourages_capability_exploration_before_text_fallback` with:

```python
def test_tool_exploration_suffix_prefers_bound_direct_tool_before_discovery():
    lower_suffix = TOOL_EXPLORATION_SUFFIX.lower()
    assert "if a bound tool clearly matches" in lower_suffix
    assert "use it directly" in lower_suffix
    assert "use `tool_search` when" in lower_suffix
```

Update `test_tool_exploration_suffix_requires_discovery_for_real_environment_actions`:

```python
def test_tool_exploration_suffix_uses_discovery_when_tool_is_missing_or_ambiguous():
    lower_suffix = TOOL_EXPLORATION_SUFFIX.lower()
    assert "missing" in lower_suffix
    assert "ambiguous" in lower_suffix
    assert "not currently bound" in lower_suffix
```

Update `test_tool_search_description_discourages_skipping_deferred_discovery`:

```python
def test_tool_search_description_respects_already_bound_tools():
    description = f"{tool_search.description} {create_tool_search_tool().description}".lower()
    assert "already bound" in description
    assert "missing" in description
    assert "ambiguous" in description
    assert "not currently bound" in description
```

- [ ] **Step 2: Run prompt tests and verify they fail**

Run:

```bash
python -m pytest tests/test_tool_search_prompt_guidance.py -q
```

Expected: fails until prompt/tool descriptions are rebalanced.

- [ ] **Step 3: Replace shared tool exploration guidance**

In `app/ai/prompts.py`, replace `TOOL_EXPLORATION_SUFFIX` with:

```python
TOOL_EXPLORATION_SUFFIX = """

TOOLS AND ENVIRONMENT:
IMPORTANT: Prompt examples are not a tool inventory. Treat bound tool schemas as the tools you can call now, and treat `tool_search` as discovery for dynamic tools.

- Treat bound tools and discovery results as scoped to the current conversation, agent, and current user/device session. Do not reuse tool availability from prompt memory, other conversations, or other client devices.
- If a bound tool clearly matches the user's requested action, use it directly.
- Use `tool_search` when the needed capability is missing, ambiguous, not currently bound, or tied to a named integration/server whose exact identifier is unknown.
- When the user asks what tools or integrations are available, call `tool_search()` in this request context. Do NOT answer from prompt memory.
- If the user names an integration and you do not know the exact server identifier, call `tool_search()` first, then inspect that server with `tool_search(server_name="...")`. Use the exact `server_name` returned by `tool_search()`. Do not invent or modify server identifiers.
- When the task is specific but no clear bound tool exists, use `tool_search(query="...")` with the actual action, target, and context.
- After `tool_search`, if `recommended_tool` is present, `confidence` is `high`, and `is_loaded` is true, call that tool next. Do not issue another `tool_search` with a synonym for the same capability.
- Only refine the search when `requires_refinement` is true, the recommended tool is not suitable for the user's actual task, or the needed integration is missing from the result.
- For in-chat structured visuals, use widget tools directly when already bound, or discover them with `tool_search(query="create widget")`. Keep widgets in-chat; use `canvas_agent` only for standalone browser artifacts."""
```

In `TOOL_CONTEXT_SUFFIX`, add one compact bullet:

```python
- If a tool result has `"status":"error"`, read `error_type`, `retryable`, and `hint`. Do not repeat the same failing call with identical arguments unless you have a concrete reason it is safe and useful.
```

- [ ] **Step 4: Update `tool_search` descriptions**

In both `tool_search()` and `create_tool_search_tool()` descriptions in `app/ai/tool_search_tool.py`, replace wording that says:

```text
Do not skip discovery because the task sounds common.
```

with:

```text
If a suitable specialized tool is already bound, use it directly. Use
tool_search when the needed capability is missing, ambiguous, or not currently
bound.
```

Keep inventory and named-integration instructions.

- [ ] **Step 5: Run prompt and tool-search tests**

Run:

```bash
python -m pytest tests/test_tool_search_prompt_guidance.py tests/test_unified_tool_search.py tests/test_tool_search_scoring.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/ai/prompts.py app/ai/tool_search_tool.py tests/test_tool_search_prompt_guidance.py
git commit -m "fix: prefer bound tools before deferred discovery"
```

---

## Task 8: Update Documentation and Contract Claims

**Files:**
- Modify: `README.md`
- Test: no new test; verify grep and focused tests.

- [ ] **Step 1: Update README tool execution contract**

In `README.md`, update the tool execution bullet near the architecture section so it accurately states:

```markdown
5. **Tool execution** — `tool_execution.execute_tool_calls` runs each tool with bounded per-tool timeout, conservative retry for retry-safe transient failures, compact model-facing error summaries, and truncated `ToolMessage` bodies (full artifacts preserved for the UI).
```

Update the deferred tool search section so it says:

```markdown
When `MCP_TOOL_SEARCH_ENABLED=true`, only lightweight discovery, pinned tools, already-loaded deferred tools, internal tools, and device-scoped loaded client tools are bound at start. Agents should use a clearly matching bound tool directly and use `tool_search` when capability is missing, ambiguous, or tied to an unknown integration/server identifier.
```

- [ ] **Step 2: Verify no docs still imply forced discovery**

Run:

```bash
rg -n "forces discovery|force discovery|Do not skip discovery|before giving a text-only" README.md app tests
```

Expected: no matches that imply every tool-capable task must call `tool_search` first. Matches in historical `plans/` files are acceptable and should not be edited.

- [ ] **Step 3: Run final focused regression suite**

Run:

```bash
python -m pytest \
  tests/test_tool_error_policy.py \
  tests/test_tool_execution_recovery.py \
  tests/test_tool_execution_rendering.py \
  tests/test_graph_tool_budget.py \
  tests/test_rag_tool_loop_finalization.py \
  tests/test_graph_planning_subagents.py \
  tests/test_tool_search_prompt_guidance.py \
  tests/test_unified_tool_search.py \
  tests/test_tool_search_scoring.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document resilient tool execution policy"
```

---

## Verification Checklist

- [x] `tool_execution_timeout` is enforced by `execute_tool_calls()`.
- [x] `tool_execution_max_retries` is used only for safe transient retries.
- [x] Unknown side-effect tools are not blindly retried.
- [x] MCP closed/broken session reconnect behavior still works.
- [x] Missing/unbound tools produce matching ToolMessages.
- [x] Empty execution maps do not leave unresolved AI tool calls.
- [x] Tool error ToolMessages remain under 500 characters in focused tests.
- [x] Full diagnostics are present in artifacts/logs, not model context.
- [x] Graph-level ReAct loops force final synthesis after repeated same tool/error/argument failures.
- [x] RAG loops force final synthesis after repeated same tool/error/argument failures.
- [x] Isolated subagent workers have both iteration and repeated-error bounds.
- [x] Planning's existing consecutive-error behavior is preserved (untouched; new generic breaker is additive).
- [x] Shared prompt guidance prefers clear bound tools before `tool_search`.
- [x] `tool_search` remains available for missing, ambiguous, named-integration, and inventory requests.
- [x] No individual external tool names are hardcoded into retry policy (metadata `retry_safe`/`idempotent` + existing `TOOL_LOADING_TOOLS`).

## Rollout Notes

- Keep `tool_execution_consecutive_errors_limit` default at `3` to match Planning's current default behavior.
- Keep model-facing error summaries compact and stable; do not add stack traces or full exception text to ToolMessages.
- Prefer tool metadata such as `retry_safe` or `idempotent` for automatic retry eligibility. This allows future tools to opt in without central hardcoded name lists.
- If a future tool needs richer recovery semantics, add metadata on the tool, not prompt instructions for a specific tool name.
- If prompt size grows, remove prose before adding new rules. The model needs concise policy, not an exhaustive runtime manual.

## Execution Handoff

Plan complete and saved to `retry_tool_binding_fix.md`. Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

---

## Implementation Progress Log

Inline execution via `superpowers:executing-plans`. Each task follows the TDD loop (failing test → implement → verify pass) and is committed before moving on.

### Environment note (applies to all tasks)
- The app runtime is `.venv`; the system Python lacks pytest. Run tests with `.venv/Scripts/python.exe -m pytest ...`.
- An `rtk` shell hook rewrites/compresses command output, and it collapses pytest output to a misleading `Pytest: No tests collected`. To see real pytest results, prefix with `rtk proxy` (e.g. `rtk proxy .venv/Scripts/python.exe -m pytest ...`).

### Task 1 — Add Compact Tool Error Policy — DONE
- Created `app/ai/tool_error_policy.py` and `tests/test_tool_error_policy.py` exactly as specified.
- Verification: `tests/test_tool_error_policy.py` → 5 passed.
- Design decisions: none beyond the plan. Implementation matches the plan snippet verbatim. `ConnectionError` classifies as `network` (retryable) via the substring match, which is what the auto-retry test relies on.

### Task 2 — Enforce Timeout and Conservative Retries — DONE
- `app/ai/tool_execution.py`: added `_tool_execution_timeout_seconds()`, `_tool_execution_max_retries()`, and `invoke_tool_with_policy()`; the normal-execution branch now routes through the policy helper; the MCP reconnect branch wraps the post-reconnect invoke in `asyncio.wait_for(..., timeout=...)` and returns a compact `session` error on reconnect failure.
- Verification: `tests/test_tool_error_policy.py tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py` → 17 passed; `ruff check` clean on all touched files.
- Design decisions:
  - Introduced the shared local helper `_append_tool_error_output(...)` inside `execute_tool_calls()` now (Task 2) rather than Task 3. The plan endorses this helper ("Prefer a small local helper"); doing it once keeps the policy-error and reconnect-failure paths DRY and makes Task 3's pre-invocation error paths trivial.
  - Dropped the unused `as exc` bindings the plan snippet had on `except asyncio.TimeoutError` to keep the module lint-clean (the message is rebuilt from `timeout_seconds`).
  - No existing test asserted the old "MCP session lost…" string, so converting that path to compact JSON broke nothing.
  - Moved the appended test imports (`asyncio`, `json`, `settings`) to the top of `test_tool_execution_recovery.py` instead of mid-file to avoid E402 (the repo enforces `E` rules at line-length 100).
  - Left the outer generic `except Exception` handler in `execute_tool_calls` intact as a safety net for success-path post-processing; tool-call failures are now handled inside `invoke_tool_with_policy`.

### Task 3 — Resolve Pending Tool Calls Even When Binding Fails — DONE
- `app/ai/graph.py`: removed the `if not tool_map: return state` early exit in `_tool_node` and the `if not tool_map: return [], [], []` early exit in `_execute_agent_tool_calls`, so an empty execution map now yields one compact error ToolMessage per pending tool call.
- `app/ai/tool_execution.py`: converted the missing-name, missing-tool (client + non-client), and device-binding pre-invocation errors to compact JSON via the shared `_append_tool_error_output` helper.
- Verification: `tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py tests/test_graph_tool_budget.py` → 22 passed; `ruff check` clean.
- Design decision (deviation from plan, needed): the plan's `test_..._missing_tool_name_returns_compact_argument_error` assumed a nameless tool call yields a falsy `tool_name`, but `normalize_tool_call` substitutes the sentinel `"unknown"` when no name is provided (so the original `if not tool_name:` branch was effectively dead code). Changed the guard to `if not tool_name or tool_name == "unknown":` so an absent/blank name is correctly classified as an `argument` error. Risk is negligible — no real tool is named "unknown", and a tool call literally named "unknown" would have been a not-found anyway. Added a clarifying comment at the branch.

### Task 4 — Add Generic Repeated Tool Failure Breakers — DONE
- `app/core/config.py`: added `tool_execution_consecutive_errors_limit` (default 3) and registered it in the `_positive_int` validator (≥ 1).
- `app/ai/graph.py`: added `_tool_error_signature` (static) + `_update_tool_error_streak`; called the latter at the end of `_apply_tool_outputs_to_state` and `_rag_tools_node`; added force-final-synthesis breakers to `_route_tool_output` (graph ReAct) and `_should_continue_rag` (RAG loop).
- Verification: `tests/test_graph_tool_budget.py tests/test_rag_tool_loop_finalization.py` → 17 passed; `ruff check` clean on graph + tests; config E501 count unchanged at 42 (no new warnings); `settings.tool_execution_consecutive_errors_limit == 3`.
- Design decisions:
  - Wrapped the new config `description=` across lines to stay under the repo's 100-char E501 limit (existing config descriptions exceed it, but I avoid adding new violations).
  - The RAG-error artifacts in Task 4 carry `error_type="unknown"` (the `error_type`/`retryable` fields are populated in Task 5); the streak tracker tolerates this via its `or "unknown"` fallback.

### Task 5 — Bring RAG Document Tool Errors Into the Same Compact Shape — DONE
- `app/ai/rag_tool_actions.py`: added `compact_rag_tool_error(...)` (+ `json`/`tool_error_policy` imports); converted the validation/missing-target/invalid-argument returns of `execute_search_documents_action` (scan/read/search/grep/list/view_images no-arg cases, document-not-found, and unknown-action) to compact JSON.
- `app/ai/graph.py`: both `_rag_tools_node` and the isolated RAG worker now parse compact-JSON tool results and copy `error_type`/`retryable` onto the artifact.
- Verification: `tests/test_rag_tool_loop_finalization.py tests/test_rag_agent.py` → 44 passed; graph budget suite → 8 passed; `ruff check` clean.
- Design decisions / scoping:
  - Deliberately left genuine empty-state results (`"No search results found"`, `"No documents found in this conversation"`, `"No images found for document …"`) as plain text. They are not failures; rendering them as `status:error` could wrongly trip the new consecutive-error breaker on an empty corpus.
  - Left the generic `except Exception` catch-all (`"Error executing {action}: …"`) as-is. It is still detected as an error by the graph's `result.startswith("Error")` fallback; classifying it would require threading a timeout value and the `classify_tool_error` import into this module for marginal benefit.
  - "Invalid regex" mentioned in the plan is handled inside `rag_agent.grep_document` (not in this file's scope), so it surfaces via the catch-all rather than a dedicated compact branch here.

### Task 6 — Bound Subagent Worker Tool Loops — DONE
- `app/ai/graph.py` `_run_agent_in_isolated_context`: added an iteration cap and a repeated-same-error breaker to both the generic worker loop (`react_agent_max_iterations` cap) and the RAG worker loop (`agentic_max_iterations` cap). Both reuse `_tool_error_signature` and emit `pause_reason="worker_max_iterations"` or `"consecutive_tool_errors"`.
- Tests: added the 4 new tests and renamed the two `has_no_subagent_iteration_cap` tests to `allows_*_worker_under_iteration_cap` (assertions unchanged; their 3 iterations stay under the default caps of 50/10).
- Verification: `tests/test_graph_planning_subagents.py` → 41 passed; `tests/test_event_streaming_subagents.py` → 6 passed; `ruff check` clean.
- Process note: the three new "stops_after_*" tests use infinitely-looping fake agents, so before the cap exists they HANG rather than fail an assertion. I confirmed the missing-cap state by observing the hang (killed the run), then implemented the caps and re-ran — they now terminate and pass. (Ran pytest under a `timeout` guard to avoid runaway loops during iteration.)
- Design decision: the RAG worker tracks per-iteration errors via a `rag_iteration_start = len(accumulated_artifacts)` slice (the RAG loop appends artifacts one-per-tool-call rather than returning a single per-iteration list like the generic worker), then checks `accumulated_artifacts[rag_iteration_start:]`.

### Task 7 — Rebalance Tool Search Prompting Toward Bound Tools First — DONE
- `app/ai/prompts.py`: replaced `TOOL_EXPLORATION_SUFFIX` with bound-tool-first guidance (kept inventory, named-integration, server-identifier, recommended-tool, and runtime-scoping rules); added a compact-error-handling bullet to `TOOL_CONTEXT_SUFFIX`.
- `app/ai/tool_search_tool.py`: replaced the "Do not skip discovery…" wording in both descriptions with bound-tool-first phrasing.
- Tests: rewrote the 4 affected tests + docstring; the other 7 invariants still hold against the new suffix.
- Verification: `tests/test_tool_search_prompt_guidance.py tests/test_unified_tool_search.py tests/test_tool_search_scoring.py` → 37 passed; `ruff check` clean.
- Design decisions:
  - Replaced the full two-sentence discovery block in the tool descriptions (not just the single "Do not skip discovery" sentence) to avoid leaving contradictory "call tool_search first for real environment actions" guidance next to the new bound-first message.
  - Debug note: my first reflow wrapped "not currently\nbound" across a docstring line, breaking the `"not currently bound"` substring assertion. Reflowed so the phrase stays contiguous on one line.

### Task 8 — Update Documentation and Contract Claims — DONE
- `README.md`: updated the tool-execution architecture bullet (bounded timeout + conservative retry + compact error summaries) and the deferred tool search section (bound-tool-first, with `tool_search` for missing/ambiguous/unknown-integration capability).
- Verification: grep for `forces discovery|force discovery|Do not skip discovery|before giving a text-only` across `README.md app tests` → no matches. Final focused regression suite (9 files: tool_error_policy, tool_execution_recovery, tool_execution_rendering, graph_tool_budget, rag_tool_loop_finalization, graph_planning_subagents, tool_search_prompt_guidance, unified_tool_search, tool_search_scoring) → **116 passed**.
- Note: `rg` is not on PATH in this shell; used the harness Grep (ripgrep-backed) for the doc audit.

### Final Verification — DONE
- Focused regression suite (9 plan-listed files) → **116 passed**.
- Full suite (`tests/`) → **1282 passed, 1 failed**. The single failure is `tests/client_backend/test_live_server_integration.py::test_live_document_upload_list_get_task_and_delete_flow` (`KeyError: 'document'` at line 472).
  - Confirmed pre-existing/environmental: checked out the pre-work base commit `d8f571e` and the same test failed identically there. It exercises the document-upload/indexing subsystem, which this plan never touched.
- `ruff check` clean on every file changed by this plan; `app/core/config.py` E501 count unchanged at 42 (all pre-existing baseline).
- Follow-up commit `style: wrap long lines in tool error policy hints` — the Task 2 E501 wrap of `tool_error_policy.py` had not been staged in the Task 2 commit; committed separately so the repo's committed state is lint-clean.

### Commits (branch `Thai-Postgre-FastAPI`, not pushed)
1. `feat: add compact tool error policy`
2. `feat: enforce compact tool execution retry policy`
3. `fix: resolve pending tool calls on empty tool maps`
4. `feat: add generic repeated tool failure breaker`
5. `feat: normalize rag tool error summaries`
6. `fix: bound isolated worker tool loops`
7. `fix: prefer bound tools before deferred discovery`
8. `docs: document resilient tool execution policy`
9. `style: wrap long lines in tool error policy hints`

### Post-Implementation Review Follow-Ups

Verification on 2026-06-29 confirms the main retry/tool-binding refactor works, but there are three follow-up items before calling this fully production-polished:

1. `app/ai/rag_tool_actions.py`: the broad `except Exception` fallback still returns legacy free-form text (`Error executing {action}: ...`). Normal validation paths are compact JSON, but unexpected RAG helper exceptions should also use `compact_rag_tool_error(...)` so the model and loop breaker receive `status`, `error_type`, `retryable`, and `hint`.
2. `app/ai/tool_execution.py`: if MCP reconnect succeeds but the post-reconnect retry fails, the final compact error is classified from the original session exception rather than the retry exception. Preserve the final retry exception and build the model/artifact error from that failure so diagnostics match the actual terminal failure.
3. `app/ai/tool_search_tool.py` / `app/core/config.py`: normal discovery output is compact enough by default (`top_k=3`, descriptions truncated, match reasons capped), but inventory mode can still be noisy in real MCP environments (`inventory_default_top_k=20`, max 50). Reduce the default inventory size to around 8-10 or add a compact/default output mode that returns `recommended_tool` plus top candidates, with match reasons only when refinement is required or debug scoring is enabled.

These are follow-ups, not blockers for the implemented retry surface: the focused regression suite still passes (`116 passed`), and the full-suite blocker remains the pre-existing live document upload contract failure.
