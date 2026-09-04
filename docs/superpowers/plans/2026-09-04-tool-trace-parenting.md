# LangSmith API Migration and Tool Trace Parenting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove LangSmith's January 2027 legacy query dependency and keep every conversation-owned MCP/provider invocation beneath its active LangChain/LangSmith tool run instead of creating unrelated root traces.

**Architecture:** First migrate the repository's RAG evaluation reads from the legacy `Client.get_test_results()` → `Client.list_runs()` path to the async SmithDB-backed `Client.runs.query()` API and raise the LangSmith SDK floor. Then carry the active `RunnableConfig` from `ToolCallRequest.runtime.config` through the product execution pipeline and into every nested `ainvoke`. The product web/image tools created by the focused-web plan accept injected `ToolRuntime` and forward its config to child tools; bounded payloads stay at that boundary.

**Tech Stack:** Python 3.10+, LangSmith Python SDK >=0.10.15, SmithDB v2 run queries, LangChain `RunnableConfig`, LangChain middleware and callbacks, LangGraph `ToolRuntime`, pytest/pytest-asyncio.

## Global Constraints

- Preserve the existing tool authorization, approval, receipt, retry, timeout, offload, artifact, and image ordering.
- Never place credentials, unrestricted prompts, full page bodies, user IDs, conversation IDs, generation IDs, or blob IDs in metric labels.
- A conversation-owned Tavily or Brave call must not appear as a root trace.
- Direct administrative provider tests may remain roots only when explicitly tagged diagnostic.
- Do not change model-visible tool output in this plan.
- Keep every new parameter optional outside framework middleware so legacy tests and non-graph callers remain compatible.
- Do not call `Client.list_runs()`, `Client.get_test_results()`, or `POST /api/v1/runs/query`; these sunset in LangSmith Cloud on 2027-01-31.
- Do not enable OpenTelemetry as a response to the legacy banner. Trace ingestion (`/runs/multipart`) is not one of the endpoint families named in this deprecation.
- SmithDB `runs.query()` returns only the last 24 hours by default; every experiment query must pass the experiment project's `start_time` as `min_start_time`.

---

## File Structure

- Create `app/evaluation/rag/langsmith_queries.py`: query experiment root runs through SmithDB and aggregate feedback.
- Modify `scripts/evaluate_rag.py`: await the new metrics query instead of calling `get_test_results()`.
- Modify `pyproject.toml` and `environment.yml`: require a SmithDB-capable LangSmith SDK.
- Create `tests/test_langsmith_smithdb_migration.py`: lock down v2 query arguments and forbid legacy methods.
- Modify `tests/test_rag_evaluation_cli.py`: exercise async comparison metrics without network access.
- Modify `app/ai/tool_execution.py`: thread `RunnableConfig` through generic tool invocation and retry layers.
- Modify `app/ai/workflow/middleware.py`: take the active config from `ToolCallRequest.runtime.config` and hand it to the executor.
- Modify `app/ai/web_tools.py`: inject `ToolRuntime` and forward the parent config to Tavily and image discovery.
- Modify `app/ai/image_discovery_flow.py`: forward the config into Brave invocation.
- Modify `tests/test_tool_execution_control_flow.py`: lock down generic nested callback ancestry and retry propagation.
- Modify `tests/test_specialist_middleware.py`: verify middleware passes the runtime config without changing pipeline ordering.
- Modify `tests/test_web_tools.py`: verify Tavily and Brave receive the same config.
- Create `tests/test_tool_trace_parenting.py`: callback-level integration test proving nested calls have a parent run.

## Research Finding: What the LangSmith Banner Means

The banner's exact 2027-01-31 date matches LangSmith's SmithDB migration, not
trace ingestion. LangSmith marks the v1 runs query/retrieve endpoints, legacy
dataset experiment-run endpoint, sharing/public-read endpoints, and annotation
queue run endpoints with `Deprecation: true` and that sunset date. The
repository has one matching call site: `scripts/evaluate_rag.py` calls
`Client.get_test_results()`, and LangSmith 0.10.9 implements that method by
calling `Client.list_runs()`, which uses `POST /api/v1/runs/query`.

The current environment pins `langsmith==0.10.9`; the official SmithDB guide
requires Python SDK `langsmith>=0.10.15`. Ordinary trace writes through the
LangChain callback integration use the ingestion API and do not themselves
explain this warning.

Primary references:

- [SmithDB SDK migration overview](https://docs.langchain.com/langsmith/smithdb-sdk-migration)
- [Run query migration: `list_runs` to `runs.query`](https://docs.langchain.com/langsmith/smithdb-sdk-migration-query-runs)
- [Experiment-run migration](https://docs.langchain.com/langsmith/smithdb-sdk-migration-experiments)
- [LangSmith deprecation policy](https://docs.langchain.com/langsmith/endpoint-deprecation)
- [Cloud changelog entry naming the 2027-01-31 endpoint sunset](https://docs.langchain.com/langsmith/changelog)

### Task 1: Migrate RAG Evaluation Reads to SmithDB

**Files:**
- Create: `app/evaluation/rag/langsmith_queries.py`
- Modify: `scripts/evaluate_rag.py:145-250`
- Modify: `pyproject.toml:47-57`
- Modify: `environment.yml:195-205`
- Create: `tests/test_langsmith_smithdb_migration.py`
- Modify: `tests/test_rag_evaluation_cli.py:105-135`

**Interfaces:**
- Consumes: `Client.aread_project(project_name=..., include_stats=True)` and `Client.runs.query(project_ids=..., is_root=True, min_start_time=..., selects=...)`.
- Produces: `async def experiment_metrics(client: Any, experiment_name: str) -> dict[str, float]` and `async def comparison_metrics(client: Any, candidate_name: str, baseline_name: str) -> tuple[dict[str, float], dict[str, float]]`.

- [ ] **Step 1: Write failing SDK-floor and query-contract tests**

Create a fake async runs resource that records the v2 query:

```python
class FakeRuns:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)

        async def iterate():
            for row in self.rows:
                yield row

        return iterate()


@pytest.mark.asyncio
async def test_experiment_metrics_use_smithdb_v2_with_full_time_window():
    started = datetime(2025, 1, 2, tzinfo=timezone.utc)
    project = SimpleNamespace(
        id=uuid4(),
        start_time=started,
        feedback_stats={"groundedness": {"avg": 0.8}},
        session_feedback_stats={"abstention_recall": {"avg": 0.7}},
    )
    runs = FakeRuns([
        SimpleNamespace(feedback_stats={"document_recall_at_5": {"avg": 1.0}}),
        SimpleNamespace(feedback_stats={"document_recall_at_5": {"avg": 0.5}}),
    ])
    client = FakeClient(project=project, runs=runs)

    metrics = await experiment_metrics(client, "baseline")

    assert metrics == {
        "document_recall_at_5": 0.75,
        "groundedness": 0.8,
        "abstention_recall": 0.7,
    }
    assert runs.calls == [{
        "project_ids": [str(project.id)],
        "is_root": True,
        "min_start_time": started,
        "selects": ["ID", "FEEDBACK_STATS"],
    }]
```

Add a dependency test using `importlib.metadata.version` and
`packaging.version.Version` that requires `langsmith>=0.10.15`. Add a source
inventory test that scans production Python files under `app/` and `scripts/`
and fails on `.list_runs(`, `.get_test_results(`, `.get_experiment_results(`,
or literal `/api/v1/runs/query`.

- [ ] **Step 2: Run the tests and verify the expected failures**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_langsmith_smithdb_migration.py tests/test_rag_evaluation_cli.py
```

Expected: the new module is missing, the installed SDK floor test reports
0.10.9, and the inventory test identifies `get_test_results()`.

- [ ] **Step 3: Raise the SDK floor with a narrow environment pin**

Set:

```toml
"langsmith>=0.10.15,<1.0.0",
```

in `pyproject.toml`, and update the reproducible environment from
`langsmith==0.10.9` to `langsmith==0.10.18`, the latest 0.10 patch available at
plan authoring time. Rebuild the environment before running the green test;
do not depend on the current virtual environment silently satisfying the new
metadata.

- [ ] **Step 4: Implement the async SmithDB metrics query**

```python
async def experiment_metrics(client: Any, experiment_name: str) -> dict[str, float]:
    project = await client.aread_project(
        project_name=experiment_name,
        include_stats=True,
    )
    totals: dict[str, list[float]] = {}
    async for run in client.runs.query(
        project_ids=[str(project.id)],
        is_root=True,
        min_start_time=project.start_time,
        selects=["ID", "FEEDBACK_STATS"],
    ):
        for key, stats in (run.feedback_stats or {}).items():
            if isinstance(stats, dict) and stats.get("avg") is not None:
                totals.setdefault(key, []).append(float(stats["avg"]))

    metrics = {key: sum(values) / len(values) for key, values in totals.items()}
    for source in (project.feedback_stats or {}, project.session_feedback_stats or {}):
        for key, stats in source.items():
            if isinstance(stats, dict) and stats.get("avg") is not None:
                metrics[key] = float(stats["avg"])
    if not metrics:
        raise ValueError(
            f"baseline experiment has no deterministic feedback: {experiment_name}"
        )
    return metrics


async def comparison_metrics(
    client: Any,
    candidate_name: str,
    baseline_name: str,
) -> tuple[dict[str, float], dict[str, float]]:
    candidate, baseline = await asyncio.gather(
        experiment_metrics(client, candidate_name),
        experiment_metrics(client, baseline_name),
    )
    return candidate, baseline
```

In `scripts/evaluate_rag.py`, replace both synchronous calls with one
`asyncio.run(comparison_metrics(...))`. Keep `list_examples`, `evaluate`, and
`read_project`/`aread_project`: the SmithDB migration guide does not deprecate
those dataset-write/project-lookup paths.

- [ ] **Step 5: Run focused tests and the complete offline evaluation**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_langsmith_smithdb_migration.py tests/test_rag_evaluation_cli.py tests/test_rag_evaluation_metrics.py
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --offline
.\.venv\Scripts\python.exe -m ruff check app/evaluation/rag/langsmith_queries.py scripts/evaluate_rag.py tests/test_langsmith_smithdb_migration.py tests/test_rag_evaluation_cli.py
```

Expected: all tests pass, the offline evaluation exits zero, and the inventory
test finds no legacy query methods/endpoints.

- [ ] **Step 6: Run an authenticated deprecation canary**

In a non-production LangSmith project, run one online RAG evaluation and one
baseline comparison with a dedicated API key. Confirm egress/debug logs contain
`POST /api/v2/runs/query` and no `POST /api/v1/runs/query`. Check that responses
do not carry `Deprecation: true` or `Sunset: 2027-01-31`. Because the UI warning
is workspace-wide and may use a lookback window, a banner that remains after
this canary means another service/API key is still calling one of the endpoint
families listed in the changelog; identify that caller rather than changing
trace ingestion.

- [ ] **Step 7: Commit Task 1**

```powershell
git add app/evaluation/rag/langsmith_queries.py scripts/evaluate_rag.py pyproject.toml environment.yml tests/test_langsmith_smithdb_migration.py tests/test_rag_evaluation_cli.py
git commit -m "fix: migrate LangSmith evaluation reads to SmithDB"
```

### Task 2: Carry RunnableConfig Through the Generic Tool Pipeline

**Files:**
- Modify: `app/ai/tool_execution.py:1371-1635`
- Modify: `app/ai/workflow/middleware.py:387-470`
- Test: `tests/test_tool_execution_control_flow.py`
- Test: `tests/test_specialist_middleware.py`

**Interfaces:**
- Consumes: `request.runtime.config: RunnableConfig` from LangChain `ToolCallRequest`.
- Produces: `execute_tool_calls(..., runnable_config: RunnableConfig | None = None)` and matching optional keyword parameters on `invoke_tool`, `invoke_tool_attempt`, and `invoke_tool_with_policy`.

- [ ] **Step 1: Write failing tests for config propagation**

Add a config-aware fake tool and assert the exact callback object reaches every retry:

```python
class ConfigSpyTool:
    name = "config_spy"
    coroutine = True

    def __init__(self):
        self.configs = []

    async def ainvoke(self, args, config=None):
        self.configs.append(config)
        return "ok"


@pytest.mark.asyncio
async def test_execute_tool_calls_forwards_runnable_config():
    callback = object()
    config = {"callbacks": [callback], "tags": ["conversation-tool"]}
    tool = ConfigSpyTool()

    outputs, _, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": tool.name, "args": {}}],
        tool_map={tool.name: tool},
        runnable_config=config,
    )

    assert outputs[0]["content"] == "ok"
    assert tool.configs == [config]
```

In `tests/test_specialist_middleware.py`, construct a `ToolCallRequest` double whose `runtime.config` is the same dictionary and monkeypatch `execute_tool_calls`; assert the captured `runnable_config is runtime.config`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_execution_control_flow.py tests/test_specialist_middleware.py
```

Expected: the new tests fail because `execute_tool_calls` has no `runnable_config` parameter and middleware does not pass it.

- [ ] **Step 3: Add optional config parameters without changing call ordering**

Use `RunnableConfig` from `langchain_core.runnables` and thread it through the call chain:

```python
async def invoke_tool(
    tool: Any,
    tool_args: Any,
    *,
    runnable_config: RunnableConfig | None = None,
) -> Any:
    if getattr(tool, "coroutine", None):
        return await tool.ainvoke(tool_args, config=runnable_config)
    ainvoke = getattr(tool, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(tool_args, config=runnable_config)
    invoke = getattr(tool, "invoke", None)
    if callable(invoke):
        return await asyncio.to_thread(invoke, tool_args, config=runnable_config)
    if callable(tool):
        return await asyncio.to_thread(tool, tool_args)
    raise TypeError("Tool has no invoke/ainvoke and is not callable")
```

Add the same optional keyword to `invoke_tool_attempt`, `invoke_tool_with_policy`, and `execute_tool_calls`, and pass it on every attempt, including an MCP reconnect retry. Do not pass config to a plain Python callable because that is not part of its interface.

- [ ] **Step 4: Pass middleware runtime config into the executor**

In `ToolExecutionMiddleware.awrap_tool_call`, derive the config defensively and pass it to `_execute`:

```python
runtime = getattr(request, "runtime", None)
runnable_config = getattr(runtime, "config", None)
return await self._execute(call, tool_map, runnable_config=runnable_config)
```

Update `_execute`:

```python
async def _execute(
    self,
    call: dict[str, Any],
    tool_map: dict[str, Any],
    *,
    runnable_config: RunnableConfig | None,
) -> ToolMessage:
    self._scope.worker_event("start", call)
    with self._scope.execution_context():
        outputs, artifacts, images = await execute_tool_calls(
            tool_calls=[call],
            tool_map=tool_map,
            capture_images=True,
            device_id=self._scope.device_id,
            agent=self._scope.agent,
            conversation_id=self._scope.conversation_id,
            user_id=self._scope.user_id,
            runnable_config=runnable_config,
        )

    self.artifacts.extend(artifacts)
    self.images.extend(images)
    output = outputs[0] if outputs else {}
    status = "error" if _is_error(artifacts) else "success"
    self._scope.worker_event("end", call, status=status)
    return ToolMessage(
        content=str(output.get("content") or ""),
        tool_call_id=str(output.get("tool_call_id") or call.get("id") or ""),
        name=str(output.get("name") or call.get("name") or "tool"),
        status=status,
    )
```

Derive `runnable_config` once at the top of `awrap_tool_call`. Pass it to
`_execute` both in the non-mutation branch and inside the receipt callback's
`invoke` closure; receipt handling must still wrap the actual invocation.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_execution_control_flow.py tests/test_tool_execution_recovery.py tests/test_tool_execution_policy.py tests/test_specialist_middleware.py tests/test_specialist_tool_pipeline.py
```

Expected: all pass; config reaches the initial call and every retry.

- [ ] **Step 6: Commit Task 2**

```powershell
git add app/ai/tool_execution.py app/ai/workflow/middleware.py tests/test_tool_execution_control_flow.py tests/test_specialist_middleware.py
git commit -m "fix: propagate tracing config through tool execution"
```

### Task 3: Parent Product Web and Image Provider Calls

**Files:**
- Modify: `app/ai/web_tools.py`
- Modify: `app/ai/image_discovery_flow.py:215-245`
- Test: `tests/test_web_tools.py`

**Interfaces:**
- Consumes: LangGraph-injected `runtime: ToolRuntime` on the `web_search`, `web_open`, and `image_search` coroutines.
- Produces: `_run_search(..., runnable_config: RunnableConfig | None)` and `discover_images(..., runnable_config: RunnableConfig | None = None)`.

- [ ] **Step 1: Make the web fake tools record config**

Change the test fake without weakening existing argument assertions:

```python
class _FakeTool:
    def __init__(self, name, payload, delay=0):
        self.name = name
        self.payload = payload
        self.delay = delay
        self.calls = []
        self.configs = []

    async def ainvoke(self, args, config=None):
        self.calls.append(dict(args))
        self.configs.append(config)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.payload
```

Add tests that invoke each product tool with `config={"tags": ["parent"]}` and assert Tavily search, Tavily extract, and Brave receive a config containing that tag.

- [ ] **Step 2: Run the web tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_tools.py
```

Expected: the new config assertions fail because nested calls currently use `tool.ainvoke(args)`.

- [ ] **Step 3: Inject ToolRuntime and forward config to both child tasks**

Import `ToolRuntime` and annotate the injected argument so it is excluded from the public tool schema:

```python
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig

async def _search(
    query: str,
    objective: str,
    freshness: Freshness = "timeless",
    start_date: date | None = None,
    end_date: date | None = None,
    locale: str | None = None,
    include_domains: list[str] | None = None,
    max_results: int = 5,
    runtime: ToolRuntime | None = None,
) -> str:
    runnable_config = runtime.config if runtime is not None else None
```

Apply the same injected argument to `_open` and `_image_search`. Pass `runnable_config` into provider helpers and `discover_images`. Invoke child tools with:

```python
await tool.ainvoke(args, config=runnable_config)
```

Keep `asyncio.create_task` usage; context propagation and the explicit config both preserve ancestry.

- [ ] **Step 4: Preserve direct-test compatibility**

The `runtime` argument must remain optional so direct helper tests and legacy invocations without graph injection still work. Assert none of the three public input schemas contains `runtime`.

- [ ] **Step 5: Run composite and image regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_tools.py tests/test_brave_image_search_server.py tests/test_image_preview_stream.py tests/test_rich_response_streaming.py
```

Expected: all pass; provider calls receive the parent config and cancellation tests remain green.

- [ ] **Step 6: Commit Task 3**

```powershell
git add app/ai/web_tools.py app/ai/image_discovery_flow.py tests/test_web_tools.py
git commit -m "fix: parent nested web provider traces"
```

### Task 4: Prove Trace Ancestry at the Callback Boundary

**Files:**
- Create: `tests/test_tool_trace_parenting.py`
- Modify: `docs/operations/routing-v2-rollout.md`

**Interfaces:**
- Consumes: the config-aware execution pipeline from Tasks 2-3.
- Produces: a deterministic regression test and a live LangSmith verification procedure.

- [ ] **Step 1: Add a callback ancestry integration test**

Use a real `StructuredTool` and a recording callback:

```python
class RecordingHandler(BaseCallbackHandler):
    def __init__(self):
        self.starts = []

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self.starts.append((run_id, parent_run_id, serialized.get("name")))


@pytest.mark.asyncio
async def test_nested_tool_run_has_parent_callback_id():
    @tool
    async def child_lookup(query: str) -> str:
        """Return one bounded lookup result."""
        return "result"

    handler = RecordingHandler()
    parent = RunnableLambda(
        lambda value: value,
        name="conversation_parent",
    )

    async def invoke_child(value, config):
        return await child_lookup.ainvoke({"query": value}, config=config)

    chain = parent | RunnableLambda(invoke_child, name="tool_dispatch")
    await chain.ainvoke("query", config={"callbacks": [handler]})

    child = next(item for item in handler.starts if item[2] == "child_lookup")
    assert child[1] is not None
```

Add a second test around `execute_tool_calls` using the same handler to prove the production seam, not only native LangChain composition.

- [ ] **Step 2: Run the new integration test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_trace_parenting.py
```

Expected: PASS after Tasks 2-3.

- [ ] **Step 3: Document live trace verification**

Add this canary check to the rollout guide:

```markdown
1. Start one chat turn that calls `web_search`, `web_open`, and `image_search`.
2. Open the conversation trace in LangSmith.
3. Verify Tavily and Brave runs have non-null parent IDs beneath their product tools.
4. Query the same time window for root runs named `tavily_search`,
   `tavily_extract`, or `brave_image_search`; expect zero conversation-owned roots.
5. Diagnostic roots are acceptable only with the `diagnostic` tag.
```

- [ ] **Step 4: Run trace, streaming, and middleware regression suites**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_langsmith_smithdb_migration.py tests/test_rag_evaluation_cli.py tests/test_tool_trace_parenting.py tests/test_tool_execution_control_flow.py tests/test_tool_execution_recovery.py tests/test_specialist_middleware.py tests/test_specialist_tool_pipeline.py tests/test_web_tools.py tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py
.\.venv\Scripts\python.exe -m ruff check app/ai/tool_execution.py app/ai/workflow/middleware.py app/ai/web_tools.py app/ai/image_discovery_flow.py tests/test_tool_trace_parenting.py tests/test_tool_execution_control_flow.py tests/test_specialist_middleware.py tests/test_web_tools.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit Task 4**

```powershell
git add tests/test_tool_trace_parenting.py docs/operations/routing-v2-rollout.md
git commit -m "test: lock down child tool trace ancestry"
```

## Acceptance Checklist

- [ ] Runtime and environment dependency declarations satisfy `langsmith>=0.10.15`.
- [ ] Production Python contains no legacy run-query method or `/api/v1/runs/query` literal.
- [ ] RAG comparison metrics use `runs.query()` with project UUID, root filter, explicit selects, and the project's full time window.
- [ ] An authenticated RAG evaluation canary emits no deprecated response header or v1 run-query request.
- [ ] `ToolCallRequest.runtime.config` reaches the raw tool on every attempt.
- [ ] Nested Tavily and Brave invocations receive the same config.
- [ ] Existing cancellation, timeout, retry, receipt, artifact, and image tests pass.
- [ ] Callback tests observe a non-null parent for conversation-owned child tools.
- [ ] The live LangSmith canary contains no conversation-owned provider root runs.
- [ ] No model-visible output or public stream contract changed.

## Execution Handoff

Task 1 can execute immediately and independently because it touches only the
evaluation/read side of LangSmith. Execute Tasks 2-4 after the focused-web plan
so trace parenting targets the durable product-tool boundaries. The plan remains
independent of generation lifecycle persistence and can merge before the
Continue/Stop plan.
