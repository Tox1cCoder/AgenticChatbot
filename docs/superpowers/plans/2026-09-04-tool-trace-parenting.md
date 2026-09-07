# LangSmith API Migration and Tool Trace Parenting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove LangSmith's January 2027 legacy query dependency and keep every conversation-owned MCP/provider invocation beneath its active LangChain/LangSmith tool run instead of creating unrelated root traces.

**Architecture:** First migrate the repository's RAG evaluation reads from the legacy `Client.get_test_results()` → `Client.list_runs()` path to the async SmithDB-backed `Client.runs.query()` API and raise the LangSmith SDK floor. Then establish, as tested contract, that the ambient `contextvars` run context already parents every nested MCP/provider invocation beneath its product tool — and that explicitly forwarding an upstream `RunnableConfig` is what would break it.

> **Revised 2026-09-07 after the three-plan review.** Tasks 2-4 originally
> threaded `ToolCallRequest.runtime.config` through the pipeline and injected
> `ToolRuntime` into the product tools. Both mechanisms were measured against
> the production seam and rejected: injection never fires on this path because
> `ToolExecutionMiddleware` bypasses the framework tool handler, and forwarding
> the upstream config demotes each provider run from *child of its product tool*
> to *sibling of it*. Task 2 records the measurements. Task 1 has landed and is
> amended here for the SDK's feedback-statistics type.

**Tech Stack:** Python 3.10+, LangSmith Python SDK >=0.10.15, SmithDB v2 run queries, `langchain_core` callback managers and `var_child_runnable_config` context propagation, LangChain middleware, pytest/pytest-asyncio.

## Global Constraints

- Preserve the existing tool authorization, approval, receipt, retry, timeout, offload, artifact, and image ordering.
- Never place credentials, unrestricted prompts, full page bodies, user IDs, conversation IDs, generation IDs, or blob IDs in metric labels.
- A conversation-owned Tavily or Brave call must not appear as a root trace.
- Direct administrative provider tests may remain roots only when explicitly tagged diagnostic.
- Do not change model-visible tool output in this plan.
- Add no new parameter to the tool execution pipeline: the ancestry this plan protects comes from the ambient context, and a config parameter is the thing that overrides it.
- Assert exact ancestry (`provider.parent_run_id == product.run_id`). A non-null parent is also true of the wrong, flattened topology.
- Do not call any client method the installed SDK marks deprecated, nor the supported helpers that reach v1 behind a suppressed warning (`get_test_results`, `read_thread`, `run_is_shared`), nor any raw `/api/v1/` runs, sharing, annotation-queue-run, or dataset experiment-run path. All sunset in LangSmith Cloud on 2027-01-31. The ban list is derived from the SDK, not written down.
- Do not enable OpenTelemetry as a response to the legacy banner. Trace ingestion (`/runs/multipart`) is not one of the endpoint families named in this deprecation.
- SmithDB `runs.query()` returns only the last 24 hours by default; every experiment query must pass the experiment project's `start_time` as `min_start_time`.

---

## File Structure

- Create `app/evaluation/rag/langsmith_queries.py`: query experiment root runs through SmithDB and aggregate feedback.
- Modify `scripts/evaluate_rag.py`: await the new metrics query instead of calling `get_test_results()`. Its other client calls — `list_examples()`, `evaluate()` with a list of examples, `aread_project()` — are not on a sunset path.
- Modify `pyproject.toml` and `environment.yml`: require a SmithDB-capable LangSmith SDK.
- Create `tests/test_langsmith_smithdb_migration.py`: lock down v2 query arguments and forbid legacy methods.
- Modify `tests/test_rag_evaluation_cli.py`: exercise async comparison metrics without network access.
- Modify `app/ai/tool_execution.py`: comment only, recording why `invoke_tool` takes no `config`.
- Create `tests/test_tool_trace_parenting.py`: callback-level integration tests proving each nested provider run is a child of its product tool, across the async path, the synchronous path and a retry.
- Modify `docs/operations/routing-v2-rollout.md`: the live LangSmith canary.
- Unchanged: `app/ai/workflow/middleware.py`, `app/ai/web_tools.py`, `app/ai/image_discovery_flow.py`, `tests/test_web_tools.py`, `tests/test_specialist_middleware.py`.

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

### Completed 2026-09-07: what "one matching call site" missed

The paragraph above is right about the families and wrong about the scope of
the guard. Audited against installed `langsmith==0.10.18`:

- **Seventeen client methods carry the SDK's own deprecation marker**, each
  naming its replacement and the Jan 31 2027 date: `list_runs`, `read_run`,
  `read_thread`, `list_threads`, `get_run_url`, `share_run`, `unshare_run`,
  `read_run_shared_link`, `read_shared_run`, `list_shared_runs`,
  `evaluate_run`, `aevaluate_run`, `get_experiment_results` on `Client`, plus
  four repeats on `AsyncClient`. The original guard listed four string markers
  and so covered three of them.
- **`get_test_results()` is no longer marked deprecated in 0.10.18** — it
  branches on `client.info.instance_flags` and calls `list_runs()` inside
  `suppress_deprecation_warning()` unless the workspace is SmithDB-only. It
  therefore reaches `POST /api/v1/runs/query` while emitting nothing at all.
  Two other methods do the same: `read_thread` and `run_is_shared`. This is the
  dangerous shape, and it is why the guard now derives its ban list from the
  SDK rather than from prose: a helper can look clean, warn about nothing, and
  still be on the sunset path.
- **The evaluation helpers are safe on the path this repository uses.**
  `_load_traces_for_experiment()` also falls back to `list_runs()`, but it is
  reached only from `evaluate_existing()` and comparative evaluation.
  `scripts/evaluate_rag.py` calls `client.evaluate(target, data=examples, ...)`
  with a list of examples, which does not touch it. `list_examples()` and
  `aread_project()` are not deprecated.
- **`client.runs` requires backend >= 0.16.0**, but `_check_backend_version`
  only logs a warning — an older workspace degrades loudly, it does not raise.

Two defects in the landed `runs.query()` call, both from the guide:

- `min_start_time` defaults to **one day ago**, and `start_time` is optional on
  the project schema. Passing it through unchecked turned an older experiment
  into zero runs, reported as "no deterministic feedback" rather than as the
  truncation it was. Now refused before the query is made.
- `page_size` defaults to 100 (max 1000). The async iterator pages on its own,
  so this was a round-trip count rather than a correctness bug: three requests
  for the 210-case golden dataset, now one.

Verified against the guide: `FEEDBACK_STATS` is a valid `selects` value,
`async for` auto-paginates through `AsyncItemsCursorPostPagination`, and
`project_ids` takes UUID strings.

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

- [x] **Step 1: Write failing SDK-floor and query-contract tests**

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

> **Amended 2026-09-07.** The fake above returns `SimpleNamespace` rows with
> plain dictionaries. The installed SDK does not: `Client.runs.query()` yields
> SmithDB `Run` models whose `feedback_stats` values are `FeedbackStats`
> instances, while `aread_project` returns project statistics as dictionaries.
> An aggregator written against `isinstance(stats, Mapping)` therefore drops
> every per-root-run metric in production while passing every test. Build the
> run rows with `langsmith._openapi_client.types.run.Run`, and read each entry
> through a helper that accepts a mapping key or an `avg` attribute.

Add a dependency test using `importlib.metadata.version` and
`packaging.version.Version` that requires `langsmith>=0.10.15`. Add a source
inventory test that scans production Python files under `app/` and `scripts/`.

**Amended 2026-09-07:** do not hardcode the markers. Derive the ban list from
the installed SDK — `langsmith._internal._beta_decorator.deprecated` produces
one shared code object per decoration, so identity against a probe names every
deprecated method exactly, and the failure message can quote the SDK's own
replacement guidance. Add to it the methods whose own source calls
`suppress_deprecation_warning`, which are supported, silent, and still on v1.
Keep a separate scan for raw `/api/v1/` path literals, anchored so it cannot
also match the `/api/v2/runs/query` replacement.

- [x] **Step 2: Run the tests and verify the expected failures**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_langsmith_smithdb_migration.py tests/test_rag_evaluation_cli.py
```

Expected: the new module is missing, the installed SDK floor test reports
0.10.9, and the inventory test identifies `get_test_results()`.

- [x] **Step 3: Raise the SDK floor with a narrow environment pin**

Set:

```toml
"langsmith>=0.10.15,<1.0.0",
```

in `pyproject.toml`, and update the reproducible environment from
`langsmith==0.10.9` to `langsmith==0.10.18`, the latest 0.10 patch available at
plan authoring time. Rebuild the environment before running the green test;
do not depend on the current virtual environment silently satisfying the new
metadata.

- [x] **Step 4: Implement the async SmithDB metrics query**

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
        for key, average in _averages(getattr(run, "feedback_stats", None)).items():
            totals.setdefault(key, []).append(average)

    metrics = {key: sum(values) / len(values) for key, values in totals.items()}
    metrics.update(_averages(getattr(project, "feedback_stats", None)))
    metrics.update(_averages(getattr(project, "session_feedback_stats", None)))
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

- [x] **Step 5: Run focused tests and the complete offline evaluation**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_langsmith_smithdb_migration.py tests/test_rag_evaluation_cli.py tests/test_rag_evaluation_metrics.py
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --offline
.\.venv\Scripts\python.exe -m ruff check app/evaluation/rag/langsmith_queries.py scripts/evaluate_rag.py tests/test_langsmith_smithdb_migration.py tests/test_rag_evaluation_cli.py
```

Expected: all tests pass and the inventory test finds no legacy query methods
or endpoints. The offline evaluation exits **1**, not zero: the repository has
no configured local RAG target, so `--offline` without `--target` is a refusal
that `tests/test_rag_evaluation_cli.py` asserts. Pass `--target MODULE:FUNCTION`
to actually run one.

- [ ] **Step 6: Run an authenticated deprecation canary**

In a non-production LangSmith project, run one online RAG evaluation and one
baseline comparison with a dedicated API key. Confirm egress/debug logs contain
`POST /api/v2/runs/query` and no `POST /api/v1/runs/query`. Check that responses
do not carry `Deprecation: true` or `Sunset: 2027-01-31`. Because the UI warning
is workspace-wide and may use a lookback window, a banner that remains after
this canary means another service/API key is still calling one of the endpoint
families listed in the changelog; identify that caller rather than changing
trace ingestion.

- [x] **Step 7: Commit Task 1**

```powershell
git add app/evaluation/rag/langsmith_queries.py scripts/evaluate_rag.py pyproject.toml environment.yml tests/test_langsmith_smithdb_migration.py tests/test_rag_evaluation_cli.py
git commit -m "fix: migrate LangSmith evaluation reads to SmithDB"
```

### Task 2: Prove the Ambient Run Context Already Parents Every Nested Call

> **Revised 2026-09-07 after the three-plan review.** The original Task 2
> threaded `ToolCallRequest.runtime.config` through `execute_tool_calls` into
> `tool.ainvoke(args, config=...)`. Measurement at the production seam shows
> that mechanism is unnecessary and actively harmful, so the task is now to
> establish the existing behavior as a tested contract instead of replacing it.

**Files:**
- Create: `tests/test_tool_trace_parenting.py`
- Modify: `app/ai/tool_execution.py` (comment only, at `invoke_tool`)

**What was measured, and how**

Three probes against real `StructuredTool`s and a recording
`BaseCallbackHandler`, with no network access:

1. A `RunnableLambda` node calling `execute_tool_calls`, whose tool calls a
   nested tool with no `config` argument. Result: the product tool is a child
   of the node run, and the nested provider tool is a child of the product
   tool. Nothing is a root.
2. The same shape inside a compiled LangGraph `StateGraph` node, which is how
   `rag_execution.py`, `tool_loop.py` and `ToolExecutionMiddleware` actually
   reach `execute_tool_calls`. Same result.
3. The same shape with the upstream config forwarded explicitly, as the
   original Task 2 and Task 3 proposed, run twice: once with LangSmith tracing
   disabled and once with it enabled.

   - Tracing **off**: the provider becomes a **sibling** of the product tool
     under the node. The forwarded config still carries the *parent's*
     callback manager, so forwarding replaces the active child context rather
     than extending it.
   - Tracing **on**: ancestry stays correct. `langsmith`'s current-run-tree
     contextvar supplies the parent and the mistake is invisible.

   This condition matters and the review's own reproduction did not name it.
   Forwarding is not merely unnecessary; it is a defect that only manifests
   where tracing is off — local development, CI, and any degraded run — which
   is exactly where nobody is looking at a trace tree.

`asyncio.create_task` in `invoke_tool_attempt` and `asyncio.to_thread` in
`invoke_tool` both copy the current `contextvars` context, so the run manager
survives every hop the pipeline makes, including each retry attempt.

The conclusion the original plan got backwards: ancestry is carried by
`langchain_core`'s `var_child_runnable_config` contextvar, which every
`Runnable` sets to its own child config while it executes. Passing no config is
what makes a nested call inherit the caller. Passing the *upstream* config is
what flattens it, whenever nothing else happens to be holding a run tree.

Two operational facts fell out of the measurement and belong in the record:

- The test suite runs with `LANGSMITH_TRACING=true` and a live API key from the
  environment file, so every local run emits traces to the `sample-chatbot`
  project. That pollutes the Task 4 canary's search for conversation-owned
  provider roots, and it is why the ancestry tests pin `tracing_context(enabled=False)`
  rather than trusting the ambient state.
- Because of the above, a trace-topology test written without that guard passes
  or fails depending on the developer's environment file.

**Interfaces:** unchanged. No production signature gains a `runnable_config`
parameter, and `ToolExecutionMiddleware` keeps discarding `request.runtime`.

- [x] **Step 1: Write the ancestry regression at the production seam**

`tests/test_tool_trace_parenting.py` drives `execute_tool_calls` from inside a
LangChain run and asserts the exact topology, not merely a non-null parent:

```python
child = _start(handler, "provider_tool")
parent = _start(handler, "product_tool")
assert child.parent_run_id == parent.run_id
```

Cover: the nested async path, the synchronous `invoke`/`to_thread` path, a
retry that reaches the provider twice, and the blanket property that no tool
run recorded during a conversation-owned call has a null parent. Wrap each in
`langsmith.run_helpers.tracing_context(enabled=False)` so the assertion tests
this pipeline's own context propagation rather than LangSmith's run-tree
fallback.

A non-null-parent assertion alone is not enough. The sibling topology that
config forwarding produces also has a non-null parent — it is the wrong one.

Verify the guard by mutation: monkeypatch `web_tools._call_provider` to forward
the node's config, and confirm the ancestry assertion fails. A trace test that
cannot fail is worse than none, because it reports good news about a tree
nobody is checking.

- [x] **Step 2: Run it and confirm it passes against unmodified production code**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_trace_parenting.py
```

Expected: PASS with no production change. This is the point of the revision.
A test that only passes after a change nobody needed would have locked in the
regression.

- [x] **Step 3: Record why `invoke_tool` takes no config**

Add a comment at `invoke_tool` in `app/ai/tool_execution.py` stating that the
absent `config` argument is deliberate and what breaks if one is added. The
regression from Step 1 is the enforcement; the comment is what stops someone
writing the change in the first place.

- [x] **Step 4: Commit Task 2**

```powershell
git add app/ai/tool_execution.py tests/test_tool_trace_parenting.py
git commit -m "test: lock down nested tool trace ancestry"
```

### Task 3: Prove the Product Web and Image Tools Keep That Ancestry

> **Revised 2026-09-07.** The original Task 3 injected `ToolRuntime` into the
> three product tools and forwarded `runtime.config` into Tavily and Brave.
> Both halves are wrong: `ToolExecutionMiddleware.awrap_tool_call` bypasses the
> framework tool handler that performs runtime injection, so the argument would
> arrive as `None`; and forwarding the config is the flattening measured in
> Task 2. The plan's references to `_run_search`, to sibling child tasks and to
> retained `asyncio.create_task` usage are also stale — the focused-web plan
> replaced them with the `_call_provider` / `_discover` split, and `web_tools.py`
> creates no tasks at all.

**Files:**
- Modify: `tests/test_tool_trace_parenting.py`
- No production change to `app/ai/web_tools.py` or `app/ai/image_discovery_flow.py`

**Interfaces:** unchanged. `_call_provider` and `_discover` keep calling
`tool.ainvoke(args)` with no config, which is what parents them.

- [x] **Step 1: Assert the real product tools sit above their providers**

Drive `create_web_search_tool`, `create_web_open_tool` and
`create_image_search_tool` through `execute_tool_calls` with the Tavily and
Brave fakes built as real `StructuredTool`s, so each provider call produces an
observable run. Assert for all three:

```python
assert provider.parent_run_id == product.run_id
```

The existing `_FakeTool` in `tests/test_web_tools.py` is a bare object and
creates no run, so it cannot answer this question. Leave it alone — it owns the
argument mapping — and build the traced doubles here.

- [x] **Step 2: Assert no conversation-owned provider call is a root**

Collect every recorded tool start from one turn that calls all three product
tools and assert none has `parent_run_id is None`. This is the machine-checkable
half of the live canary in Task 4.

- [x] **Step 3: Run the composite and image regressions**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_trace_parenting.py tests/test_web_tools.py tests/test_brave_image_search_server.py tests/test_image_preview_stream.py tests/test_rich_response_streaming.py
```

Expected: all pass, with cancellation and image ordering unchanged.

- [x] **Step 4: Commit Task 3**

```powershell
git add tests/test_tool_trace_parenting.py
git commit -m "test: pin web and image provider trace ancestry"
```

### Task 4: Document Live Verification and Run the Regression Set

**Files:**
- Modify: `docs/operations/routing-v2-rollout.md`

**Interfaces:**
- Consumes: the ancestry regressions from Tasks 2-3.
- Produces: a live LangSmith verification procedure.

- [x] **Step 1: Document live trace verification**

Add this canary to the rollout guide:

```markdown
1. Start one chat turn that calls `web_search`, `web_open`, and `image_search`.
2. Open the conversation trace in LangSmith.
3. Verify each Tavily and Brave run sits directly beneath its product tool run,
   not merely beneath the same node. A provider run that is a sibling of its
   product tool means someone reintroduced explicit config forwarding.
4. Query the same window for root runs named `tavily_search`, `tavily_extract`,
   or `brave_image_search`; expect zero conversation-owned roots.
5. Diagnostic roots are acceptable only with the `diagnostic` tag.
6. Exclude runs produced by local test execution. The suite inherits
   `LANGSMITH_TRACING=true` from the environment file and writes into the same
   project, so an unfiltered window mixes test traces with real turns.
```

State plainly that offline tests cannot substitute for this: they prove the
callback topology, not that the deployed workspace records it.

- [x] **Step 2: Run trace, streaming, and middleware regression suites**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_langsmith_smithdb_migration.py tests/test_rag_evaluation_cli.py tests/test_tool_trace_parenting.py tests/test_tool_execution_control_flow.py tests/test_tool_execution_recovery.py tests/test_specialist_middleware.py tests/test_specialist_tool_pipeline.py tests/test_web_tools.py tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py
.\.venv\Scripts\python.exe -m ruff check app/ai/tool_execution.py app/ai/web_tools.py app/ai/image_discovery_flow.py app/evaluation/rag/langsmith_queries.py tests/test_tool_trace_parenting.py
```

Expected: all tests pass and Ruff reports no errors.

- [x] **Step 3: Commit Task 4**

```powershell
git add docs/operations/routing-v2-rollout.md
git commit -m "docs: record live trace ancestry canary"
```

## Acceptance Checklist

- [x] Runtime and environment dependency declarations satisfy `langsmith>=0.10.15`.
- [x] Production Python calls no client method the installed SDK marks deprecated, and none of the supported helpers that reach v1 behind a suppressed warning. The ban list is derived from the SDK, so an SDK bump extends it.
- [x] Production Python contains no legacy v1 endpoint literal, across all four sunset families (runs query/retrieve, dataset experiment-run, sharing/public-read, annotation-queue runs).
- [x] RAG comparison metrics use `runs.query()` with project UUID, root filter, explicit selects, and the project's full time window — refusing a project with no `start_time` rather than silently reading the last 24 hours.
- [x] Feedback aggregation reads the SDK's `FeedbackStats` objects, and its tests use that type rather than dictionaries.
- [ ] An authenticated RAG evaluation canary emits no deprecated response header or v1 run-query request. **(needs live credentials)**
- [x] No production call site passes an upstream `RunnableConfig` into a nested tool invocation.
- [x] Callback tests assert `provider.parent_run_id == product.run_id`, not merely a non-null parent.
- [x] Nested Tavily and Brave invocations are children of their product tool, across the async path, the synchronous path, and a retry.
- [x] Existing cancellation, timeout, retry, receipt, artifact, and image tests pass.
- [ ] The live LangSmith canary contains no conversation-owned provider root runs. **(needs live credentials)**
- [x] No model-visible output or public stream contract changed.

## Execution Handoff

Task 1 has landed (`f1d0772`), amended 2026-09-07 for the SDK feedback shape.
Tasks 2-4 execute after the focused-web plan, and are test and documentation
work only: the behavior they were written to create already exists, and the
mechanism originally proposed would have removed it. The plan remains
independent of generation lifecycle persistence and can merge before the
Continue/Stop plan.
