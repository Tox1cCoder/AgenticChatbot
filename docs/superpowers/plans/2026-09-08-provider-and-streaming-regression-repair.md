# Provider and Streaming Regression Repair Plan

> **For Codex:** Execute each task test-first and preserve the intentional working-tree reversions.

**Goal:** Restore incremental specialist output and make OpenAI reasoning models compatible with function tools without weakening reasoning.

**Architecture:** `ModelFactory` owns provider transport/reasoning parameters. Event-stream normalization uses the outer graph namespace to distinguish public specialist subgraphs from private control-plane model nodes. Both client adapters continue consuming the same canonical events.

**Tech stack:** Python, FastAPI, LangChain/LangGraph, pytest.

---

### Task 1: Reproduce nested specialist streaming

**Files:**
- Modify: `tests/test_internal_node_streams_stay_private.py`
- Modify: `tests/test_event_streaming_langgraph_normalizer.py`

1. Add a v3 message-envelope case with `langgraph_node="model"` and namespace `search_agent:<run-id>`; assert its text is public.
2. Add matching internal namespaces (`route`, `planning_actions`) and assert their text stays private.
3. Add a real nested `create_agent` streaming test that asserts multiple deltas precede the terminal state.
4. Run the focused tests and confirm the public specialist cases fail for the current structural filter.

### Task 2: Repair stream source classification

**Files:**
- Modify: `app/services/event_streaming/langchain_v3.py`
- Modify: `app/services/event_streaming/graph_public_projection.py`

1. Add one helper that extracts the first namespace node and determines whether a source belongs to a public specialist.
2. Use it in the v3 translator and tuple fallback projection while retaining internal-tag and planning-worker precedence.
3. Run the Task 1 tests and the existing AI SDK/Streamlit stream contract tests.
4. Refactor redundant comments/docstrings only after green.

### Task 3: Reproduce OpenAI reasoning/tool incompatibility

**Files:**
- Add: `tests/test_model_factory_runtime.py`
- Modify: `tests/test_specialist_middleware.py`

1. Patch provider constructors at the network boundary and build resolved OpenAI runtime configs.
2. Assert default reasoning for `gpt-5.6-luna` selects the Responses API, explicit reasoning is propagated, and explicit `none` does not enable reasoning.
3. Assert the fallback warning contains the failed provider plus safe exception type/message.
4. Run focused tests and confirm they fail against current construction/logging.

### Task 4: Centralize runtime model configuration

**Files:**
- Modify: `app/ai/model_factory.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/workflow/middleware.py`
- Modify: `app/ai/reasoning_controls.py` if a shared effective-default helper is needed

1. Map resolved provider reasoning settings in `create_model_from_runtime`.
2. Select OpenAI Responses transport whenever effective reasoning is not `none`; pass explicit effort unchanged.
3. Reuse the factory path from the legacy agent instead of maintaining a second OpenAI kwargs builder.
4. Preserve fallback reasoning fields where available and improve the provider warning without logging request payloads or keys.
5. Run model, reasoning, runtime override, and middleware tests.
6. Remove redundant model-factory docstrings/comments after behavior is green.

### Task 5: Verify finalization and both client paths

**Files:**
- Modify existing workflow/client tests only if the reproduced contract lacks coverage.

1. Run a deterministic specialist tool loop through the graph and assert it produces a non-empty finalized response.
2. Feed its canonical deltas through AI SDK and internal/Streamlit projection; assert text arrives incrementally and completion does not duplicate it.
3. Run the focused regression suite, then the broader workflow/event-stream/model suites.
4. Inspect the diff for accidental edits to the reverted description-limit files.
5. Run formatting/lint/type checks configured by the repository and report any unrelated pre-existing failures separately.

