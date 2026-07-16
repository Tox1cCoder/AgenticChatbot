# Tool Search Action-Intent Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make agents discover, load, and execute the correct device-action tool on the first turn for explicit requests such as opening a URL in a browser, without vendor-specific production rules or changes to HITL.

**Architecture:** Preserve the public `tool_search(query, top_k, server_name)` API and deferred-loading flow. Refactor the deterministic capability layer so it models the requested action separately from the resource, infers tool capabilities from names/descriptions/schemas rather than server brands, and ranks an active-device direct opener first with an active-device command/process executor as the generic fallback. Carry private capability metadata through the server/client merge so a remote server shell cannot displace a suitable active-device action tool. Keep an optional semantic ranker as a documented extension point only; do not add an LLM or embedding call in this phase.

**Tech Stack:** Python 3.10+, LangChain tools, MCP server/client catalogs, Pydantic, pytest

---

Branch: current working tree | Date: 2026-07-15 | Revised: 2026-07-16 after codebase review | Spec: this document | Input: the attached three-turn YouTube/browser trace and the approved deterministic-first design.

## Summary

The reported behavior is a tool-discovery failure, not an HITL failure. In the captured trace, an explicit request to open a YouTube highlight in the user's browser required three user turns and nine discovery/config calls before `client__desktop_commander__start_process` was loaded and executed.

The defect reproduces locally:

- `open browser url` ranks `tavily_extract` at score `75`, confidence `high`, and marks it autoload-eligible.
- The local process launcher is absent or low-confidence because the query shares no literal `command`, `shell`, or `process` token with it.
- `open link` and `open youtube in browser` can return no candidates.
- Existing focused tests pass (`47 passed`) because the golden matrix covers shell/file/config actions but not external launch versus web extraction.

The normal discovery payload is already compact: local measurements ranged from roughly 45 to 230 tokens for zero to three results. A seven-tool per-server inventory was roughly 390 tokens. The large trace came primarily from repeated incorrect searches and inventory fallbacks, not from the default discovery response. Argument hints are currently unbounded, however, so this plan caps that model-facing field, adds a representative compact-output regression budget and observability, and does not claim an absolute byte limit for arbitrary user queries or invokable tool names.

## Technical Context

Language/Version: Python 3.10+.

Primary dependencies: LangChain 1.x, LangGraph 1.x, Pydantic, MCP adapters, pytest.

Runtime topology:

- `app/` owns multi-agent orchestration, shared prompts, server MCP discovery, deferred loading, and tool execution.
- `client_backend/` syncs device-local MCP tools to the server. The relevant browser action is currently exposed as a client tool such as `client__desktop_commander__start_process`.
- `tool_search` merges server and active-device client candidates, ranks them on one comparable scale, and autoloads only the high-confidence top candidate.

Testing: focused unit/integration tests under `tests/`, plus the offline evaluator at `scripts/evaluate_tool_search_accuracy.py`.

Target platforms: Windows, macOS, and Linux clients. The LLM remains responsible for constructing the platform-appropriate command after a generic command/process executor is loaded.

## Constitution Check

No `.specify/memory/constitution.md` or active `.specify` template exists in this repository. Apply these established repository constraints:

- Keep the public `tool_search` input and output fields backward-compatible.
- Keep deferred loading, per-agent allowlists, device isolation, tool aliases, and same-turn tool-map refresh intact.
- Do not hardcode YouTube, Desktop Commander, Tavily, Windows, or a particular browser in production ranking rules.
- Do not eager-bind the full MCP catalog.
- Do not add a model call to the tool-search hot path in this phase.
- Do not change HITL configuration or approval enforcement.
- Use TDD: every behavior change begins with a failing regression.
- Preserve all user changes already present in the working tree. At review time these include content changes in `app/ai/agent_config.py` and `tests/test_chat_agent_thinking_level.py`; re-run `git status --short` before every commit rather than relying on this snapshot.

## Current Findings

### Finding 1: A URL Is Mistaken For An Extraction Request

`app/ai/tool_search_profiles.py::infer_query_intent()` currently adds `web_extract` whenever the query contains a URL/resource term such as `url`, `page`, or `article`. It does not require an inspection verb such as `extract`, `read`, or `summarize`.

Consequently, `open browser url` is interpreted as content extraction even though `open` expresses an external action.

### Finding 2: Process Execution Is Recognized Only Through Shell Vocabulary

The process launcher is correctly profiled as `shell_exec` from its tool name and required `command` argument. However, launch requests like `open link` do not contain the current shell vocabulary (`run`, `execute`, `command`, `shell`, `python`, or `script`). The scorer therefore has no capability bridge between the requested external action and a general command/process executor.

### Finding 3: Web Capability Inference Contains A Vendor Check

`infer_tool_profile()` currently uses `if "tavily" in server_name.lower()` to assign `web_search`, `web_extract`, `web_map`, and `web_crawl`. That makes otherwise generic capability inference depend on a server brand and conflicts with the requirement to support arbitrary MCP servers dynamically.

### Finding 4: Prompt Policy Is Mostly Correct

`SEARCH_SYSTEM_PROMPT` already says to use an execution tool when the user wants an action on their device. `TOOL_EXPLORATION_SUFFIX` already tells the model to call a high-confidence loaded recommendation next. The prompt cannot compensate when discovery confidently recommends the wrong capability. One generic same-turn completion invariant will make the policy explicit, but prompt-string tests prove guidance only; same-turn behavior still requires tool-map regressions and manual runtime verification.

### Finding 5: Default Output Size Is Not The Primary Defect

Discovery defaults to three public candidates, 120-character purposes, and at most two match reasons. The response includes useful control fields (`recommended_tool`, `requires_refinement`, and `next_action`) that stop synonym loops when ranking is correct. Removing these fields would make orchestration less reliable.

Per-server inventory can be larger because its default is 20 tools, but it should be a fallback rather than the normal path for a concrete action request. Fix first-search ranking and measure serialized response bytes before changing inventory semantics.

### Finding 6: Cross-Origin Ties Prefer Server Tools

`tool_search` ranks server and active-device catalogs separately, then merges them by score with a server-first tie break. Two equally profiled process executors therefore put the remote server executor ahead of the active-device executor unless they share the same qualified tool identity. External-open discovery must explicitly prefer active-device action candidates and must not autoload a server shell as a substitute when no active-device action tool is available.

### Finding 7: One Focused Baseline Test Is Stale

The planned isolation verification currently reports `1 failed, 41 passed`: `tests/test_client_tool_scope.py::test_deferred_binding_keeps_hand_off_available` expects a static `hand_off`, while `BaseAgent` intentionally accepts only a graph-injected handoff through `internal_tools`. Repair this stale test in a separate baseline-only commit before feature work so later red/green results are attributable to this change.

## Functional Requirements

FR-001: Queries that combine an external-action verb (`open`, `launch`, `play`, or equivalent supported vocabulary) with an external target (`browser`, `URL`, `link`, `website`, application, or media) must express an `external_open` capability.

FR-002: A URL/resource term by itself must not imply `web_extract`. Extraction requires an inspection/content verb plus a web resource target.

FR-003: Within the active-device action candidates, a structured tool that directly opens or launches a URL must rank above a generic command/process executor.

FR-004: When no active-device direct opener exists, an active-device generic command/process executor must be the high-confidence, autoload-eligible fallback for an explicit external-open request.

FR-005: Web search, extraction, mapping, and crawling capabilities must be inferred from tool names, descriptions, and argument schemas, independent of the MCP server name.

FR-006: The exact invokable client tool name returned by discovery must be autoloaded in the active conversation/device/session scope and recommended for the next call. A server-side shell/process tool must never be autoloaded as a substitute for an external-open request on the user's device.

FR-007: Shared prompt guidance must direct agents to complete an explicit action in the same turn after a suitable high-confidence tool is loaded. Tool-map tests must prove same-turn callability, and manual verification must confirm model follow-through; prompt-string tests alone are not behavioral proof.

FR-008: Existing `extract/read/summarize known URL`, web search, site map, site crawl, shell, file, and configuration golden cases must retain their current top-ranked tools.

FR-009: Normal discovery output must retain its current public contract, cap each model-facing `arg_hints` string at 160 characters, and keep the representative three-candidate maximum-compact-field fixture at or below 2000 UTF-8 bytes. This regression budget is not an absolute limit for arbitrary query text or exact invokable tool names.

FR-010: Tool-search debug logs must expose response byte size and next-action outcome without adding raw tool descriptions or numeric scoring diagnostics to model-facing output.

FR-011: The implementation must not modify HITL behavior. Existing MCP-server/tool approval configuration continues to decide whether a selected tool call requires approval.

FR-012: The implementation must not weaken client-device isolation, custom-agent allowlists, ambiguous-tool aliases, or deferred-state loading limits.

FR-013: `ToolSearchInput.top_k` documentation must state the configured discovery default of three without changing the public input fields.

## Non-Goals

- Do not add an LLM, embedding model, external reranker, or translation service to `tool_search`.
- Do not make `tool_search` execute the platform command itself; it discovers and loads the executor, while the agent supplies tool arguments.
- Do not introduce a YouTube-specific, browser-specific, Desktop Commander-specific, Tavily-specific, or Windows-specific production rule.
- Do not change multi-agent routing. `search_agent` is a valid owner for a request that first locates a current video and then acts on the user's device.
- Do not redesign global or per-server inventory mode.
- Do not change the public tool-search input/output schema.
- Do not change HITL defaults, settings, persistence, UI, interrupts, or resume behavior.
- Do not implement direct multilingual deterministic token dictionaries in this phase. The agent may continue producing concise English capability queries from multilingual user requests, as it did in the captured trace.

## Design

### 1. Separate Requested Action From Resource

Keep `QueryIntent` as the lightweight deterministic representation, but make `action_verbs` and `target_terms` meaningful instead of treating every query token as a target.

The external-open rule is compositional:

```text
external_open = launch_action AND external_target
```

Examples:

| Query | Capability | Why |
|---|---|---|
| `open browser URL` | `external_open` | `open` action + browser/URL target |
| `play video in browser` | `external_open` | `play` action + media/browser target |
| `extract this URL` | `web_extract` | extraction action + web resource |
| `summarize this article` | `web_extract` | content-read action + article target |
| `URL` | none | resource without an action is ambiguous |
| `open file` | none from this new family | local file opening is outside this external-web action rule |

### 2. Infer Tool Capabilities Without Server Brands

Tool profiles remain deterministic and catalog-agnostic:

- Direct external opener: an `open`/`launch` name or description plus a URL/link/browser argument or description signal.
- Shell/process executor: existing `start_process`, required `command`, or shell+command schema signals.
- Web extractor: `extract` name plus URL argument, or equivalent description/schema evidence.
- Web search: search name/query argument plus web/internet/news description evidence.
- Web map/crawl: map/crawl name plus URL argument.

Server identity remains provenance and scoping metadata, never a capability signal.

### 3. Rank Active-Device Direct Action, Then Safe Generic Fallback

For `external_open` intent:

1. A direct `external_open` tool receives the strongest capability-compatibility score.
2. A `shell_exec` tool receives a high-confidence fallback score.
3. `web_extract` and `web_search` tools receive negative adjustments because they inspect or locate content rather than perform the requested external action.
4. During the unified merge, active-device candidates with `external_open` or `shell_exec` capability are promoted ahead of server-side action candidates for `external_open` intent.
5. Server-side action candidates remain visible for diagnosis but are marked ineligible for autoload under device external-open intent. If no active-device action candidate exists, discovery returns no loaded recommendation and requires refinement instead of opening a browser on the server.

The existing margin rule remains unchanged: only the single top candidate is autoload-eligible when it is high-confidence and leads the second candidate by at least eight points.

The scoring helper will return adjustment reasons together with the numeric adjustment. Compatibility reasons are placed before lexical reasons and deduplicated before the existing two-reason cap, so the model actually sees `direct external opener` or `can execute launch command` rather than losing those explanations to truncation.

### 4. Preserve Deferred Loading And Execution

The execution path remains unchanged after unified discovery selects the client process launcher. The merge gains private capability-aware origin ordering, but loading and invocation continue through the existing path:

1. `_execute_tool_search()` creates a `ClientToolReference` using the exact active device/session identity.
2. `ConversationToolSet.autoload_client_tools()` persists it for the active conversation and agent key.
3. The public result becomes `is_loaded=true`.
4. `_build_recommendation()` returns `next_action="call_recommended_tool"`.
5. The graph's existing post-`tool_search` refresh binds the loaded client tool for same-turn invocation.
6. Existing HITL policy independently approves, rejects, or directly runs the call.

### 5. Compact Output And Observability

Keep discovery fields unchanged. Add a private 160-character `arg_hints` compactor applied to scored, tuple, and plain descriptor paths, correct the `top_k` description to say three, and add one private serializer used by both public `tool_search` constructors. The serializer produces the current compact JSON and logs:

- mode
- serialized byte count
- result count
- recommended tool name, if any
- `next_action`
- whether refinement was required

Do not log tool arguments, raw descriptions, command contents, or query text in this serializer. Query logging remains governed by the existing `mcp_tool_search_log_queries` setting. The 2000-byte test is a representative regression budget over bounded public fields, not runtime truncation of exact tool names or arbitrary query text.

### 6. Semantic Fallback Extension Point

Do not implement a semantic fallback now. The deterministic profiler/scorer stays behind the existing `rank_tool_candidates()` boundary. If production telemetry later shows a material low-confidence/no-result rate, a future ranker can be evaluated behind that function without changing catalogs or the public tool contract.

Trigger criteria for considering that follow-up:

- more than 5% of concrete capability searches return `refine_search` after vocabulary tuning; or
- the same user turn averages more than 1.5 `tool_search` calls for concrete actions; or
- a maintained multilingual golden set cannot reach 95% top-1 accuracy deterministically.

## File Plan

Create:

- `tests/test_tool_search_action_intent.py`
  - Pure intent/profile/ranking regressions for external action versus content inspection.

Modify:

- `app/ai/tool_search_profiles.py`
  - Own generic action/resource vocabulary and schema-derived capability profiles.
  - Remove the vendor-specific web capability branch.

- `app/ai/tool_search_scoring.py`
  - Own compatibility weights, confidence, margin, autoload eligibility, and human-readable match reasons.

- `app/ai/tool_search_tool.py`
  - Carry private capability metadata through merge results, prefer active-device action candidates for external-open intent, cap public argument hints, correct the `top_k` description, and add compact response-size/outcome logging.

- `app/ai/prompts.py`
  - Add one generic same-turn action completion invariant to `TOOL_EXPLORATION_SUFFIX`.

- `tests/test_unified_tool_search.py`
  - Add competing server/client action regressions, no-client-action safety coverage, and compact-output coverage.

- `tests/test_client_tool_scope.py`
  - Repair the stale handoff baseline so it matches graph-owned dynamic handoff injection before feature work begins.

- `tests/test_tool_search_prompt_guidance.py`
  - Pin the generic follow-through rule and continued absence of vendor-specific guidance.

- `scripts/evaluate_tool_search_accuracy.py`
  - Expand the offline golden matrix with external-open and extraction contrast cases using synthetic server identities.

No changes expected:

- `app/ai/agents/base_agent.py`
- `app/ai/agents/search_agent.py`
- `app/ai/deferred_tool_state.py`
- `app/ai/deferred_tool_binding.py`
- `app/ai/tool_execution.py`
- `app/ai/graph.py`
- `app/ai/hitl.py` and all HITL configuration/service/UI files

## Implementation Tasks

### Task 0: Restore A Green Isolation Baseline

**Files:**

- Modify: `tests/test_client_tool_scope.py`
- Test: `tests/test_client_tool_scope.py`

- [x] **Step 1: Reproduce the stale handoff expectation before feature work**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_client_tool_scope.py::test_deferred_binding_keeps_hand_off_available -q
```

Expected on the reviewed baseline: fail because `BaseAgent` no longer creates a static handoff fallback; graph code owns the live target roster and injects `hand_off` through `internal_tools`.

- [x] **Step 2: Update the regression to exercise graph-injected handoff preservation**

Replace `test_deferred_binding_keeps_hand_off_available` with:

```python
def test_deferred_binding_keeps_graph_injected_hand_off_available(monkeypatch):
    from app.ai.hand_off_tool import create_hand_off_tool

    agent = _BindingTestAgent(agent_config_key="canvas")
    agent.tools = []
    agent.mcp_manager = None

    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.get_available_skill_summaries",
        lambda **kwargs: [],
    )

    hand_off = create_hand_off_tool(["search_agent"])
    tools = agent._get_tools_for_binding(
        conversation_id="conversation-1",
        internal_tools=[hand_off],
    )

    assert "hand_off" in [tool.name for tool in tools]
```

This changes only the stale test setup. Do not add a static handoff fallback to production code.

- [x] **Step 3: Verify the isolation baseline is green**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_client_tool_scope.py tests/test_client_tool_isolation.py tests/test_client_invocation_isolation.py tests/test_hitl_client_and_deferred.py tests/test_tool_execution_recovery.py -q
```

Expected: all collected tests pass. Record the count in the implementation notes so later failures can be attributed to feature work.

- [x] **Step 4: Commit the baseline-only test repair**

```powershell
git add tests/test_client_tool_scope.py
git commit -m "test: align handoff binding regression with graph injection"
```

### Task 1: Pin External-Action Intent And Vendor-Neutral Profiles

**Files:**

- Create: `tests/test_tool_search_action_intent.py`
- Test: `tests/test_tool_search_action_intent.py`

- [x] **Step 1: Create the failing intent tests**

Create `tests/test_tool_search_action_intent.py` with these imports, fixtures, and tests:

```python
from __future__ import annotations

import pytest

from app.ai.mcp_tool_catalog import ToolDescriptor
from app.ai.tool_search_profiles import infer_query_intent, infer_tool_profile
from app.ai.tool_search_scoring import rank_tool_candidates


def _tool(
    name: str,
    description: str,
    args: list[str],
    required: list[str],
    *,
    server: str = "synthetic_server",
) -> ToolDescriptor:
    return ToolDescriptor(
        tool_name=name,
        server_name=server,
        description=description,
        arg_names=args,
        required_arg_names=required,
        schema_fingerprint=f"fp-{server}-{name}",
    )


@pytest.mark.parametrize(
    "query",
    [
        "open browser url",
        "open link",
        "open application or URL on desktop",
        "browser open",
        "open youtube in browser",
        "open url in browser play youtube video",
    ],
)
def test_external_action_queries_infer_external_open(query: str):
    intent = infer_query_intent(query)

    assert "external_open" in intent.capabilities
    assert "web_extract" not in intent.capabilities


@pytest.mark.parametrize(
    "query",
    [
        "extract this URL",
        "read this web page",
        "summarize this article",
        "inspect content at this URL",
    ],
)
def test_content_queries_infer_web_extract(query: str):
    intent = infer_query_intent(query)

    assert "web_extract" in intent.capabilities
    assert "external_open" not in intent.capabilities


def test_url_without_action_does_not_guess_extract_or_open():
    intent = infer_query_intent("https://example.com")

    assert "web_extract" not in intent.capabilities
    assert "external_open" not in intent.capabilities


def test_tool_profiles_do_not_depend_on_server_brand():
    first = infer_tool_profile(
        tool_name="extract_url",
        server_name="alpha",
        description="Extract page content from known URLs.",
        arg_names=["urls"],
        required_arg_names=["urls"],
    )
    second = infer_tool_profile(
        tool_name="extract_url",
        server_name="beta",
        description="Extract page content from known URLs.",
        arg_names=["urls"],
        required_arg_names=["urls"],
    )

    assert first.capabilities == second.capabilities
    assert "web_extract" in first.capabilities
```

- [x] **Step 2: Run the new tests and verify the expected failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tool_search_action_intent.py -q
```

Expected: the external-open cases fail because `external_open` does not exist, `open ... URL` incorrectly contains `web_extract`, and arbitrary-server extract profiles lack `web_extract`.

- [x] **Step 3: Commit only the failing regression file**

```powershell
git add tests/test_tool_search_action_intent.py
git commit -m "test: reproduce tool search browser action mismatch"
```

### Task 2: Refactor Query And Tool Capability Profiles

**Files:**

- Modify: `app/ai/tool_search_profiles.py`
- Test: `tests/test_tool_search_action_intent.py`
- Test: `tests/test_tool_search_accuracy.py`
- Test: `tests/test_tool_search_scoring.py`

- [x] **Step 1: Replace the action/resource vocabulary constants**

In `app/ai/tool_search_profiles.py`, retain the existing shell/file/config constants and replace the web-only constants with:

```python
_EXTERNAL_OPEN_ACTION_TERMS = {"open", "launch", "play", "show"}
_EXTERNAL_OPEN_TARGET_TERMS = {
    "app",
    "application",
    "browser",
    "link",
    "media",
    "url",
    "urls",
    "video",
    "webpage",
    "website",
}
_CONTENT_READ_ACTION_TERMS = {
    "analyze",
    "analyse",
    "extract",
    "fetch",
    "inspect",
    "read",
    "summarize",
    "summarise",
}
_WEB_RESOURCE_TERMS = {
    "article",
    "content",
    "page",
    "source",
    "url",
    "urls",
    "webpage",
    "website",
}
_WEB_SEARCH_TERMS = {"web", "search", "current", "recent", "news", "source", "sources"}
_WEB_MAP_TERMS = {"map", "sitemap", "site", "structure", "pages", "urls", "discover"}
_WEB_CRAWL_TERMS = {"crawl", "site", "website", "docs", "documentation", "section", "pages"}
_URL_ARGUMENT_TERMS = {"link", "uri", "url", "urls"}
```

- [x] **Step 2: Make `infer_query_intent()` compositional**

Replace the current web capability block and target assignment with this logic while preserving existing shell/file/config inference:

```python
    external_open_actions = tokens & _EXTERNAL_OPEN_ACTION_TERMS
    external_open_targets = tokens & _EXTERNAL_OPEN_TARGET_TERMS
    content_read_actions = tokens & _CONTENT_READ_ACTION_TERMS
    web_resource_targets = tokens & _WEB_RESOURCE_TERMS

    if external_open_actions and external_open_targets:
        capabilities.add("external_open")
    if content_read_actions and web_resource_targets:
        capabilities.add("web_extract")
    if tokens & _WEB_SEARCH_TERMS and tokens & {
        "web",
        "search",
        "current",
        "recent",
        "news",
    }:
        capabilities.add("web_search")
    if tokens & _WEB_MAP_TERMS and tokens & {
        "map",
        "sitemap",
        "structure",
        "discover",
        "urls",
    }:
        capabilities.add("web_map")
    if tokens & _WEB_CRAWL_TERMS and tokens & {
        "crawl",
        "site",
        "website",
        "docs",
        "documentation",
    }:
        capabilities.add("web_crawl")

    action_verbs.update(
        tokens
        & (
            _SHELL_TERMS
            | _FILE_SEARCH_TERMS
            | _FILE_EDIT_TERMS
            | _FILE_WRITE_TERMS
            | _EXTERNAL_OPEN_ACTION_TERMS
            | _CONTENT_READ_ACTION_TERMS
        )
    )
    target_terms = external_open_targets | web_resource_targets
```

Delete the old `_WEB_EXTRACT_TERMS` rule so `url` alone cannot create extraction intent.

- [x] **Step 3: Replace the server-name web profile branch with schema-derived rules**

In `infer_tool_profile()`, keep the existing shell/file/config/process rules, delete `if "tavily" in server_name.lower(): ...`, and add:

```python
    url_arg_tokens = (arg_tokens | required_arg_tokens) & _URL_ARGUMENT_TERMS
    direct_open_signal = bool(
        ({"open", "launch"} & name_tokens)
        and (url_arg_tokens or {"browser", "link", "url"} & description_tokens)
    )
    if direct_open_signal:
        capabilities.add("external_open")

    if "extract" in name_tokens and url_arg_tokens:
        capabilities.add("web_extract")
    if (
        "search" in name_tokens
        and "query" in (arg_tokens | required_arg_tokens)
        and {"internet", "news", "online", "source", "web"} & description_tokens
    ):
        capabilities.add("web_search")
    if "map" in name_tokens and url_arg_tokens:
        capabilities.add("web_map")
    if "crawl" in name_tokens and url_arg_tokens:
        capabilities.add("web_crawl")
```

- [x] **Step 4: Add the compact purpose for direct openers**

At the start of `_compact_purpose()` capability checks, add:

```python
    if "external_open" in capabilities:
        return "Open an external URL or resource in a local application."
```

- [x] **Step 5: Run the profile and existing accuracy suites**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tool_search_action_intent.py tests/test_tool_search_accuracy.py tests/test_tool_search_scoring.py -q
```

Expected: all intent/profile assertions pass. Ranking assertions for external-open fallback are not present yet; all pre-existing shell/file/web assertions remain green.

- [x] **Step 6: Commit the profile refactor**

```powershell
git add app/ai/tool_search_profiles.py tests/test_tool_search_action_intent.py
git commit -m "refactor: model tool actions independently of resources"
```

### Task 3: Rank Direct Openers And Process Executors Correctly

**Files:**

- Modify: `tests/test_tool_search_action_intent.py`
- Modify: `app/ai/tool_search_scoring.py`
- Test: `tests/test_tool_search_accuracy.py`
- Test: `tests/test_tool_search_scoring.py`

- [x] **Step 1: Add failing ranking tests**

Append to `tests/test_tool_search_action_intent.py`:

```python
def _action_candidates(*, include_direct_opener: bool) -> list[ToolDescriptor]:
    candidates = [
        _tool(
            "start_process",
            "Start a shell command or local process.",
            ["command", "timeout_ms", "shell"],
            ["command"],
            server="local_runtime",
        ),
        _tool(
            "extract_url",
            "Extract page content from known URLs.",
            ["urls"],
            ["urls"],
            server="web_content",
        ),
        _tool(
            "web_search",
            "Search the web for current sources.",
            ["query"],
            ["query"],
            server="web_content",
        ),
    ]
    if include_direct_opener:
        candidates.append(
            _tool(
                "open_url",
                "Open a URL in the default browser.",
                ["url"],
                ["url"],
                server="local_runtime",
            )
        )
    return candidates


def test_direct_opener_outranks_process_fallback():
    ranked = rank_tool_candidates(
        query="open URL in browser",
        candidates=_action_candidates(include_direct_opener=True),
    )

    assert ranked[0].tool.tool_name == "open_url"
    assert ranked[0].confidence == "high"
    assert ranked[0].autoload_eligible is True
    assert "direct external opener" in ranked[0].match_reasons
    assert ranked[1].tool.tool_name == "start_process"


@pytest.mark.parametrize(
    "query",
    [
        "open browser url",
        "open link",
        "open application or URL on desktop",
        "browser open",
        "open youtube in browser",
        "open url in browser play youtube video",
    ],
)
def test_process_executor_is_high_confidence_external_open_fallback(query: str):
    ranked = rank_tool_candidates(
        query=query,
        candidates=_action_candidates(include_direct_opener=False),
    )

    assert ranked[0].tool.tool_name == "start_process"
    assert ranked[0].confidence == "high"
    assert ranked[0].autoload_eligible is True
    assert "can execute launch command" in ranked[0].match_reasons
    assert all(item.tool.tool_name != "extract_url" for item in ranked[:2])


def test_extract_query_still_prefers_extractor():
    ranked = rank_tool_candidates(
        query="extract content from this URL",
        candidates=_action_candidates(include_direct_opener=True),
    )

    assert ranked[0].tool.tool_name == "extract_url"
    assert ranked[0].confidence == "high"
    assert ranked[0].autoload_eligible is True
```

- [x] **Step 2: Verify the ranking tests fail for the intended reason**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tool_search_action_intent.py -q
```

Expected: intent/profile tests pass; process-fallback ranking tests fail because `shell_exec` has no compatibility score for `external_open`, and the new match reason is missing.

- [x] **Step 3: Return score adjustments with reasons**

In `_score_candidate()` in `app/ai/tool_search_scoring.py`, replace:

```python
    score += _capability_specific_adjustment(intent, profile)
```

with:

```python
    adjustment, adjustment_reasons = _capability_specific_adjustment(intent, profile)
    score += adjustment
    reasons = adjustment_reasons + reasons
```

Change `_capability_specific_adjustment()` to return `tuple[float, list[str]]`. Preserve all existing adjustment branches and add the new branch first:

```python
def _capability_specific_adjustment(
    intent: QueryIntent,
    profile: ToolCapabilityProfile,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if "external_open" in intent.capabilities:
        if "external_open" in profile.capabilities:
            score += 55.0
            reasons.append("direct external opener")
        elif "shell_exec" in profile.capabilities:
            score += 65.0
            reasons.append("can execute launch command")
        if "web_extract" in profile.capabilities:
            score -= 60.0
        if "web_search" in profile.capabilities:
            score -= 30.0

    if "shell_exec" in intent.capabilities:
        if "shell_exec" in profile.capabilities:
            score += 35.0
        if "file_search" in profile.capabilities:
            score -= 18.0
        if "process_interaction" in profile.capabilities:
            score += 30.0

    if "file_write" in intent.capabilities:
        if "file_write" in profile.capabilities:
            score += 30.0
        if (
            "file_edit" in profile.capabilities
            and "edit" not in intent.tokens
            and "patch" not in intent.tokens
        ):
            score -= 10.0

    if "file_edit" in intent.capabilities and "file_edit" in profile.capabilities:
        score += 30.0

    if "file_search" in intent.capabilities and "file_search" in profile.capabilities:
        score += 30.0

    if "config_read" in intent.capabilities and "config_read" in profile.capabilities:
        score += 25.0

    if "web_search" in intent.capabilities:
        if "web_search" in profile.capabilities:
            score += 30.0
        if "web_extract" in profile.capabilities:
            score -= 8.0
        if "web_crawl" in profile.capabilities:
            score -= 15.0

    if "web_extract" in intent.capabilities:
        if "web_extract" in profile.capabilities:
            score += 35.0
        if "web_search" in profile.capabilities and {"url", "urls"} & intent.tokens:
            score -= 12.0

    if "web_map" in intent.capabilities:
        if "web_map" in profile.capabilities:
            score += 35.0
        if "web_crawl" in profile.capabilities and "crawl" not in intent.tokens:
            score -= 12.0

    if "web_crawl" in intent.capabilities:
        if "web_crawl" in profile.capabilities:
            score += 35.0
        if "web_map" in profile.capabilities and "content" in intent.tokens:
            score -= 8.0

    return score, reasons
```

The existing capability-overlap score adds 40 points to a direct opener, so its total compatibility is 95. A process executor fallback receives 65 and remains high-confidence when no direct opener exists. The extractor penalty prevents a URL argument match from defeating the requested action. Prepending adjustment reasons ensures the existing two-reason output cap retains the explanation that matters to the model.

- [x] **Step 4: Keep match reasons deterministic and non-duplicated**

Before constructing `ToolSearchScore` in `_score_candidate()`, normalize reasons while preserving the compatibility-first order:

```python
    unique_reasons = list(dict.fromkeys(reasons))
```

Then pass:

```python
        match_reasons=unique_reasons or ["weak lexical match"],
```

- [x] **Step 5: Run new and existing scorer tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tool_search_action_intent.py tests/test_tool_search_accuracy.py tests/test_tool_search_scoring.py tests/test_unified_tool_search.py -q
```

Expected: all tests pass; existing shell/file/config/web rankings remain unchanged.

- [x] **Step 6: Commit the scoring change**

```powershell
git add app/ai/tool_search_scoring.py tests/test_tool_search_action_intent.py
git commit -m "fix: rank external actions above web inspection"
```

### Task 4: Prefer Active-Device Action Tools During Unified Merge

**Files:**

- Modify: `app/ai/tool_search_tool.py`
- Modify: `tests/test_unified_tool_search.py`
- Test: `tests/test_unified_tool_search.py`

- [x] **Step 1: Add the end-to-end integration regression**

Append this test to `tests/test_unified_tool_search.py`:

```python
@pytest.mark.asyncio
async def test_browser_action_prefers_client_process_over_remote_direct_opener(monkeypatch):
    from app.ai.client_tool_catalog import ClientToolDescriptor
    from app.ai.mcp_tool_catalog import ToolDescriptor
    from app.ai.tool_context import ToolContext
    from app.ai.tool_search_scoring import rank_tool_candidates
    from app.ai.tool_search_tool import _execute_tool_search

    server_tools = [
        ToolDescriptor(
            tool_name="open_url",
            server_name="remote_runtime",
            description="Open a URL in the default browser on the server host.",
            arg_names=["url"],
            required_arg_names=["url"],
            schema_fingerprint="fp-remote-open",
        ),
        ToolDescriptor(
            tool_name="extract_url",
            server_name="web_content",
            description="Extract page content from known URLs.",
            arg_names=["urls"],
            required_arg_names=["urls"],
            schema_fingerprint="fp-extract",
        ),
        ToolDescriptor(
            tool_name="web_search",
            server_name="web_content",
            description="Search the web for current sources.",
            arg_names=["query"],
            required_arg_names=["query"],
            schema_fingerprint="fp-search",
        ),
    ]
    client_tool = ClientToolDescriptor(
        tool_name="client__desktop_commander__start_process",
        server_name="desktop_commander",
        description="Start a shell command or local process.",
        arg_names=["command", "timeout_ms", "shell", "origin"],
        required_arg_names=["command"],
        qualified_tool_id="desktop_commander::start_process",
        origin="client_mcp",
        device_id="device-123",
        session_id="session-7",
        catalog_version=7,
        tool_instance_id="instance-7",
    )

    class FakeServerCatalog:
        def search_scored(self, query=None, top_k=5, server_name=None, allowlist=None):
            return rank_tool_candidates(query=query, candidates=server_tools)[:top_k]

        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            return []

        def is_ambiguous(self, tool_name):
            return False

    class FakeClientCatalog:
        tool_count = 1
        session_id = "session-7"
        catalog_version = 7

        def search_scored(self, query=None, top_k=5, server_name=None, allowlist=None):
            return rank_tool_candidates(query=query, candidates=[client_tool])[:top_k]

        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            return []

    class DeferredStateStub:
        def __init__(self):
            self.server_refs = []
            self.client_refs = []

        def autoload(self, **kwargs):
            self.server_refs.extend(kwargs["references"])
            return kwargs["references"]

        def autoload_client_tools(self, **kwargs):
            self.client_refs.extend(kwargs["references"])
            return kwargs["references"]

    state = DeferredStateStub()

    async def fake_get_global_mcp_manager():
        return object()

    async def fake_get_tool_catalog(_manager):
        return FakeServerCatalog()

    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_global_mcp_manager",
        fake_get_global_mcp_manager,
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_catalog",
        fake_get_tool_catalog,
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_client_tool_catalog",
        lambda device_id, user_id: FakeClientCatalog(),
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_deferred_tool_state",
        lambda: state,
    )
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(
            conversation_id="conversation-1",
            user_id="user-1",
            agent_key="search",
            device_id="device-123",
            tool_scope="default",
        ),
    )

    result = await _execute_tool_search(
        query="open url in browser play youtube video"
    )

    expected_name = "client__desktop_commander__start_process"
    assert result["recommended_tool"] == {
        "tool_name": expected_name,
        "confidence": "high",
        "is_loaded": True,
    }
    assert result["next_action"] == "call_recommended_tool"
    assert result["requires_refinement"] is False
    assert result["results"][0]["tool_name"] == expected_name
    assert state.server_refs == []
    assert [ref.tool_name for ref in state.client_refs] == [expected_name]


def test_external_open_server_executor_is_not_autoload_eligible_without_client_action():
    from app.ai.mcp_tool_catalog import ToolDescriptor
    from app.ai.tool_search_scoring import rank_tool_candidates
    from app.ai.tool_search_tool import _merge_search_results

    remote_process = ToolDescriptor(
        tool_name="start_process",
        server_name="remote_runtime",
        description="Start a shell command or local process on the server host.",
        arg_names=["command"],
        required_arg_names=["command"],
        schema_fingerprint="fp-remote-process",
    )
    server_results = rank_tool_candidates(
        query="open URL in browser",
        candidates=[remote_process],
    )

    public, internal = _merge_search_results(
        server_results=server_results,
        client_results=[],
        query="open URL in browser",
        top_k=3,
    )

    assert public[0]["tool_name"] == "start_process"
    assert internal[0]["_autoload_eligible"] is False
```

- [x] **Step 2: Run both regressions and verify the intended failures**

Run after Tasks 1-3 so capability scores exist but before changing merge behavior:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_unified_tool_search.py::test_browser_action_prefers_client_process_over_remote_direct_opener tests/test_unified_tool_search.py::test_external_open_server_executor_is_not_autoload_eligible_without_client_action -q
```

Expected: both fail. The remote direct opener currently sorts ahead of the client process tool, and a lone remote process executor remains autoload-eligible.

- [x] **Step 3: Carry private capability metadata into the merge**

In `app/ai/tool_search_tool.py`, import both profile helpers:

```python
from .tool_search_profiles import infer_query_intent, infer_tool_profile
```

In `_public_result_from_scored()`, add private capability metadata beside the other internal score fields:

```python
        internal["_capabilities"] = sorted(score_meta.profile.capabilities)
```

Do not add `_capabilities` to `public`; it is merge metadata only.

In `_search_item_to_dicts()`, normalize private capabilities for legacy tuple/plain inputs as well. Add after the branch that selects `public_dict` and `internal_dict`, before assigning `_score`:

```python
    if "_capabilities" not in internal_dict:
        if hasattr(item, "tool"):
            descriptor = item.tool
        elif isinstance(item, tuple):
            descriptor = item[0]
        else:
            descriptor = item
        profile = infer_tool_profile(
            tool_name=str(
                getattr(descriptor, "tool_name", getattr(descriptor, "name", "")) or ""
            ),
            server_name=str(getattr(descriptor, "server_name", "") or ""),
            description=str(getattr(descriptor, "description", "") or ""),
            arg_names=list(getattr(descriptor, "arg_names", []) or []),
            required_arg_names=list(getattr(descriptor, "required_arg_names", []) or []),
        )
        internal_dict["_capabilities"] = sorted(profile.capabilities)
```

This keeps the safety rule consistent for production scored catalogs and compatibility fakes without exposing metadata to the model.

- [x] **Step 4: Add external-open origin prioritization**

Add immediately before `_merge_search_results()`:

```python
_EXTERNAL_ACTION_TOOL_CAPABILITIES = {"external_open", "shell_exec"}


def _prioritize_external_action_results(
    chosen_results: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    query: str | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    intent = infer_query_intent(query)
    if "external_open" not in intent.capabilities:
        return chosen_results

    client_actions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    remaining: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for public, internal in chosen_results:
        capabilities = set(internal.get("_capabilities") or [])
        is_action_tool = bool(capabilities & _EXTERNAL_ACTION_TOOL_CAPABILITIES)
        is_client_action = bool(internal.get("is_client_tool")) and is_action_tool

        if is_action_tool and not internal.get("is_client_tool"):
            internal["_autoload_eligible"] = False

        if is_client_action:
            client_actions.append((public, internal))
        else:
            remaining.append((public, internal))

    return [*client_actions, *remaining]
```

In `_merge_search_results()`, apply the helper after deduplication and before truncation:

```python
    chosen_results = _prioritize_external_action_results(
        chosen_results,
        query=query,
    )
    chosen_results = chosen_results[:top_k]
```

This is a stable partition: capability scoring still orders direct openers above process fallbacks within the active-device partition, while remote action tools remain visible but cannot be autoloaded for a user-device external-open request.

- [x] **Step 5: Run unified, isolation, and execution-refresh regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_unified_tool_search.py tests/test_client_tool_isolation.py tests/test_client_tool_scope.py tests/test_client_invocation_isolation.py tests/test_hitl_client_and_deferred.py tests/test_tool_execution_recovery.py -q
```

Expected: all tests pass, proving the scoring change did not bypass device/session scoping or deferred execution refresh.

- [x] **Step 6: Commit the unified merge change**

```powershell
git add app/ai/tool_search_tool.py tests/test_unified_tool_search.py
git commit -m "fix: prefer active-device tools for external actions"
```

### Task 5: Require Same-Turn Follow-Through In Shared Prompt Guidance

**Files:**

- Modify: `tests/test_tool_search_prompt_guidance.py`
- Modify: `app/ai/prompts.py`
- Test: `tests/test_tool_search_prompt_guidance.py`

- [x] **Step 1: Add failing prompt contract tests**

Append to `tests/test_tool_search_prompt_guidance.py`:

```python
def test_tool_guidance_requires_same_turn_action_completion():
    normalized = " ".join(TOOL_EXPLORATION_SUFFIX.lower().split())

    assert "complete the action in the same turn" in normalized
    assert "do not claim you cannot act" in normalized


def test_tool_guidance_does_not_name_browser_action_vendors():
    normalized = TOOL_EXPLORATION_SUFFIX.lower()

    assert "desktop commander" not in normalized
    assert "desktop-commander" not in normalized
    assert "youtube" not in normalized
```

- [x] **Step 2: Verify the same-turn contract fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tool_search_prompt_guidance.py -q
```

Expected: the same-turn completion test fails because the invariant is not yet present; existing no-vendor tests remain green.

- [x] **Step 3: Add one generic bullet to `TOOL_EXPLORATION_SUFFIX`**

In `app/ai/prompts.py`, insert immediately after the rule that calls a high-confidence loaded recommendation:

```text
- When the user explicitly asks you to perform an action and a suitable tool is loaded, complete the action in the same turn. Do not claim you cannot act, stop at instructions, or merely describe the tool unless the tool call fails or policy blocks it.
```

Do not add platform, server, product, or command examples.

- [x] **Step 4: Run prompt guidance and same-turn binding regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tool_search_prompt_guidance.py tests/test_client_tool_isolation.py::test_refresh_tool_map_after_search_uses_active_session_scope tests/test_client_tool_isolation.py::test_refresh_tool_map_uses_tool_state_key_for_custom_agent -q
```

Expected: all tests pass. The prompt test proves the policy text is present; the refresh tests separately prove a loaded client tool becomes callable in the same turn. Neither test claims to prove model compliance, which remains part of Manual Verification.

- [x] **Step 5: Commit the prompt invariant**

```powershell
git add app/ai/prompts.py tests/test_tool_search_prompt_guidance.py
git commit -m "fix: require same-turn execution after tool discovery"
```

### Task 6: Guard Output Size And Log Search Outcomes

**Files:**

- Modify: `app/ai/tool_search_tool.py`
- Modify: `tests/test_unified_tool_search.py`
- Test: `tests/test_unified_tool_search.py`

- [x] **Step 1: Add failing serializer and size-budget tests**

Append to `tests/test_unified_tool_search.py`:

```python
def test_compact_tool_search_output_stays_within_representative_budget(caplog):
    import json
    import logging

    from app.ai.mcp_tool_catalog import ToolDescriptor
    from app.ai.tool_search_scoring import rank_tool_candidates
    from app.ai.tool_search_tool import (
        _merge_search_results,
        _serialize_tool_search_output,
    )

    arg_names = [
        f"parameter_{index}_with_a_deliberately_long_schema_name"
        for index in range(20)
    ]
    tools = [
        ToolDescriptor(
            tool_name=f"open_url_variant_{index}",
            server_name="remote_runtime",
            description="x" * 200,
            arg_names=arg_names,
            required_arg_names=arg_names,
            schema_fingerprint=f"fp-{index}",
        )
        for index in range(3)
    ]
    scored = rank_tool_candidates(query="open URL in browser", candidates=tools)
    public, _internal = _merge_search_results(
        server_results=scored,
        client_results=[],
        query="open URL in browser",
        top_k=3,
    )

    result = {
        "query": "open URL in browser",
        "mode": "discovery",
        "recommended_tool": None,
        "results": public,
        "requires_refinement": True,
        "next_action": "refine_search",
        "loaded_count": 0,
        "more_available": False,
    }

    with caplog.at_level(logging.DEBUG, logger="app.ai.tool_search_tool"):
        payload = _serialize_tool_search_output(result)

    assert json.loads(payload) == result
    assert len(public) == 3
    assert all(len(item["arg_hints"]) <= 160 for item in public)
    assert all(item["arg_hints"].endswith("...") for item in public)
    assert len(payload.encode("utf-8")) <= 2000
    assert "response_bytes=" in caplog.text
    assert "next_action=refine_search" in caplog.text
    assert "open URL in browser" not in caplog.text


def test_tool_search_top_k_description_matches_configured_default():
    from app.ai.tool_search_tool import ToolSearchInput

    description = ToolSearchInput.model_fields["top_k"].description or ""
    assert "Defaults to 3" in description
```

- [x] **Step 2: Verify the serializer test fails because the helper is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_unified_tool_search.py::test_compact_tool_search_output_stays_within_representative_budget tests/test_unified_tool_search.py::test_tool_search_top_k_description_matches_configured_default -q
```

Expected: the budget test fails because `_serialize_tool_search_output` does not exist, and the schema-description test fails because it still says five.

- [x] **Step 3: Add the argument-hint compactor and correct the schema description**

Add to `app/ai/tool_search_tool.py` after `_ALLOWLIST_UNSET`:

```python
_ARG_HINTS_MAX_CHARS = 160


def _compact_arg_hints(value: Any) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= _ARG_HINTS_MAX_CHARS:
        return normalized
    return normalized[: _ARG_HINTS_MAX_CHARS - 3] + "..."
```

In `_search_item_to_dicts()`, apply the cap once after all scored/tuple/plain branches and before returning:

```python
    public_dict["arg_hints"] = _compact_arg_hints(public_dict.get("arg_hints"))
```

In `ToolSearchInput.top_k`, change only the description text:

```python
            "Maximum number of tools to return. Defaults to 3. "
```

- [x] **Step 4: Add the private compact serializer**

Add after `_compact_arg_hints()`:

```python
def _serialize_tool_search_output(result: dict[str, Any]) -> str:
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    recommended = result.get("recommended_tool")
    recommended_name = (
        recommended.get("tool_name") if isinstance(recommended, dict) else None
    )
    logger.debug(
        "tool_search output: mode=%s response_bytes=%d results=%d "
        "recommended=%s next_action=%s requires_refinement=%s",
        result.get("mode"),
        len(payload.encode("utf-8")),
        len(result.get("results") or []),
        recommended_name,
        result.get("next_action"),
        result.get("requires_refinement"),
    )
    return payload
```

- [x] **Step 5: Use the serializer in both tool constructors**

In `tool_search()` and the nested `tool_search_impl()`, replace the two direct `json.dumps(...)` returns with:

```python
    return _serialize_tool_search_output(result)
```

- [x] **Step 6: Run serializer, schema, and output regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_unified_tool_search.py tests/test_tool_search_prompt_guidance.py -q
```

Expected: all tests pass; public field names remain unchanged, argument hints are compacted consistently across result paths, and the configured default is documented accurately.

- [x] **Step 7: Commit compact formatting and observability without changing the API fields**

```powershell
git add app/ai/tool_search_tool.py tests/test_unified_tool_search.py
git commit -m "chore: bound and measure compact tool search responses"
```

### Task 7: Expand The Offline Golden Accuracy Matrix

**Files:**

- Modify: `scripts/evaluate_tool_search_accuracy.py`
- Test: `scripts/evaluate_tool_search_accuracy.py`

- [x] **Step 1: Extend the evaluator case contract**

Change `GoldenCase` to:

```python
@dataclass(frozen=True)
class GoldenCase:
    query: str
    expected_top: str
    expected_confidence: str = "high"
```

Extend `CASES` with:

```python
    GoldenCase("open browser url", "start_process"),
    GoldenCase("open link", "start_process"),
    GoldenCase("open application or URL on desktop", "start_process"),
    GoldenCase("browser open", "start_process"),
    GoldenCase("open youtube in browser", "start_process"),
    GoldenCase("open url in browser play youtube video", "start_process"),
    GoldenCase("extract content from known URL", "extract_url"),
    GoldenCase("search the web for current news", "web_search"),
```

- [x] **Step 2: Add vendor-neutral web candidates to the evaluator**

Append these descriptors in `_desktop_tools()` or rename it to `_candidate_tools()` and return the combined list:

```python
        ToolDescriptor(
            "extract_url",
            "web_content",
            "Extract page content from known URLs.",
            ["urls"],
            ["urls"],
            "fp7",
        ),
        ToolDescriptor(
            "web_search",
            "web_content",
            "Search the web for current news and sources.",
            ["query"],
            ["query"],
            "fp8",
        ),
```

- [x] **Step 3: Check both top-1 tool and confidence**

Replace the evaluator loop condition with:

```python
        top = ranked[0].tool.tool_name if ranked else None
        confidence = ranked[0].confidence if ranked else "none"
        print(
            f"{case.query}: top={top} confidence={confidence} "
            f"expected={case.expected_top}/{case.expected_confidence}"
        )
        if top != case.expected_top or confidence != case.expected_confidence:
            failures.append(
                (
                    case.query,
                    f"{top}/{confidence}",
                    f"{case.expected_top}/{case.expected_confidence}",
                )
            )
```

Keep the existing failure rendering compatible with the three string values.

- [x] **Step 4: Run the evaluator**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tool_search_accuracy.py
```

Expected: exit code `0`; every case prints the expected top tool with `high` confidence.

- [x] **Step 5: Commit the production-tuning matrix**

```powershell
git add scripts/evaluate_tool_search_accuracy.py
git commit -m "test: expand tool search golden action matrix"
```

### Task 8: Focused And Full Verification

**Files:** Existing tests and scripts only.

- [x] **Step 1: Run focused capability and orchestration tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tool_search_action_intent.py tests/test_tool_search_accuracy.py tests/test_tool_search_scoring.py tests/test_unified_tool_search.py tests/test_tool_search_prompt_guidance.py tests/test_client_tool_isolation.py tests/test_client_tool_scope.py tests/test_client_invocation_isolation.py tests/test_hitl_client_and_deferred.py tests/test_tool_execution_recovery.py -q
```

Expected: all collected tests pass with no warnings introduced by this change.

- [x] **Step 2: Run the offline evaluator**

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tool_search_accuracy.py
```

Expected: exit code `0`; all shell/file/config/external-open/web-extract/web-search cases match top-1 and confidence expectations.

- [x] **Step 3: Run lint on changed Python files**

```powershell
.\.venv\Scripts\python.exe -m ruff check app/ai/tool_search_profiles.py app/ai/tool_search_scoring.py app/ai/tool_search_tool.py app/ai/prompts.py tests/test_client_tool_scope.py tests/test_tool_search_action_intent.py tests/test_unified_tool_search.py tests/test_tool_search_prompt_guidance.py scripts/evaluate_tool_search_accuracy.py
```

Expected: exit code `0`.

- [x] **Step 4: Run the full test suite**

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests pass. Task 0 removes the known stale focused failure before feature work. If a different environment-dependent integration test fails, reproduce it on the pre-feature commit after Task 0 before classifying it as pre-existing; do not silently waive failures.

- [x] **Step 5: Review the final diff for forbidden scope**

Run:

```powershell
git diff --check
git diff --name-only
git diff -- app/ai/tool_search_profiles.py app/ai/tool_search_scoring.py app/ai/tool_search_tool.py app/ai/prompts.py tests/test_client_tool_scope.py tests/test_tool_search_action_intent.py tests/test_unified_tool_search.py tests/test_tool_search_prompt_guidance.py scripts/evaluate_tool_search_accuracy.py
```

Expected:

- no whitespace errors;
- no production references to YouTube, Desktop Commander, Tavily, Windows, or browser executable names;
- no HITL files changed;
- no public `ToolSearchInput` or `ToolSearchOutput` field changes;
- unrelated user modifications remain untouched.

- [x] **Step 6: Commit final verification-only adjustments, if any**

If verification required formatting changes, stage only files in this plan:

```powershell
git add app/ai/tool_search_profiles.py app/ai/tool_search_scoring.py app/ai/tool_search_tool.py app/ai/prompts.py tests/test_client_tool_scope.py tests/test_tool_search_action_intent.py tests/test_unified_tool_search.py tests/test_tool_search_prompt_guidance.py scripts/evaluate_tool_search_accuracy.py
git commit -m "fix: make tool discovery action aware"
```

If no adjustments were required, do not create an empty commit.

## Implementation Progress Notes

### Task 0 (done 2026-07-16)

- Step 1 reproduced the expected failure exactly: `test_deferred_binding_keeps_hand_off_available` failed with `assert 'hand_off' in ['tool_search']` because `BaseAgent` no longer creates a static handoff fallback.
- Step 3 baseline: **42 passed, 1 warning** across `test_client_tool_scope.py`, `test_client_tool_isolation.py`, `test_client_invocation_isolation.py`, `test_hitl_client_and_deferred.py`, `test_tool_execution_recovery.py`. The warning is a pre-existing `langchain-community` DeprecationWarning from `app/services/document_processing_service.py:19`, unrelated to this change.
- Committed as baseline-only test repair; user working-tree changes (`app/ai/agent_config.py`, `app/ai/tool_execution.py`, `tests/test_chat_agent_thinking_level.py`) left unstaged.

### Task 1 (done 2026-07-16)

- Red run matched the plan: 7 failed / 5 passed. All six external-open queries failed on missing `external_open`; `test_tool_profiles_do_not_depend_on_server_brand` failed with `capabilities=set()` for the non-Tavily server. The four content-read cases and the bare-URL case already passed under the old rules (their failing assertion was the `external_open not in` guard, which trivially held).

### Task 2 (done 2026-07-16)

- Implemented exactly as specified. Verified `intent.target_terms`/`intent.action_verbs` are consumed nowhere outside `tool_search_profiles.py` (grep across the repo), so narrowing `target_terms` from "all tokens" to external/web targets is behavior-safe.
- Green run: 32 passed across intent + accuracy + scoring suites; no pre-existing shell/file/config/web assertion regressed.

### Task 3 (done 2026-07-16)

- Red run matched the plan: 7 failed / 13 passed. `extract_url` outranked `start_process` for every external-open query (URL-arg lexical overlap with no capability bridge), and `direct external opener` was missing from match reasons. `test_extract_query_still_prefers_extractor` already passed pre-change.
- `_capability_specific_adjustment()` now returns `tuple[float, list[str]]`; adjustment reasons are prepended before lexical reasons and deduplicated via `dict.fromkeys` before the existing two-reason cap.
- Green run: 56 passed (intent + accuracy + scoring + unified).

### Task 4 (done 2026-07-16)

- Red run matched the plan: the remote `open_url` sorted ahead of the client `start_process` in the merged results, and the lone remote process executor stayed `_autoload_eligible=True`.
- Implemented exactly as specified: `_capabilities` carried through scored and legacy tuple/plain paths, plus `_prioritize_external_action_results()` applied after dedup and before truncation.
- Green run: 60 passed (unified + all five isolation/scope/HITL/recovery suites), confirming device/session scoping and deferred refresh unchanged.

### Task 5 (done 2026-07-16)

- Red run: only `test_tool_guidance_requires_same_turn_action_completion` failed; vendor-absence guard was already green.
- The invariant bullet was inserted in `app/ai/prompts.py` directly after the `recommended_tool`/high/`is_loaded` rule, before the refinement rule. 15 tests green including both same-turn tool-map refresh regressions.

### Task 6 (done 2026-07-16)

- Red run: budget test failed on missing `_serialize_tool_search_output` import; description test failed on "Defaults to 5".
- `_compact_arg_hints()` is applied once in `_search_item_to_dicts()` after the `_capabilities` normalization, covering scored, tuple, and plain paths. Both `tool_search()` and `tool_search_impl()` now return via `_serialize_tool_search_output`.
- Verified the configured default is 3 (`mcp_tool_search_default_top_k` in `app/core/config.py:1017`) before changing the description text.
- Green run: 33 passed (unified + prompt guidance).

### Task 7 (done 2026-07-16)

- First evaluator run failed one case: `search file contents` ranked `start_search` top-1 but at `medium` confidence — see Design Decision 1 below for the root cause and fix. After the fix, exit code 0 with all 14 cases top-1/high.
- `scripts/` is gitignored (`.gitignore:72`); the evaluator was force-added (`git add -f`), matching the repo precedent of the two already-tracked scripts.

### Task 8 (done 2026-07-16)

- Focused suites: **117 passed, 1 warning** (the pre-existing langchain-community deprecation).
- Offline evaluator: exit 0.
- Ruff: initially 6 pre-existing E501s in the evaluator's original one-line fixtures; reformatted to multi-line style (the file was newly tracked, so the lines fell in scope). Ruff now exits 0 on all nine planned files.
- Full suite: **5 failed, 1793 passed, 14 skipped**. All five proven pre-existing/environmental, not silently waived:
  - `test_live_document_upload_list_get_task_and_delete_flow`, both `test_mcp_global_allowlist` cases, and `test_widget_runtime` truncation reproduce identically on the post-Task-0 baseline commit (`efb2a401`) in a clean worktree.
  - `test_brave_image_search_defaults` fails on `assert settings.brave_search_api_key == ""` only when run from the repo root (local `.env` supplies a key); it passes in a clean worktree without `.env`. No commit in this change touches `tests/test_brave_image_search_config.py` or `app/core/config.py` (verified via `git diff efb2a401..HEAD --name-only`).
- Final diff scope review: `git diff --check` clean; the only vendor-string match in the production diff is the *removal* of the `tavily` branch; no HITL files changed; no public `ToolSearchInput`/`ToolSearchOutput` field changes (only the `top_k` description text); user working-tree modifications untouched.

## Design Decisions Made During Implementation

1. **`web_search` intent now requires a search verb plus a web-context token** (`app/ai/tool_search_profiles.py`). The plan's Task 2 snippet kept the legacy rule `tokens & _WEB_SEARCH_TERMS and tokens & {web, search, current, recent, news}`, which the single token `search` satisfies on both sides. That latent defect surfaced in Task 7: adding the vendor-neutral `web_search` fixture tool made `search file contents` tie `start_search` and `web_search` at 85 points (margin 0 → `medium`), failing the plan's own exit-0/high-confidence expectation. Fixed compositionally, consistent with the plan's action+target design: `_WEB_SEARCH_ACTION_TERMS = {search, find, lookup}` AND `_WEB_SEARCH_CONTEXT_TERMS = {web, internet, online, news, current, recent}`. Two new pinning regressions were added first and observed failing (`test_file_content_search_does_not_infer_web_search`, `test_web_context_search_still_infers_web_search`). This also means Task 7 touched `app/ai/tool_search_profiles.py` and `tests/test_tool_search_action_intent.py` beyond the plan's file list.
2. **Evaluator committed with `git add -f`.** `.gitignore:72` ignores `scripts/`, but `scripts/benchmark_ingestion.py` and `scripts/start_mineru_service.ps1` are already force-tracked, and the plan requires the golden matrix in version control. Followed the existing precedent rather than editing `.gitignore`.
3. **Evaluator fixture lines reformatted for lint.** The six original one-line `ToolDescriptor` fixtures exceeded the 100-char limit; since committing the file put them in lint scope, they were rewrapped to the multi-line style already used by the two new web descriptors. No values changed; evaluator still exits 0.
4. **Baseline attribution used a temporary git worktree** at the Task 0 commit instead of stashing, so the user's unstaged changes (`app/ai/agent_config.py`, `app/ai/tool_execution.py`) were never touched during verification. The worktree was removed afterward.

## Remaining Work

- **Manual Verification (steps 1-12 below) has not been run.** It requires a live desktop sidecar with a process/command launcher and HITL toggling; this is user-driven runtime verification.
- The two dirty user files and `tests/test_chat_agent_thinking_level.py` remain uncommitted in the working tree, as required.

## Manual Verification

Use an active desktop sidecar whose MCP catalog exposes a process/command launcher and whose existing settings allow the call without HITL approval.

1. Start a new conversation so no deferred tools are already loaded.
2. Ask in Vietnamese: `bật cho tôi youtube highlight trận Pháp với Tây Ban Nha`.
3. Confirm the selected agent searches for a current/relevant video URL as needed.
4. Confirm its first action-oriented `tool_search` call recommends and loads the active device's command/process tool rather than a URL extractor.
5. Confirm the agent calls that tool in the same user turn with a platform-appropriate open command.
6. Confirm the browser opens the selected URL.
7. Confirm the assistant reports success only after the tool returns success.
8. Repeat with `mở youtube trên trình duyệt`; confirm one user turn is sufficient.
9. Enable HITL for the same MCP server in settings and repeat; confirm the existing approval interrupt appears. Approve it and confirm execution resumes normally. This verifies HITL remained independent rather than bypassed.
10. Ask `extract the content from this URL: https://example.com`; confirm an extractor, not the process launcher, is recommended.
11. Enable an unrelated server-side MCP tool that can open URLs or start processes while the active device launcher is still available. Repeat `open this URL in my browser`; confirm discovery recommends the active-device tool, not the server executor.
12. Disconnect or disable the active-device action tools while leaving the server executor enabled. Repeat the request; confirm the server executor may be listed but is not loaded or recommended as a substitute for the user's device.

## Acceptance Criteria

- Every captured discovery phrase (`open browser url`, `open link`, `open application or URL on desktop`, `browser open`, `open youtube in browser`, and `open url in browser play youtube video`) ranks an active-device command/process executor high and autoload-eligible when no active-device direct opener exists.
- An active-device direct structured URL opener outranks an active-device generic command/process executor when both exist.
- An unrelated server-side direct opener or process executor cannot displace an available active-device action tool and is never autoloaded as the substitute for a user-device external-open request.
- `extract/read/summarize` plus a web resource still ranks the extractor first.
- The server-plus-client integration test returns `client__desktop_commander__start_process` as a loaded high-confidence recommendation for the captured query.
- The shared prompt explicitly directs same-turn completion, same-turn tool-map refresh tests pass, and manual verification confirms the agent does not stop at a capability disclaimer after a suitable action tool is loaded.
- No production capability inference depends on an MCP server brand.
- Every model-facing `arg_hints` value is at most 160 characters, and the representative three-result maximum-compact-field fixture stays at or below 2000 UTF-8 bytes.
- The public `tool_search` input/output fields are unchanged, and the `top_k` description correctly states the default of three.
- Client isolation, allowlists, aliases, deferred loading, and same-turn tool-map refresh tests pass.
- HITL files and behavior are unchanged.
- The expanded offline evaluator exits successfully.
- The full test suite passes or any unrelated environment failure is proven against the unchanged base.

## Risks And Mitigations

- Risk: Generic launch vocabulary could classify an informational request as an action.
  - Mitigation: require both an external-action verb and an external target; `URL` alone is not enough.

- Risk: A shell executor is powerful and could outrank a safer structured tool.
  - Mitigation: active-device direct `external_open` tools receive a higher compatibility score; active-device shell execution is only the fallback. Server action tools are ineligible substitutes for user-device external-open intent. Existing HITL remains the independent policy boundary.

- Risk: Origin preference could hide a useful remote tool for a non-device task.
  - Mitigation: apply the active-device partition only to compositional `external_open` intent. Remote candidates remain visible, and all shell/file/config ranking outside this intent is unchanged.

- Risk: Removing the vendor-specific web profile branch could regress current web tools whose descriptions are poor.
  - Mitigation: golden fixtures use generic schema/name evidence, existing Tavily behavior tests remain in the focused suite, and production catalogs should provide descriptive names/schemas. If a real tool lacks sufficient metadata, improve its MCP description rather than add a server-name exception.

- Risk: English deterministic vocabulary may miss non-English `tool_search` queries.
  - Mitigation: current agents already formulate English capability queries from multilingual requests. Measure no-result rates; consider a semantic fallback only when the documented trigger is met.

- Risk: The 2000-byte representative budget could be mistaken for an absolute runtime cap.
  - Mitigation: cap only `arg_hints`, preserve exact invokable tool names and query text, document the fixture as a regression budget, and use byte-size telemetry to detect real outliers.

- Risk: Prompt follow-through could make an agent retry a failed action aggressively.
  - Mitigation: the invariant explicitly stops when a tool fails or policy blocks it; existing tool error/retry guidance remains authoritative.

## Rollout And Monitoring

1. Deploy behind the existing `mcp_tool_search_enabled` flag; no new flag is required because the public behavior is a correctness fix.
2. Keep `mcp_tool_search_debug_scores=false` in production.
3. Aggregate existing tool-search logs plus the new serializer log to monitor:
   - `next_action=refine_search` rate for concrete queries;
   - average `tool_search` calls per user turn;
   - response byte percentiles by mode;
   - recommended-tool-to-successful-call conversion;
   - no-result queries worth adding to the golden matrix.
4. Roll back the profile/scoring/unified-merge commits together if extraction, search top-1 accuracy, or device-origin selection regresses.
5. Do not respond to isolated misses with vendor exceptions. Add a generic capability family and regression case, or evaluate the documented semantic fallback when thresholds justify it.

## Definition Of Done

- The implementation satisfies all functional requirements and acceptance criteria.
- Every new test was observed failing before its corresponding production change.
- Focused tests, offline evaluation, lint, and the full suite have been run with recorded results.
- Manual verification demonstrates both immediate execution under no-HITL configuration and the unchanged approval interrupt when HITL is enabled.
- Documentation contains no unresolved placeholders or ambiguous optional implementation branches.
- The final diff contains only scoped tool-search, prompt, evaluator, and test changes, including the separate Task 0 baseline-test repair.
