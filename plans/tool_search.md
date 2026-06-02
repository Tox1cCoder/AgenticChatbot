# Tool Search Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `tool_search` return the right callable tool on the first search for common production requests, with compact model-facing results that reduce repeat searches and protect the context window.

**Architecture:** Keep the public `tool_search(query, top_k, server_name)` interface backward-compatible, but upgrade ranking from raw token overlap to deterministic intent-aware scoring. Add compact recommendation metadata, load only high-confidence recommended tools, and fix the alias-to-execution-map path so every returned callable name is actually callable in the same turn.

**Tech Stack:** Python 3.10+, LangChain tools, MCP server/client catalogs, Pydantic settings, pytest

---

## Spec

### Problem Statement

The current `tool_search` can return and autoload tools that share weak description tokens with the query instead of the tool that actually satisfies the user's intent. In the observed production trace:

- `tool_search(query="run shell command")` returned `start_search`, `get_config`, and `interact_with_process` above `start_process`.
- `start_process` was present but ranked fourth and `is_loaded=false`.
- The agent then searched again with related phrases like `apply_patch`, `write file`, and `run python script`.

This is not just an agent prompt issue. The search output made the correct next action unclear and failed to load the most likely tool.

### User Stories

1. As an agent, when I search for a common capability such as "run shell command", I get one clear recommended tool that is loaded and callable.
2. As an agent, when the first search is sufficient, I can call the recommended tool instead of issuing synonym searches.
3. As a production operator, I can evaluate ranking quality against stable golden queries before changing thresholds.
4. As a developer, every public `tool_name` returned by `tool_search` is present in the same-turn execution map when `is_loaded=true`.
5. As a model-context consumer, `tool_search` results are concise, structured, and avoid dumping long MCP tool descriptions unless explicitly debugging.

### Functional Requirements

- `tool_search(query="run shell command")` must rank `start_process` above `start_search`, `get_config`, and `interact_with_process` when those tools match the sample desktop-commander-like catalog.
- `tool_search(query="run python script")` must rank `start_process` first for the same catalog.
- `tool_search(query="search file contents")` must rank `start_search` first.
- `tool_search(query="write file")` must rank `write_file` above `edit_block` unless the query includes edit/patch/replace language.
- `tool_search(query="apply_patch")`, `tool_search(query="patch file")`, and `tool_search(query="edit existing file")` must rank `edit_block` first.
- `tool_search(query="get server config")` must rank `get_config` first.
- Public results must keep the existing keys `tool_name`, `description`, `arg_hints`, and `is_loaded`.
- Public results must add compact keys that help the model decide:
  - `confidence`: `high`, `medium`, or `low`
  - `match_reasons`: at most two short strings
  - `requires_refinement`: response-level boolean
  - `recommended_tool`: response-level object or `null`
  - `next_action`: response-level string such as `call_recommended_tool`, `refine_search`, or `inspect_inventory`
- `description` in public results must be a compact purpose string, not the first 200 characters of a long MCP prompt.
- Model-facing JSON must be compact, not pretty-printed with two-space indentation.
- Full raw descriptions and numeric scores must not be included in normal model-facing output.
- Debug output may include score breakdowns only when an explicit debug setting is enabled.

### Non-Goals

- Do not add embeddings or a vector index for tool search in this iteration.
- Do not maintain a large hand-authored synonym dictionary for every tool.
- Do not inject full tool inventories into system prompts.
- Do not remove inventory mode.
- Do not change custom-agent allowlist semantics except where result ranking and formatting naturally apply.
- Do not change unrelated router behavior.

### Production Constraints

- Context budget matters. Discovery responses should default to three public candidates and include only compact descriptions and reasons.
- The result contract must be deterministic and testable. Scores can change internally, but golden queries should have stable top results and confidence.
- Tool loading must be truthful. `is_loaded=true` means the exact returned `tool_name` can be found in the same-turn execution map.
- Backward compatibility matters. Existing callers that read `results[*].tool_name`, `description`, `arg_hints`, and `is_loaded` must keep working.

## Current Findings

- Scoring is mostly token overlap plus whole-query name match. It does not distinguish high-value fields such as tool name and required args from low-value long descriptions.
- Long descriptions can contain broad words such as "command", "file", or "shell", allowing config/search/process-interaction tools to beat the direct execution tool.
- A local reproduction with the plan's desktop-commander-like fixture confirms the failure class:
  - `run shell command` currently lets `get_config` tie `start_process`; name sort places `get_config` first.
  - `run python script` currently returns no matching tool.
  - `search file contents` currently ranks `write_file` above `start_search`.
  The exact production order can vary by catalog text, but the underlying scoring defect is present.
- `mcp_tool_search_autoload_top_k=3` can load the wrong top three tools when ranking is wrong.
- Public results currently expose truncated raw descriptions, which wastes context and can make tool purpose ambiguous.
- `ToolReference.call_name` is stored and returned for ambiguous tools, but `get_deferred_tools_for_binding()` currently returns the raw MCP tool object name. A public alias such as `brave__search` can therefore appear callable while the execution map contains `search`.
- The prompt tells agents to refine if results are weak or ambiguous, but the result shape does not explicitly say whether refinement is needed.
- Client-side tool search currently uses `ClientToolCatalog.search()` rather than a scored API, so `_merge_search_results()` assigns synthetic scores (`1000 - idx`) to client results. Any ranking change must give server and client catalogs the same scored result contract before merging.
- Deferred tool state must use the same key everywhere. Custom agents bind loaded tools by `tool_state_key`, so tool execution context, same-turn refresh, snapshot hydration/persistence, and recovery must not fall back to `agent_config_key` when `tool_state_key` is present.

## Target Public Result Shape

Normal discovery result:

```json
{
  "query": "run shell command",
  "mode": "discovery",
  "recommended_tool": {
    "tool_name": "start_process",
    "confidence": "high",
    "is_loaded": true
  },
  "results": [
    {
      "tool_name": "start_process",
      "description": "Start a shell command or local process.",
      "arg_hints": "(command*, timeout_ms*, shell)",
      "is_loaded": true,
      "confidence": "high",
      "match_reasons": ["matches shell command intent", "requires command argument"]
    },
    {
      "tool_name": "interact_with_process",
      "description": "Send input to an already running process.",
      "arg_hints": "(pid*, input*, timeout_ms)",
      "is_loaded": false,
      "confidence": "medium",
      "match_reasons": ["process control match"]
    }
  ],
  "requires_refinement": false,
  "next_action": "call_recommended_tool",
  "loaded_count": 1,
  "more_available": true
}
```

Low-confidence result:

```json
{
  "query": "do computer thing",
  "mode": "discovery",
  "recommended_tool": null,
  "results": [
    {
      "tool_name": "start_process",
      "description": "Start a shell command or local process.",
      "arg_hints": "(command*, timeout_ms*, shell)",
      "is_loaded": false,
      "confidence": "low",
      "match_reasons": ["weak process match"]
    }
  ],
  "requires_refinement": true,
  "next_action": "refine_search",
  "loaded_count": 0,
  "more_available": false
}
```

## File Map

- Create: `app/ai/tool_search_profiles.py`
  - Builds compact capability profiles for tools.
  - Infers query intent from the user's search phrase.
  - Produces compact purpose strings and match reasons.
- Modify: `app/ai/tool_search_scoring.py`
  - Replace unweighted token-overlap ranking with intent-aware field-weighted ranking.
  - Return score metadata, confidence, and reasons.
- Modify: `app/ai/mcp_tool_catalog.py`
  - Build and cache profiles for server MCP tools.
  - Return scored metadata from `search_scored()`.
  - Use compact public descriptions.
- Modify: `app/ai/client_tool_catalog.py`
  - Mirror server scoring and result formatting for device-scoped client tools.
  - Add `search_scored()` so client and server results are merged using comparable scores instead of synthetic fallback scores.
- Modify: `app/ai/tool_search_tool.py`
  - Add `recommended_tool`, `requires_refinement`, `next_action`, compact JSON formatting, and debug-gated score breakdowns.
  - Change autoload from "top N above threshold" to "recommended high-confidence candidate, plus optional same-capability companion only when configured".
- Modify: `app/ai/deferred_tool_binding.py`
  - Wrap server tools with alias names when `ToolReference.call_name` differs from raw `tool_name`.
- Modify: `app/ai/tool_execution.py`
  - Refresh and recover tools by returned `call_name` as well as raw names.
  - Use `agent.tool_state_key` when reading deferred state.
- Modify: `app/ai/graph.py`
  - Pass `agent.tool_state_key` into tool execution context and deferred snapshot hydration/persistence so custom-agent loaded tools are not stored under the shared `custom` key.
- Modify: `app/core/config.py`
  - Add result-format and confidence settings.
  - Reduce default discovery result count and autoload aggressiveness.
- Modify: `app/ai/prompts.py`
  - Instruct agents to call a high-confidence loaded `recommended_tool` instead of issuing synonym searches.
- Create: `tests/test_tool_search_accuracy.py`
  - Golden query ranking and autoload tests.
- Modify: `tests/test_tool_search_scoring.py`
  - Unit tests for scoring metadata and confidence thresholds.
- Modify: `tests/test_unified_tool_search.py`
  - Contract tests for result shape and compact formatting.
- Modify: `tests/test_client_tool_isolation.py`
  - Alias binding and same-turn execution-map tests.
- Modify: `tests/test_tool_execution_recovery.py`
  - Recovery tests for aliased server tools.
- Modify: `tests/test_tool_search_prompt_guidance.py`
  - Prompt tests for recommendation handling and no repeated synonym searches.
- Create: `scripts/evaluate_tool_search_accuracy.py`
  - Offline golden-query evaluator for production tuning.

## Implementation Tasks

### Task 1: Add Golden Accuracy Tests

**Files:**
- Create: `tests/test_tool_search_accuracy.py`
- Modify: `tests/test_tool_search_scoring.py`

- [x] **Step 1: Write failing tests for the sample failure**

Create `tests/test_tool_search_accuracy.py`:

```python
from __future__ import annotations

from app.ai.mcp_tool_catalog import ToolDescriptor
from app.ai.tool_search_scoring import rank_tool_candidates


def _tool(
    name: str,
    description: str,
    args: list[str],
    required: list[str] | None = None,
) -> ToolDescriptor:
    return ToolDescriptor(
        tool_name=name,
        server_name="desktop_commander",
        description=description,
        arg_names=args,
        required_arg_names=required or [],
        schema_fingerprint=f"fp-{name}",
    )


def _desktop_tools() -> list[ToolDescriptor]:
    return [
        _tool(
            "start_search",
            "Start a streaming search that can return results progressively. "
            "Search files by path and pattern.",
            ["path", "pattern", "searchType", "filePattern", "ignoreCase", "maxResults"],
            ["path", "pattern"],
        ),
        _tool(
            "get_config",
            "Get the complete server configuration as JSON. Config includes "
            "blockedCommands and shell command policy.",
            [],
            [],
        ),
        _tool(
            "interact_with_process",
            "Send input to a running process and receive the response.",
            ["pid", "input", "timeout_ms", "wait_for_prompt"],
            ["pid", "input"],
        ),
        _tool(
            "start_process",
            "Start a new terminal process with intelligent state detection. "
            "Primary tool for command execution and data processing.",
            ["command", "timeout_ms", "shell", "verbose_timing"],
            ["command", "timeout_ms"],
        ),
        _tool(
            "create_directory",
            "Create a new directory or ensure a directory exists.",
            ["path"],
            ["path"],
        ),
        _tool(
            "edit_block",
            "Apply surgical edits to files. Best for patching existing text.",
            ["file_path", "old_string", "new_string", "expected_replacements"],
            ["file_path"],
        ),
        _tool(
            "write_file",
            "Write or append to file contents.",
            ["path", "content", "mode"],
            ["path", "content"],
        ),
    ]


def _rank(query: str):
    return rank_tool_candidates(query=query, candidates=_desktop_tools())


def test_run_shell_command_prefers_start_process():
    ranked = _rank("run shell command")

    assert ranked[0].tool.tool_name == "start_process"
    assert ranked[0].confidence == "high"
    assert "shell_exec" in ranked[0].profile.capabilities
    assert ranked[0].autoload_eligible is True
    assert all(
        item.tool.tool_name not in {"start_search", "get_config"}
        for item in ranked[:2]
    )


def test_run_python_script_prefers_start_process():
    ranked = _rank("run python script")

    assert ranked[0].tool.tool_name == "start_process"
    assert ranked[0].confidence == "high"


def test_search_file_contents_prefers_start_search():
    ranked = _rank("search file contents")

    assert ranked[0].tool.tool_name == "start_search"
    assert "file_search" in ranked[0].profile.capabilities


def test_write_new_file_prefers_write_file_over_edit_block():
    ranked = _rank("write file")

    assert ranked[0].tool.tool_name == "write_file"
    assert ranked[1].tool.tool_name != "write_file"


def test_patch_existing_file_prefers_edit_block():
    ranked = _rank("apply_patch")

    assert ranked[0].tool.tool_name == "edit_block"


def test_get_server_config_prefers_get_config():
    ranked = _rank("get server config")

    assert ranked[0].tool.tool_name == "get_config"
```

- [x] **Step 2: Run the new tests and verify they fail**

Run:

```bash
python -m pytest tests/test_tool_search_accuracy.py -q
```

Expected: fail because `rank_tool_candidates` does not exist yet.

- [x] **Step 3: Add scoring metadata unit tests**

Append to `tests/test_tool_search_scoring.py`:

```python
def test_score_metadata_marks_weak_description_only_matches_low_confidence():
    from app.ai.tool_search_scoring import rank_tool_candidates
    from app.ai.mcp_tool_catalog import ToolDescriptor

    tools = [
        ToolDescriptor(
            tool_name="get_config",
            server_name="desktop_commander",
            description="Configuration includes blocked shell commands.",
            arg_names=[],
            required_arg_names=[],
            schema_fingerprint="fp-config",
        ),
        ToolDescriptor(
            tool_name="start_process",
            server_name="desktop_commander",
            description="Start a terminal process.",
            arg_names=["command", "timeout_ms", "shell"],
            required_arg_names=["command"],
            schema_fingerprint="fp-process",
        ),
    ]

    ranked = rank_tool_candidates(query="run shell command", candidates=tools)

    assert ranked[0].tool.tool_name == "start_process"
    assert ranked[0].confidence == "high"
    assert ranked[1].tool.tool_name == "get_config"
    assert ranked[1].confidence in {"low", "medium"}
    assert ranked[1].autoload_eligible is False
```

- [x] **Step 4: Run the scoring tests and verify failures are limited to the new API**

Run:

```bash
python -m pytest tests/test_tool_search_accuracy.py tests/test_tool_search_scoring.py -q
```

Expected: existing scoring tests pass or fail only where the new return type is not implemented yet.

### Task 2: Build Tool Profiles and Query Intents

**Files:**
- Create: `app/ai/tool_search_profiles.py`
- Modify: `app/ai/text_normalization.py`
- Test: `tests/test_tool_search_accuracy.py`

- [x] **Step 1: Add lightweight normalization helpers**

Modify `app/ai/text_normalization.py`:

```python
def split_identifier_tokens(value: str | None) -> list[str]:
    """Split snake/camel/kebab identifiers into normalized tokens."""
    raw = str(value or "").replace("-", "_")
    expanded: list[str] = []
    current = ""
    for char in raw:
        if char == "_":
            if current:
                expanded.append(current)
                current = ""
            continue
        if current and char.isupper() and current[-1].islower():
            expanded.append(current)
            current = char
            continue
        current += char
    if current:
        expanded.append(current)
    return tokenize_text(" ".join(expanded))
```

- [x] **Step 2: Create profile and intent dataclasses**

Create `app/ai/tool_search_profiles.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .text_normalization import split_identifier_tokens, tokenize_text


@dataclass(frozen=True)
class QueryIntent:
    raw_query: str
    tokens: set[str]
    capabilities: set[str] = field(default_factory=set)
    action_verbs: set[str] = field(default_factory=set)
    target_terms: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ToolCapabilityProfile:
    tool_name: str
    server_name: str
    name_tokens: set[str]
    arg_tokens: set[str]
    required_arg_tokens: set[str]
    description_tokens: set[str]
    capabilities: set[str]
    purpose: str


_SHELL_TERMS = {"run", "execute", "shell", "command", "terminal", "process", "python", "script"}
_FILE_SEARCH_TERMS = {"search", "find", "grep", "pattern", "contents", "content"}
_FILE_EDIT_TERMS = {"edit", "patch", "apply", "replace", "modify", "surgical"}
_FILE_WRITE_TERMS = {"write", "create", "append", "save"}
_CONFIG_TERMS = {"config", "configuration", "settings", "blockedcommands"}


def infer_query_intent(query: str | None) -> QueryIntent:
    raw_query = str(query or "").strip()
    tokens = set(tokenize_text(raw_query))
    capabilities: set[str] = set()
    action_verbs: set[str] = set()
    target_terms: set[str] = set(tokens)

    if tokens & _SHELL_TERMS and (tokens & {"run", "execute", "command", "shell", "python", "script"}):
        capabilities.add("shell_exec")
    if tokens & _FILE_SEARCH_TERMS and tokens & {"file", "files", "contents", "content", "pattern"}:
        capabilities.add("file_search")
    if tokens & _FILE_EDIT_TERMS:
        capabilities.add("file_edit")
    if tokens & _FILE_WRITE_TERMS and tokens & {"file", "files", "content", "contents"}:
        capabilities.add("file_write")
    if tokens & _CONFIG_TERMS:
        capabilities.add("config_read")

    action_verbs.update(tokens & (_SHELL_TERMS | _FILE_SEARCH_TERMS | _FILE_EDIT_TERMS | _FILE_WRITE_TERMS))
    return QueryIntent(
        raw_query=raw_query,
        tokens=tokens,
        capabilities=capabilities,
        action_verbs=action_verbs,
        target_terms=target_terms,
    )


def infer_tool_profile(
    *,
    tool_name: str,
    server_name: str = "",
    description: str = "",
    arg_names: list[str] | None = None,
    required_arg_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolCapabilityProfile:
    arg_names = arg_names or []
    required_arg_names = required_arg_names or []
    name_tokens = set(split_identifier_tokens(tool_name))
    arg_tokens = set(token for arg in arg_names for token in split_identifier_tokens(arg))
    required_arg_tokens = set(
        token for arg in required_arg_names for token in split_identifier_tokens(arg)
    )
    description_tokens = set(tokenize_text(description))
    all_tokens = name_tokens | arg_tokens | required_arg_tokens | description_tokens
    capabilities: set[str] = set()

    if (
        {"start", "process"} <= name_tokens
        or "command" in required_arg_tokens
        or ("shell" in arg_tokens and "command" in arg_tokens)
    ):
        capabilities.add("shell_exec")
    if "search" in name_tokens and ({"path", "pattern"} & required_arg_tokens):
        capabilities.add("file_search")
    if "edit" in name_tokens or "patch" in name_tokens or {"old", "new", "string"} <= arg_tokens:
        capabilities.add("file_edit")
    if ("write" in name_tokens and "content" in required_arg_tokens) or (
        "path" in required_arg_tokens and "content" in required_arg_tokens
    ):
        capabilities.add("file_write")
    if "config" in name_tokens and not ({"set", "write"} & name_tokens):
        capabilities.add("config_read")
    if "set" in name_tokens and "config" in name_tokens:
        capabilities.add("config_write")
    if "interact" in name_tokens and "process" in name_tokens:
        capabilities.add("process_interaction")

    purpose = _compact_purpose(tool_name, capabilities, description, all_tokens)
    return ToolCapabilityProfile(
        tool_name=tool_name,
        server_name=server_name,
        name_tokens=name_tokens,
        arg_tokens=arg_tokens,
        required_arg_tokens=required_arg_tokens,
        description_tokens=description_tokens,
        capabilities=capabilities,
        purpose=purpose,
    )


def _compact_purpose(
    tool_name: str,
    capabilities: set[str],
    description: str,
    all_tokens: set[str],
) -> str:
    if "shell_exec" in capabilities:
        return "Start a shell command or local process."
    if "file_search" in capabilities:
        return "Search file contents by path and pattern."
    if "file_edit" in capabilities:
        return "Apply focused edits to existing file text."
    if "file_write" in capabilities:
        return "Write or append file contents."
    if "config_read" in capabilities:
        return "Read server configuration."
    if "process_interaction" in capabilities:
        return "Send input to an already running process."
    first_line = " ".join(str(description or "").strip().split())
    if first_line:
        return first_line[:120]
    return f"Use {tool_name}."
```

- [x] **Step 3: Run profile tests through golden ranking tests**

Run:

```bash
python -m pytest tests/test_tool_search_accuracy.py -q
```

Expected: still fail because the scorer has not consumed profiles.

### Task 3: Replace Ranking With Intent-Aware Scoring

**Files:**
- Modify: `app/ai/tool_search_scoring.py`
- Test: `tests/test_tool_search_accuracy.py`, `tests/test_tool_search_scoring.py`

- [x] **Step 1: Add scored result dataclass and ranking API**

Modify `app/ai/tool_search_scoring.py`:

```python
from dataclasses import dataclass
from typing import Any, Iterable

from .tool_search_profiles import (
    QueryIntent,
    ToolCapabilityProfile,
    infer_query_intent,
    infer_tool_profile,
)


@dataclass(frozen=True)
class ToolSearchScore:
    tool: Any
    score: float
    confidence: str
    match_reasons: list[str]
    profile: ToolCapabilityProfile
    autoload_eligible: bool


def rank_tool_candidates(
    *,
    query: str,
    candidates: Iterable[Any],
    min_relevance_score: float | None = None,
    high_confidence_margin: float = 8.0,
) -> list[ToolSearchScore]:
    intent = infer_query_intent(query)
    scored = [_score_candidate(intent, tool) for tool in candidates]
    scored = [item for item in scored if item.score >= (min_relevance_score or 1.0)]
    scored.sort(key=lambda item: (-item.score, getattr(item.tool, "tool_name", "")))

    if not scored:
        return []

    top_score = scored[0].score
    second_score = scored[1].score if len(scored) > 1 else 0.0
    margin = top_score - second_score
    result: list[ToolSearchScore] = []
    for index, item in enumerate(scored):
        confidence = _confidence_for(item.score, margin if index == 0 else 0.0)
        result.append(
            ToolSearchScore(
                tool=item.tool,
                score=item.score,
                confidence=confidence,
                match_reasons=item.match_reasons[:2],
                profile=item.profile,
                autoload_eligible=(index == 0 and confidence == "high" and margin >= high_confidence_margin),
            )
        )
    return result
```

- [x] **Step 2: Implement `_score_candidate()` with field weights**

Add to `app/ai/tool_search_scoring.py`:

```python
def _score_candidate(intent: QueryIntent, tool: Any) -> ToolSearchScore:
    tool_name = str(getattr(tool, "tool_name", getattr(tool, "name", "")) or "")
    profile = infer_tool_profile(
        tool_name=tool_name,
        server_name=str(getattr(tool, "server_name", "") or ""),
        description=str(getattr(tool, "description", "") or ""),
        arg_names=list(getattr(tool, "arg_names", []) or []),
        required_arg_names=list(getattr(tool, "required_arg_names", []) or []),
    )

    score = 0.0
    reasons: list[str] = []

    capability_overlap = intent.capabilities & profile.capabilities
    if capability_overlap:
        score += 40.0 * len(capability_overlap)
        reasons.append(f"matches {sorted(capability_overlap)[0]} intent")

    name_overlap = intent.tokens & profile.name_tokens
    if name_overlap:
        score += 14.0 * len(name_overlap)
        reasons.append(f"name match: {', '.join(sorted(name_overlap)[:2])}")

    required_arg_overlap = intent.tokens & profile.required_arg_tokens
    if required_arg_overlap:
        score += 12.0 * len(required_arg_overlap)
        reasons.append(f"required arg match: {', '.join(sorted(required_arg_overlap)[:2])}")

    arg_overlap = intent.tokens & profile.arg_tokens
    if arg_overlap:
        score += 6.0 * len(arg_overlap)
        reasons.append(f"arg match: {', '.join(sorted(arg_overlap)[:2])}")

    description_overlap = intent.tokens & profile.description_tokens
    if description_overlap:
        score += min(4.0, 1.0 * len(description_overlap))

    score += _capability_specific_adjustment(intent, profile)

    return ToolSearchScore(
        tool=tool,
        score=score,
        confidence=_confidence_for(score, 0.0),
        match_reasons=reasons or ["weak lexical match"],
        profile=profile,
        autoload_eligible=False,
    )
```

- [x] **Step 3: Add capability-specific adjustments**

Add to `app/ai/tool_search_scoring.py`:

```python
def _capability_specific_adjustment(
    intent: QueryIntent,
    profile: ToolCapabilityProfile,
) -> float:
    score = 0.0
    if "shell_exec" in intent.capabilities:
        if "shell_exec" in profile.capabilities:
            score += 35.0
        if "file_search" in profile.capabilities:
            score -= 18.0
        if "config_read" in profile.capabilities:
            score -= 16.0
        if "process_interaction" in profile.capabilities:
            score -= 8.0

    if "file_write" in intent.capabilities:
        if "file_write" in profile.capabilities:
            score += 30.0
        if "file_edit" in profile.capabilities and "edit" not in intent.tokens and "patch" not in intent.tokens:
            score -= 10.0

    if "file_edit" in intent.capabilities and "file_edit" in profile.capabilities:
        score += 30.0

    if "file_search" in intent.capabilities and "file_search" in profile.capabilities:
        score += 30.0

    if "config_read" in intent.capabilities and "config_read" in profile.capabilities:
        score += 25.0

    return score
```

- [x] **Step 4: Add confidence thresholds**

Add to `app/ai/tool_search_scoring.py`:

```python
def _confidence_for(score: float, margin: float) -> str:
    if score >= 60.0 and margin >= 8.0:
        return "high"
    if score >= 25.0:
        return "medium"
    return "low"
```

- [x] **Step 5: Preserve existing `score_tool()` tests**

Keep the existing `score_tool()` function temporarily as a compatibility shim. Update it to call the new field-weighted scorer only when enough metadata is available, or leave it unchanged until all catalogs switch to `rank_tool_candidates()`.

- [x] **Step 6: Run scoring tests**

Run:

```bash
python -m pytest tests/test_tool_search_accuracy.py tests/test_tool_search_scoring.py -q
```

Expected: all tests pass after tuning constants.

### Task 4: Integrate Ranking Into Server and Client Catalogs

**Files:**
- Modify: `app/ai/mcp_tool_catalog.py`
- Modify: `app/ai/client_tool_catalog.py`
- Test: `tests/test_tool_search_accuracy.py`, `tests/test_unified_tool_search.py`

- [x] **Step 1: Update server catalog `search_scored()` return type**

Change `McpToolCatalog.search_scored()` to return `list[ToolSearchScore]` instead of `list[tuple[ToolDescriptor, float]]`.

Implementation pattern:

```python
from .tool_search_scoring import ToolSearchScore, rank_tool_candidates


def search_scored(
    self,
    query: str,
    top_k: int = 5,
    server_name: str | None = None,
    allowlist: list[str] | None = None,
) -> list[ToolSearchScore]:
    canonical_server = self.resolve_server_name(server_name) if server_name else None
    if server_name and canonical_server is None:
        return []

    candidates = (
        self._tools_by_server.get(canonical_server, []) if canonical_server else self._tools
    )
    if allowlist:
        allowlist_set = set(allowlist)
        candidates = [
            tool for tool in candidates if self._descriptor_matches_allowlist(tool, allowlist_set)
        ]
    if not query or not query.strip():
        return []
    return rank_tool_candidates(query=query, candidates=candidates)[:top_k]
```

- [x] **Step 2: Keep `search()` backward-compatible**

Change `McpToolCatalog.search()` query mode to use `search_scored()` and unwrap tools:

```python
if not query or not query.strip():
    return candidates[:top_k]

return [item.tool for item in self.search_scored(query, top_k, server_name, allowlist)]
```

- [x] **Step 3: Apply the same pattern in `ClientToolCatalog`**

Add a `search_scored()` method to `app/ai/client_tool_catalog.py` with the same return type and update `search()` to unwrap tools. This is required before changing `_merge_search_results()` because client results currently reach the merge path as plain descriptors and receive synthetic `1000 - idx` scores.

```python
from .tool_search_scoring import ToolSearchScore, rank_tool_candidates


def search_scored(
    self,
    query: str,
    top_k: int = 5,
    server_name: str | None = None,
    allowlist: list[str] | None = None,
) -> list[ToolSearchScore]:
    canonical_server = self.resolve_server_name(server_name) if server_name else None
    if server_name and canonical_server is None:
        return []

    candidates = (
        self._tools_by_server.get(canonical_server, []) if canonical_server else self._tools
    )
    if allowlist:
        allowlist_set = set(allowlist)
        candidates = [
            tool for tool in candidates if self._descriptor_matches_allowlist(tool, allowlist_set)
        ]
    if not query or not query.strip():
        return []
    return rank_tool_candidates(query=query, candidates=candidates)[:top_k]
```

Update query mode in `ClientToolCatalog.search()`:

```python
if not query or not query.strip():
    return candidates[:top_k]

return [item.tool for item in self.search_scored(query, top_k, server_name, allowlist)]
```

- [x] **Step 4: Use scored client results in `tool_search`**

Modify the client search branch in `app/ai/tool_search_tool.py`:

```python
if query and hasattr(client_catalog, "search_scored"):
    client_results = client_catalog.search_scored(
        query=query,
        top_k=effective_top_k * 2,
        server_name=server_name,
        allowlist=effective_client_allowlist,
    )
else:
    client_results = client_catalog.search(
        query=query,
        top_k=effective_top_k * 2,
        server_name=server_name,
        allowlist=effective_client_allowlist,
    )
```

- [x] **Step 5: Update existing tests that expect tuple scores**

Search:

```bash
rg -n "search_scored|_score|score_tool|rank_and_filter" tests app/ai
```

Update tests to use `item.tool` and `item.score` where needed. Keep `_merge_search_results()` tuple handling temporarily so older test fakes and external callers that still return `(descriptor, score)` continue to work during the migration.

- [x] **Step 6: Run targeted catalog tests**

Run:

```bash
python -m pytest tests/test_tool_search_accuracy.py tests/test_unified_tool_search.py tests/test_tool_search_scoring.py -q
```

Expected: all pass.

### Task 5: Compact and Actionable `tool_search` Output

**Files:**
- Modify: `app/ai/tool_search_tool.py`
- Modify: `app/ai/mcp_tool_catalog.py`
- Modify: `app/ai/client_tool_catalog.py`
- Modify: `app/core/config.py`
- Test: `tests/test_unified_tool_search.py`

- [x] **Step 1: Add context-budget settings**

Modify `app/core/config.py`:

```python
mcp_tool_search_default_top_k: int = Field(
    default=3,
    description="Default public candidates returned for tool_search discovery queries.",
)
mcp_tool_search_max_top_k: int = Field(
    default=100,
    description=(
        "Maximum explicit top_k accepted for tool_search discovery queries. "
        "Keep high for compatibility; default output stays compact."
    ),
)
mcp_tool_search_description_max_chars: int = Field(
    default=120,
    description="Maximum characters in model-facing tool_search descriptions.",
)
mcp_tool_search_match_reasons_max: int = Field(
    default=2,
    description="Maximum match reasons exposed per search result.",
)
mcp_tool_search_debug_scores: bool = Field(
    default=False,
    description="Include explicit tool_search score diagnostics in discovery output.",
)
```

Add `mcp_tool_search_description_max_chars` and `mcp_tool_search_match_reasons_max` to the existing non-negative integer validator list. Keep inventory settings unchanged (`20` default, `50` max) because inventory is already server-summary oriented. Do not reduce `mcp_tool_search_max_top_k` in this iteration; explicit larger searches are part of inventory/debug workflows.

- [x] **Step 2: Update result models**

Modify `ToolSearchResult` and `ToolSearchOutput` in `app/ai/tool_search_tool.py`:

```python
class ToolSearchResult(BaseModel):
    tool_name: str
    description: str
    arg_hints: str
    is_loaded: bool = False
    confidence: str = "low"
    match_reasons: list[str] = Field(default_factory=list)


class RecommendedTool(BaseModel):
    tool_name: str
    confidence: str
    is_loaded: bool


class ToolSearchOutput(BaseModel):
    query: str | None
    mode: str
    resolved_server_name: str | None = None
    inventory: list[dict[str, Any]] = Field(default_factory=list)
    recommended_tool: RecommendedTool | None = None
    results: list[ToolSearchResult]
    requires_refinement: bool
    next_action: str
    loaded_count: int
    more_available: bool
```

- [x] **Step 3: Convert scored candidates into compact public results**

Add helper in `app/ai/tool_search_tool.py`:

```python
def _public_result_from_scored(item: Any, *, is_client: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    score_meta = item if hasattr(item, "tool") else None
    tool = score_meta.tool if score_meta is not None else item

    public = tool.to_search_result()
    if score_meta is not None:
        public["description"] = score_meta.profile.purpose[: settings.mcp_tool_search_description_max_chars]
        public["confidence"] = score_meta.confidence
        public["match_reasons"] = score_meta.match_reasons[
            : settings.mcp_tool_search_match_reasons_max
        ]

    internal = tool._to_internal_result()
    if score_meta is not None:
        internal["_score"] = score_meta.score
        internal["_confidence"] = score_meta.confidence
        internal["_autoload_eligible"] = score_meta.autoload_eligible
        internal["_match_reasons"] = list(score_meta.match_reasons)
        if settings.mcp_tool_search_debug_scores:
            internal["_debug_score"] = {
                "score": score_meta.score,
                "capabilities": sorted(score_meta.profile.capabilities),
            }
    return public, internal
```

- [x] **Step 4: Update `_merge_search_results()` to accept scored objects**

Change `_merge_search_results()` to call `_public_result_from_scored()` when an item has a `.tool` attribute. Preserve tuple handling only as a temporary compatibility path:

```python
if hasattr(item, "tool"):
    public_dict, internal_dict = _public_result_from_scored(item)
    real_score = float(item.score)
elif isinstance(item, tuple) and len(item) == 2:
    desc, real_score = item
    public_dict = desc.to_search_result()
    internal_dict = desc._to_internal_result()
else:
    desc = item
    real_score = float(1000 - idx)
    public_dict = desc.to_search_result()
    internal_dict = desc._to_internal_result()
```

- [x] **Step 5: Add recommendation decision helper**

Add:

```python
def _build_recommendation(
    public_results: list[dict[str, Any]],
    internal_results: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool, str]:
    if not public_results or not internal_results:
        return None, True, "refine_search"

    top_public = public_results[0]
    top_internal = internal_results[0]
    confidence = str(top_public.get("confidence") or top_internal.get("_confidence") or "low")
    is_loaded = bool(top_public.get("is_loaded"))

    if confidence == "high" and is_loaded:
        return (
            {
                "tool_name": top_public["tool_name"],
                "confidence": confidence,
                "is_loaded": is_loaded,
            },
            False,
            "call_recommended_tool",
        )
    if confidence == "medium" and is_loaded and len(public_results) == 1:
        return (
            {
                "tool_name": top_public["tool_name"],
                "confidence": confidence,
                "is_loaded": is_loaded,
            },
            False,
            "call_recommended_tool",
        )
    if confidence in {"high", "medium"} and not is_loaded:
        return None, True, "refine_search"
    return None, True, "refine_search"
```

Call `_build_recommendation()` only after the autoload pass and after public `is_loaded` flags have been updated. Inventory returns must include response-level action fields even though they do not recommend a tool:

```python
return {
    "query": None,
    "mode": "inventory",
    "resolved_server_name": resolved_server_name,
    "inventory": inventory,
    "recommended_tool": None,
    "results": [],
    "requires_refinement": False,
    "next_action": "inspect_inventory",
    "loaded_count": 0,
    "more_available": False,
}
```

Normal discovery returns:

```python
recommended_tool, requires_refinement, next_action = _build_recommendation(
    public_results,
    internal_results,
)
result = {
    "query": query,
    "mode": mode,
    "recommended_tool": recommended_tool,
    "results": public_results,
    "requires_refinement": requires_refinement,
    "next_action": next_action,
    "loaded_count": loaded_count,
    "more_available": truncated,
}
if settings.mcp_tool_search_debug_scores:
    result["debug_scores"] = [
        {
            "tool_name": public.get("tool_name"),
            "score": internal.get("_score"),
            "confidence": internal.get("_confidence"),
            "reasons": internal.get("_match_reasons", []),
        }
        for public, internal in zip(public_results, internal_results, strict=False)
    ]
```

- [x] **Step 6: Return compact JSON from tool functions**

Change both `tool_search()` and `tool_search_impl()` final returns:

```python
return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
```

Keep UI rendering responsible for pretty display.

- [x] **Step 7: Add result-shape tests**

Append to `tests/test_unified_tool_search.py`:

```python
def test_tool_search_result_description_is_compact_purpose():
    desc = ToolDescriptor(
        tool_name="start_process",
        server_name="desktop_commander",
        description="Start a new terminal process with intelligent state detection.\n\n"
        "PRIMARY TOOL FOR FILE ANALYSIS AND DATA PROCESSING\n" + "x" * 500,
        arg_names=["command", "timeout_ms", "shell"],
        required_arg_names=["command"],
        schema_fingerprint="fp-start-process",
    )

    ranked = rank_tool_candidates(query="run shell command", candidates=[desc])
    public, _ = _merge_search_results(
        server_results=ranked,
        client_results=[],
        query="run shell command",
        top_k=3,
    )

    assert public[0]["description"] == "Start a shell command or local process."
    assert len(public[0]["description"]) <= 120
    assert public[0]["confidence"] == "high"
    assert len(public[0]["match_reasons"]) <= 2
```

- [x] **Step 8: Run result-shape tests**

Run:

```bash
python -m pytest tests/test_unified_tool_search.py tests/test_tool_search_accuracy.py -q
```

Expected: pass.

### Task 6: Make Autoload Conservative and Useful

**Files:**
- Modify: `app/ai/tool_search_tool.py`
- Modify: `app/core/config.py`
- Test: `tests/test_unified_tool_search.py`, `tests/test_tool_search_accuracy.py`

- [x] **Step 1: Change default autoload cap**

Modify `app/core/config.py`:

```python
mcp_tool_search_autoload_top_k: int = Field(
    default=1,
    description="Maximum high-confidence recommended tools to automatically load after tool_search.",
)
```

- [x] **Step 2: Gate autoload by `_autoload_eligible`**

Modify the autoload loop in `_execute_tool_search()`:

```python
for internal in internal_results[:autoload_top_k]:
    if not bool(internal.get("_autoload_eligible")):
        continue
    # existing server/client reference construction follows
```

Remove broad score-threshold autoload of the first three candidates. Keep `mcp_tool_search_autoload_min_relevance_score` as a secondary safety floor for backward compatibility:

```python
if float(internal.get("_score", 0.0)) < settings.mcp_tool_search_autoload_min_relevance_score:
    continue
```

- [x] **Step 3: Add sample autoload test**

Append to `tests/test_unified_tool_search.py`:

```python
@pytest.mark.asyncio
async def test_run_shell_command_autoloads_only_start_process(monkeypatch):
    from app.ai.tool_context import ToolContext
    from app.ai.tool_search_scoring import rank_tool_candidates
    from tests.test_tool_search_accuracy import _desktop_tools

    class FakeCatalog:
        def search_scored(self, query=None, top_k=5, server_name=None, allowlist=None):
            return rank_tool_candidates(query=query, candidates=_desktop_tools())[:top_k]

        def search(self, query=None, top_k=5, server_name=None, allowlist=None):
            return [item.tool for item in self.search_scored(query, top_k, server_name, allowlist)]

        def is_ambiguous(self, tool_name):
            return False

        def get_server_inventory(self, allowlist=None):
            return []

    class DeferredStateStub:
        def __init__(self):
            self.refs = []

        def autoload(self, **kwargs):
            self.refs.extend(kwargs["references"])
            return kwargs["references"]

        def autoload_client_tools(self, **kwargs):
            return []

    state = DeferredStateStub()

    async def fake_get_global_mcp_manager():
        return object()

    async def fake_get_tool_catalog(_manager):
        return FakeCatalog()

    monkeypatch.setattr("app.ai.tool_search_tool.get_global_mcp_manager", fake_get_global_mcp_manager)
    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", fake_get_tool_catalog)
    monkeypatch.setattr("app.ai.tool_search_tool.get_deferred_tool_state", lambda: state)
    monkeypatch.setattr(
        "app.ai.tool_search_tool.get_tool_context",
        lambda: ToolContext(conversation_id="conv-1", agent_key="chat"),
    )

    result = await _execute_tool_search(query="run shell command")

    assert result["recommended_tool"]["tool_name"] == "start_process"
    assert result["recommended_tool"]["is_loaded"] is True
    assert [ref.tool_name for ref in state.refs] == ["start_process"]
    assert {ref.tool_name for ref in state.refs}.isdisjoint({"start_search", "get_config"})
```

- [x] **Step 4: Run autoload tests**

Run:

```bash
python -m pytest tests/test_unified_tool_search.py tests/test_tool_search_accuracy.py -q
```

Expected: pass.

### Task 7: Fix Alias Binding and Same-Turn Execution

**Files:**
- Modify: `app/ai/deferred_tool_binding.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/graph.py`
- Test: `tests/test_client_tool_isolation.py`, `tests/test_tool_execution_recovery.py`, `tests/test_custom_agents_tools.py`

- [x] **Step 1: Add failing alias binding test**

Append to `tests/test_client_tool_isolation.py`:

```python
def test_deferred_binding_exposes_alias_name_for_ambiguous_server_tool(monkeypatch):
    from app.ai.deferred_tool_binding import get_deferred_tools_for_binding
    from app.ai.deferred_tool_state import get_deferred_tool_state, reset_deferred_tool_state
    from app.ai.mcp_tool_catalog import ToolReference

    reset_deferred_tool_state()

    brave_tool = SimpleNamespace(name="search", description="Brave search")
    tavily_tool = SimpleNamespace(name="search", description="Tavily search")

    class FakeManager:
        _tool_index = {"search": [tavily_tool, brave_tool]}
        _server_tools = {"tavily": [tavily_tool], "brave": [brave_tool]}

        def get_server_for_tool(self, tool):
            return "brave" if tool is brave_tool else "tavily"

    try:
        get_deferred_tool_state().autoload(
            "conv-1",
            "chat",
            [ToolReference(tool_name="search", server_name="brave", call_name="brave__search")],
        )

        tools = get_deferred_tools_for_binding("conv-1", "chat", FakeManager(), [])

        assert [tool.name for tool in tools] == ["brave__search"]
    finally:
        reset_deferred_tool_state()
```

- [x] **Step 2: Add aliased server recovery test**

Append to `tests/test_tool_execution_recovery.py`:

```python
@pytest.mark.asyncio
async def test_execute_tool_calls_recovers_aliased_server_tool(monkeypatch):
    from app.ai.deferred_tool_state import get_deferred_tool_state, reset_deferred_tool_state
    from app.ai.mcp_tool_catalog import ToolReference

    invoked = []

    class _ServerTool:
        name = "search"

        async def ainvoke(self, args):
            invoked.append(args)
            return "searched"

    brave_tool = _ServerTool()

    class _Manager:
        async def get_tools(self):
            return [brave_tool]

        def get_server_for_tool(self, tool):
            return "brave"

    async def _manager():
        return _Manager()

    async def _noop_refresh(**kwargs):
        return None

    reset_deferred_tool_state()
    try:
        get_deferred_tool_state().autoload(
            conversation_id="conv-1",
            agent_key="chat",
            references=[
                ToolReference(
                    tool_name="search",
                    server_name="brave",
                    call_name="brave__search",
                )
            ],
        )
        monkeypatch.setattr("app.ai.tool_execution._refresh_tool_map_after_search", _noop_refresh)
        monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _manager)

        outputs, artifacts, _ = await execute_tool_calls(
            tool_calls=[{"id": "call-1", "name": "brave__search", "args": {"query": "x"}}],
            tool_map={},
            conversation_id="conv-1",
            user_id="user-1",
            agent=SimpleNamespace(agent_config_key="chat", tool_state_key="chat"),
        )

        assert invoked == [{"query": "x"}]
        assert outputs[0]["name"] == "brave__search"
        assert artifacts[0]["status"] == "success"
    finally:
        reset_deferred_tool_state()
```

- [x] **Step 3: Implement alias wrapper**

Add to `app/ai/deferred_tool_binding.py`. The wrapper must expose the public alias as the actual bound tool name; it must not return the raw tool when `args_schema` is missing, because that preserves the bug.

```python
import copy
from typing import Any

from langchain_core.tools import BaseTool


def _tool_with_call_name(tool: BaseTool, call_name: str | None) -> BaseTool:
    if not call_name or getattr(tool, "name", None) == call_name:
        return tool

    metadata: dict[str, Any] = {
        **(getattr(tool, "metadata", {}) or {}),
        "aliased_from_tool_name": getattr(tool, "name", ""),
        "call_name": call_name,
    }
    updates = {"name": call_name, "metadata": metadata}

    model_copy = getattr(tool, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update=updates)
        except Exception:
            pass

    legacy_copy = getattr(tool, "copy", None)
    if callable(legacy_copy):
        try:
            return legacy_copy(update=updates)
        except Exception:
            pass

    alias = copy.copy(tool)
    try:
        setattr(alias, "name", call_name)
    except Exception:
        object.__setattr__(alias, "name", call_name)
    try:
        setattr(alias, "metadata", metadata)
    except Exception:
        object.__setattr__(alias, "metadata", metadata)
    return alias
```

- [x] **Step 4: Use wrapper in deferred binding**

In `get_deferred_tools_for_binding()`, after finding `tool`, append:

```python
deferred_tools.append(_tool_with_call_name(tool, loaded.call_name))
```

To support this, make `LoadedTool` store the public call name while preserving the existing storage-key behavior. Current state already keys `ConversationToolSet.loaded` by `call_name`; the explicit field keeps snapshot/listing/recovery code from needing to infer it from dictionary keys.

```python
@dataclass
class LoadedTool:
    tool_name: str
    server_name: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    generation: int = 0
    call_name: str | None = None
```

Set it in `ConversationToolSet.add()` for both new and existing entries:

```python
if storage_key in self.loaded:
    existing = self.loaded[storage_key]
    existing.server_name = server_name
    existing.generation = generation
    existing.call_name = call_name
    existing.touch()
    return existing

loaded_tool = LoadedTool(
    tool_name=tool_name,
    server_name=server_name,
    generation=generation,
    call_name=call_name,
)
```

Update `LoadedTool.to_reference()`:

```python
def to_reference(self) -> ToolReference:
    return ToolReference(
        tool_name=self.tool_name,
        server_name=self.server_name,
        call_name=self.call_name if self.call_name != self.tool_name else None,
    )
```

Update `ConversationToolSet.list_tools()` so restored snapshots keep the public call name:

```python
def list_tools(self) -> list[ToolReference]:
    return [
        ToolReference(
            tool_name=tool.tool_name,
            server_name=tool.server_name,
            call_name=tool.call_name if tool.call_name and tool.call_name != tool.tool_name else None,
        )
        for tool in self.loaded.values()
    ]
```

- [x] **Step 5: Make execution recovery alias-aware**

In `_recover_missing_tool()`, before raw server recovery, parse aliases from loaded state:

```python
from .deferred_tool_binding import _tool_with_call_name
from .deferred_tool_state import get_deferred_tool_state

agent_key = (
    getattr(agent, "tool_state_key", None)
    or getattr(agent, "agent_config_key", None)
    or "default"
)
state = get_deferred_tool_state()
server_name = state.get_server_for_loaded_tool(conversation_id, agent_key, tool_name)
raw_tool_name = state.get_raw_tool_name_for_loaded_tool(conversation_id, agent_key, tool_name)
```

Add `get_raw_tool_name_for_loaded_tool()` to `DeferredToolState`:

```python
def get_raw_tool_name_for_loaded_tool(
    self,
    conversation_id: str | None,
    agent_key: str | None,
    tool_name: str,
) -> str | None:
    key = self._get_key(conversation_id, agent_key)
    with self._lock:
        tool_set = self._conversation_tools.get(key)
        if not tool_set:
            return None
        tool = tool_set.get(tool_name)
        return tool.tool_name if tool else None
```

Then recover by both server and raw name:

```python
if raw_tool_name and server_name:
    from .mcp_registry import get_global_mcp_manager

    manager = await get_global_mcp_manager()
    for server_tool in await manager.get_tools():
        if (
            getattr(server_tool, "name", None) == raw_tool_name
            and manager.get_server_for_tool(server_tool) == server_name
        ):
            tool_map[tool_name] = _tool_with_call_name(server_tool, tool_name)
            return tool_map[tool_name]
```

- [x] **Step 6: Align deferred-state keys in refresh and graph execution context**

Modify `_refresh_tool_map_after_search()` in `app/ai/tool_execution.py` so it uses `tool_state_key` when present:

```python
agent_key = "default"
if agent:
    agent_key = (
        getattr(agent, "tool_state_key", None)
        or getattr(agent, "agent_config_key", None)
        or "default"
    )
```

Leave `_mark_tool_used_if_deferred()` behavior unchanged after graph contexts are corrected; it already reads `ctx.agent_key`, so the fix is to supply the deferred-state key to `tool_execution_context()`.

Modify `app/ai/graph.py` in every place it enters `tool_execution_context()` or persists/hydrates deferred snapshots for an agent:

```python
agent_key = (
    getattr(agent, "tool_state_key", None)
    or getattr(agent, "agent_config_key", None)
    or getattr(agent, "agent_id", None)
    or "unknown"
)
```

Keep model routing and model configuration keys unchanged; this key alignment is only for deferred tool state.

- [x] **Step 7: Add custom-agent state-key regression**

Append to `tests/test_client_tool_isolation.py`:

```python
@pytest.mark.asyncio
async def test_refresh_tool_map_uses_tool_state_key_for_custom_agent(monkeypatch):
    from types import SimpleNamespace

    from app.ai.deferred_tool_state import get_deferred_tool_state, reset_deferred_tool_state
    from app.ai.mcp_tool_catalog import ToolReference
    from app.ai.tool_execution import _refresh_tool_map_after_search

    reset_deferred_tool_state()
    brave_tool = SimpleNamespace(name="search", description="Brave search")

    class FakeManager:
        _tool_index = {"search": [brave_tool]}
        _server_tools = {"brave": [brave_tool]}

        async def get_tools(self):
            return [brave_tool]

        def get_server_for_tool(self, tool):
            return "brave"

    async def fake_get_global_mcp_manager():
        return FakeManager()

    try:
        get_deferred_tool_state().autoload(
            conversation_id="conv-1",
            agent_key="custom_agent:abc",
            references=[
                ToolReference(
                    tool_name="search",
                    server_name="brave",
                    call_name="brave__search",
                )
            ],
        )

        monkeypatch.setattr(
            "app.ai.mcp_registry.get_global_mcp_manager",
            fake_get_global_mcp_manager,
        )
        tool_map: dict[str, object] = {}

        await _refresh_tool_map_after_search(
            tool_map=tool_map,
            agent=SimpleNamespace(
                agent_config_key="custom",
                tool_state_key="custom_agent:abc",
            ),
            conversation_id="conv-1",
            user_id="user-1",
            device_id=None,
        )

        assert "brave__search" in tool_map
        assert "search" not in tool_map
    finally:
        reset_deferred_tool_state()
```

- [x] **Step 8: Run alias and state-key tests**

Run:

```bash
python -m pytest tests/test_client_tool_isolation.py tests/test_tool_execution_recovery.py tests/test_custom_agents_tools.py -q
```

Expected: pass.

### Task 8: Update Prompt Guidance to Stop Synonym Search Loops

**Files:**
- Modify: `app/ai/prompts.py`
- Test: `tests/test_tool_search_prompt_guidance.py`

- [x] **Step 1: Add prompt rules for recommendations**

Modify `TOOL_EXPLORATION_SUFFIX`:

```text
- After `tool_search`, if `recommended_tool` is present, `confidence` is `high`,
  and `is_loaded` is true, call that tool next. Do not issue another
  `tool_search` with a synonym for the same capability.
- Only refine the search when `requires_refinement` is true, the recommended
  tool is not suitable for the user's actual task, or the needed integration is
  missing from the result.
```

- [x] **Step 2: Add prompt tests**

Append to `tests/test_tool_search_prompt_guidance.py`:

```python
def test_tool_exploration_suffix_uses_recommended_tool_without_synonym_search():
    lower = TOOL_EXPLORATION_SUFFIX.lower()

    assert "recommended_tool" in lower
    assert "do not issue another" in lower
    assert "synonym" in lower
    assert "requires_refinement" in lower
```

- [x] **Step 3: Run prompt tests**

Run:

```bash
python -m pytest tests/test_tool_search_prompt_guidance.py -q
```

Expected: pass.

### Task 9: Add Offline Accuracy Evaluation

**Files:**
- Create: `scripts/evaluate_tool_search_accuracy.py`
- Modify: `README.md`
- Test: run script manually

- [x] **Step 1: Create evaluator script**

Create `scripts/evaluate_tool_search_accuracy.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from app.ai.mcp_tool_catalog import ToolDescriptor
from app.ai.tool_search_scoring import rank_tool_candidates


@dataclass(frozen=True)
class GoldenCase:
    query: str
    expected_top: str


CASES = [
    GoldenCase("run shell command", "start_process"),
    GoldenCase("run python script", "start_process"),
    GoldenCase("search file contents", "start_search"),
    GoldenCase("write file", "write_file"),
    GoldenCase("apply_patch", "edit_block"),
    GoldenCase("get server config", "get_config"),
]


def _desktop_tools() -> list[ToolDescriptor]:
    return [
        ToolDescriptor("start_search", "desktop_commander", "Search files by path and pattern.", ["path", "pattern"], ["path", "pattern"], "fp1"),
        ToolDescriptor("get_config", "desktop_commander", "Get server configuration and blocked shell commands.", [], [], "fp2"),
        ToolDescriptor("interact_with_process", "desktop_commander", "Send input to a running process.", ["pid", "input"], ["pid", "input"], "fp3"),
        ToolDescriptor("start_process", "desktop_commander", "Start a terminal process.", ["command", "timeout_ms", "shell"], ["command"], "fp4"),
        ToolDescriptor("edit_block", "desktop_commander", "Apply surgical edits to files.", ["file_path", "old_string", "new_string"], ["file_path"], "fp5"),
        ToolDescriptor("write_file", "desktop_commander", "Write or append to file contents.", ["path", "content", "mode"], ["path", "content"], "fp6"),
    ]


def main() -> int:
    tools = _desktop_tools()
    failures = []
    for case in CASES:
        ranked = rank_tool_candidates(query=case.query, candidates=tools)
        top = ranked[0].tool.tool_name if ranked else None
        confidence = ranked[0].confidence if ranked else "none"
        print(f"{case.query}: top={top} confidence={confidence} expected={case.expected_top}")
        if top != case.expected_top:
            failures.append((case.query, top, case.expected_top))
    if failures:
        print("FAILURES:")
        for query, top, expected in failures:
            print(f"- {query}: got {top}, expected {expected}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 2: Add README command**

Add under the deferred tool search section:

~~~markdown
Run the deterministic tool-search accuracy checks before changing ranking:

```bash
python scripts/evaluate_tool_search_accuracy.py
python -m pytest tests/test_tool_search_accuracy.py tests/test_unified_tool_search.py -q
```
~~~

- [x] **Step 3: Run evaluator**

Run:

```bash
python scripts/evaluate_tool_search_accuracy.py
```

Expected: exit code 0 and each golden query reports the expected top tool.

### Task 10: Final Verification Matrix

**Files:**
- No code edits in this task

- [x] **Step 1: Run focused tool-search tests**

Run:

```bash
python -m pytest tests/test_tool_search_accuracy.py tests/test_tool_search_scoring.py tests/test_unified_tool_search.py -q
```

Expected: all pass.

- [x] **Step 2: Run binding and execution tests**

Run:

```bash
python -m pytest tests/test_client_tool_isolation.py tests/test_tool_execution_recovery.py tests/test_custom_agents_tools.py tests/test_multi_sidecar_hardening.py -q
```

Expected: all pass.

- [x] **Step 3: Run prompt and routing-adjacent tests**

Run:

```bash
python -m pytest tests/test_tool_search_prompt_guidance.py tests/test_router.py tests/test_widget_runtime.py -q
```

Expected: all pass.

- [x] **Step 4: Run offline evaluator**

Run:

```bash
python scripts/evaluate_tool_search_accuracy.py
```

Expected output includes:

```text
run shell command: top=start_process confidence=high expected=start_process
run python script: top=start_process confidence=high expected=start_process
search file contents: top=start_search confidence=high expected=start_search
write file: top=write_file confidence=high expected=write_file
apply_patch: top=edit_block confidence=high expected=edit_block
get server config: top=get_config confidence=high expected=get_config
```

- [x] **Step 5: Run full regression suite when time permits**

Run:

```bash
python -m pytest -q
```

Expected: all non-live-server tests pass. If live server integration tests fail because no server is running, document those as environment-required failures.

## Acceptance Criteria

- The sample query `run shell command` returns `start_process` as `recommended_tool`, `confidence=high`, `is_loaded=true`.
- The sample query does not autoload `start_search`, `get_config`, or `interact_with_process`.
- Server and client catalog results use the same scored result contract before merge; client results are not merged with synthetic fallback scores when `query` is present.
- `tool_search` normal discovery output includes no raw multiline MCP prompt fragments.
- Normal model-facing `tool_search` output uses compact JSON and keeps discovery responses within three public candidates by default.
- `requires_refinement=false` whenever a high-confidence loaded recommendation is present.
- `recommended_tool` is only present when the returned public `tool_name` is loaded and callable in the same turn; otherwise `requires_refinement=true`.
- Prompt guidance tells the agent to call the high-confidence loaded recommendation instead of searching synonyms.
- Ambiguous aliased tools are bound and executable under the exact public `tool_name` returned by `tool_search`.
- Custom agents store, refresh, snapshot, and recover deferred tools under `tool_state_key`, not the shared `custom` model config key.
- Existing callers that read `results[*].tool_name`, `description`, `arg_hints`, and `is_loaded` continue to work.
- Golden-query evaluator passes.

## Rollout Notes

- Ship the ranking/result-format changes behind tests first. Do not tune constants without updating golden cases.
- Keep `mcp_tool_search_debug_scores=false` in production. Enable it only for short diagnostic sessions because score breakdowns can expose extra catalog text and consume context.
- Watch for tool calls where `tool_search` is called more than once before any non-search tool. Those should drop sharply after `recommended_tool` and prompt changes.
- If production catalogs reveal large domains not covered by deterministic capability tags, add a small new capability family and golden cases before considering embeddings.

## Implementation Progress Log

Started 2026-06-01. Each entry records verification result and any design decisions
that diverged from the literal plan text.

### Task 1 — Add Golden Accuracy Tests — DONE

- Created `tests/test_tool_search_accuracy.py` and appended the scoring-metadata test
  to `tests/test_tool_search_scoring.py`, verbatim from the plan.
- Verification:
  - `pytest tests/test_tool_search_accuracy.py` → collection ImportError on
    `rank_tool_candidates` (expected, API not built yet).
  - `pytest tests/test_tool_search_scoring.py` → 9 passed, 1 failed
    (`test_score_metadata_marks_weak_description_only_matches_low_confidence`,
    ImportError on the new API). Failures limited to the new API, as the plan expects.
- Design note / risk flagged for Task 3: the literal scoring constants in the plan
  cannot satisfy BOTH `test_run_shell_command_prefers_start_process` (7-tool list,
  `get_config` must be absent from top 2) AND
  `test_score_metadata_marks_weak_description_only_matches_low_confidence` (2-tool
  list, `get_config` must be present at index 1, low confidence). Resolution, matching
  the plan's own Target Public Result Shape (lines 108-114 show `interact_with_process`
  as the medium-confidence #2 for "run shell command"): tune
  `_capability_specific_adjustment` so `process_interaction` gets a positive bump under
  `shell_exec` intent (ranking it above `config_read`), and keep `config_read` at a
  small positive (survives filtering, low confidence) rather than the plan's -16 sink.
  The plan explicitly sanctions constant tuning (Task 3 Step 6).

### Task 2 — Build Tool Profiles and Query Intents — DONE

- Added `split_identifier_tokens()` to `app/ai/text_normalization.py` (verbatim).
- Created `app/ai/tool_search_profiles.py` (verbatim): `QueryIntent`,
  `ToolCapabilityProfile`, `infer_query_intent()`, `infer_tool_profile()`,
  `_compact_purpose()`.
- Verification:
  - `split_identifier_tokens('timeout_ms')` → `['timeout','ms']`;
    `'filePattern'` → `['file','pattern']` (camel + snake handled).
  - `infer_query_intent('run shell command').capabilities` → `{'shell_exec'}`.
  - `start_process` → `{'shell_exec'}` + purpose "Start a shell command or local
    process."; `interact_with_process` → `{'process_interaction'}`;
    `get_config` → `{'config_read'}`.
  - `pytest tests/test_tool_search_accuracy.py` → still ImportError on
    `rank_tool_candidates` (scorer not yet built), as the plan expects.
- No design divergence.

### Task 3 — Replace Ranking With Intent-Aware Scoring — DONE

- Added to `app/ai/tool_search_scoring.py`: `ToolSearchScore`, `rank_tool_candidates()`,
  `_score_candidate()`, `_capability_specific_adjustment()`, `_confidence_for()`.
  Left the legacy `score_tool()`/`build_query_tokens()`/`rank_and_filter()` untouched as
  the Step 5 compatibility shim (catalogs still call them until Task 4).
- Verification (`rtk proxy python -m pytest tests/test_tool_search_accuracy.py
  tests/test_tool_search_scoring.py -v`): **16 passed** — all 6 golden accuracy tests
  plus all 10 scoring tests (including the metadata test and the legacy contracts).
- Design divergence from the literal plan (sanctioned by Task 3 Step 6 "tune constants"):
  in `_capability_specific_adjustment`, under `shell_exec` intent I set
  `process_interaction` to **+30** (plan literal: −8) and **removed** the `config_read`
  **−16** penalty. Rationale: the plan's literal constants make
  `test_run_shell_command_prefers_start_process` and the metadata test mutually
  unsatisfiable (see Task 1 note). The +30/​no-penalty tuning ranks
  `interact_with_process` as the medium-confidence #2 (matching the plan's Target Public
  Result Shape) and keeps weak `get_config` visible at a low score, so both tests pass.
  Resulting scores for "run shell command" (7-tool fixture): start_process=100 (high,
  autoload-eligible), interact_with_process=30 (medium), get_config=2 (low),
  everything else filtered (< 1.0).
- Tooling note: the RTK pytest proxy can report a misleading non-zero exit code with a
  "N passed" summary; `rtk proxy python -m pytest -v` gives trustworthy raw output and is
  used for verification from here on.

### Task 4 — Integrate Ranking Into Server and Client Catalogs — DONE

- `McpToolCatalog.search_scored()` now returns `list[ToolSearchScore]` via
  `rank_tool_candidates`; `search()` query mode unwraps `[item.tool for item in ...]`.
- Added `ClientToolCatalog.search_scored()` (mirrors server) and updated client `search()`.
- `tool_search_tool._execute_tool_search` client branch now uses `search_scored()` when
  available, falling back to `search()`.
- Step 5: no tuple-score test edits were needed — the only fake returning
  `(descriptor, score)` tuples is `test_is_loaded_false_when_autoload_score_below_threshold`,
  which still works via the merge's tuple path (preserved; rewritten in Task 5).
- Verification (`rtk proxy python -m pytest test_tool_search_accuracy
  test_unified_tool_search test_tool_search_scoring`): **28 passed**.
- Sequencing note: at this checkpoint the REAL catalog returns `ToolSearchScore` into
  `_merge_search_results`, which still only handles tuples/plain descriptors. No test
  exercises that production path (all fakes use `search()` or tuple `search_scored`), so
  tests are green; Task 5 Step 4 updates the merge to consume `.tool` objects, closing the
  gap. Kept the legacy `_rank_candidates`/`score_tool` shim per Task 3 Step 5.

### Task 5 — Compact and Actionable tool_search Output — DONE

- `app/core/config.py`: `mcp_tool_search_default_top_k` 5→3; added
  `mcp_tool_search_description_max_chars` (120), `mcp_tool_search_match_reasons_max` (2),
  `mcp_tool_search_debug_scores` (False); added the two int fields to the non-negative
  validator. `mcp_tool_search_max_top_k` left at 100.
- `tool_search_tool.py`: extended `ToolSearchResult` (confidence, match_reasons), added
  `RecommendedTool`, extended `ToolSearchOutput` (recommended_tool, requires_refinement,
  next_action). Added `_public_result_from_scored()`, `_search_item_to_dicts()` (the
  Step-4 merge normalizer handling scored/.tool, tuple, and plain-descriptor inputs), and
  `_build_recommendation()`. Wired response-level fields into the global-inventory,
  client-only-inventory, and discovery returns; added the debug-gated `debug_scores`
  block. Both `tool_search()` and `tool_search_impl()` now return compact JSON
  (`separators=(",", ":")`, `ensure_ascii=False`).
- Closes the Task 4 production gap: the real catalog's `ToolSearchScore` results now flow
  through merge correctly.
- Verification: `test_unified_tool_search` (13) + `test_tool_search_accuracy` (6) →
  **19 passed**, including the new `test_tool_search_result_description_is_compact_purpose`;
  `test_tool_search_scoring` + `test_tool_search_prompt_guidance` → **18 passed**, no
  regressions. The existing client-autoload test still passes (autoload loop unchanged
  until Task 6).
- Design notes: dropped the unused `is_client` param from `_public_result_from_scored`
  (plan included it but never used it) to keep the lints clean. Models keep
  `Field(description=...)` to match the file's existing self-documenting style. One
  pre-existing >100-char `logger.info` line in this file was left untouched (out of scope).

### Task 6 — Make Autoload Conservative and Useful — DONE

- `mcp_tool_search_autoload_top_k` default 3→1.
- Autoload loop now skips a candidate when `internal["_autoload_eligible"] is False`,
  after the existing `autoload_min_relevance_score` floor.
- Added `test_run_shell_command_autoloads_only_start_process`.
- Verification (`test_unified_tool_search` + `test_tool_search_accuracy` +
  `test_tool_search_scoring`): **30 passed**.
- Design divergence from the plan's literal gate (`if not bool(_autoload_eligible):
  continue`): I gate on `is False` rather than "not truthy". Rationale: the plan's literal
  gate would break the existing `test_execute_tool_search_preserves_client_execution_scope_for_autoload`,
  whose `FakeClientCatalog` exposes only `search()` (plain descriptors with no eligibility
  flag), and would also drop real medium-confidence client tools that previously
  autoloaded. With `is False`, scored results (key present) are gated strictly to the single
  high-confidence recommendation — satisfying the acceptance criterion that the shell query
  autoloads only `start_process` — while legacy/unscored results (key absent → None) fall
  back to the `autoload_min_relevance_score` floor, preserving prior behavior. This is the
  plan's stated "secondary safety floor for backward compatibility," applied as the gate
  when explicit eligibility is unavailable.

### Task 7 — Fix Alias Binding and Same-Turn Execution — DONE

- `deferred_tool_binding.py`: added `_tool_with_call_name()` (model_copy → legacy copy →
  copy.copy fallback) and bound deferred tools under `loaded.call_name`.
- `deferred_tool_state.py`: `LoadedTool` gained a `call_name` field; `ConversationToolSet.add()`
  sets it on new+existing entries; `to_reference()` and `list_tools()` emit it (guarded by
  `!= tool_name`); added `get_raw_tool_name_for_loaded_tool()`.
- `tool_execution.py`: `_recover_missing_tool()` now does alias-aware recovery (resolve raw
  name + server from loaded state, rebind under the public alias) before the raw-name
  recovery; `_refresh_tool_map_after_search()` derives `agent_key` from `tool_state_key` first.
- `graph.py`: the three deferred-snapshot methods and the RAG tool-execution block now derive
  the deferred-state key from `tool_state_key` first; the planning-subagent path keeps
  `agent_key` for model routing but uses a new `tool_state_key` var for its two
  `tool_execution_context` calls.
- Verification: `test_client_tool_isolation` (12, incl. 2 new) + `test_tool_execution_recovery`
  (2, incl. 1 new) + `test_custom_agents_tools` (11) → **25 passed**; `test_multi_sidecar_hardening`
  → **24 passed** (no deferred-state regression).
- Design notes: in `graph.py` the planning-subagent's single `agent_key` previously served both
  model routing and tool-execution context; I split it into `agent_key` (routing, unchanged) +
  `tool_state_key` (deferred state) to honor the plan's "keep model routing keys unchanged."
  Recovery in `_recover_missing_tool` was added as a NEW first attempt rather than replacing the
  raw-name loop, so unaliased tools still recover via the original path.

### Task 8 — Update Prompt Guidance to Stop Synonym Search Loops — DONE

- Added two bullets to `TOOL_EXPLORATION_SUFFIX` (call the high-confidence loaded
  `recommended_tool`; do not issue synonym `tool_search`; only refine when
  `requires_refinement` is true).
- Added `test_tool_exploration_suffix_uses_recommended_tool_without_synonym_search`.
- Verification (`test_tool_search_prompt_guidance.py -v`): **9 passed** (no hard-coded
  server names; all existing invariants intact).
- No design divergence.

### Task 9 — Add Offline Accuracy Evaluation — DONE

- Created `scripts/evaluate_tool_search_accuracy.py` and added the run command under the
  README "Deferred tool search" section.
- Verification: `python scripts/evaluate_tool_search_accuracy.py` → exit 0, output matches
  the plan's expected matrix exactly (all 6 golden queries `confidence=high`, correct top).
- Design divergence: added the repo-standard `sys.path` bootstrap
  (`_REPO_ROOT = Path(__file__).resolve().parent.parent`) the plan's literal script omitted
  — without it, `python scripts/evaluate_tool_search_accuracy.py` raises
  `ModuleNotFoundError: No module named 'app'`. Matches `scripts/reindex_documents.py`.

### Task 10 — Final Verification Matrix — DONE

- Step 1 (`test_tool_search_accuracy` + `test_tool_search_scoring` + `test_unified_tool_search`):
  **30 passed**.
- Step 2 (`test_client_tool_isolation` + `test_tool_execution_recovery` +
  `test_custom_agents_tools` + `test_multi_sidecar_hardening`): **49 passed**.
- Step 3 (`test_tool_search_prompt_guidance` + `test_router` + `test_widget_runtime`):
  **85 passed**.
- Step 4 (offline evaluator): exit 0, output matches the plan's expected matrix exactly.
- Step 5 (full suite, `python -m pytest -q`): **932 passed, 1 failed in ~176s**. The single
  failure is `tests/client_backend/test_live_server_integration.py::
  test_live_document_upload_list_get_task_and_delete_flow` (`KeyError: 'document'` at line 472,
  with the `excel` MCP subprocess failing to spawn — `anyio.BrokenResourceError`). This is an
  environment-required live-server test, unrelated to tool search. Confirmed pre-existing: it
  fails identically on the base tree (verified via `git stash` of `app/` changes, then restored),
  so it is NOT a regression from this work.

## Implementation Complete

All 10 tasks implemented and verified. Acceptance criteria met:
- "run shell command" → `recommended_tool=start_process`, `confidence=high`, `is_loaded=true`;
  does not autoload `start_search`/`get_config`/`interact_with_process`.
- Server and client catalogs share the `ToolSearchScore` contract before merge.
- Discovery output is compact JSON with capability purposes (no raw MCP prompt fragments),
  ≤3 default public candidates, response-level `recommended_tool`/`requires_refinement`/`next_action`.
- Ambiguous aliased tools bind and execute under the exact public `tool_name`; custom agents
  use `tool_state_key` for deferred state across binding, refresh, recovery, and graph context.
- Backward-compatible public keys (`tool_name`, `description`, `arg_hints`, `is_loaded`) preserved.
- Prompt guidance and the golden-query evaluator are in place.
