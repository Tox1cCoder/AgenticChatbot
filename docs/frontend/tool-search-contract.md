# Tool search: AI SDK frontend contract

Backend baseline: commit `d5d868ec`, 2026-11-09.

This document describes the compact `tool_search` output and the frontend changes needed to consume it. It supplements the [main AI SDK contract](../../plans/AI_SDK_FE_CONTRACT.md) and [answer-streaming contract](answer-streaming.md).

## 1. What changes for the frontend

Update any tool-search-specific schema, result card, or status calculation that requires the old response fields. A generic tool renderer that accepts arbitrary JSON may need no change.

The chat endpoints, authentication, request bodies, stream event names, and `x-vercel-ai-ui-message-stream: v1` header remain unchanged:

- `POST /api/chat/{conversationId}`
- `POST /ai/chat/{conversationId}` (alias)
- `POST /ai/resume-interrupt`
- `POST /ai/continue`

This change does not add a frontend tool-search HTTP endpoint. The backend agent calls `tool_search`; the frontend receives its input and output in the existing stream.

| Previous field or value | Current contract / frontend migration |
| --- | --- |
| `query` echoed in output | Removed. Read the query from the matching tool input, or saved artifact `args.query`. |
| Required `mode: "discovery"` | Omitted for ordinary discovery. Treat missing `mode` as `"discovery"`. Inventory modes remain explicit. |
| `recommended_tool` | Removed. Render `results`; identify loaded candidates using each result's `is_loaded`. |
| `requires_refinement` | Removed. Read `next_action`. |
| `loaded_count` | Removed. If the UI needs a count, count returned results with `is_loaded === true`. |
| `more_available: false` | Omitted. Treat missing `more_available` as false. |
| `results[].confidence`, `results[].match_reasons` | Removed from normal results. Diagnostics are available only in optional `debug_scores`. |
| `next_action: "call_recommended_tool"` | Replaced by `"call_a_loaded_tool"`. |
| Empty results always implying a search miss | Inspect `next_action`, `error`, and `unavailable_sources`; a catalog outage is distinct from no matches. |

## 2. AI SDK transport

`tool_search` returns a JSON string internally. The AI SDK adapter parses that string, so a normal live stream delivers an **object** in `tool-output-available.output`.

Illustrative tool-event excerpt, with unrelated events and optional render metadata omitted:

```text
data: {"type":"tool-input-start","toolCallId":"search-1","toolName":"tool_search"}

data: {"type":"tool-input-available","toolCallId":"search-1","toolName":"tool_search","input":{"query":"attendance today"}}

data: {"type":"tool-output-available","toolCallId":"search-1","output":{"results":[],"next_action":"refine_search"}}

```

Correlate input and output by `toolCallId`. The output event does not repeat `toolName`; retain that association from the input events or let the SDK maintain it.

Outer protocol keys use camel case (`toolCallId`, `toolName`). Keys inside the tool's `input` and `output` remain snake case. Do not rename them in a proxy.

Optional rendering metadata arrives immediately before the matching output as:

```json
{
  "type": "data-tool-render",
  "data": {
    "toolCallId": "search-1",
    "render": {
      "version": 1,
      "type": "text",
      "model_content": "{\"results\":[],\"next_action\":\"refine_search\"}",
      "text": "{\"results\":[],\"next_action\":\"refine_search\"}"
    }
  },
  "transient": true
}
```

Keep existing generic render support. A tool-search-specific card should read the structured `output`; render text is a fallback, not a second set of search results. Do not add a `render` key to the standard `tool-output-available` event.

## 3. Payload types

These types describe the current payload, before any frontend normalization. Keep decoders tolerant of additional fields for rolling deployments and saved history.

```typescript
type ToolSearchNextAction =
  | "call_a_loaded_tool"
  | "refine_search"
  | "inspect_inventory"
  | "report_unavailable";

interface ToolSearchResult {
  tool_name: string;     // Exact callable name; preserve aliases/prefixes.
  description: string;
  arg_hints: string;     // Display hint, not an executable argument schema.
  is_loaded: boolean;
  source_server?: string; // Present when needed to disambiguate server aliases.
}

interface ToolSearchServer {
  server_name: string;
  description: string;
  tool_count: number;
}

interface ToolSearchOutput {
  next_action: ToolSearchNextAction;
  mode?: "discovery" | "inventory" | "per_server_inventory";
  results?: ToolSearchResult[];
  inventory?: ToolSearchServer[];
  resolved_server_name?: string;
  more_available?: true;
  error?: "catalog_unavailable";
  unavailable_sources?: Array<"server" | "client">;
  guidance?: string;
  debug_scores?: Array<Record<string, unknown>>;
}
```

- Discovery and per-server inventory include `results`, including an empty array when no tools were returned.
- Global inventory uses `mode: "inventory"` and `inventory`; it omits `results`.
- Per-server inventory uses `mode: "per_server_inventory"` and `results`.
- `next_action` is always present. It is guidance for the backend agent, not a command for the browser to execute.
- `resolved_server_name`, when present, is canonical. Do not guess a server name from a tool prefix.
- `more_available` is emitted only when true. It indicates additional matches, not another page token.
- `unavailable_sources` means discovery is incomplete for those sources in this request. It does not mean every integration on that source is down.
- `error` is emitted for a catalog failure when no results could be returned. Partial responses can have `unavailable_sources` and `guidance` without `error`.
- `guidance` may also explain intentionally hidden provider tools, without an outage. Do not classify every response containing guidance as an error.
- `debug_scores` is optional diagnostic output. Ordinary UI behavior must not depend on it.

## 4. Payload examples

### No matching tools

```json
{"results":[],"next_action":"refine_search"}
```

### Loaded result

```json
{
  "results": [
    {
      "tool_name": "read_file",
      "description": "Read a local text file.",
      "arg_hints": "(path*)",
      "is_loaded": true
    }
  ],
  "next_action": "call_a_loaded_tool"
}
```

### Global inventory

```json
{
  "mode": "inventory",
  "inventory": [
    {"server_name":"files","description":"Local file tools","tool_count":3}
  ],
  "next_action": "inspect_inventory"
}
```

### Per-server inventory

```json
{
  "results": [
    {
      "tool_name": "read_file",
      "description": "Read a local text file.",
      "arg_hints": "(path*)",
      "is_loaded": true
    }
  ],
  "mode": "per_server_inventory",
  "resolved_server_name": "files",
  "more_available": true,
  "next_action": "inspect_inventory"
}
```

Inventory can report loaded results while its next action remains `inspect_inventory`. Do not infer the next action solely from the loaded count.

### Catalog unavailable

```json
{
  "results": [],
  "unavailable_sources": ["client"],
  "guidance": "Check the unavailable integration connection. Rewording the query will not restore its catalog; use suitable results from available sources if present.",
  "error": "catalog_unavailable",
  "next_action": "report_unavailable"
}
```

### Partial failure with a usable result

```json
{
  "results": [
    {
      "tool_name": "read_file",
      "description": "Read a local text file.",
      "arg_hints": "(path*)",
      "is_loaded": true
    }
  ],
  "unavailable_sources": ["client"],
  "guidance": "Check the unavailable integration connection. Rewording the query will not restore its catalog; use suitable results from available sources if present.",
  "next_action": "call_a_loaded_tool"
}
```

## 5. Frontend behavior

| `next_action` | Suggested tool-card behavior |
| --- | --- |
| `call_a_loaded_tool` | Show matching results and loaded indicators. The backend agent owns the subsequent tool call. |
| `refine_search` | Show no matches or the returned unloaded candidates. This is not a connection failure. |
| `inspect_inventory` | Show server inventory or per-server tool results, including an empty inventory. |
| `report_unavailable` | Show the connection/catalog warning and recovery guidance. This can also accompany returned unloaded candidates. |

1. End the search card's loading state when its output arrives, including an empty result or domain failure.
2. Show a warning alongside healthy results when `unavailable_sources` is nonempty; keep those results visible.
3. Check the payload's `error` and `next_action`. A catalog failure can arrive in `tool-output-available`; do not wait for `tool-output-error`, a stream-level `error`, or an HTTP failure to display it. A successfully delivered tool output does not establish catalog health.
4. Keep the assistant stream running after a tool-domain warning. Handle stream-level failures separately under the [answer-streaming contract](answer-streaming.md).
5. Do not automatically reissue searches, execute a returned tool, or restart the chat stream based on `next_action`. The backend owns orchestration. `more_available` alone must not start a fetch loop.
6. Treat descriptions, argument hints, and guidance as text. A loaded flag reports availability at discovery time; it does not grant frontend execution permission or guarantee availability later.

## 6. Saved history and rollout compatibility

Persisted assistant messages expose tool artifacts in `metadata.tool_artifacts`. For a tool-search artifact, the relevant fields are `tool: "tool_search"`, `tool_call_id`, `args`, `output`, and optional `render`.

Unlike the parsed live AI SDK `output`, an artifact's `output` is normally a JSON **string**. Normalize object/string representations before validating the payload:

```typescript
function decodeToolOutput(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value; // Preserve a generic text fallback for non-JSON/truncated history.
  }
}
```

Apply this only as a decoding step; validate the decoded shape before using it as `ToolSearchOutput`. Do not call `JSON.parse` unconditionally on a live object. Saved artifact previews may be truncated, so parsing failures must not break conversation rendering.

Older messages and older servers may still contain the removed fields. During rollout:

- Accept and ignore old extra fields.
- Map legacy `next_action: "call_recommended_tool"` to `"call_a_loaded_tool"` in the display adapter.
- Accept explicit legacy `mode: "discovery"` and `more_available: false`.
- Keep generic fallback rendering for an unknown shape/action instead of failing the entire message.
- Use the tool input/artifact args to display the query. Do not require it in output.
- Do not infer current callable tools from saved results or reuse their loaded flags for a new request.

There is no payload version field or new feature flag for this change. A permissive display adapter allows the frontend to deploy before or alongside the backend.

## 7. Frontend acceptance checks

- The exact minimal empty response renders without missing-field errors.
- Live object output and saved string output render the same tool-search details.
- All four `next_action` values finish the tool card's loading state.
- Global inventory works without `results`; discovery works without `mode`.
- Partial outages display warnings while preserving healthy candidates.
- Catalog errors are visible even when the SDK reports an available output.
- Missing `more_available`, absent diagnostics, legacy payloads, and unknown extra fields are tolerated.
- Input/output/render correlation uses `toolCallId`, including concurrent search calls.
- No automatic retry or tool execution is triggered by rendering the payload.

## Backend references

- [Tool-search response builder and schemas](../../app/ai/tool_search_tool.py)
- [AI SDK tool input/output projection](../../app/services/event_streaming/ai_sdk_v6.py)
- [Live output normalization](../../app/services/event_streaming/ai_sdk_projection.py)
- [Persisted tool artifact builder](../../app/ai/tool_execution.py)
- [Tool-search contract regressions](../../tests/test_tool_search_output_contract.py)
- [Strict AI SDK wire tests](../../tests/test_ai_sdk_wire_fixtures.py)
