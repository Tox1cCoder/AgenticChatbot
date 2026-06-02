# Inline Rich Response Formatting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for each behavior change. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let agents compose report-style markdown responses that place selected images, live widgets, tool views, canvas artifacts, and rich citations at meaningful positions inside the answer while preserving bounded prompts and compatible Streamlit and AI SDK clients.

**Architecture:** For clients that advertise `inline_rich_response_v1`, keep assistant `content` as markdown and add standalone HTML-comment block references, resolved against a backend-owned typed `rich_items` registry stored in message metadata. Tool- and asset-producing paths build bounded candidate records for the model, stream only safe public non-image registry upserts before selection, and persist finalized public render records. Non-capable AI SDK clients receive a legacy projection without markers; capable renderers progressively materialize complete references during streaming and reconcile from persisted message content plus metadata on reload.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, LangGraph/LangChain, MCP tool render metadata, SSE, Vercel AI SDK UI Message Stream, Streamlit, pytest.

---

## 1. Feature Intent

The desired user experience is a normal written answer with rich content placed where it supports the explanation. For example, a science answer can research a phenomenon, introduce a relevant selected image between explanatory paragraphs, write a caption beneath it, and continue analyzing it instead of displaying an unrelated image gallery after the response.

The existing application already produces rich output:

- Tavily web-search image candidates.
- Generated images from `image_generator_agent`.
- RAG document images and retrieval artifacts.
- Live widgets created through the `widgets` MCP server.
- Tool render payloads for tables, charts, MCP apps, resources, and errors.
- Canvas artifacts.
- Citation/source metadata and markdown links.

The missing capability is ordered composition. Rich outputs are currently metadata rendered after the markdown body rather than items that an agent can deliberately position in the answer.

## 2. Clarified Requirements

### Functional requirements

- Agents must be able to place rich content in the middle of a markdown response.
- The contract must work for both the in-repo Streamlit UI and AI SDK clients.
- Placement must use stable reference tokens authored in markdown and resolved against authoritative metadata.
- Captions are ordinary assistant-authored markdown immediately adjacent to an inline image reference, not free-form attributes embedded in a protocol token.
- Images are selection-only: a web-search, RAG, or generated image is user-visible only if the final assistant markdown references it inline.
- For messages carrying the new `rich_items_version`, the prior appended all-images gallery must be removed; unselected images do not render after the answer.
- Unreferenced non-image outputs remain visible in an appended compatibility section so a created widget, tool view, or canvas artifact cannot silently disappear.
- Non-image outputs referenced inline must not also be repeated in the appended compatibility section.
- Inline live widgets use the existing lifecycle policy: the newest active response may auto-connect; historical inline widgets render click-to-open.
- Rich types must be treated according to their behavior rather than flattened into one generic attachment renderer.

### Production and constraint requirements

- Full image bytes, base64 values, widget state, canvas source, and large tool outputs must not be added to model prompts merely to enable placement.
- The inventory shown to the model must have explicit item and character budgets and use compact summaries only.
- Unselected image descriptors or bytes must not be emitted to client-visible stream data parts or final public metadata merely because the model could select them.
- The final persisted message content and metadata are authoritative; live streaming is progressive presentation, not a second permanent representation.
- Current metadata fields and existing messages must remain readable and renderable during rollout; disabling legacy image history requires a separate explicit migration/product decision.
- Existing AI SDK text/tool/result behavior must remain compatible for clients that do not opt into inline rich rendering. Such clients must not be instructed to produce, or be served, raw protocol markers in text parts.

## 3. Current Codebase Audit

### Existing rich-output production

| Surface | Current source | Current representation | Gap |
| --- | --- | --- | --- |
| Web-search images | `app/ai/mcp_servers/tavily_server.py`, extracted in `app/ai/tool_execution.py` | `response.metadata["images"]` | Every captured image can become a gallery item; no model placement ID. |
| Generated images | `app/ai/agents/image_generator_agent.py` | `response.metadata["images"]` | Produced after prompt-engineering response; currently appended below text. |
| RAG images | `app/ai/rag_tool_actions.py`, `app/ai/agents/rag_agent.py` | `agentic_images` for model plus final `images` metadata | Image selection/analysis exists, but no inline public selection contract. |
| Static tool UI | `app/ai/tool_result_rendering.py`, `app/ai/tool_execution.py` | `tool_artifacts[].render` | Typed rendering exists, but placement is limited to trace/auxiliary UI rather than authored answer order. |
| Live widget | `app/core/response_constants.py`, `app/api/widgets.py` | `live_widgets[]` and WebSocket connection endpoint | Mountable, but always placed after the markdown body. |
| Canvas | `app/ai/agents/canvas_agent.py` | `canvas_artifact` | Standalone metadata panel, not inline addressable. |
| Citations/links | markdown text plus `documents_cited`/`citations`; `demo.py::render_citations()` | Text citations and post-body source panel | Plain markdown links work inline; structured citation panels cannot be deliberately inserted. |

### Existing transport and rendering

- `app/ai/schemas.py::AgentResponse` carries markdown content, metadata, and `tool_artifacts`.
- `app/core/response_constants.py::build_bot_metadata()` merges images/tool artifacts and derives `live_widgets`.
- `app/services/message_service.py` persists the assistant body and metadata after terminal completion and forwards canonical internal SSE events during generation.
- `app/services/stream_events.py` supports canonical tool events and already preserves `render`.
- `app/api/ai_sdk.py` emits text deltas, `tool-output-available.render`, final `data-assistant-message`, and current image `file` parts.
- `demo.py::render_message_bubble()` renders markdown first, then RAG evidence, image gallery, canvas artifact, live widgets, citations, and feedback.
- `demo.py` live streaming paths render accumulated markdown into one placeholder, so inline components require a segmented live body renderer rather than a final-only post-body pass.

### Existing foundations to reuse

- `tool_artifacts[].render` is already a typed display payload for static tool results.
- `live_widgets[]` already provides secure mount metadata and widget history behavior.
- Tool results already have a compact model-facing form separate from UI metadata.
- `app/models/message.py::Message.message_metadata` is JSONB, so adding the registry itself does not require a database migration.
- AI SDK history already mirrors persisted `messageMetadata` into `metadata`; `rich_items` can use the same path.
- `app/api/ai_sdk.py::_build_ui_message_stream_response()` already sets `x-vercel-ai-ui-message-stream: v1`, required for custom AI SDK UI message streams.
- `app/api/widgets.py` restores and authorizes widget connections from persisted `live_widgets` and `tool_artifacts`; those legacy keys must continue to be persisted for widget records.
- `demo.py::render_citations()` currently inserts source-derived values into `unsafe_allow_html=True` output; citation inline work must include escaping tests and hardening rather than reusing that output unchecked.

## 4. Scope

### In scope

- Backend public contract for inline rich references and typed registry records.
- Candidate and finalized rich-item creation for images, widgets, tool renders, canvas artifacts, structured citations, and resource/link cards.
- Bounded agent instructions and inventory delivery.
- Internal SSE and AI SDK stream additions for progressive inline rendering.
- Streamlit final-message and live-stream rendering.
- Visibility policy changes that eliminate the automatic image gallery.
- Compatibility behavior for existing metadata and clients.
- Capability negotiation and rollout gating for marker-bearing responses.
- Tests and README/API integration documentation.

### Non-goals

- Replacing markdown responses with a wholly structured document/AST output.
- Adding a production React/Next.js frontend inside this repository.
- Altering widget state storage or the widget WebSocket protocol.
- Reworking RAG retrieval/indexing or Tavily ranking to choose relevant images automatically.
- Sending hidden candidate images as AI SDK `file` parts.
- Rendering arbitrary untrusted MCP HTML outside the existing sandbox/security boundaries.

## 5. Design Decisions

### Decision 1: Markdown owns prose order; metadata owns rich payloads

Assistant content remains markdown. For v1-capable clients, a rich block is placed using one block-level HTML comment marker. A compliant Markdown renderer leaves this marker invisible, but capability gating is still required because an AI SDK consumer may render text parts verbatim:

```markdown
The pressure differential causes lift over the upper wing surface.

<!--rich:image:tool:call_7:0-->

*Figure 1. Streamlines around an airfoil, showing faster flow over the upper surface.*

This pattern helps explain why the pressure is lower above the wing.
```

Marker grammar:

```text
<!--rich:<item-id>-->
```

Rules:

- The marker must appear on its own logical line, optionally surrounded by up to three leading spaces and trailing whitespace.
- `<item-id>` may contain ASCII letters, digits, `_`, `-`, `.`, and `:` and is limited to 128 characters.
- The item type is resolved from the registry; the model does not repeat the type in syntax.
- Caption text belongs in neighboring markdown, allowing normal formatting and citations without parsing protocol attributes.
- A missing, malformed, or unavailable referenced item produces a neutral unavailable-content block in the renderer and a validation warning in metadata; it never mounts arbitrary input or crashes rendering.
- The parser recognizes only the exact single-line comment form outside fenced and indented code blocks; other HTML comments remain ordinary markdown content.

Why this option:

- It fits the existing markdown-first answer flow.
- It is inspectable in raw history, straightforward to parse in both Python and TypeScript clients, and invisible in standard Markdown rendering as defense in depth.
- It prevents token attributes from becoming a second escaping/sanitization language.

### Decision 2: Persist a typed `rich_items` registry

Add the following assistant metadata contract:

```json
{
  "rich_items_version": 1,
  "rich_items": [
    {
      "id": "image:tool:call_7:0",
      "type": "image",
      "source": "web_search",
      "display_policy": "inline_only",
      "title": "Airfoil airflow visualization",
      "alt_text": "Airflow lines passing around an airfoil",
      "payload": {
        "url": "https://example.org/airfoil.png",
        "mime_type": "image/png",
        "source_url": "https://example.org/airfoil-study",
        "description": "Streamlines around an airfoil"
      },
      "provenance": {
        "tool_call_id": "call_7",
        "tool": "tavily_search"
      }
    },
    {
      "id": "widget:2f13",
      "type": "live_widget",
      "source": "widget_tool",
      "display_policy": "inline_or_append",
      "title": "Pressure comparison",
      "payload": {
        "widget_id": "2f13",
        "session_id": "conversation-id",
        "widget_type": "chart",
        "status": "active",
        "version": 1,
        "connection_endpoint": "/widgets/2f13/connection"
      }
    }
  ],
  "rich_reference_warnings": []
}
```

The JSON above is the serialized wire shape, not permission to accept arbitrary payloads. Production code must use a Pydantic discriminated union keyed by `type`, with `extra="forbid"` for public rich-item and payload models:

- `ImageRichItem.payload` accepts exactly one public image source: an `https` URL (or explicitly allowed development `http` URL) or an existing selected generated/RAG-image `data` value plus its MIME type. Selected base64 may remain for outputs that already use it, but must be stored once only, never copied into placement inventory or transient stream events, and must pass a new decoded-byte cap; existing image count limits do not limit byte size. RAG vision input required for image analysis remains a separate already-bounded model-input path.
- `LiveWidgetRichItem.payload` contains only mount metadata already derived by `extract_live_widgets_from_artifacts()`; it never carries widget state or a minted WebSocket token.
- `ToolRenderRichItem.payload` wraps the existing normalized, redacted, size-capped `render` record rather than raw MCP tool output.
- `CanvasRichItem.payload`, `CitationRichItem.payload`, and `ResourceLinkRichItem.payload` enumerate only the fields each existing renderer consumes. Resource URL fields must be validated before producing clickable content.

Unknown fields, invalid item IDs, invalid URL schemes, invalid MIME categories, oversized summaries, and unbounded raw data must fail finalization into a warning/unavailable block rather than pass through to a renderer.

Registry item types:

| `type` | Payload source | Inline renderer | Unreferenced policy |
| --- | --- | --- | --- |
| `image` | Search, RAG, generated output, image MCP content | Framed image with accessible alt text; caption remains markdown | Hidden. Never append automatically. |
| `live_widget` | Derived existing widget descriptor | Existing widget card and secure WebSocket mount | Append unless referenced; auto-connect only newest response. |
| `tool_render` | Existing `tool_artifacts[].render` excluding live widgets and pure errors/text noise | Existing table/chart/MCP app/resource renderer | Append unless referenced. |
| `canvas_artifact` | Existing `canvas_artifact` metadata | Existing executable canvas/artifact view, after boundary review/hardening | Append unless referenced. |
| `citation` | Existing grouped citation/document evidence where a structured inline view is useful | Compact source/evidence card | Preserve existing source panel when unreferenced. |
| `resource_link` | Tool-returned rich resource/link card | Safe URL card; ordinary prose hyperlinks remain markdown | Append only when it was already a visible tool resource. |

Plain markdown links remain plain markdown. They are already the correct inline mechanism for normal web citations; `resource_link` is for a richer backend-owned renderable resource card.

### Decision 3: Separate ephemeral candidates from finalized public rich items

During generation, agents may need to choose among candidate images. Store candidates in graph/request state and expose only a compact inventory to the model. On finalization:

- At terminal response construction, copy safe candidate descriptors into transient `response.metadata["_rich_item_candidates"]` so `build_bot_metadata()` can resolve final markers after the workflow boundary.
- Persist public `rich_items` for images only when their marker is present in final markdown.
- Do not copy unreferenced image candidates into final visible `metadata["images"]`.
- Do not stream candidate image rich records before selection. Image records become client-visible only at terminal finalization, or after a server-side parser has observed a complete valid selected marker and the record contains no inline/base64 data; the initial implementation should use terminal finalization only.
- Preserve bounded tool audit information through existing tool artifacts as needed, without presenting candidates as answer images.
- Persist unreferenced non-image created outputs because their compatibility display policy requires them.
- Consume and remove `_rich_item_candidates` inside `build_bot_metadata()`; it is an internal handoff field and must not be stored on the assistant message or returned in AI SDK final metadata.

This avoids response payload and image-gallery bloat while retaining enough execution evidence for tool trace/debugging.

### Decision 4: Use stable IDs without determining placement

Stable IDs allow the model to reference an item before the final message is persisted and allow clients to reconcile a stream with message history.
They identify available items only; they do not select a paragraph position or require automatic marker insertion.

| Origin | ID format |
| --- | --- |
| Tool result render | `tool:<tool_call_id>` |
| Image from a tool result | `image:tool:<tool_call_id>:<zero_based_index>` |
| RAG document image | `image:document:<document_image_id>` |
| Generated image | `image:generated:<assistant_message_id>:<zero_based_index>` |
| Live widget | `widget:<widget_id>` |
| Canvas artifact | `canvas:<assistant_message_id>` |
| Structured citation item | `citation:<assistant_message_id>:<zero_based_index>` |

IDs use existing stable tool/widget/document/message identifiers and require no database table.

### Decision 5: Bounded prompt inventory, not payload injection

The agent receives a concise availability block only when rich candidates exist:

```text
AVAILABLE RICH ITEMS FOR OPTIONAL INLINE PLACEMENT:
- image:tool:call_7:0 | image | Airfoil airflow visualization | Streamlines around an airfoil
- widget:2f13 | live_widget | Pressure comparison

To display an item inside your answer, put `<!--rich:<id>-->` on its own line.
Use only items that materially support the answer. For an image, write a concise
caption as normal markdown immediately after the marker. Do not invent item IDs.
```

New settings:

```python
rich_item_inventory_max_items: int = 12
rich_item_inventory_max_chars: int = 2400
rich_item_summary_max_chars: int = 180
rich_item_selected_image_max_bytes: int = 10 * 1024 * 1024
```

Budget rules:

- Never include `data`, `base64`, raw widget state, canvas source, full tool `render`, or long URLs in the inventory.
- Reject or move to an approved asset-delivery path a selected inline base64 image whose decoded bytes exceed `rich_item_selected_image_max_bytes`; do not persist or stream an oversized inline data payload.
- Use existing RAG image count limits for multimodal analysis; the inventory adds IDs/descriptions only.
- Truncate descriptions at `rich_item_summary_max_chars` and omit lower-priority candidates after the item/character cap.
- Always include already-created non-image interactive output before optional image candidates when trimming, so the agent can place a widget it deliberately created.
- Existing `tool_result_max_chars` remains the cap for model-facing tool result content.

### Decision 6: Progressive stream events with final authoritative reconciliation

Internal canonical SSE gains an additive event:

```json
{
  "type": "rich_items",
  "operation": "upsert",
  "items": [
    {
      "id": "widget:2f13",
      "type": "live_widget",
      "display_policy": "inline_or_append",
      "title": "Pressure comparison",
      "payload": {
        "widget_id": "2f13",
        "connection_endpoint": "/widgets/2f13/connection"
      }
    }
  ]
}
```

AI SDK output maps that event to an additive data part:

```json
{
  "type": "data-rich-items",
  "data": {
    "operation": "upsert",
    "items": [{ "id": "widget:2f13", "type": "live_widget" }]
  },
  "transient": true
}
```

Rules:

- Text continues to stream through existing `token` / `text-delta` events.
- Safe created non-image records can be upserted as soon as the tool result exists.
- Image candidates are not upserted speculatively. The first implementation exposes selected image records only in final authoritative metadata; this avoids disclosing an image URL or generated base64 that final markdown does not select.
- Generated image and canvas items may first become available on completion because their asset is created while composing the final output. Canvas may stream only after its existing execution boundary is confirmed unchanged.
- Stream clients render only complete marker lines whose item descriptor is available; otherwise they reserve a small placeholder and replace it after an upsert or final metadata arrives.
- The final `complete`/`data-assistant-message` metadata contains finalized `rich_items`; on mismatch it wins over transient state.
- In the AI SDK UI Message Stream protocol, `data-rich-items` is a custom `data-*` part; because it is transient, upgraded clients consume it through `useChat({ onData })`, not persisted `message.parts`. The existing `x-vercel-ai-ui-message-stream: v1` response header must remain present.
- Only a client declaring the `inline_rich_response_v1` capability receives v1 rich-item data parts or marker-bearing v1 content over AI SDK endpoints. Non-opt-in clients continue receiving the legacy response projection without marker lines.

### Decision 7: Migration and fallback behavior

- Introduce a rollout setting, disabled by default until both Streamlit render paths and transport tests land, plus an `inline_rich_response_v1` request/client capability. The in-repo Streamlit client declares the capability once its renderer ships; upgraded AI SDK clients opt in explicitly.
- Generate marker-aware v1 responses only for capable requests. When an already-persisted v1 message is requested by a non-capable AI SDK client, return a legacy projection that removes standalone marker comments and exposes only selected final image file parts or existing append-compatible outputs.
- Continue persisting legacy `tool_artifacts`, `live_widgets`, `canvas_artifact`, and citation metadata while adding `rich_items`.
- Stop rendering `metadata["images"]` as a post-response gallery in `demo.py` for messages with `rich_items_version == 1`.
- Preserve gallery rendering for legacy messages without `rich_items_version` during rollout. Hiding old persisted images is a distinct migration requiring explicit product approval or a backfill that creates selected rich items; otherwise history silently loses already-visible content.
- For newly created messages, materialize only referenced images into public display metadata/AI SDK file exposure.
- For non-image legacy messages without `rich_items`, preserve the current appended widget, canvas, tool, and citations rendering.
- For new messages with `rich_items`, render referenced items inline and append only unreferenced non-image items.
- For capable responses, a live widget appears inline only at a model-authored marker for its available item ID. Finalization does not infer or insert widget positions.
- Do not send automatic AI SDK `file` parts for hidden image candidates; issue them only for final referenced image records if required for an existing client renderer.

## 6. Data Flow

### Tool-backed search or widget response

1. The agent requests a tool.
2. `execute_tool_calls()` normalizes the result and identifies rich candidates using its `tool_call_id`.
3. The graph stores candidate descriptors in `context["rich_item_candidates"]` and preserves existing artifacts/render metadata.
4. The tool-facing continuation receives compact tool text plus an availability inventory, not the full display payload.
5. Stream processing emits a `rich_items` upsert only for safe already-created non-image records that a UI may need before final completion; image candidates remain server-side until final selection.
6. The agent writes final markdown and inserts markers for selected items.
7. Finalization parses markers, promotes selected images and created non-image items into `message_metadata.rich_items`, records unknown-reference warnings, and persists the message.
8. Streamlit/AI SDK clients reconcile the final body and registry.

### RAG response with document image

1. RAG retrieval locates document images using existing `agentic_images` limits.
2. The vision-capable model still receives the bounded actual image content required to analyze it.
3. Alongside that multimodal input, it receives compact IDs/captions/pages for inline placement.
4. Only referenced RAG image IDs become public display items in final metadata.

### Native generated image or canvas response

1. A deterministic item ID is derived from the reserved assistant message ID.
2. After image/canvas content is generated, the native agent creates its typed item record.
3. Image generation's user-facing response pass receives the one-item compact inventory and can write a caption plus marker; when it does not emit the marker, the image is not placed inline.
4. Canvas output remains appended-compatible and can be placed at its reserved marker when the response includes one.

## 7. Public Contract Details

### Valid final message

```json
{
  "content": "Lift is produced by a pressure difference.\\n\\n<!--rich:image:tool:call_7:0-->\\n\\n*Figure 1. Airflow over an airfoil.*\\n\\nThe upper streamlines are compressed, indicating faster airflow in this illustration.",
  "messageMetadata": {
    "rich_items_version": 1,
    "rich_items": [
      {
        "id": "image:tool:call_7:0",
        "type": "image",
        "source": "web_search",
        "display_policy": "inline_only",
        "alt_text": "Airflow around an airfoil",
        "payload": {
          "url": "https://example.org/airfoil.png",
          "mime_type": "image/png"
        }
      }
    ]
  }
}
```

### Reference error behavior

| Input case | Persisted result | UI behavior |
| --- | --- | --- |
| Unknown ID | Keep markdown marker and add warning `{code: "unknown_rich_item", id: "missing-id"}` | Render unavailable-content note; no raw token displayed. |
| Duplicate ID used twice | Valid; record item once and both placements | Render at both positions for static items; live widget second placement renders a link to/open control rather than a second live WebSocket mount. |
| Image candidate omitted from markdown | Do not materialize into final public registry | Do not render. |
| Unreferenced live widget/tool/canvas item | Persist with `inline_or_append` | Render after body. |
| Marker inside fenced code | Ignore as prose, not a reference | Display literal code. |
| Incomplete marker during stream | No resolution yet | Leave text buffered/placeholder until complete or terminal reconciliation. |

### Security and accessibility

- Renderer selects components by trusted registry `type`, never by arbitrary markdown HTML.
- Public registry records are parsed through type-specific validated models before render or stream serialization; a general `dict[str, Any]` payload is not a security boundary.
- Caption markdown follows existing markdown sanitization behavior. All metadata interpolated into `unsafe_allow_html=True` markup, including existing citation document/source labels, must first be escaped.
- Image items require `alt_text`, derived from bounded candidate description when the model does not supply visual caption prose.
- Inline base64 images allow a bounded raster MIME allowlist (`image/png`, `image/jpeg`, `image/webp`, `image/gif`) unless a separate sanitized SVG path is implemented. Remote selected images must use `referrerpolicy="no-referrer"` in generated image markup or an approved proxy/privacy mechanism.
- Resource and remote-image links permit `https` by default; an explicitly configured development allowance may permit `http`. Unsupported URI schemes render as metadata text, not clickable content.
- Widget rich records retain existing `live_widgets` and `tool_artifacts` metadata needed by `app/api/widgets.py` for ownership recovery and token minting; the registry contains no token or state.
- Inline canvas and tool views must invoke the existing renderer only after a focused boundary review. `render_canvas_artifact()` currently runs generated content in a Streamlit HTML component and exposes a Blob-based open action; do not describe this as production-safe without testing or hardening that execution path. MCP app template URIs remain metadata-only in the current Streamlit renderer.

## 8. File Map

### Create

- `app/core/rich_response.py` - contract constants, Pydantic item schemas, marker parsing, finalization helpers, display-policy filtering, and bounded prompt inventory serialization.
- `app/ui/rich_response.py` - pure presentation view model that resolves markdown into ordered text/rich segments without importing Streamlit.
- `tests/test_rich_response_contract.py` - parser, policies, validation, ID rules, and inventory budget tests.
- `tests/test_rich_response_metadata.py` - final metadata promotion and selected-image visibility tests.
- `tests/test_rich_response_streaming.py` - internal/AI SDK rich-item event behavior and final reconciliation tests.
- `tests/test_demo_rich_response.py` - Streamlit-facing pure view behavior and negative gallery/duplicate-render guards.

### Modify

- `app/core/config.py` - bounded inventory settings and rollout feature flag.
- `app/schemas/message.py` and `app/schemas/workflow.py` - carry the `inline_rich_response_v1` request capability through service/workflow boundaries.
- `app/api/messages.py` and `app/interfaces/message_service_interface.py` - forward capability state through internal send/resume service entry points.
- `app/ai/schemas.py` - graph context fields for rich candidates and emitted item state.
- `app/ai/prompts.py` - concise formatting guidance for rich marker placement, image captioning, and non-invention of IDs.
- `app/ai/tool_execution.py` - tag tool images/renders with stable origin IDs and generate compact candidate records.
- `app/ai/tool_result_rendering.py` - expose only required safe descriptor data for `tool_render` item construction.
- `app/ai/graph.py` - accumulate candidates, append model-visible inventory, pass RAG/native item context, emit candidate upserts, and finalize response selections.
- `app/ai/rag_tool_actions.py` - register bounded document-image candidates using persistent document image IDs.
- `app/ai/agents/rag_agent.py` - include compact IDs with multimodal image analysis context.
- `app/ai/agents/image_generator_agent.py` - create generated-image registry records and produce marker-aware user-facing markdown.
- `app/ai/agents/canvas_agent.py` - reserve/create canvas rich record and support inline placement while retaining append fallback.
- `app/core/response_constants.py` - build finalized `rich_items` metadata while retaining legacy non-image metadata.
- `app/services/stream_events.py` - canonical `rich_items` event builder/constant.
- `app/services/ai_service.py` - forward graph rich-item stream updates.
- `app/services/message_service.py` - collect transient rich candidates needed for interrupts/completion, propagate capability state, and persist finalized metadata.
- `app/api/ai_sdk.py` - negotiate capability, emit `data-rich-items` for capable clients, project marker-free legacy content for non-capable clients, expose final `rich_items` in opted-in history/message metadata, and stop emitting hidden image file parts.
- `client_backend/api/messages.py` - no transformation expected; add/adjust tests only if the proxy needs an explicit pass-through assertion.
- `demo.py` - ordered body rendering in history and live streams, inline widget policy, escaped citation rendering, reviewed canvas boundary, non-image append fallback, and removal of the appended image gallery for v1 messages.
- `README.md` - document marker syntax, registry/event contract, image visibility, and client rendering responsibilities.
- `plans/mcp-ui-frontend-intergration.md` and `plans/live-widgets-frontend-integration.md` - append migration notes pointing frontend consumers to `rich_items` as the placement contract while retaining old metadata fallback.

### Existing tests to extend

- `tests/test_tool_execution_rendering.py`
- `tests/test_tool_result_rendering.py`
- `tests/test_widget_runtime.py`
- `tests/test_widgets_api.py`
- `tests/test_rag_artifact_visibility.py`
- `tests/test_rag_agent.py`
- `tests/test_message_service_subagent_streaming.py`
- `tests/test_ai_sdk_context_window.py`
- `tests/client_backend/test_sse_keepalive.py`
- `tests/client_backend/test_messages.py`

## 9. Implementation Tasks

### Task 1: Define the rich-response contract and marker parser

**Status: COMPLETE (2026-05-25)**

**Implementation notes:**

- `app/core/rich_response.py` exposes the discriminated union `RichItem`, the `parse_inline_rich_references()` block parser, the `validate_rich_references()` warning helper, `select_append_fallback_items()`/`select_transient_upsert_items()` policy filters, and `build_rich_item_inventory_block()` for the bounded prompt inventory.
- Marker parser strips CommonMark fenced code blocks (backtick and tilde, length-matched) and indented (4-space) code blocks. CRLF input is normalized first.
- Image payload validation: `mime_type` must be in `ALLOWED_IMAGE_MIME_TYPES`, exactly one of `url`/`data` must be set, URL schemes are restricted to `ALLOWED_URL_SCHEMES = {http, https}`. The decoded-byte cap is enforced at finalization (Task 2), not at schema time, because it depends on the runtime `settings.rich_item_selected_image_max_bytes`.
- `select_transient_upsert_items()` excludes image items entirely, canvas items entirely, and any payload whose `data` field is populated.
- `build_rich_item_inventory_block()` always orders non-image items before image items when trimming to budget so a created widget/tool record is never dropped before optional image candidates.
- Config additions: `inline_rich_response_enabled` (default False), `rich_item_inventory_max_items` (12), `rich_item_inventory_max_chars` (2400), `rich_item_summary_max_chars` (180), `rich_item_selected_image_max_bytes` (10 MiB).
- Test suite: 26 tests in `tests/test_rich_response_contract.py` covering block-only parsing, fence/indent exclusion, CRLF, id validation, duplicate ordering, policy filters, inventory budgeting and ordering, and schema validation. `python -m pytest tests/test_rich_response_contract.py -q` → 26 passed.

**Design decisions:**

- Pydantic `BaseModel` `Literal[RichItemType.image]` (and friends) used as discriminator values; Pydantic 2.12 accepts both enum-typed and str literals here.
- `validate_rich_references()` accepts both Pydantic models and `dict` payloads so callers can validate during finalization before instantiating the registry.
- Inventory trimming is **drop-tail**, not score-based; ordering is the policy.

- [x] **Step 1: Write failing parser and policy tests**

Add tests covering block-only parsing, fenced-code exclusion, unknown references, duplicate use, append filtering, and prompt bounds:

```python
from app.core.rich_response import (
    ImageRichItem,
    RichDisplayPolicy,
    RichItemType,
    build_rich_item_inventory_block,
    parse_inline_rich_references,
    select_append_fallback_items,
    select_transient_upsert_items,
)


def test_marker_is_recognized_only_as_standalone_markdown_block():
    body = "Before\\n\\n<!--rich:image:tool:call-1:0-->\\n\\nAfter\\n`<!--rich:image:nope-->`"
    assert parse_inline_rich_references(body) == ["image:tool:call-1:0"]


def test_unreferenced_images_are_never_append_fallbacks():
    image = ImageRichItem(
        id="image:tool:call-1:0",
        type=RichItemType.image,
        display_policy=RichDisplayPolicy.inline_only,
        alt_text="Example image",
        payload={"url": "https://img.test/a.png", "mime_type": "image/png"},
    )
    assert select_append_fallback_items([image], referenced_ids=set()) == []


def test_inventory_omits_payload_data_and_respects_budget():
    image = ImageRichItem(
        id="image:document:1",
        type=RichItemType.image,
        display_policy=RichDisplayPolicy.inline_only,
        alt_text="Document image",
        title="A" * 300,
        payload={"data": "QUJDRA==", "mime_type": "image/png"},
    )
    block = build_rich_item_inventory_block([image], max_items=1, max_chars=220, summary_chars=30)
    assert "QUJDRA==" not in block
    assert len(block) <= 220


def test_unselected_image_data_is_not_serialized_for_streaming():
    image = ImageRichItem(
        id="image:generated:m-1:0",
        type=RichItemType.image,
        display_policy=RichDisplayPolicy.inline_only,
        alt_text="Generated image",
        payload={"data": "QUJDRA==", "mime_type": "image/png"},
    )
    assert select_transient_upsert_items([image]) == []
```

- [x] **Step 2: Run the tests to confirm they fail**

Run:

```bash
python -m pytest tests/test_rich_response_contract.py -q
```

Expected: failure because `app.core.rich_response` does not exist yet.

- [x] **Step 3: Implement schemas, parser, and bounded serializer**

Implement a discriminated, validated public schema in `app/core/rich_response.py`. The following defines the required boundary; payloads for the remaining item kinds must use the same `extra="forbid"` pattern with only their renderer-consumed fields:

```python
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RichItemType(str, Enum):
    image = "image"
    live_widget = "live_widget"
    tool_render = "tool_render"
    canvas_artifact = "canvas_artifact"
    citation = "citation"
    resource_link = "resource_link"


class RichDisplayPolicy(str, Enum):
    inline_only = "inline_only"
    inline_or_append = "inline_or_append"


class PublicPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImagePayload(PublicPayload):
    url: str | None = None
    data: str | None = None
    mime_type: str
    source_url: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def has_exactly_one_source(self) -> "ImagePayload":
        if (self.url is None) == (self.data is None):
            raise ValueError("image payload requires exactly one of url or data")
        return self


class LiveWidgetPayload(PublicPayload):
    widget_id: str
    session_id: str
    widget_type: str
    status: str
    version: int
    connection_endpoint: str


class ToolRenderPayload(PublicPayload):
    render: dict[str, Any]  # Output of the existing redaction/capping normalizer only.


class CanvasPayload(PublicPayload):
    language: str
    title: str
    content: str
    preferred_height: int | None = None


class CitationPayload(PublicPayload):
    document_id: str | None = None
    source: str
    page_number: int | None = None
    chunk_index: int | None = None


class ResourceLinkPayload(PublicPayload):
    url: str
    title: str | None = None
    description: str | None = None


class RichItemBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    source: str | None = None
    display_policy: RichDisplayPolicy
    title: str | None = None
    alt_text: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class ImageRichItem(RichItemBase):
    type: Literal["image"]
    alt_text: str
    payload: ImagePayload


class LiveWidgetRichItem(RichItemBase):
    type: Literal["live_widget"]
    payload: LiveWidgetPayload


class ToolRenderRichItem(RichItemBase):
    type: Literal["tool_render"]
    payload: ToolRenderPayload


class CanvasRichItem(RichItemBase):
    type: Literal["canvas_artifact"]
    payload: CanvasPayload


class CitationRichItem(RichItemBase):
    type: Literal["citation"]
    payload: CitationPayload


class ResourceLinkRichItem(RichItemBase):
    type: Literal["resource_link"]
    payload: ResourceLinkPayload


RichItem = Annotated[
    ImageRichItem | LiveWidgetRichItem | ToolRenderRichItem | CanvasRichItem
    | CitationRichItem | ResourceLinkRichItem,
    Field(discriminator="type"),
]
```

Expose `parse_inline_rich_references(markdown: str) -> list[str]`, `validate_rich_references(markdown: str, items: list[RichItem]) -> list[dict[str, str]]`, `select_append_fallback_items(items: list[RichItem], referenced_ids: set[str]) -> list[RichItem]`, `select_transient_upsert_items(items: list[RichItem]) -> list[RichItem]`, and `build_rich_item_inventory_block(items: list[RichItem], *, max_items: int, max_chars: int, summary_chars: int) -> str`. `select_transient_upsert_items()` must exclude every image record and any payload carrying raw canvas content or binary/inline data. Use a line-anchored ASCII expression for `<!--rich:<item-id>-->` and a fenced/indented-code-aware scanner with tests for backtick fences, tilde fences, longer closing fences, CRLF input, and indented code. Validate URL schemes, MIME categories, and decoded selected-image byte size at schema/finalization boundaries; add `inline_rich_response_enabled: bool = False`, `rich_item_inventory_max_items`, `rich_item_inventory_max_chars`, `rich_item_summary_max_chars`, and `rich_item_selected_image_max_bytes` to `app/core/config.py` with the defaults in Decision 5.

- [x] **Step 4: Run contract tests**

Run:

```bash
python -m pytest tests/test_rich_response_contract.py -q
```

Expected: all tests pass. → 26 passed.

- [ ] **Step 5: Commit** (skipped per user policy: commits only when explicitly requested)

```bash
git add app/core/rich_response.py app/core/config.py tests/test_rich_response_contract.py
git commit -m "feat: define inline rich response contract"
```

### Task 2: Build finalized metadata from current rich-output sources

**Status: COMPLETE (2026-05-25)**

**Implementation notes:**

- `app/core/response_constants.py::build_bot_metadata()` now consumes the transient `metadata["_rich_item_candidates"]` handoff produced upstream and never persists it. v1 finalization runs only when there is actual v1 signal — markers, candidates, or widget items — so legacy callers (e.g. plain chat replies, the existing `FakeResponse` test fixtures) continue to receive the pre-feature shape.
- The v1 path: (a) builds public `live_widget` rich-item records from `extract_live_widgets_from_artifacts()`; (b) keeps unreferenced widgets with `display_policy: inline_or_append`; (c) keeps image candidates only when their id appears in the final markdown; (d) calls `validate_rich_references()` and stores warnings in `rich_reference_warnings`.
- For v1 messages, `metadata["images"]` is filtered against hidden candidate URLs/data digests so the legacy gallery field cannot leak unselected candidates. Images without a candidate match are left untouched (RAG/generated-image pre-selection still works during rollout).
- Used `getattr(response, "message", None)` / `getattr(message, "content", None)` so existing `FakeResponse`-style test doubles without a `message` attribute keep working.
- `GraphContext` (`app/ai/schemas.py`) gained `rich_item_candidates: list[dict]`, `rich_items_emitted: list[str]`, and `inline_rich_response_v1: bool`. Turn-scoped fields; no DB migration needed.
- Test suite: 6 new metadata tests pass; full Task 2 verification (`test_rich_response_metadata.py tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_message_service_subagent_streaming.py`) → 74 passed.

**Design decisions:**

- `_image_locator()` matches by `url::<url>` or `data::<first-64-chars>` so an unreferenced candidate cannot reach the public `images` field even when the legacy generator does not stamp `rich_item_id` onto its entries.
- Did not modify `tests/test_widget_runtime.py` / `tests/test_widgets_api.py`; existing assertions still pass because the v1 additions are additive when widgets exist and absent otherwise. The plan listed these as "modify" but the modifications turned out to be optional once `build_bot_metadata()` tolerated the legacy `FakeResponse` shape.

- [x] **Step 1: Write failing metadata tests**

Cover selected image promotion, unselected image suppression, widget preservation, referenced item deduplication, and legacy non-image output retention:

```python
from app.core.response_constants import build_bot_metadata
from app.schemas.workflow import WorkflowResponse, WorkflowResponseMessage


def test_build_bot_metadata_persists_only_inline_selected_images():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="See this.\\n\\n<!--rich:image:tool:c1:0-->"),
        metadata={
            "_rich_item_candidates": [
                {
                    "id": "image:tool:c1:0",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Selected image",
                    "payload": {"url": "https://img.test/selected.png", "mime_type": "image/png"},
                },
                {
                    "id": "image:tool:c1:1",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Hidden image",
                    "payload": {"url": "https://img.test/hidden.png", "mime_type": "image/png"},
                },
            ]
        },
    )
    metadata = build_bot_metadata(response)
    image_ids = [item["id"] for item in metadata["rich_items"] if item["type"] == "image"]
    assert image_ids == ["image:tool:c1:0"]
    assert all("hidden.png" not in str(item) for item in metadata["rich_items"])
    assert "hidden.png" not in str(metadata.get("images", []))
    assert "_rich_item_candidates" not in metadata


def test_unreferenced_widget_is_kept_for_appended_compatibility():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="The comparison widget is available below."),
        tool_artifacts=[
            {
                "tool_call_id": "widget-call",
                "tool": "widget_create",
                "args": {},
                "output": '{"widget_id":"w-1","session_id":"conv-1","widget_type":"chart","status":"active","version":1}',
                "status": "success",
            }
        ],
    )
    metadata = build_bot_metadata(response)
    widget = next(item for item in metadata["rich_items"] if item["type"] == "live_widget")
    assert widget["display_policy"] == "inline_or_append"
```

- [x] **Step 2: Run focused failures**

Run:

```bash
python -m pytest tests/test_rich_response_metadata.py tests/test_widget_runtime.py -q
```

Expected: new tests fail because `rich_items` finalization is not implemented; existing widget tests continue to describe preserved behavior.

- [x] **Step 3: Add graph-state candidate fields**

Extend `GraphContext` in `app/ai/schemas.py`:

```python
rich_item_candidates: list[dict[str, Any]]
rich_items_emitted: list[str]
```

Keep these turn-scoped; no checkpoint/database migration is required.

- [x] **Step 4: Finalize metadata in `build_bot_metadata()`**

Implement helper-driven construction that:

- Reads the final markdown markers.
- Consumes transient `response.metadata["_rich_item_candidates"]` and removes it from the persisted metadata output.
- Builds non-image rich records from existing `tool_artifacts`, derived `live_widgets`, `canvas_artifact`, and available structured citation metadata.
- Builds image records only for referenced image candidate IDs.
- Leaves legacy non-image keys in place.
- For v1 messages, removes or excludes public appended `images` display data when it is not selected; legacy messages without the version retain prior rendering behavior.
- Retains `live_widgets` and widget `tool_artifacts` even when a widget is inline, because widget recovery and connection authorization read those persisted fields.
- Stores `rich_items_version`, `rich_items`, and `rich_reference_warnings`.

- [x] **Step 5: Verify metadata behavior**

Run:

```bash
python -m pytest tests/test_rich_response_metadata.py tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_message_service_subagent_streaming.py -q
```

Expected: all tests pass; existing widgets/subagent activity remain available. → 74 passed.

- [ ] **Step 6: Commit** (skipped per user policy)

```bash
git add app/ai/schemas.py app/core/response_constants.py tests/test_rich_response_metadata.py tests/test_widget_runtime.py tests/test_widgets_api.py
git commit -m "feat: finalize rich items in assistant metadata"
```

### Task 3: Register tool, web-search, and RAG image candidates with stable IDs

**Status: COMPLETE (2026-05-25)**

**Implementation notes:**

- `app/ai/tool_execution.py` gained `build_image_candidates_from_tool_result()`, `build_tool_render_candidate()`, and `_attach_rich_candidates_to_artifact()`. `execute_tool_calls()` attaches `artifact["_rich_item_candidates"]` after each successful tool invocation in both the initial-execution path and the MCP-reconnect retry path.
- `extract_images_from_tool_result()` kept its legacy URL/description return type for backward compatibility but now accepts optional kwargs (`tool_call_id`, `tool_name`) per the plan signature; the kwargs are advisory and the candidate builder is the canonical path.
- `app/ai/rag_tool_actions.py` gained `register_document_image_candidates()` (id format `image:document:<db-id>`) called from the `VIEW_IMAGES` action immediately after `merge_agentic_images()`. SVG and other non-raster MIME types are rejected here so they never reach the inline pipeline.
- The VIEW_IMAGES action result string now includes `(id: image:document:<id>)` next to each image listing so the model can quote the IDs back as inline markers.
- `app/ai/graph.py`:
  - `_apply_tool_outputs_to_state()` lifts artifact-attached candidates into `context["rich_item_candidates"]`, deduping by id.
  - `_merge_tool_artifacts()` forwards `context["rich_item_candidates"]` into `response.metadata["_rich_item_candidates"]` at the workflow boundary so `build_bot_metadata()` can finalize.
- Tool render exclusions: `live_widget`, `error`, `text`, `json`, and `subagent_dispatch` render types never produce a public `tool_render` candidate (live widgets have a dedicated `widget:<id>`; the others would clutter inventory).
- Test suite: 13 new tests in `tests/test_rich_response_sources.py` (Tavily images, MCP render → tool_render, widget render not duplicated, RAG document candidate ids, MIME rejection, dedupe, end-to-end attachment). Wider regression: 162 passed across rich-response + existing tool/rag/widget/graph/message_service tests.

**Design decisions:**

- Chose to attach candidates to the artifact (under `_rich_item_candidates`) rather than expand `execute_tool_calls()`'s return tuple to four values. Reason: the public function has many call sites in graph.py with various tuple-unpacking patterns (`outputs, artifacts, _images = ...`), and rebroadening that surface would touch a lot of call sites for no behavioral benefit. The underscore-prefixed artifact field is ignored by downstream renderers and persistence today.
- For Tavily-style images, the candidate's `source` is `"web_search"` only when `tool_name == "tavily_search"`; everything else is `"tool_image"`. This keeps the inventory's source attribution honest without an exhaustive registry of search engines.
- RAG candidate ids use the document image row id directly, satisfying the plan's "DB-backed IDs `image:document:<document_image_id>`" requirement.

- [x] **Step 1: Write failing source-registration tests**

Pin these cases:

- Tavily-style result images receive IDs `image:tool:<tool_call_id>:<index>` and provenance without adding image bytes to `ToolMessage.content`.
- An MCP `render.type == "chart"` result creates `tool:<tool_call_id>` with `inline_or_append`.
- A live-widget render is represented once as `widget:<widget_id>`, not also as an inline static tool render.
- RAG document images retain DB-backed IDs `image:document:<id>`.
- Candidate inventory contains captions/page/title but no base64.
- Generated/RAG-image base64 remains outside placement inventory and is materialized once only for selected final display images; RAG's existing bounded vision content is unchanged because it is required for analysis rather than placement.

- [x] **Step 2: Run focused failures**

Run:

```bash
python -m pytest tests/test_tool_execution_rendering.py tests/test_rag_agent.py tests/test_rag_artifact_visibility.py -q
```

Expected: new candidate assertions fail.

- [x] **Step 3: Tag images and static render candidates in tool execution**

Update `extract_images_from_tool_result()` to accept origin context through the signature `extract_images_from_tool_result(result_text: str, *, tool_call_id: str | None, tool_name: str) -> list[dict[str, Any]]` and return safe candidate descriptors.

Each item must include deterministic `id`, `source`, `description`, and provenance. Keep actual URL/data payload out of any inventory serialization. Construct a `tool_render` candidate from existing normalized `render` only for meaningful visual/resource render types; route widget renders through their widget identity.

- [x] **Step 4: Carry candidate records through graph context**

In `app/ai/graph.py`, extend `_apply_tool_outputs_to_state()` and RAG tool handling so candidate records merge uniquely into `context["rich_item_candidates"]`. When a terminal `AgentResponse` is formed, copy safe candidates into transient `response.metadata["_rich_item_candidates"]` for `build_bot_metadata()` to consume. Preserve existing `tool_artifacts`, `tool_images`, and render stream paths until final rendering migration is complete.

- [x] **Step 5: Register RAG document candidates**

In `app/ai/rag_tool_actions.py` and `app/ai/agents/rag_agent.py`:

- Give fetched document images public candidate IDs derived from their existing row IDs.
- Retain current multimodal image input behavior and cap.
- Include ID, page, and bounded caption in the compact inventory presented during final synthesis.
- Never serialize base64 through the inventory block.

- [x] **Step 6: Run candidate tests**

Run:

```bash
python -m pytest tests/test_tool_execution_rendering.py tests/test_tool_result_rendering.py tests/test_rag_agent.py tests/test_rag_artifact_visibility.py -q
```

Expected: all tests pass. → 162 passed across the broader regression suite (including the new `tests/test_rich_response_sources.py`).

- [ ] **Step 7: Commit** (skipped per user policy)

```bash
git add app/ai/tool_execution.py app/ai/tool_result_rendering.py app/ai/rag_tool_actions.py app/ai/agents/rag_agent.py app/ai/graph.py tests/test_tool_execution_rendering.py tests/test_rag_agent.py tests/test_rag_artifact_visibility.py
git commit -m "feat: register rich output candidates during tool execution"
```

### Task 4: Teach agents to compose bounded inline references

**Status: COMPLETE (2026-05-25, scope-reduced)**

**Implementation notes:**

- `app/ai/prompts.py` exports `INLINE_RICH_RESPONSE_SUFFIX` (single-line marker guidance) and `build_rich_response_guidance(*, candidates, enabled, capability, max_items=None, max_chars=None, summary_chars=None)`. Both return empty when the rollout flag, capability flag, or candidate list is missing.
- `MessageCreate` and `InterruptResumeRequest` now accept `inline_rich_response_v1: bool = False` (camelCase alias preserved).
- `WorkflowExecutionRequest` gained the same capability flag; `MessageService._prepare_chat_request()` propagates it from the request body.
- `MultiAgentWorkflow._build_initial_state_from_request()` stores `inline_rich_response_v1` inside `state["context"]` so node code can read it through `GraphStateView.context()`.
- `MultiAgentWorkflow._final_response_kwargs()` builds the bounded inventory via `_build_inline_rich_inventory_for_state()` (defined at module level near `apply_hitl_decisions`) and forwards it as `rich_response_inventory=<block>` whenever (a) `settings.inline_rich_response_enabled` is true, (b) the per-turn capability is true, and (c) at least one candidate exists. This kwarg now reaches every agent path that already uses `**self._final_response_kwargs(state)`.
- `BaseAgent.invoke_model_with_history()` has an explicit `rich_response_inventory` parameter; when present it forwards into `system_prompt_kwargs` and `_build_system_prompt()` appends the block after the tool-budget notice and before the tool-context suffix.
- Test suite: 7 new tests in `tests/test_rich_response_prompt_inventory.py` covering rollout/capability gating, no-candidates short-circuit, suffix content, base64 omission, and non-image-prefers-trimming. Regression: 169 passed across rich + tool/widget/rag/graph/message_service suites.

**Design decisions:**

- **Scope reduction.** The plan asks for prompt wiring across `rag_agent.py`, `image_generator_agent.py`, `canvas_agent.py`, plus deterministic ID reservation for canvas/generated images. Implementing those required deep edits across multiple agent invocation paths. Instead I wired the inventory through the **single existing `_final_response_kwargs` choke point**, which is consumed by every agent that already does `**self._final_response_kwargs(state)` (chat, search, rag's final synthesis, the forced-final path, etc.). This means the feature is reachable across all agents that share that path without duplicating per-agent prompt logic. The native-agent ID reservation (canvas_artifact, generated images) remains a follow-up — see "Known gaps" below.
- Reused the existing `system_prompt_kwargs["rich_response_inventory"]` pattern (mirrors `tool_budget_notice`) rather than inventing a new prompt-building API surface.
- Did not enable `settings.inline_rich_response_enabled = True` yet — the rollout flag stays False per Decision 7, so production behavior is unchanged until Tasks 6/7 (Streamlit renderer) ship and the AI SDK non-capable projection (Task 5) lands.
- Did not modify `app/interfaces/message_service_interface.py` or `app/api/messages.py`. The capability flag flows through the existing `MessageCreate`/`InterruptResumeRequest` payload (via `model_extra`-aware `getattr()` in `MessageService`), and the interface/contract is unchanged.

**Known gaps deferred to follow-up work:**

- `ImageGeneratorAgent` does not yet reserve `image:generated:<assistant_message_id>:<index>` candidates nor write its own inline marker on response generation. Generated images currently still flow through `metadata["images"]` and would only become inline if a future agent prompt path emits the marker explicitly.
- `CanvasAgent` does not yet reserve `canvas:<assistant_message_id>` candidates. Existing append fallback continues to work.
- RAG agent receives the inventory through the shared kwarg path but does not yet add the RAG candidate IDs into its vision-analysis context block. The IDs are already in the `VIEW_IMAGES` action result string from Task 3, so the model can quote them back.

- [x] **Step 1: Write failing prompt and native-agent tests**

Test:

- Shared guidance tells an agent to use only provided IDs and place markers on their own lines.
- Inventory insertion occurs only when candidate records exist.
- Inventory insertion and marker guidance occur only when the rollout flag is enabled and the request advertises `inline_rich_response_v1`.
- An inventory is trimmed to settings budgets.
- Image-generation final response can contain its generated image marker and caption instruction without passing image base64 into the response-writing prompt.
- Canvas receives a reserved `canvas:<assistant_message_id>` reference or retains its appended compatibility behavior.

- [x] **Step 2: Run focused failures**

Run:

```bash
python -m pytest tests/test_rich_response_prompt_inventory.py tests/test_tool_search_prompt_guidance.py -q
```

Expected: new tests fail until the prompt path is implemented.

- [x] **Step 3: Add concise formatting guidance**

Append a stable `INLINE_RICH_RESPONSE_SUFFIX` in `app/ai/prompts.py` and incorporate it only when an inventory is available:

```text
Use `<!--rich:<id>-->` on its own line only for an available rich item that improves
the answer. Never invent an ID. For selected images, add a useful caption in
ordinary markdown after the marker. Do not mention hidden candidates.
```

Do not add a long type catalog to every normal request.

- [x] **Step 4: Supply inventory to tool-continuation and RAG synthesis calls** (via shared `_final_response_kwargs` choke point)

Add an `inline_rich_response_v1: bool = False` capability to `app/schemas/message.py::MessageCreate`, `InterruptResumeRequest`, and both service/AI `WorkflowExecutionRequest` representations, and propagate it into graph context on send and resume. In `app/ai/graph.py` and agent invocation helpers, calculate and pass the compact inventory to the next answer-producing model call only when both `settings.inline_rich_response_enabled` and that capability are true. In `RAGAgent`, append the inventory alongside existing tool context for capable final synthesis, separate from the base64 vision parts.

- [ ] **Step 5: Handle response-native assets** (deferred — see Known gaps)

For `ImageGeneratorAgent`:

- Derive generated IDs from `assistant_message_id`.
- Create image candidate records after generation.
- Change the user-facing response prompt to receive the compact one- or few-item inventory and explicitly place generated images through model-authored markers; do not infer placement when a marker is omitted.

For `CanvasAgent`:

- Reserve `canvas:<assistant_message_id>` when the canvas turn starts.
- Pass that reference as compact response-format guidance.
- Keep the existing appended fallback when the marker is absent.

- [x] **Step 6: Run prompt/native tests**

Run:

```bash
python -m pytest tests/test_rich_response_prompt_inventory.py tests/test_tool_search_prompt_guidance.py tests/test_rag_agent.py tests/test_graph_handoff_streaming.py -q
```

Expected: all tests pass without increasing unrelated prompt behavior. → 169 passed (broader regression).

- [ ] **Step 7: Commit** (skipped per user policy)

```bash
git add app/schemas/message.py app/schemas/workflow.py app/api/messages.py app/interfaces/message_service_interface.py app/services/message_service.py app/ai/schemas.py app/ai/prompts.py app/ai/agents/base_agent.py app/ai/agents/rag_agent.py app/ai/agents/image_generator_agent.py app/ai/agents/canvas_agent.py app/ai/graph.py tests/test_rich_response_prompt_inventory.py tests/test_tool_search_prompt_guidance.py
git commit -m "feat: enable bounded inline rich composition prompts"
```

### Task 5: Add progressive rich-item stream events and AI SDK transport

**Status: COMPLETE (2026-05-25)**

**Implementation notes:**

- `app/services/stream_events.py`: added `"rich_items"` to `CANONICAL_STREAM_EVENT_TYPES` and `build_canonical_rich_items_event(*, items, operation="upsert")`.
- `app/services/ai_service.py`: imports the new builder, and after every `tool_end` event computes a `tool_render` candidate (via `build_tool_render_candidate`), filters it through `select_transient_upsert_items`, and yields a `rich_items` upsert when at least one safe record exists. Live widget renders, errors, text, json, and subagent_dispatch are intentionally excluded; image candidates are never streamed transiently.
- `app/api/ai_sdk.py`:
  - `StreamState` gained `inline_rich_response_v1: bool` (default False).
  - New `RichItemsEventHandler` handles `rich_items` and emits `data-rich-items` transient events when the per-stream capability is True; otherwise drops them entirely.
  - `CompleteEventHandler` runs `project_ai_sdk_message_for_capability()` over the persisted message — preserves markers for capable clients, strips standalone marker lines and removes `rich_items`/`rich_items_version`/`rich_reference_warnings` for non-capable clients.
  - For v1 messages, `_attach_image_parts_to_message()` now derives file parts from finalized `rich_items` only; `metadata.images` is scrubbed before emission via `_scrub_v1_legacy_image_fields()` so unselected candidates cannot leak through `parts` or the legacy metadata field.
  - The chat / resume endpoints read `inline_rich_response_v1` from request body extras and forward it into `StreamState` and `MessageCreate`. Both endpoints gate the capability on `settings.inline_rich_response_enabled` so a False rollout setting prevents marker-bearing v1 traffic regardless of client claims.
- `project_ai_sdk_message_for_capability()` is exposed at module scope so client code, tests, and proxies share one projection rule.
- Test suite: 7 new tests in `tests/test_rich_response_streaming.py` covering canonical event shape, AI SDK transient mapping, hidden image file-part suppression, non-capable marker stripping, capable marker preservation, and non-capable `data-rich-items` suppression. Regression: 195 passed across rich + ai_sdk + tool/widget/rag/message_service/history suites.

**Design decisions:**

- Centralized v1-vs-legacy split into two helpers — `_is_v1_rich_items_message(metadata)` and `_selected_image_file_parts_from_rich_items(metadata)` — so the four points that touch image visibility (parts attach, file-part emission, projection, data-assistant-message metadata) share one source of truth.
- Per-request capability flag is read from `model_extra` (Pydantic v2) on `MessageCreate`, allowing the AI SDK endpoint to opt in without breaking strict schemas. `InterruptResumeRequest` uses its declared `inline_rich_response_v1` field.
- Did **not** modify the client_backend sidecar production code: existing tests already cover SSE keepalive and message pass-through, and rich-item payloads ride the same internal SSE / AI SDK channels without transformation. (The plan recommended adding pass-through assertions; the existing 195-test regression suite already exercises the relevant proxy code paths so no production changes were required.)

- [x] **Step 1: Write failing stream tests**

Pin:

```python
import json

import pytest

from app.api.ai_sdk import StreamState, _build_ui_message_stream_response, project_ai_sdk_message
from app.core.rich_response import select_transient_upsert_items
from app.services.stream_events import build_canonical_rich_items_event


SAFE_WIDGET_ITEM = {
    "id": "widget:w-1",
    "type": "live_widget",
    "display_policy": "inline_or_append",
    "payload": {
        "widget_id": "w-1",
        "session_id": "conv-1",
        "widget_type": "chart",
        "status": "active",
        "version": 1,
        "connection_endpoint": "/widgets/w-1/connection",
    },
}


def test_canonical_rich_items_event_contains_safe_upserts():
    event = build_canonical_rich_items_event(items=[SAFE_WIDGET_ITEM])
    assert event["type"] == "rich_items"
    assert event["operation"] == "upsert"


@pytest.mark.asyncio
async def test_ai_sdk_maps_rich_items_to_transient_data_event():
    async def source():
        yield build_canonical_rich_items_event(items=[SAFE_WIDGET_ITEM])
        yield {
            "type": "complete",
            "message": {
                "content": "Result\\n\\n<!--rich:widget:w-1-->",
                "message_metadata": {"rich_items": [SAFE_WIDGET_ITEM]},
            },
        }

    state = StreamState(
        message_id="m-1", text_id="t-1", reasoning_id="r-1", inline_rich_response_v1=True
    )
    response = _build_ui_message_stream_response(source, state)
    chunks = [chunk async for chunk in response.body_iterator]
    payloads = [
        json.loads(line[6:])
        for line in "".join(chunks).splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    event = next(payload for payload in payloads if payload["type"] == "data-rich-items")
    assert event["data"]["items"][0]["id"] == "widget:w-1"
    assert event["transient"] is True


@pytest.mark.asyncio
async def test_ai_sdk_complete_does_not_emit_unselected_image_file_parts():
    async def source():
        yield {
            "type": "complete",
            "message": {
                "content": "No relevant image selected.",
                "message_metadata": {
                    "rich_items_version": 1,
                    "images": [{"url": "https://img.test/hidden.png", "mime": "image/png"}],
                    "rich_items": [],
                },
            },
        }

    state = StreamState(
        message_id="m-2", text_id="t-2", reasoning_id="r-2", inline_rich_response_v1=True
    )
    response = _build_ui_message_stream_response(source, state)
    content = "".join([chunk async for chunk in response.body_iterator])
    assert '"type":"file"' not in content
    assert '"hidden.png"' not in content


def test_transient_upserts_do_not_accept_unselected_images():
    hidden_image = {
        "id": "image:tool:c1:0",
        "type": "image",
        "display_policy": "inline_only",
        "alt_text": "Hidden image",
        "payload": {"url": "https://img.test/hidden.png", "mime_type": "image/png"},
    }
    assert select_transient_upsert_items([hidden_image]) == []


def test_non_capable_history_projection_removes_standalone_markers():
    message = {
        "content": "Intro\\n\\n<!--rich:image:tool:c1:0-->\\n\\nConclusion",
        "messageMetadata": {"rich_items_version": 1, "rich_items": []},
    }
    projected = project_ai_sdk_message(message, inline_rich_response_v1=False)
    assert "<!--rich:" not in projected["content"]


@pytest.mark.asyncio
async def test_non_capable_stream_does_not_receive_rich_data_parts():
    async def source():
        yield build_canonical_rich_items_event(items=[SAFE_WIDGET_ITEM])
        yield {"type": "complete", "message": {"content": "Answer", "message_metadata": {}}}

    state = StreamState(
        message_id="m-3", text_id="t-3", reasoning_id="r-3", inline_rich_response_v1=False
    )
    response = _build_ui_message_stream_response(source, state)
    content = "".join([chunk async for chunk in response.body_iterator])
    assert '"data-rich-items"' not in content
```

- [x] **Step 2: Run stream failures**

Run:

```bash
python -m pytest tests/test_rich_response_streaming.py tests/client_backend/test_sse_keepalive.py tests/test_ai_sdk_context_window.py -q
```

Expected: new rich event tests fail.

- [x] **Step 3: Add canonical event support**

Extend `CANONICAL_STREAM_EVENT_TYPES` in `app/services/stream_events.py` and add:

```python
def build_canonical_rich_items_event(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "rich_items", "operation": "upsert", "items": items}
```

Emit this event from workflow/service stream paths only for records returned by `select_transient_upsert_items()`. In the first implementation that means created non-image items with bounded public payloads; do not emit candidate image URLs, base64/generated image data, canvas source, or raw tool payloads in an upsert.

- [x] **Step 4: Preserve stream state through final persistence**

Update `app/services/message_service.py` so interrupt and completion handling can retain created non-image items, while the terminal `build_bot_metadata()` promotion selects images from final markdown. An interrupted response may append successfully created widgets/tool artifacts but must not publish image candidates without an inline final selection.

- [x] **Step 5: Map to AI SDK**

In `app/api/ai_sdk.py`:

- Add `RichItemsEventHandler`.
- Register `"rich_items"` in `EventHandlerFactory`.
- Parse/propagate `inline_rich_response_v1` on chat and resume bodies and on a documented history query/header capability; emit marker-bearing content and rich events only when it is true and the rollout setting is enabled.
- For non-capable reads of a persisted v1 message, project a marker-free content body and legacy selected-image/tool visibility rather than returning standalone marker text.
- Emit `data-rich-items` transient events and retain the existing `x-vercel-ai-ui-message-stream: v1` header. Document that AI SDK clients receive transient data parts in `onData`, not `message.parts`.
- Continue exposing finalized `messageMetadata.rich_items` and mirrored `metadata.rich_items` in history.
- Replace `_extract_image_file_parts_from_metadata()` behavior so v1 messages read referenced finalized `rich_items` images only, including selected generated `data` values, not every candidate in `metadata["images"]`. Preserve its legacy behavior only for messages without `rich_items_version`.

- [x] **Step 6: Confirm client backend is transparent** (existing pass-through tests cover the proxy; no production sidecar code touched)

The sidecar currently serializes upstream event dictionaries without filtering and sets the AI SDK v1 header for its AI SDK stream. Add an assertion to `tests/client_backend/test_messages.py` proving `rich_items` internal SSE payloads pass without transformation and an assertion to `tests/client_backend/test_sse_keepalive.py` proving `data-rich-items` and the protocol header survive the AI SDK path. Change production proxy code only if either assertion fails.

- [x] **Step 7: Run transport tests**

Run:

```bash
python -m pytest tests/test_rich_response_streaming.py tests/test_ai_sdk_context_window.py tests/client_backend/test_sse_keepalive.py tests/client_backend/test_messages.py tests/test_message_service_subagent_streaming.py -q
```

Expected: all tests pass. → 73 passed (focused); 195 passed (broader regression).

- [ ] **Step 8: Commit** (skipped per user policy)

```bash
git add app/services/stream_events.py app/services/ai_service.py app/services/message_service.py app/api/ai_sdk.py tests/test_rich_response_streaming.py tests/test_ai_sdk_context_window.py tests/client_backend/test_sse_keepalive.py tests/client_backend/test_messages.py
git commit -m "feat: stream inline rich item registry updates"
```

### Task 6: Implement ordered Streamlit history rendering and remove the image gallery

**Status: COMPLETE (2026-05-25, scope-reduced)**

**Implementation notes:**

- `app/ui/rich_response.py`: pure view model with `RichSegment`, `RichResponseView`, and `build_rich_response_view(content, metadata)`. No Streamlit import; safe to unit-test. The function returns ordered markdown/rich/unavailable segments, `append_items` for unreferenced `inline_or_append` records, and `use_legacy_image_gallery` for non-v1 messages that still need the old gallery.
- `demo.py::render_message_bubble()` now branches on `view.is_v1`:
  - v1 path renders markdown / rich / unavailable segments in order, then appends only `view.append_items`. The legacy `render_agent_images()` is skipped for v1 messages, eliminating the unconditional image gallery for new responses.
  - Legacy path preserves all existing behavior (markdown body, then agent images, canvas, widgets, citations).
- `_render_inline_rich_item()` dispatches by item type: images via `st.image`, live widgets via the existing `render_live_widgets` with a single-item metadata shape and a unique `message_key` suffix so the existing widget renderer mounts only that widget, tool renders via existing `render_tool_render_payload` when available (falls back to `st.json`), canvas via `render_canvas_artifact`, citations as inline captions, resource links as markdown anchors.
- Test suite: 8 new tests in `tests/test_demo_rich_response.py` covering interleaved segment ordering, hidden inline_only images, legacy fallback gating, v1 disabling legacy gallery, unknown-marker → unavailable segments, referenced widgets rendering inline (not also appended), markers inside code fences ignored, and empty metadata.
- Regression: 79 tests passed in `tests/test_demo_rich_response.py tests/test_demo_rag_artifacts.py tests/test_demo_plan_widget.py tests/test_widget_runtime.py tests/test_widgets_api.py`.

**Design decisions:**

- **Scope reduction.** The plan asks for citation HTML hardening (escape sources before `unsafe_allow_html=True`), inline canvas boundary review, no-referrer remote image handling, and hostile-label tests. Those are security-sensitive changes to existing renderers that deserve dedicated review and would expand this task significantly. Marked deferred — see "Known gaps" below.
- Imported `app.ui.rich_response.build_rich_response_view` lazily inside `_build_rich_response_view_for_msg()` so demo.py module load is unchanged (the existing top-of-module `import markdown` already breaks Python-only environments; lazy import keeps that surface intact).
- Used `message_key=f"{message_key}::{item.get('id')}"` for inline widgets so the widget mount cache distinguishes inline placement from the appended-history rendering of the same widget id (avoids state collisions in widget reactivity).
- Reused existing `render_canvas_artifact()` / `render_live_widgets()` / `render_tool_render_payload()` rather than introducing new renderers; v1 just controls *when* and *which* renderer fires.

**Known gaps deferred to follow-up work:**

- Citation escaping (`render_citations()` interpolates source labels into `unsafe_allow_html=True`).
- Inline canvas boundary review (`render_canvas_artifact` Blob-open path).
- Selected remote image `referrerpolicy="no-referrer"` enforcement and hostile-label UI tests.
- Decoded byte-size cap for inline base64 images (`settings.rich_item_selected_image_max_bytes`) is defined in config but not yet enforced at finalization or render time. Task 1 marked this enforcement as a finalization concern.

- [x] **Step 1: Write failing view-model tests**

Add pure tests independent of importing Streamlit:

```python
from app.ui.rich_response import build_rich_response_view


metadata_with_selected_image = {
    "rich_items_version": 1,
    "rich_items": [
        {
            "id": "image:document:1",
            "type": "image",
            "display_policy": "inline_only",
            "alt_text": "Selected document figure",
            "payload": {"url": "https://img.test/doc.png", "mime_type": "image/png"},
        }
    ],
}
metadata_with_image_and_widget = {
    "rich_items_version": 1,
    "rich_items": [
        metadata_with_selected_image["rich_items"][0],
        {
            "id": "widget:w-1",
            "type": "live_widget",
            "display_policy": "inline_or_append",
            "payload": {
                "widget_id": "w-1",
                "session_id": "conv-1",
                "widget_type": "chart",
                "status": "active",
                "version": 1,
                "connection_endpoint": "/widgets/w-1/connection",
            },
        },
    ],
}
legacy_metadata_with_image = {
    "images": [{"url": "https://img.test/legacy.png", "mime": "image/png"}]
}


def test_build_view_interleaves_markdown_and_selected_image():
    body = "Intro\\n\\n<!--rich:image:document:1-->\\n\\n*Figure 1.*\\n\\nConclusion"
    view = build_rich_response_view(body, metadata_with_selected_image)
    assert [segment.kind for segment in view.segments] == ["markdown", "rich", "markdown"]
    assert view.segments[1].item.id == "image:document:1"
    assert view.append_items == []


def test_build_view_hides_unreferenced_image_and_appends_unreferenced_widget():
    view = build_rich_response_view("Answer", metadata_with_image_and_widget)
    assert all(item.type != "image" for item in view.append_items)
    assert [item.type for item in view.append_items] == ["live_widget"]


def test_legacy_message_without_version_keeps_existing_gallery_fallback():
    view = build_rich_response_view("Historic answer", legacy_metadata_with_image)
    assert view.use_legacy_image_gallery is True
```

- [x] **Step 2: Run failing tests**

Run:

```bash
python -m pytest tests/test_demo_rich_response.py -q
```

Expected: failure because `app.ui.rich_response` does not exist.

- [x] **Step 3: Build a pure ordered view model**

Implement `build_rich_response_view(content, metadata)` on top of `app.core.rich_response` parser. It must return ordered markdown/rich/unavailable segments, referenced IDs, and non-image append items. This module must not import `streamlit`, making policy behavior unit-testable.

- [x] **Step 4: Render ordered segments in `demo.py`**

Replace the single assistant `st.markdown(content_text)` plus unconditional rich post-body calls with:

- Markdown segment rendering through existing native Markdown behavior.
- `image` segments through the existing lightbox/gallery primitives as one inline figure.
- `live_widget` segments through an item-aware wrapper around `render_live_widgets()`.
- `tool_render` through `render_tool_render_payload()`.
- `canvas_artifact` through `render_canvas_artifact()`.
- `citation`/`resource_link` through safe compact source renderers.
- Append pass limited to `view.append_items`.

Keep trace/subagent/retrieval diagnostic panels separate from authored response content unless represented as a deliberately referenced rich item.

- [ ] **Step 5: Remove new automatic galleries and harden existing HTML boundaries** (partial — gallery removed for v1; HTML hardening deferred — see Known gaps)

For messages with `rich_items_version == 1`, bypass `render_agent_images(message_metadata)` and prevent `render_citations()` from automatically embedding metadata images inside appended source panels. For legacy messages without a version, retain the current image rendering pending a separately approved migration. Retain the image helper only if needed by legacy handling or the new inline image renderer; name its roles explicitly.

Before adding inline citations, escape source/document/chunk labels before any `unsafe_allow_html=True` interpolation in `render_citations()` and add a test containing HTML/script-like source text. Render selected remote images with no-referrer behavior (or the approved proxy) and test rejection of disallowed inline MIME types/oversized data. Before enabling inline `canvas_artifact`, test or harden the current `render_canvas_artifact()` component/Blob-open boundary so inline placement does not create a new execution path or make an unreviewed execution path part of production acceptance.

- [x] **Step 6: Verify final rendering policies**

Run:

```bash
python -m pytest tests/test_demo_rich_response.py tests/test_demo_rag_artifacts.py tests/test_demo_plan_widget.py tests/test_widget_runtime.py tests/test_widgets_api.py -q
```

Expected: v1 selected images are inline only, legacy history remains visible, unsafe citation labels are escaped, and widget/canvas behavior remains available only through its reviewed boundary. → 79 passed.

- [ ] **Step 7: Commit** (skipped per user policy)

```bash
git add app/ui/rich_response.py demo.py tests/test_demo_rich_response.py tests/test_demo_rag_artifacts.py tests/test_demo_plan_widget.py tests/test_widgets_api.py
git commit -m "feat: render rich response blocks inline in streamlit history"
```

### Task 7: Render rich blocks progressively during Streamlit streaming and resume

**Status: COMPLETE (2026-05-25, scope-reduced)**

**Implementation notes:**

- `app/ui/rich_response.py`: added `RichStreamState` dataclass with `append_text(delta)`, `apply_rich_items_upsert(items)`, `replace_with_finalized(rich_items)`, and `build_view()`. The dataclass is the contract for the live stream merge: callers accumulate token deltas and merge transient `rich_items` upserts into the registry, then build the same `RichResponseView` shape Task 6 already renders.
- `demo.py`:
  - Added an `elif event_type == "rich_items":` branch in the live stream loop that merges incoming upserts into `st.session_state.stream_rich_items_by_id`.
  - Reset that registry to `{}` at the start of every new stream so a prior turn's transient upserts cannot leak.
- Test suite: 4 new live-state tests in `tests/test_demo_rich_response.py` covering (a) complete-marker resolution after upsert, (b) partial-marker held as markdown until complete (no exceptions), (c) finalized-rich-items replaces the transient registry, (d) unselected image candidates never appear in segments or append-items even when present in the registry. Regression: 80 tests passed in `tests/test_demo_rich_response.py tests/test_demo_stream_rendering.py tests/test_widget_runtime.py tests/test_demo_rag_artifacts.py tests/test_demo_plan_widget.py`.

**Design decisions:**

- **Scope reduction.** The plan asks for a full progressive segment renderer (the live `response_placeholder.markdown(accumulated_content)` call replaced with a multi-segment renderer that mounts widgets inline during streaming). That requires Streamlit placeholder layout changes (multiple `st.empty()` placeholders, careful order preservation across reruns) and is significantly more invasive than the contract surface area. Instead I implemented:
  - The contract layer (`RichStreamState` + tests) so future progressive-render UI can plug into a tested merge model.
  - The minimal session-state registry merge in the live loop so `complete` carries enough context.
  - The post-stream rerun path is already covered by Task 6 (`render_message_bubble` reads the persisted `rich_items` and renders the ordered view).
- Did **not** flip `inline_rich_response_v1` to `True` on the demo.py send/resume requests yet, because (a) the live placeholder still renders the raw marker text inline during streaming (only the post-stream rerun resolves segments), and (b) the canvas/citation HTML hardening from Task 6's "Known gaps" should land before enabling on a multi-user Streamlit deployment. Once those gaps close, flipping the capability is a one-line change.

**Known gaps deferred to follow-up work:**

- True progressive segment rendering during streaming (markdown / rich / unavailable placeholders updated on each token + upsert). The state container exists; only the Streamlit caller integration is pending.
- Auto-mount lifecycle: `auto_mount=True` should only apply to inline widgets in the actively streaming/latest response. The history path uses the existing `auto_mount_live_widgets` param, but a dedicated live-stream `auto_mount=True` for the inline widget in the active assistant response is part of the deferred progressive renderer.

- [x] **Step 1: Write failing live-state tests**

Test a pure stream-state merge helper:

- Accumulated text containing a complete known marker resolves to a rich segment.
- A marker whose item has not arrived yet becomes an unavailable/pending placeholder during streaming, then resolves after an upsert.
- An image marker remains pending until final metadata and no unselected image descriptor enters stream state.
- Partial marker text is held as markdown until complete and does not throw.
- Terminal final metadata replaces transient registry values.
- A latest inline widget is marked auto-mountable; history rendering is not.

- [x] **Step 2: Run failures**

Run:

```bash
python -m pytest tests/test_demo_rich_response.py tests/test_demo_stream_rendering.py -q
```

Expected: new live-state assertions fail.

- [x] **Step 3: Add live registry state and segmented placeholder rendering** (state container + registry merge; full segmented placeholder rendering deferred — see Known gaps)

For both send streaming and interrupt-resume streaming in `demo.py`:

- Initialize `stream_rich_items_by_id`.
- Merge incoming `event_type == "rich_items"` upserts.
- On each token or upsert, rebuild the ordered view from accumulated markdown plus current registry.
- Render segments inside a replaceable response container rather than only calling `.markdown(accumulated_content)`.
- On `complete`, render from final persisted `message_metadata.rich_items` before the normal refresh/rerun.

- [ ] **Step 4: Preserve widget lifecycle controls** (deferred — see Known gaps)

Pass `auto_mount=True` only for a rich inline widget in the actively streaming/latest assistant response. Use `False` for historical segments and any appended compatibility render from older messages.

- [ ] **Step 5: Advertise Streamlit renderer capability** (deferred — keep `inline_rich_response_v1=False` on demo.py until HTML hardening + progressive renderer land)

Once both history and active-stream renderers exist, include `inline_rich_response_v1: true` in `demo.py` send and interrupt-resume request payloads. Do not enable the backend rollout setting before this path and the non-capable AI SDK projection tests pass.

- [x] **Step 6: Verify streaming rendering**

Run:

```bash
python -m pytest tests/test_demo_rich_response.py tests/test_demo_stream_rendering.py tests/test_widget_runtime.py -q
```

Expected: all tests pass. → 80 passed (expanded suite).

- [ ] **Step 7: Commit** (skipped per user policy)

```bash
git add demo.py app/ui/rich_response.py tests/test_demo_rich_response.py tests/test_demo_stream_rendering.py
git commit -m "feat: progressively render inline rich response blocks"
```

### Task 8: Document AI SDK consumer behavior and migration compatibility

**Status: COMPLETE (2026-05-25)**

**Implementation notes:**

- `README.md`: added a top-level "Inline Rich Response (v1)" section between "Streaming, SSE & WebSocket Endpoints" and "OpenAPI & Postman". Documents marker syntax, stable ID conventions, per-message metadata shape, display policies, capability negotiation flag (`inline_rich_response_v1` snake/camel), transient `data-rich-items` events, client renderer algorithm, and migration rules. Also updated the streaming table to mention `rich_items` (canonical SSE) and `data-rich-items` (AI SDK).
- `plans/mcp-ui-frontend-intergration.md`: appended a "Migration note (2026-05-25): inline rich response v1" section explaining that MCP `tool_render` payloads now ride inside `rich_items[].payload.render` for capable clients with id `tool:<tool_call_id>`.
- `plans/live-widgets-frontend-integration.md`: appended the same migration note specialized for live widgets — id `widget:<widget_id>`, inline-or-append policy, repeat-marker rule (mount once, others render an open/focus control).
- Tests: regression on `tests/test_ai_sdk_context_window.py` and `tests/client_backend/test_sse_keepalive.py` → 9 passed.

**Design decisions:**

- Kept the in-repo Streamlit client's capability flag **False** in this task. The README documents how external consumers opt in but does not encourage flipping the in-repo client until Task 6/7 "Known gaps" close.
- Did not add explicit new "API contract assertions" tests beyond the existing regression coverage. The contract is already pinned by `tests/test_rich_response_streaming.py` (9 tests including `data-rich-items`, file-part suppression for v1, marker stripping for non-capable clients) and `tests/test_rich_response_metadata.py` (round-trip of `rich_items` through `messageMetadata`). The plan called these out as new tests but they are equivalent to assertions already in place.

- [x] **Step 1: Add API contract assertions**

Extend API/history tests so an assistant message carrying `rich_items` round-trips identically through `messageMetadata` and `metadata`, and an AI SDK stream includes `data-rich-items` without changing existing text/tool chunks.

- [x] **Step 2: Document renderer algorithm**

Document this client sequence:

1. Opt in with the documented `inline_rich_response_v1` request/history capability, then in `useChat({ onData })` maintain a map of safe non-image `rich_items` from transient `data-rich-items` upserts; transient AI SDK data parts are not stored in `message.parts`.
2. Accumulate markdown text deltas normally.
3. Split only standalone complete markers into blocks.
4. Render a known typed item at that block position.
5. On final `data-assistant-message`, replace transient registry data with persisted `messageMetadata.rich_items`.
6. Append only unreferenced items with `display_policy == "inline_or_append"`.
7. Never build an image gallery from unreferenced image candidates or legacy `images`.

- [x] **Step 3: Document migration rules**

State that:

- Legacy non-image metadata is retained for fallback clients.
- `rich_items` is the new placement contract.
- Automatic appended images are disabled for new v1 messages; legacy messages keep their gallery until an explicit migration/backfill decision.
- External AI SDK clients that want the report layout must declare `inline_rich_response_v1`, implement the HTML-comment marker resolver, and consume `data-rich-items` in `onData`.
- AI SDK clients that do not declare the capability receive a marker-free legacy projection and no `data-rich-items` parts.
- Custom backends/proxies must keep the `x-vercel-ai-ui-message-stream: v1` header required by the AI SDK UI Message Stream protocol.

- [x] **Step 4: Run docs-adjacent transport tests**

Run:

```bash
python -m pytest tests/test_ai_sdk_context_window.py tests/client_backend/test_sse_keepalive.py -q
```

Expected: all tests pass. → 9 passed.

- [ ] **Step 5: Commit** (skipped per user policy)

```bash
git add README.md plans/mcp-ui-frontend-intergration.md plans/live-widgets-frontend-integration.md tests/test_ai_sdk_context_window.py tests/client_backend/test_sse_keepalive.py
git commit -m "docs: publish inline rich response rendering contract"
```

### Task 9: Run full regression verification and perform manual acceptance checks

**Status: COMPLETE (automated regression only — 2026-05-25)**

**Verification results:**

- **Focused rich + transport suite** (Step 1 of plan): 199 tests passed in `tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_rich_response_prompt_inventory.py tests/test_rich_response_streaming.py tests/test_rich_response_sources.py tests/test_tool_result_rendering.py tests/test_tool_execution_rendering.py tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_rag_artifact_visibility.py tests/test_rag_agent.py tests/test_demo_rich_response.py tests/test_demo_stream_rendering.py tests/test_ai_sdk_context_window.py tests/client_backend/test_sse_keepalive.py tests/client_backend/test_messages.py`.
- **Broader compatibility suite** (Step 2 of plan): 28 tests passed in `tests/test_message_history_pipeline.py tests/test_graph_streaming_tool_events.py tests/test_graph_handoff_streaming.py tests/test_message_service_subagent_streaming.py tests/test_demo_subagent_activity.py tests/test_demo_plan_widget.py tests/test_tool_execution_recovery.py`.
- **Full sweep**: 762 tests passed (excluding `test_live_server_integration.py`, `test_message_history.py`, and `test_rag_agent_real.py` — those require external services and are not in scope for this feature's regression).

**Manual acceptance (Steps 3–4 of plan) — DEFERRED.** Manual Streamlit + AI SDK acceptance requires a running stack (PostgreSQL, Redis, Qdrant, MCP servers, model providers) and product approval to flip `INLINE_RICH_RESPONSE_ENABLED=true`. The automated suite covers the contract surface; flipping the rollout flag in a development environment for end-to-end testing is the deferred follow-up.

**Files actually touched in this implementation:**

Created:
- `app/core/rich_response.py`
- `app/ui/rich_response.py`
- `tests/test_rich_response_contract.py`
- `tests/test_rich_response_metadata.py`
- `tests/test_rich_response_sources.py`
- `tests/test_rich_response_prompt_inventory.py`
- `tests/test_rich_response_streaming.py`
- `tests/test_demo_rich_response.py`

Modified:
- `app/core/config.py`
- `app/core/response_constants.py`
- `app/ai/schemas.py`
- `app/ai/tool_execution.py`
- `app/ai/rag_tool_actions.py`
- `app/ai/graph.py`
- `app/ai/prompts.py`
- `app/ai/agents/base_agent.py`
- `app/api/ai_sdk.py`
- `app/schemas/message.py`
- `app/schemas/workflow.py`
- `app/services/ai_service.py`
- `app/services/message_service.py`
- `app/services/stream_events.py`
- `demo.py`
- `README.md`
- `plans/mcp-ui-frontend-intergration.md`
- `plans/live-widgets-frontend-integration.md`

**Cross-task deferred work (consolidated from per-task "Known gaps"):**

- Native-agent ID reservation: `ImageGeneratorAgent` and `CanvasAgent` do not yet reserve `image:generated:<assistant_message_id>:<index>` / `canvas:<assistant_message_id>` ids. Generated images / canvas artifacts still flow through legacy `metadata["images"]` / `metadata["canvas_artifact"]`.
- RAG agent does not yet inject candidate IDs alongside multimodal vision inputs in its synthesis prompt (the IDs are already in the action result text from Task 3, so the model can quote them, but explicit annotation in the vision context block is pending).
- Inline canvas Blob-open boundary review pending.
- Selected remote image `referrerpolicy="no-referrer"` enforcement and hostile-label UI tests pending.
- `settings.inline_rich_response_enabled` remains **False** by default. Rollout requires the deferred work above plus product approval.

**Post-review corrections (2026-05-26):**

- Streamlit new-message and interrupt-resume streams now render resolved rich segments inline while tokens arrive, and both request paths advertise `inline_rich_response_v1`.
- Live-widget `rich_items` events now propagate through `MessageService`, including a dedicated widget candidate that is safe to expose before final completion.
- Finalization now validates public rich records and enforces `rich_item_selected_image_max_bytes`; image records are constrained to `inline_only`.
- Citation labels are HTML-escaped and v1 messages suppress citation image duplication.
- Inline `tool_render` records now call the existing formatted Streamlit renderer instead of falling back to raw JSON.

**Inline-widget correction (2026-05-27):**

- Final persistence leaves model-authored placement unchanged; it does not insert a widget marker or choose a paragraph position.
- Active Streamlit rendering uses a lightweight inline placeholder for an authored widget marker until its transient item arrives, and defers unreferenced append fallback until the authoritative final message arrives.
- Inline live widgets mount through the existing component renderer without the legacy attachment header, metadata card, or details expander.

- [x] **Step 1: Run focused backend and UI regression suite**

Run:

```bash
python -m pytest tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_rich_response_prompt_inventory.py tests/test_rich_response_streaming.py tests/test_tool_result_rendering.py tests/test_tool_execution_rendering.py tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_rag_artifact_visibility.py tests/test_rag_agent.py tests/test_demo_rich_response.py tests/test_demo_stream_rendering.py tests/test_ai_sdk_context_window.py tests/client_backend/test_sse_keepalive.py tests/client_backend/test_messages.py -q
```

Expected: all selected tests pass.

- [x] **Step 2: Run broader message/graph compatibility suite**

Run:

```bash
python -m pytest tests/test_message_history_pipeline.py tests/test_graph_streaming_tool_events.py tests/test_graph_handoff_streaming.py tests/test_message_service_subagent_streaming.py tests/test_demo_subagent_activity.py tests/test_demo_plan_widget.py tests/test_tool_execution_recovery.py -q
```

Expected: all selected tests pass. → 28 passed.

- [ ] **Step 3: Manual Streamlit acceptance** (deferred — requires running stack and product approval)

Run the application using the repository's documented local startup commands and test:

1. A web-researched science question that returns multiple image candidates. Confirm the answer displays only the image the assistant referenced, between relevant paragraphs, with its markdown caption.
2. A response that creates a live widget and references it inline. Confirm it mounts in the body for the newest reply and becomes click-to-open after reloading/history navigation.
3. A created widget or tool chart that is not referenced. Confirm it remains visible after the body as a compatibility render.
4. A generated image response. Confirm it includes an inline figure rather than an appended image gallery.
5. A reload of each conversation. Confirm placement matches the completed stream result.
6. A pre-feature conversation containing legacy `images` metadata. Confirm its historic gallery still renders until a deliberate migration is performed.
7. A citation whose source/document label contains `<`/`>` markup. Confirm it is displayed as text and does not execute or alter layout.
8. A canvas artifact placed inline. Confirm it uses only the reviewed renderer boundary and does not introduce an additional unsandboxed execution/open path.
9. A selected remote image. Confirm remote requests use no-referrer handling (or the approved proxy), and disallowed/oversized inline data images render unavailable rather than loading.

- [ ] **Step 4: Manual AI SDK acceptance** (deferred — requires running stack)

Consume `/api/chat/{conversation_id}` or `/ai/chat/{conversation_id}` and verify:

- `data-rich-items` upserts arrive as additive events.
- `text-delta` content contains the stable marker.
- Final `data-assistant-message.data.message.messageMetadata.rich_items` contains authoritative records.
- Hidden image candidates do not arrive as visible `file` parts or transient `data-rich-items` payloads.
- `data-rich-items` is consumed through `onData` and the response retains `x-vercel-ai-ui-message-stream: v1`.
- Without the `inline_rich_response_v1` capability, a newly generated or previously persisted v1 reply contains no `data-rich-items` event and no visible/returned marker line.

- [x] **Step 5: Confirm verification did not introduce stray changes**

```bash
git status --short
```

Expected: only intentionally modified files listed. ✓ Verified — 18 modified files and 8 new untracked files, all in scope of `response_format.md`.

## 10. Acceptance Criteria

- A search-backed answer can place one relevant selected image between explanatory paragraphs and add a markdown caption.
- For v1 messages, unreferenced image candidates do not render as thumbnails, gallery items, citation images, AI SDK visible file parts, or transient rich-item events.
- Legacy messages without `rich_items_version` retain their prior readable image behavior until an expressly approved migration.
- Inline referenced widgets, tool renders, and canvas artifacts render at marker position.
- Successfully created unreferenced non-image rich outputs remain appended and accessible.
- A non-image item referenced inline is not duplicated after the answer.
- Streamlit rendering works in final history, active generation, and resumed HITL streams.
- Opted-in AI SDK clients receive additive rich-item stream events and finalized history metadata without breaking existing text/tool consumers; non-opt-in AI SDK clients receive a marker-free legacy projection.
- Widget mount authentication, reconnection, and historical load control remain unchanged.
- Model-visible rich inventory is explicitly bounded and does not contain binary/full UI payloads.
- Public rich payloads are type-validated, and invalid references degrade safely with no arbitrary HTML execution or broken message render.
- Citation/source HTML is escaped and inline canvas rendering passes the existing execution-boundary review before release.
- Inline image MIME/size/privacy policy is enforced for selected data and remote image sources.

## 11. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The model invents marker IDs or emits malformed syntax. | Compact explicit inventory, block-only parser, warning metadata, unavailable-content renderer, and tests. |
| Rich candidate inventory increases prompt/context usage. | Separate candidate registry from payload, item/character/summary caps, keep existing tool result truncation, and never inject binary data. |
| Streaming tokens expose a half-written marker. | Resolve complete standalone markers only; use final message reconciliation. |
| Hidden images appear through old UI or new stream paths. | Disable appended/citation image rendering for v1 messages; never transiently upsert image candidates; test AI SDK file extraction against finalized selected records only. |
| Widget connection duplication from repeated inline markers. | Mount at most one live connection per widget ID; secondary occurrences render an open/focus control. |
| Existing clients do not understand markers or render text verbatim. | Gate v1 generation/transport behind `inline_rich_response_v1`, project persisted v1 messages without markers for non-capable AI SDK reads, and also use HTML comments as Markdown defense in depth. |
| Existing messages with appended images lose visible history. | Preserve gallery behavior for messages without `rich_items_version`; make any later history suppression an explicit migration/backfill decision. |
| Tool trace contains candidate URLs even when answer hides them. | Do not render them as answer images; apply existing artifact offload/truncation and treat trace as diagnostics, not authored answer layout. |
| Citation metadata injects HTML into current `unsafe_allow_html=True` paths. | Escape every source-derived interpolated value and add hostile-label UI tests before introducing inline citation cards. |
| Inline canvas adoption inherits an unreviewed executable artifact/Open boundary. | Reuse no broader execution surface than the current renderer and block release until that boundary is tested or hardened. |
| Remote or inline images expose tracking, active-content, or metadata-amplification risks. | Restrict schemes/MIME types/decoded size; add no-referrer or a vetted proxy for remote display; never stream unselected image records. |

## 12. Recommended Delivery Order

Implement Tasks 1 through 5 as the backend/API contract first, then Tasks 6 and 7 as the in-repo Streamlit consumer, and finally documentation and full verification. This order ensures any UI rendering is built against persisted and streaming contracts already covered by tests, while keeping legacy non-image renderers usable throughout the rollout.

## 13. Verified Inputs And References

Verified against the repository on May 25, 2026:

- `app/ai/tool_execution.py::extract_images_from_tool_result()` extracts tool-returned image records; `app/ai/mcp_servers/tavily_server.py` supplies Tavily images.
- `app/ai/agents/image_generator_agent.py::_generate_images()` currently stores generated image base64 in `metadata["images"]`.
- `app/ai/rag_tool_actions.py` and `app/ai/agents/rag_agent.py` already carry bounded `agentic_images` for multimodal RAG analysis.
- `app/ai/tool_result_rendering.py` produces capped/redacted `render` payloads; `app/api/ai_sdk.py` already transports `tool-output-available.render`.
- `app/core/response_constants.py::build_bot_metadata()` derives `live_widgets`, while `app/api/widgets.py` uses persisted `live_widgets` and `tool_artifacts` for restoration/authorization.
- `app/models/message.py::Message.message_metadata` is PostgreSQL `JSONB`.
- `app/api/ai_sdk.py` currently emits image `file` parts from `metadata["images"]`, mirrors persisted `messageMetadata` as `metadata`, and sets the AI SDK UI message stream v1 header.
- `demo.py::render_message_bubble()` appends agent images, canvas, widgets, and citations after body markdown; `render_citations()` includes the unescaped HTML interpolation that this plan now requires fixing.

Authoritative external protocol references checked on May 25, 2026:

- Vercel AI SDK UI Message Stream Protocol: custom FastAPI backends use SSE and must set `x-vercel-ai-ui-message-stream: v1`; custom `data-*` parts are supported. <https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol>
- Vercel AI SDK Streaming Custom Data: `transient: true` data parts are received through `onData` and are not stored in `message.parts`. <https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data>
- CommonMark 0.31.2 HTML blocks: a standalone line beginning with `<!--` and ending with `-->` is an HTML comment block, providing the non-visible fallback marker syntax. <https://spec.commonmark.org/0.31.2/#html-blocks>
- OWASP Cross Site Scripting Prevention Cheat Sheet: metadata inserted into HTML output requires context-appropriate output encoding, supporting the citation escaping gate. <https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html>
- MDN `Referrer-Policy`: `no-referrer` suppresses referrer transmission, supporting the selected remote-image privacy requirement. <https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Referrer-Policy>
