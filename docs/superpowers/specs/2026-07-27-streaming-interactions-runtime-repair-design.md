# Streaming, Interactions, and Runtime Repair Design

## Problem

Four related production failures affect the chat experience:

1. Streamlit can show live reasoning or tool activity while a response streams,
   then lose that execution trace after terminal persistence and rerender.
2. Generated-image references use protected relative URLs such as
   `/chat-images/{image_id}`. The checked-out sidecar now exposes the required
   proxy, but the running sidecar predates that route and returns 404.
3. `widget_create` exposes a required `widget_type` even though the application
   supports only one implementation: a sandboxed HTML interaction. Models can
   omit this redundant field and fail validation before the interaction is
   created.
4. `AIService.execute_request_stream` keeps a `ContextVar` token open across a
   consumer-facing `yield`. Closing that async generator from a different task
   context can raise `ValueError: Token was created in a different Context`.

The desired interaction behavior follows the same product direction as Gemini's
interactive simulations and Dynamic View: generate a focused interactive
experience for the current explanation rather than ask the model to select a
renderer type.

## Goals

- Make a completed Streamlit response retain the same reasoning and tool trace
  that was visible while streaming.
- Preserve equivalent tool input/output information on the AI SDK stream.
- Deliver protected generated images through the local sidecar without making
  image bytes public.
- Remove `widget_type` from the active interaction contract end-to-end.
- Render every new interaction through one validated sandboxed-HTML path.
- Make stream cleanup safe on completion, disconnect, cancellation, and early
  generator closure for both Streamlit and AI SDK consumers.
- Add regression coverage at the component and transport boundaries.

## Non-Goals

- Supporting multiple widget renderer types.
- Migrating durable application data to a new interaction framework.
- Making protected image URLs publicly readable.
- Replacing the existing AI SDK UI Message Stream contract.
- Redesigning unrelated MCP tools or the broader capability architecture.

## Considered Approaches

### Contract hardening (selected)

Remove `widget_type` from the model-facing tool, runtime records, public tool
results, persisted live-widget metadata, and renderer decisions. Treat valid
`html` plus `height` state as the interaction contract. Fix trace persistence and
stream ownership at their shared backend boundaries, and use the existing
authenticated media proxy after restarting stale processes.

This approach removes the source of the widget validation failure, gives both
clients one canonical behavior, and avoids maintaining a meaningless type field.

### Optional `widget_type` compatibility field

Keep the field but default it to `html`. This is smaller, but it leaves a useless
model parameter, keeps renderer branching alive, and permits old assumptions to
reappear.

### New `interaction_create` API

Introduce a new tool and migrate widget storage and clients. The naming would be
clean, but the migration cost is disproportionate because the existing widget
runtime already provides the required sandbox, lifecycle, and connection model.

## Architecture

### Canonical interaction contract

The active creation tool is:

```text
widget_create(session_id, initial_state, title="")
```

`session_id` is still rebound to the authenticated active conversation before
execution. `initial_state` remains a JSON string with this validated shape:

```json
{
  "html": "<!doctype html>...",
  "height": 620,
  "caption": "Optional caption"
}
```

The runtime validates that the state is an object, contains non-empty HTML, uses
a numeric height within the existing limits, and stays within the state-size
budget. The HTML continues to render in the existing sandboxed iframe.

`WidgetRecord`, `WidgetStore.create`, `WidgetStore.restore`, Redis hashes,
in-memory records, MCP tool results, `live_widgets` metadata, rich-item payloads,
and compact model-facing tool output no longer require or emit `widget_type`.
Update and close operations identify an interaction only by `widget_id`.

### Legacy compatibility

Widget records have a short TTL, so no durable database migration is required.
Readers tolerate and ignore a legacy `widget_type` key in Redis hashes or message
metadata. A legacy structured record that lacks valid `html` and `height` is not
mounted; the renderer shows the existing safe unavailable/error state. New writes
never include the field.

### Renderer behavior

Streamlit derives an interaction mount from `widget_id`, lifecycle status,
version, connection endpoint, and validated state. It does not branch on or
require a type discriminator. Both transient rich-item rendering and persisted
message rendering use this same mount path.

The renderer retains existing security boundaries: authenticated state fetch,
sandboxed iframe execution, bounded height, no ambient authentication inside the
generated document, and no unsanitized HTML injected into the parent page.

### Execution-trace terminal handoff

Live Streamlit trace state remains a projection of canonical stream events.
Before clearing in-flight state or rerunning, terminal handling reconciles that
projection with the final persisted message. Tool start/end events must become
`tool_artifacts` on the terminal message, including validation failures, and
thinking/reasoning summaries must remain in message metadata.

After completion, the persisted message is the source of truth. The ordinary
message renderer reconstructs the trace from `thinking_summary`,
`reasoning_summary`, and `tool_artifacts`. The AI SDK path continues to emit
standard tool input/output parts and projects the same terminal message metadata;
it does not need a Streamlit-specific trace data part.

### Protected image delivery

Final generated images remain persisted before narrative completion and use a
protected relative reference. The canonical server owns authorization and bytes.
The local sidecar proxies both:

```text
GET /chat-images/{image_id}
GET /api/chat-images/{image_id}
```

The proxy requires a valid local session, adds canonical-server credentials,
streams the response, preserves approved media/cache headers, and enforces the
existing size ceiling. Streamlit fetches the reference with authentication and
renders a data URI. AI SDK clients receive the protected reference and must use
the credentialed media-fetch contract rather than a bare unauthenticated image
element.

The current 404 is also an operational stale-process condition: the running
sidecar was started before the route commit. Verification therefore includes a
controlled service restart and live route probes.

### Usage-context stream ownership

No `bind_usage_context` scope may remain open across a consumer-facing `yield`.
The AI service advances and closes its underlying workflow stream inside the
bound usage context, then yields the resulting mapped event after the token has
been reset in the same context that created it.

This boundary applies before Streamlit or AI SDK transport adaptation, so both
clients receive identical cleanup behavior. Normal completion, early `aclose`,
disconnect, cancellation, mapping failure, and source failure all close the
underlying stream exactly once without leaking an unobserved task exception.

## Error Handling

- Malformed interaction JSON returns the existing actionable parse error.
- Missing/empty HTML, invalid height, or oversized state fails validation before
  storage and produces a persisted error tool artifact.
- Expired, closed, or legacy non-HTML records render a bounded unavailable/error
  state instead of executing unvalidated content.
- Media proxy authentication, not-found, content-type, declared-size, streaming,
  and upstream failures retain typed HTTP outcomes without exposing upstream
  internals.
- Stream cancellation is not converted into a user-visible model error. The
  underlying source is closed and any existing partial-message policy remains in
  force.
- Context cleanup errors are prevented by ownership rather than suppressed.

## Testing Strategy

### Interaction contract and renderer

- Assert the generated MCP schema for `widget_create` has no `widget_type` field.
- Create, update, retrieve, list, close, restore, and expire interactions without
  a type parameter in both in-memory and Redis-compatible record shapes.
- Assert tool results, compact outputs, live-widget metadata, rich items, and API
  responses omit `widget_type`.
- Assert legacy records containing the field remain readable and the field is
  ignored.
- Assert the Streamlit renderer mounts a valid interaction without a type field
  and safely rejects legacy structured state.
- Retain tests for malformed JSON, invalid HTML, height bounds, state size,
  optimistic versioning, session binding, and sandbox configuration.

### Trace persistence

- Drive a tool validation failure through the canonical workflow and assert a
  terminal error artifact is persisted.
- Drive successful tool start/end events through internal SSE and assert the
  completed message rerender shows the execution trace after live state clears.
- Verify thinking-only and reasoning-only responses retain their post-response
  panels.
- Verify AI SDK tool input/output parts and terminal metadata remain intact.

### Image delivery

- Exercise authenticated and unauthenticated reads through both sidecar route
  aliases.
- Verify the sidecar forwards a protected canonical image response, approved
  headers, streaming chunks, and typed failures.
- Verify Streamlit resolves a protected reference and finalizes without losing
  the image.
- Verify AI SDK final file/reference parts remain present.
- After restarting services, probe port 8100 and confirm the route returns an
  authentication/not-found outcome rather than router-level 404.

### Context lifecycle

- Close an AI-service stream after its first event and assert the usage context
  resets without `ValueError` or unobserved task exceptions.
- Cover normal completion, source exception, mapping exception, cancellation,
  and explicit `aclose`.
- Run the same lifecycle through internal SSE and AI SDK adapters.

## Acceptance Criteria

- A model can create a valid interactive HTML experience without sending
  `widget_type`, and no active public payload contains that field.
- Streamlit mounts the interaction after streaming and after message reload.
- A response that showed live trace information shows the corresponding persisted
  trace after completion.
- Generated images load through the restarted sidecar using both supported route
  aliases and remain protected.
- Closing either client stream cannot produce a cross-context token reset error or
  an unretrieved async-task exception.
- Targeted regression suites and the broader relevant test suite pass cleanly.
