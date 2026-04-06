# Live UI Widgets Implementation Plan

## Summary

- Build live, server-driven widgets as a normal chat complement, not as a replacement for the existing `canvas_agent`.
- Keep `canvas_agent` for explicit standalone artifact generation like websites, pages, and self-contained interactive code.
- Implement live UI as a canonical backend tool surface named `widgets`, discovered through the existing MCP and `tool_search` flow.
- Surface widget mount data through assistant message metadata using a new sibling key, `live_widgets`.
- Reuse the existing AI SDK SSE and message-history contracts instead of inventing a new stream event.

## Updated Codebase Constraints

- Current server MCP tools are loaded from `app/ai/mcp_config.json` by `MCPManager` and run as separate stdio processes.
- A stdio MCP server does not share FastAPI request state, SQLAlchemy sessions, `ToolContext`, or in-memory Python globals with the main app.
- Because of that, any widget runtime used by both:
  - widget MCP tools, and
  - `/widgets/...` HTTP or WebSocket endpoints
  must use a cross-process shared store.
- Redis is therefore required for real end-to-end widget flows if `widgets` remains an out-of-process stdio MCP server.
- An in-memory widget store is still useful for isolated unit tests, but it is not sufficient for real MCP plus HTTP integration because the MCP server and API server are different processes.

## Locked Decisions

- Canvas coexistence:
  - `canvas_agent` remains the path for explicit website, web app, or standalone sandbox artifact requests.
  - Normal visual aids inside chat, RAG, or search turns stay with the currently selected agent and use widget tools when helpful.
  - Do not reuse `canvas_artifact` for live widgets.
- Agent scope:
  - v1 target agents are `chat_agent`, `rag_agent`, and `search_agent`.
  - `planning_agent`, `image_generator_agent`, and `canvas_agent` should not intentionally use widgets in v1.
  - Hard enforcement needs code changes, not only prompt changes, because the current tool-binding path exposes all MCP tools when per-agent allowlists are empty.
- MCP consistency:
  - Implement the feature as a real FastMCP server under `app/ai/mcp_servers/`.
  - Register it in `app/ai/mcp_config.json`.
  - Add `widgets` to `MCPManager.DEFAULT_SERVERS`.
  - Keep tool names unique, snake_case, and JSON-safe.
- Deferred loading:
  - Widget tools remain normal server tools and are discovered through `tool_search`.
  - Prompt guidance should explicitly steer agents to `tool_search(..., server_name="widgets")` when a live visual aid would help.
  - If any relevant allowlist is non-empty, add `widgets` to the allowlists for `chat`, `rag`, and `search`.
- HITL:
  - Widget tools remain server-side tools and must not use the `client__` prefix or the device runtime bridge.
  - Do not add widget tools to `hitl_tools_require_approval`.
  - If a later tool in the same turn triggers HITL, any widgets already created earlier in the run must still be surfaced on the persisted interrupt assistant message.
- State backend:
  - Use a Redis-backed shared widget runtime for real flows.
  - Keep an in-memory implementation only for isolated tests and non-MCP unit coverage.
  - Do not add Postgres tables or Alembic changes in v1.
- Sidecar or local backend compatibility:
  - Keep widget execution and state canonical on the server.
  - `client_backend` phase 1 only adds frontend connection compatibility for HTTP.
  - Do not sync widget tools into the device runtime catalog and do not route widget operations through the sidecar tool-dispatch WebSocket.
  - Add a dedicated sidecar phase after the direct server path is working.
  - The sidecar phase should implement a proper widget relay with local fan-out or multiplexing, not just a naive 1:1 pass-through.

## Public Interfaces

### MCP tools

- `widget_create(session_id, widget_type, initial_state, title=None)`
- `widget_update(widget_id, state, version=None)`
- `widget_get_state(widget_id)`
- `widget_close(widget_id)`
- `session_list_widgets(session_id)`

### Server HTTP and WebSocket surfaces

- `POST /widgets/{widget_id}/connection`
  - Authenticated server endpoint that returns fresh connection info for a widget:
  - `widget_id`
  - `session_id`
  - `widget_type`
  - `title`
  - `status`
  - `version`
  - `ws_url`
  - `token`
  - `expires_at`
- `WS /widgets/{widget_id}/connect?session_id=<conversation_id>&token=<signed_widget_token>`

### Client-backend compatibility surfaces

- `POST /widgets/{widget_id}/connection`
  - Proxy the canonical server endpoint through `client_backend`.
- Phase 1:
  - No widget WebSocket relay.
  - The frontend uses the returned `ws_url` and connects directly to the canonical server.
- Sidecar phase:
  - `WS /widgets/{widget_id}/connect`
  - `client_backend` terminates the local browser connection and relays to the canonical server.
  - Prefer local fan-out or multiplexing so one desktop session does not need one upstream socket per local widget surface unless necessary.

### Assistant message metadata

Add a new metadata key:

```json
{
  "live_widgets": [
    {
      "widget_id": "uuid",
      "session_id": "conversation-id",
      "widget_type": "table",
      "title": "Optional title",
      "status": "active",
      "version": 1,
      "connection_endpoint": "/widgets/{widget_id}/connection"
    }
  ]
}
```

Rules:

- Persist `live_widgets` in normal assistant messages.
- Persist the same `live_widgets` metadata on interrupt assistant messages if widget tools already succeeded earlier in the same run.
- Do not persist long-lived bearer credentials in message metadata.
- Use the connection endpoint to mint fresh short-lived widget connection tokens.

### WebSocket event contract

Server to client:

- `widget_state_sync`
- `widget_update`
- `widget_close`
- `ping`

Client to server:

- `user_state_patch`
- `pong`

`user_state_patch` merge rule:

- Use shallow merge in v1.
- Increment the widget version after a successful patch merge.

## Real End-to-End Flow

1. The user asks for something that would be clearer as a live table, chart, dashboard, chooser, or structured form inside chat.
2. The router keeps the request on `chat_agent`, `rag_agent`, or `search_agent` unless the user clearly asked for a standalone website or app artifact.
3. The active agent calls `tool_search(query="...", server_name="widgets")`.
4. Deferred loading autoloads widget tools the same way it already autoloads other MCP tools.
5. The agent calls `widget_create(...)` or `widget_update(...)` on the `widgets` MCP server.
6. The widget tool writes canonical widget state into the shared widget runtime store and returns structured JSON describing the widget.
7. Tool execution stores both:
  - the normal stringified output used for model context, and
  - structured widget output retained in tool artifacts for backend metadata derivation.
8. During response finalization, the backend derives `live_widgets` from structured widget artifacts and merges it into normal assistant message metadata.
9. The existing AI SDK SSE path emits the normal `data-assistant-message` payload. No new widget-specific SSE event is required.
10. The frontend sees `live_widgets` either in the streamed assistant message or from `/ai/conversations/{conversation_id}/messages`.
11. The frontend calls `POST /widgets/{widget_id}/connection` with normal bearer auth.
12. The server revalidates conversation ownership, mints a short-lived widget token, and returns `ws_url`.
13. The frontend opens the widget WebSocket and receives `widget_state_sync` plus later `widget_update` events.
14. If the user interacts with the widget, the frontend sends `user_state_patch`.
15. On a later turn, the agent can call `widget_get_state(widget_id)` to inspect current widget state.

## Interaction With Other MCP Servers

- `widgets` is just another server-owned MCP surface in the current architecture.
- It does not replace or bypass other MCP servers such as `tavily`, `time`, `calculator`, `weather`, or future servers.
- A normal turn can chain tools across servers, for example:
  - `tavily` or `search` tools fetch live data
  - `calculator` computes totals
  - `widgets` renders the final comparison table or dashboard
- There is no separate execution path for widgets. They use the same:
  - `tool_search`
  - deferred autoload
  - MCP execution
  - tool artifact capture
  - AI SDK SSE completion path
- Avoid conflicts by keeping widget tool names unique and prefixed consistently, for example `widget_create`, `widget_update`, and `widget_get_state`.
- If two servers expose the same tool name, the current tool catalog already treats that as ambiguous. Ambiguous tools are not autoloaded unless the server is specified, which is why the prompt should prefer `server_name="widgets"` for widget discovery.

## Implementation Status

**Last updated: 2026-04-03**

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 — Redis foundation | ✅ DONE | `REDIS_URL` in `.env.example`; settings description updated; startup validation log in `app/main.py` |
| Phase 1 — Core backend widget flow | ✅ DONE | Widget runtime, MCP server, HTTP/WS API, `live_widgets` metadata — all implemented and tested |
| Phase 2 — Workflow hardening | ✅ DONE | Prompt updates, router disambiguation, agent scoping, interrupt preservation |
| Phase 3 — Sidecar relay (Phase 1) | ✅ DONE | `POST /widgets/{widget_id}/connection` proxied through `client_backend`; WS relay deferred to later phase |

### Design decisions made during implementation

- **Widget store fallback**: `get_widget_store()` tries Redis first; falls back to `InMemoryWidgetStore` with a warning. This allows unit tests to run without Redis while making the out-of-process MCP + HTTP path require Redis.
- **Artifact extraction approach**: Rather than adding a structured output field to `build_tool_artifact`, we parse the raw JSON output of `widget_create`/`widget_update` artifacts during `build_bot_metadata`. The discriminator is the `tool` name. This avoids changing the artifact schema.
- **Interrupt widget preservation**: Tool artifacts are accumulated from streamed `tool` events during message creation and resume. They are passed to `_persist_interrupt_bot_message` via a new optional `tool_artifacts` parameter so `live_widgets` can be derived and stored on pause messages.
- **Agent widget exclusion**: Implemented as a code-level set `_WIDGET_EXCLUDED_AGENT_KEYS = {"canvas", "image_generator", "planning"}` in `base_agent.py` rather than allowlist config, to avoid accidentally exposing widgets to those agents when allowlists are empty.
- **Router guidance**: Added in-chat visual aid examples explicitly routing to `chat_agent` in `ROUTER_SYSTEM_PROMPT` to prevent canvas_agent routing for non-artifact visual requests.
- **Token TTL**: Widget connection tokens are minted with 5-minute TTL (`WIDGET_TOKEN_TTL_SECONDS = 300`). Tokens are never persisted in message metadata — only `connection_endpoint` is stored and fresh tokens are minted on-demand via `POST /widgets/{widget_id}/connection`.

---

## Delivery Phases

### Phase 0. Redis foundation and environment setup

- Treat Redis as required for the real end-to-end widget path while `widgets` is an out-of-process stdio MCP server.
- Local setup:
  - run Redis locally, for example with Docker
  - `docker run -d --name sample-chatbot-redis -p 6379:6379 redis`
- Environment:
  - add `REDIS_URL=redis://localhost:6379/0` to `.env`, or explicitly reuse the existing Redis URL already used by Celery
  - keep `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` aligned with the same Redis instance unless there is a reason to isolate them
- Repo updates:
  - document `REDIS_URL` in `.env.example`
  - update `README.md` so Redis is described as required for widgets, not only optional for Celery and HITL
- Verification:
  - confirm the API health check passes at `/health/redis`
  - add widget-runtime startup validation that logs whether Redis-backed storage is active

### Phase 1. Core backend widget flow ✅ DONE

- Build the Redis-backed shared widget runtime → `app/services/widget_runtime.py`
- Add the `widgets` MCP server → `app/ai/mcp_servers/widgets_server.py`
- Registered in `app/ai/mcp_config.json`, added to `MCPManager.DEFAULT_SERVERS`
- Add server `/widgets/...` HTTP and WebSocket surfaces → `app/api/widgets.py`
- Wired into `app/main.py` and `app/api/__init__.py`
- Persist `live_widgets` in assistant messages via `extract_live_widgets_from_artifacts` in `app/core/response_constants.py`

### Phase 2. Widget-aware workflow hardening ✅ DONE

- Derive `live_widgets` during normal response finalization via `build_bot_metadata`
- Preserve `live_widgets` on persisted interrupt assistant messages via `tool_artifacts` param on `_persist_interrupt_bot_message`
- Tightened router and prompt behavior: `ROUTER_SYSTEM_PROMPT` now routes in-chat visuals to `chat_agent`; `TOOL_EXPLORATION_SUFFIX` includes widget discovery guidance
- Code-level agent scoping: `_WIDGET_EXCLUDED_AGENT_KEYS` in `base_agent.py` blocks widget tools from canvas, image_generator, planning agents

### Phase 3. Proper sidecar relay (Phase 1 connection proxy) ✅ DONE (Phase 1 only)

- Add a dedicated `client_backend` widget WebSocket relay endpoint
- Keep it separate from `/device-runtime/{device_id}/connect`
- Proxy widget connection minting through `client_backend`
- Terminate widget sockets locally and relay upstream to the canonical server
- Add local fan-out or multiplexing so the sidecar can reduce local reconnection churn and avoid unnecessary duplicate upstream subscriptions
- Keep widget tools server-owned and do not route widget MCP execution through the sidecar tool-dispatch bridge
- Treat this phase as the preferred production shape when the desktop architecture expects all realtime browser traffic to stay behind the local sidecar

## Implementation Changes

### 1. Shared widget runtime service

- Add a widget runtime module, for example `app/services/widget_runtime.py`, with:
  - `WidgetRecord`
  - `WidgetStore` interface
  - `RedisWidgetStore`
  - `InMemoryWidgetStore`
  - `WidgetConnectionManager`
  - `WidgetTokenService`
- Store fields at minimum:
  - `widget_id`
  - `session_id`
  - `widget_type`
  - `title`
  - `state`
  - `status`
  - `version`
  - `created_at`
  - `updated_at`
  - `expires_at`
- Validate:
  - `session_id` maps to an existing conversation identifier
  - max serialized state size is 256 KB
  - optimistic version checks for `widget_update`
  - TTL eviction of stale widgets
- Behavior:
  - `widget_create` starts at version `1`
  - `widget_update` is full-state replacement
  - `user_state_patch` is shallow merge into stored state
  - `widget_close` is terminal in v1
- Security note:
  - Because stdio MCP tools do not automatically receive authenticated request context, ownership must be revalidated when issuing widget connection tokens.
  - If stricter tool-time ownership validation is required later, add an explicit MCP execution context bridge or move widgets to an in-process server-owned tool surface.

### 2. FastMCP server integration

- Add `app/ai/mcp_servers/widgets_server.py` using the same `FastMCP(...); @mcp.tool(); mcp.run(transport="stdio")` pattern as the current built-in servers.
- Add the `widgets` server entry to `app/ai/mcp_config.json`.
- Update `MCPManager.DEFAULT_SERVERS` to reserve `widgets` as a managed core server.
- Implement widget tools against the shared widget runtime service, not against process-local globals.
- Make tool descriptions explicit enough for `tool_search`, with phrasing around:
  - showing a live table in chat
  - rendering a chart or dashboard in chat
  - collecting structured user input in chat
  - reading widget state back on the next turn

### 3. Prompt and router updates

- Update shared prompt guidance for `chat_agent`, `rag_agent`, and `search_agent`:
  - when a structured live visual would materially improve the answer, call `tool_search(..., server_name="widgets")`
  - create or update a widget and then continue the textual answer normally
- Update router guidance so requests like:
  - "show me a comparison table in chat"
  - "render a quick dashboard for this result"
  - "let me click through options in the chat"
  stay with the normal agent path rather than being routed to `canvas_agent`
- Keep the existing website and web-page examples routed to `canvas_agent`.
- Add hard scoping for widget usage if needed:
  - either add explicit allowlist settings for `canvas_agent` and `image_generator_agent`
  - or add targeted code-level exclusions when binding tools for those agents

### 4. Workflow and metadata integration

- Do not rely on the frontend to parse generic tool outputs.
- Extend `build_tool_artifact(...)` so widget tool results retain structured JSON output in addition to the current truncated string output.
- Add a deterministic `extract_live_widgets_from_artifacts(...)` helper during response finalization:
  - inspect structured widget tool artifacts
  - derive `live_widgets`
  - merge them into `build_bot_metadata(...)`
- Extend the interrupt path so the same derived widget metadata is available to `_persist_interrupt_bot_message(...)`.
  - This likely needs `_persist_interrupt_bot_message(...)` to accept extra derived metadata, because interrupt-message persistence does not currently merge accumulated context outputs automatically.
- Keep the current SSE contract intact:
  - no new `data-live-widget` stream event
  - continue using the existing `data-assistant-message` payload and message-history APIs

### 5. Server API integration

- Add `app/api/widgets.py`.
- Export it from `app/api/__init__.py`.
- Wire it in `app/main.py` dependency wiring and router registration.
- Keep widget routing additive and namespaced under `/widgets`.
- Use signed short-lived widget connection tokens instead of bearer auth on browser WebSocket query params.
- Keep token scope narrow:
  - user ID
  - conversation ID
  - widget ID
  - expiry
- Reuse the existing JWT machinery where practical, but with a widget-specific claim set and short TTL.

### 6. Client-backend compatibility

- Add `client_backend/api/widgets.py`, or extend existing proxy routing with a dedicated widget path.
- Proxy `POST /widgets/{widget_id}/connection` to the canonical server.
- Phase 1:
  - keep widget WebSocket relay out of scope
  - the frontend connects directly to the canonical server widget WebSocket
- Phase 3:
  - add `WS /widgets/{widget_id}/connect` to `client_backend`
  - build a dedicated relay service for upstream widget sockets
  - add local fan-out or multiplexing where feasible so the sidecar can act as a proper local realtime gateway instead of a thin pass-through
- Do not register widget tools in the client tool catalog and do not expose them through `client__...`.
- Keep the existing device runtime contract unchanged.

### 7. Canvas separation

- Preserve the existing `canvas_artifact` behavior exactly.
- Add `live_widgets` as a sibling metadata key.
- Allow both keys to coexist on message history without collision.
- Do not mutate `canvas_agent` artifact parsing or storage logic.

### 8. Optional repo-local frontend work

- AI SDK or assistant-ui clients can mount widgets from `live_widgets` immediately once the backend surfaces are ready.
- The in-repo Streamlit demo currently renders `canvas_artifact` only.
- If `demo.py` is a target for this feature, add a dedicated `render_live_widgets(...)` path instead of overloading the canvas renderer.
- If `demo.py` is not a target, keep backend acceptance independent from Streamlit changes.

## Compatibility Notes

### Canvas

- `canvas_agent` remains the authored-artifact path.
- Live widgets are mounted from safe server-side state descriptors during normal chat turns.
- The two systems can share the same frontend host, but they remain separate backend metadata contracts.

### Deferred tool loading

- Widget tools must participate in the same deferred-loading system as other server MCP tools.
- The plan must not introduce a special execution path that bypasses `tool_search`, allowlists, or MCP catalog indexing.
- If field testing later shows under-discovery, promote widget tools to pinned tools as a follow-up, not as part of the initial implementation.

### HITL and interrupt resume

- Widget tools should not require approval by default.
- If a run pauses later on a separate approved tool, the interrupt message must still include `live_widgets`.
- Resume flow remains unchanged except for preserving widget metadata across the pause boundary.

### Sidecar or local backend

- This feature must not alter the canonical ownership split from `plan.md`.
- The server remains the owner of widget state, orchestration, streaming, and HITL.
- `client_backend` only adds connection compatibility for frontend HTTP calls in v1.
- No changes to the existing device-runtime queue, tool dispatch, or skill or MCP catalog sync are required.

## Test Plan

### Unit tests

- Widget store CRUD and TTL eviction
- Redis store behavior
- In-memory store behavior for isolated unit tests
- Max-state-size enforcement
- Optimistic version mismatch behavior
- `user_state_patch` shallow merge and version increment
- Token mint and verify with scope validation

### MCP integration tests

- `widgets` FastMCP server loads through the existing `MCPManager`
- `/mcp/tools` includes the new widget tools with sanitized schemas
- `tool_search(server_name="widgets")` returns widget tools without name collisions

### Workflow and metadata tests

- Completed assistant responses include `message_metadata.live_widgets`
- `/ai/conversations/{conversation_id}/messages` returns `live_widgets` in both `messageMetadata` and mirrored `metadata`
- AI SDK streaming emits the existing `data-assistant-message` payload with `live_widgets`
- Interrupt assistant messages also persist `live_widgets` when widgets were created before the pause

### Router and agent tests

- Explicit website or app requests still route to `canvas_agent`
- In-chat visual-aid requests stay on `chat_agent`, `rag_agent`, or `search_agent`
- Agents can discover widget tools via `tool_search(server_name="widgets")`
- `planning_agent` behavior remains unchanged

### Client-backend compatibility tests

- Proxied widget connection endpoint works through `client_backend`
- Sidecar phase tests:
  - local widget relay WebSocket works through `client_backend`
  - local fan-out or multiplexing behavior does not interfere with existing `/device-runtime/...` flows
- Existing runtime bridge and sidecar tool flows stay unchanged

### Regression tests

- `canvas_artifact` still renders and persists unchanged
- `canvas_artifact` and `live_widgets` can coexist in message history
- Existing HITL resume, deferred client-tool binding, and AI SDK SSE behavior remain green

## Acceptance Criteria

- Phase 1 acceptance:
  - a normal `chat_agent`, `rag_agent`, or `search_agent` turn can create a live widget without switching to `canvas_agent`
  - the frontend can mount the widget using assistant message metadata plus the connection endpoint
  - widget state updates arrive over WebSocket and survive reconnect via `widget_state_sync`
  - user-driven widget state is available to the agent on the next turn through `widget_get_state`
  - deferred MCP loading, current streaming, existing HITL flows, and sidecar local-backend behavior remain compatible
  - `canvas_artifact` remains unchanged and can coexist with `live_widgets`
- Phase 3 acceptance:
  - the frontend can keep widget realtime traffic behind `client_backend`
  - widget relay behavior remains isolated from the device runtime bridge
  - sidecar fan-out or multiplexing avoids unnecessary duplicate upstream widget subscriptions for the same local session

## Assumptions

- Redis is available in any environment that runs the full end-to-end widget flow while `widgets` is an out-of-process stdio MCP server
- `session_id` is the existing conversation or thread ID string
- Widget state is runtime state and does not require DB persistence in v1
- The frontend can extend its existing message renderer to support a live widget session mode without changing the meaning of `canvas_artifact`
