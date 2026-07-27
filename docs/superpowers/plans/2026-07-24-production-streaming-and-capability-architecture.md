# Production Image Streaming and Capability Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Use `superpowers:test-driven-development` for every behavior change and
> `superpowers:verification-before-completion` before closing a phase.

**Goal:** Make generated images appear reliably and early through both the internal
Streamlit stream and the Vercel AI SDK stream, make MCP tool-list responses strictly
server-scoped, and establish one production-grade capability model for backend MCP
tools, device-local MCP tools, skills, defaults, and Custom Agents.

**Architecture:** Ship the user-visible repairs first, behind compatibility-preserving
contracts. Generated final images are persisted as soon as they exist and streamed by
authenticated reference instead of as multi-megabyte base64 SSE frames. MCP list
operations become server-scoped in the catalog layer, not filtered as an optional
presentation concern. A subsequent capability layer separates stable account-wide
intent from deployment policy and volatile device/session bindings, resolving one
fail-closed effective capability set for every agent invocation.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, SQLAlchemy/PostgreSQL JSONB,
Redis-backed client runtime state, LangChain MCP adapters, Streamlit, Vercel AI SDK
UI Message Stream, pytest, Ruff.

---

## Spec Kit Inputs

- Existing image design:
  `docs/superpowers/specs/2026-07-17-image-streaming-design.md`
- Existing image hardening plan:
  `docs/superpowers/plans/2026-07-22-streaming-images-subagents-hardening.md`
- Existing MCP device-isolation design:
  `docs/superpowers/specs/2026-07-23-mcp-config-v2-device-isolation-design.md`
- Existing Custom Agent capability design:
  `docs/superpowers/specs/2026-07-23-custom-agent-device-capability-resolution-design.md`
- Existing frontend contracts:
  `plans/AI_SDK_FE_CONTRACT.md`,
  `plans/AI_SDK_FE_CONTRACT_UPDATES.md`,
  `plans/SKILLS_MCP_HITL_FE_CONTRACT.md`,
  `plans/CUSTOM_AGENTS_FE_CONTRACT.md`
- Official protocol references:
  [AI SDK transient data parts](https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data),
  [Claude Code MCP scopes](https://docs.anthropic.com/en/docs/claude-code/mcp)

## Planning Assumptions

1. "Image streaming" means **early delivery**: show provider partial previews when
   available, and show each complete image as soon as generation finishes, before
   narrative/final message completion. It does not require synthesizing progressive
   pixels for providers such as Gemini that only return a complete image.
2. The local backend on `127.0.0.1:8100` remains the public origin for Streamlit and
   local AI SDK clients; it must proxy protected image reads to the canonical server.
3. Backend MCP servers are deployment-managed. Sidecar MCP servers and all skills
   are device-local. Custom Agent definitions are account-wide, but local capability
   execution is always resolved against the device attached to the current request.
4. Compatibility routes and legacy Custom Agent refs remain readable during a
   deprecation window. New writes use the canonical contracts defined below.
5. Missing optional local capabilities degrade an agent with explicit warnings.
   They never authorize a different device and never silently broaden tool scope.

## Baseline Evidence and Current-State Findings

### Verification performed on 2026-07-24

- Image/MCP adapter baseline: **74 passed**.
- Custom Agent/device/skill/MCP isolation baseline: **133 passed**.
- The live `127.0.0.1:8100` MCP endpoint returned `401` without a local session,
  correctly preventing an unauthenticated production-data inspection.
- The current source contains commit `b55d83a`, which added the `serverName` alias
  to both MCP list endpoints. The reported all-server response is therefore either
  a stale process/build, a query-loss path not represented by the current tests, or
  an endpoint/catalog boundary that still needs live contract coverage.

Passing tests do not prove the reported paths work. Existing image tests use tiny
in-memory payloads and call adapters directly; they do not exercise the full HTTP
stream, the sidecar re-proxy, realistic generated-image sizes, authenticated media
reads, or a real AI SDK consumer.

### Independent full-path verification (2026-07-24)

Every current-state finding below and every Source Map path was re-checked directly
against the working tree by full-file review (not by test output). Result: the plan
is well-grounded. All six confirmed image defects, the resume-sink gap, the transient
`data-image-preview` contract, the post-load MCP filtering, the `client_only`
fail-open, the `enabledByDefault` conflation, the dead `skill_settings`, and the
process-local MCP registry all reproduce at the cited code. Five statements are
tightened so implementers do not build against an inaccurate premise:

| # | Plan statement | Correction | Evidence |
|---|---|---|---|
| 1 | Oversized preview is "silently" dropped | Not silent — emits `logger.info`; the real gap is no metric and no stream event | `app/ai/image_generation/emitter.py:70-78`; cap default 4,000,000 at `app/core/config.py:512` |
| 2 | Same-name tools "need stable IDs and call aliases" | Stable IDs + deterministic aliases already exist; the defect is one inconsistent bare-name dedup, not a missing system | aliases `app/ai/tool_search_tool.py:75-87`, `app/ai/deferred_tool_binding.py:25-66`, `app/ai/client_tool_catalog.py:52,57`; bare-name collapse `app/ai/agents/custom_agent.py:270-278` |
| 3 | Provenance is guessed from a bare tool name | Provenance is already stamped at load; only backend resolution falls back to a fragile `id()`-map. T006 must consume the stamped field, not add it | stamp `app/core/mcp_adapter_utils.py:230-248`, `app/ai/mcp_integration.py:307`; fragile lookup `app/ai/mcp_integration.py:366-368` |
| 4 | Empty per-agent allowlist means "all tools" | True only for the settings-driven base-agent path; Custom Agents already fail closed | base `app/ai/agents/base_agent.py:387-401`; custom `app/ai/custom_agent_runtime.py:262-305` |
| 5 | Scoped MCP load must be added from scratch | Sidecar already has an unused `get_tools_by_server()`; the endpoint just calls `get_all_tools()` then filters | unused `client_backend/services/local_mcp_manager.py:229-231` vs `client_backend/api/mcp.py:394-395` |

The baseline test counts (74 image/MCP, 133 isolation) are the plan author's; they
were not re-run in this pass. Migrations already present were confirmed:
`d7e8f9a0b1c2` (chat_images) and `j1k2l3m4n5o6` (creates `skill_settings`).

### Confirmed image delivery defects

1. `ChatImageStorageService` returns protected relative URLs such as
   `/chat-images/{image_id}`.
2. The canonical server exposes `GET /chat-images/{image_id}`, but
   `client_backend` does not expose or proxy that route.
3. `demo.py` uses the local backend as `API_BASE_URL`, so its authoritative image
   reload requests are sent to port 8100, where the route is absent.
4. Streamlit clears the transient preview on `complete`; if the authoritative
   reference cannot be fetched, the image disappears or never becomes durable.
5. AI SDK terminal `file` parts can contain the same relative URL. A browser image
   element cannot add the Bearer token automatically, so a bare `<img src>` is not
   a sufficient protected-media contract.
6. `_normalize_image_item_to_file_part` does not explicitly recognize protected
   relative URLs in every path and can treat an arbitrary relative string as loose
   base64.

### High-probability image delivery defects requiring characterization

1. `ImagePreviewPublisher` drops an inline base64 payload above
   `image_stream_preview_max_b64_chars` (default 4,000,000 characters) with only an
   info-level log — no metric and no stream event, so no consumer can observe the
   drop. A normal generated image can exceed this.
2. Gemini returns a complete image rather than progressive partials. If that final
   payload exceeds the SSE cap, no early image event exists at all.
3. AI SDK projects previews as `transient: true` `data-image-preview` parts.
   Per the AI SDK contract, transient parts are available only through `useChat`
   `onData`; they never appear in `message.parts`.
4. Resume streams intentionally install no preview sink, so an image generated
   after a HITL resume cannot emit an early preview.
5. Current tests do not assert event ordering through
   canonical server -> sidecar -> consumer, nor that exactly one `[DONE]` survives
   the proxy.

### MCP and capability architecture findings

1. The query-alias regression is fixed in current source, but filtering still occurs
   after obtaining the full catalog. A strict server request should be scoped in the
   manager/catalog operation itself.
2. `enabledByDefault` currently conflates installed/available, administrator-enabled,
   globally default, and model-bound. Those are separate policy decisions.
3. Empty per-agent allowlists can mean "all tools", which is too implicit for a
   production permission boundary. Verified scope: this "empty means all" holds only
   for the settings-driven base-agent path (`app/ai/agents/base_agent.py:387-401`);
   Custom Agents already fail closed — empty refs bind no tools
   (`app/ai/custom_agent_runtime.py:262-305`).
4. Tool collections are deduplicated by public name in several binding paths.
   Same-name tools from different servers need stable IDs and deterministic model
   call aliases. Verified nuance: stable IDs and call aliases already exist
   (`app/ai/tool_search_tool.py:75-87`, `app/ai/deferred_tool_binding.py:25-66`,
   `app/ai/client_tool_catalog.py:52,57`) and `base_agent._deduplicate_tools` already
   keys on `(name, server_name)`; the remaining defect is the Custom Agent binding
   path collapsing by bare name (`app/ai/agents/custom_agent.py:270-278`), so this is
   a consistency fix, not a new aliasing system.
5. `client_only` is downgraded to `default` when `device_id` is absent. This is
   fail-open: a request intending local-only tools can become eligible for backend
   tools.
6. Custom Agents correctly rebase saved logical client refs onto the active device,
   but the public schemas still mix stable intent with volatile device/session/catalog
   identity and retain legacy `server_default` and server-skill shapes.
7. All active skills are device-local, while the unused `skill_settings` model,
   repository, migration, and README entry describe per-user server-side toggles.
8. Backend MCP manager state is process-local. Production multi-worker deployments
   need deterministic catalog versions and coordinated configuration/reload behavior.

---

## Product Requirements

### Image streaming

- **FR-IMG-001:** A provider partial preview, when supplied and within the explicit
  transient budget, is visible before the final image.
- **FR-IMG-002:** Every provider final image is persisted and an authenticated image
  reference is emitted before narrative completion.
- **FR-IMG-003:** Final image delivery must not depend on the base64 SSE size cap.
- **FR-IMG-004:** Streamlit and AI SDK streams use the same canonical image event
  semantics and preserve ordering through the sidecar.
- **FR-IMG-005:** A terminal message and a reloaded conversation render the same
  authoritative image.
- **FR-IMG-006:** Protected image bytes are never made public merely to support
  `<img src>`.
- **FR-IMG-007:** Oversized/dropped partial previews produce a metric and structured
  diagnostic status; they are not silently discarded.
- **FR-IMG-008:** A resumed run has the same early-image behavior as a new run.
- **FR-IMG-009:** Disconnect/cancellation does not leave unbounded preview queues or
  orphan repeated storage writes.

### MCP catalog endpoints

- **FR-MCP-001:** A server-scoped request returns tools from exactly one server or
  a typed not-found response.
- **FR-MCP-002:** The unscoped compatibility request is the only operation allowed
  to return tools from multiple servers.
- **FR-MCP-003:** Server ownership is attached when tools are loaded and retained in
  a stable descriptor; it is never guessed from a bare tool name.
- **FR-MCP-004:** Canonical server and sidecar endpoints expose equivalent scope
  metadata and camelCase response aliases.
- **FR-MCP-005:** Unknown, disabled, failed, and empty servers have distinct,
  documented outcomes.
- **FR-MCP-006:** MCP server names are passed as encoded query/path parameters, not
  string-concatenated URLs.

### Capabilities, skills, defaults, and Custom Agents

- **FR-CAP-001:** Every capability has one stable logical ID independent of its live
  process/session binding.
- **FR-CAP-002:** Availability, default policy, model binding, user approval, and
  execution authorization are separate stages.
- **FR-CAP-003:** Device-local capability execution requires exact
  user + device + session + catalog version + instance identity.
- **FR-CAP-004:** The same account on two devices never borrows a capability,
  credential, approval, or local path from the other device.
- **FR-CAP-005:** A request without device context receives no device-local tools or
  skills. `client_only` without a device remains client-only and resolves to an
  explicit unavailable/empty result.
- **FR-CAP-006:** Custom Agents persist account-wide desired capabilities using
  stable IDs; request-time resolution creates volatile execution bindings.
- **FR-CAP-007:** Missing desired capabilities degrade the Custom Agent without
  broadening its allowlist.
- **FR-CAP-008:** Deployment defaults are explicit versioned policy, not inferred
  from whether an MCP server happens to be enabled.
- **FR-CAP-009:** Skills are instruction packages. Any executable action they expose
  is a separately identified tool and passes the same authorization/HITL gate.
- **FR-CAP-010:** Tool name collisions are deterministic and visible to the model
  through stable call aliases.
- **FR-CAP-011:** Catalog versions and configuration hashes are deterministic across
  server workers.
- **FR-CAP-012:** All capability resolution and execution decisions are auditable
  without logging secrets, prompt bodies, or local filesystem contents.

## Non-Goals

- Creating synthetic progressive image frames for providers that do not expose them.
- Synchronizing sidecar MCP credentials, local skill files, or local enabled states
  between devices.
- Allowing a canonical server worker to execute device-local tools directly.
- Removing all compatibility fields in the first release.
- Turning every backend MCP tool into a global default.
- Replacing MCP itself or rewriting the agent graph in one release.

---

## Target Architecture

### Capability scopes

| Scope | Owner | Examples | Persistence |
|---|---|---|---|
| Deployment | Administrator/release | Backend MCP definitions, internal tools, default policy | Versioned app config / secret manager |
| Account | Authenticated user | Custom Agent prompt/model and desired stable capability IDs | PostgreSQL |
| Device | User installation | Sidecar MCP config/secrets, installed skills, local approvals | Device profile + live server snapshot |
| Request | Authenticated turn | Active device, resolved tools, denials, HITL grants | Immutable run context / audit record |

The scopes compose; they do not overwrite each other. A request resolver computes
the intersection of availability, deployment policy, account intent, device binding,
and request denials.

### Stable capability identity

Use a canonical string plus structured fields:

```text
internal::<tool_name>
server_mcp::<server_name>::<qualified_tool_id>
client_mcp::<server_name>::<qualified_tool_id>
skill::<skill_name>
```

`qualified_tool_id` is the MCP server's logical tool identity, not a model-facing
alias and not a device instance ID.

Introduce a shared descriptor in `app/ai/capabilities/models.py`:

```python
class CapabilityDescriptor(BaseModel):
    capability_id: str
    kind: Literal["internal_tool", "mcp_tool", "skill"]
    origin: Literal["deployment", "device"]
    server_name: str | None = None
    logical_tool_id: str | None = None
    display_name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None
    schema_hash: str
    risk: Literal["read", "write", "destructive", "external_side_effect"]
    default_eligible: bool = False
```

Live device tools add a separate binding:

```python
class DeviceCapabilityBinding(BaseModel):
    capability_id: str
    user_id: str
    device_id: str
    session_id: str
    catalog_version: int
    tool_instance_id: str
    call_alias: str
```

Never persist `DeviceCapabilityBinding` as Custom Agent intent.

### Resolution pipeline

```text
Definition/catalog
       |
       v
Deployment policy ---- Account Custom Agent intent
       |                         |
       +------------+------------+
                    v
          Active device snapshot
                    |
                    v
        Request denials / HITL policy
                    |
                    v
       EffectiveCapabilitySet (immutable)
                    |
                    v
         Exact execution authorization
```

`EffectiveCapabilitySet` contains bound model tools, missing desired IDs, warnings,
catalog/config versions, and authorization metadata. Base agents and Custom Agents
must call the same resolver.

### Default policy

Replace inference from `enabledByDefault` with a versioned policy manifest:

```json
{
  "schemaVersion": 1,
  "profiles": {
    "chat": {
      "alwaysBound": ["internal::hand_off"],
      "deferredEligible": [
        "server_mcp::time::time::get_current_time",
        "server_mcp::brave_image_search::brave_image_search::search"
      ],
      "pinned": []
    }
  }
}
```

Exact IDs in the real manifest must be generated/validated against the catalog; the
example above is illustrative. Enabling a server makes it available to policy. It
does not automatically grant or bind every tool.

### Multi-device behavior

| Scenario | Result |
|---|---|
| Same account, device A request, capability installed on A | Resolve and execute on A |
| Same account, device A request, capability only on B | Degraded/missing; never scan B |
| Same account, no active device, server capabilities selected | Server capabilities may run per policy |
| Same account, no active device, local capability selected | Degraded/missing |
| `client_only`, no active device | Empty unavailable local scope; no server fallback |
| Device reconnects with new session/catalog version | Rebind stable intent to the new exact binding |
| Same MCP server/tool names on A and B | Resolve using request device; execution metadata remains distinct |
| Device catalog changes during resolution | Retry one consistent snapshot read or fail degraded; never mix versions |

### Image delivery contract

Retain `image_preview` as the compatibility event name and version its data:

```json
{
  "type": "image_preview",
  "data": {
    "schema_version": 2,
    "item_id": "image-preview-0",
    "image_index": 0,
    "status": "partial",
    "seq": 1,
    "media_type": "image/png",
    "delivery": {
      "kind": "inline",
      "data_url": "data:image/png;base64,..."
    }
  }
}
```

Final delivery is always by protected reference:

```json
{
  "type": "image_preview",
  "data": {
    "schema_version": 2,
    "item_id": "image-preview-0",
    "image_index": 0,
    "status": "final",
    "seq": 2,
    "media_type": "image/png",
    "delivery": {
      "kind": "reference",
      "image_id": "uuid",
      "url": "/chat-images/uuid"
    }
  }
}
```

Rules:

- Inline delivery is permitted only for transient provider partials under a small,
  explicit wire budget.
- Final bytes are stored once before the final event and reused by message metadata.
- Both the canonical server and sidecar expose the same protected relative URL.
- Streamlit fetches the reference server-side with its Bearer token.
- AI SDK `onData` handles the transient update and uses an authenticated fetch helper
  to create/revoke a browser Blob URL. Terminal `file` parts use the same helper.
- A client that ignores preview events still renders the terminal persisted image.
- Schema-v1 inline events remain readable during the compatibility window.

### MCP endpoint contract

Canonical route:

```http
GET /mcp/servers/{server_name}/tools
```

Compatibility route:

```http
GET /mcp/tools?serverName={server_name}
```

Scoped response:

```json
{
  "success": true,
  "message": "MCP tools retrieved successfully",
  "data": {
    "scope": {"kind": "server", "serverName": "brave_image_search"},
    "tools": [],
    "totalCount": 0,
    "serversCount": 1,
    "catalogVersion": "sha256:..."
  }
}
```

`serversCount` is `1` for a known scoped server, even if it currently exposes zero
tools. Unknown server returns `404 MCP_SERVER_NOT_FOUND`; disabled server returns
`409 MCP_SERVER_DISABLED` (or `200` plus an explicit disabled state if compatibility
requirements demand it—choose once and lock it in tests). Failed initialization
returns `503 MCP_SERVER_UNAVAILABLE` with a sanitized diagnostic ID.

---

## Constitution Check

- **Authentication:** PASS. All catalogs and media reads remain authenticated.
- **Media confidentiality:** PASS. Final image URLs stay protected; clients fetch
  with credentials rather than making storage public.
- **Device isolation:** PASS. Stable intent can move between the user's devices, but
  a live binding is created only from the request device.
- **Fail-closed scopes:** REQUIRED CHANGE. Remove `client_only -> default` fallback.
- **Least privilege:** REQUIRED CHANGE. Separate availability/default/binding/approval.
- **Backward compatibility:** PASS WITH DEPRECATION. Existing routes and refs remain
  readable while new contracts are introduced additively.
- **Multi-worker consistency:** REQUIRED CHANGE. Replace process-dependent versions
  with deterministic catalog/config hashes and coordinated reload policy.
- **Observability:** REQUIRED CHANGE. Eliminate silent preview drops and implicit
  capability broadening.

## Source Map

### Create

- `app/ai/capabilities/__init__.py`
- `app/ai/capabilities/models.py`
- `app/ai/capabilities/catalog.py`
- `app/ai/capabilities/policy.py`
- `app/ai/capabilities/resolver.py`
- `app/ai/capability_policy.json`
- `client_backend/api/chat_images.py`
- `tests/test_image_stream_http_contract.py`
- `tests/client_backend/test_image_stream_proxy.py`
- `tests/test_mcp_tools_http_scope.py`
- `tests/test_capability_catalog.py`
- `tests/test_capability_policy.py`
- `tests/test_effective_capability_resolver.py`
- `scripts/verify_image_streaming_contract.py`
- `scripts/verify_capability_isolation.py`

### Modify

- `app/ai/image_generation/emitter.py`
- `app/ai/image_generation/models.py`
- `app/ai/agents/image_generator_agent.py`
- `app/ai/graph.py`
- `app/ai/tool_scope.py`
- `app/ai/agents/base_agent.py`
- `app/ai/agents/custom_agent.py`
- `app/ai/mcp_integration.py`
- `app/ai/mcp_registry.py`
- `app/ai/deferred_tool_state.py`
- `app/ai/tool_search_tool.py`
- `app/services/chat_image_service.py`
- `app/services/message_service.py`
- `app/services/event_streaming/events.py`
- `app/services/event_streaming/graph_public_projection.py`
- `app/services/event_streaming/internal_sse.py`
- `app/services/event_streaming/ai_sdk_v6.py`
- `app/services/event_streaming/ai_sdk_projection.py`
- `app/services/event_streaming/subagents.py`
- `app/services/mcp_service.py`
- `app/api/mcp.py`
- `app/schemas/mcp.py`
- `app/schemas/custom_agent.py`
- `app/services/custom_agent_capability_resolver.py`
- `app/services/custom_agent_service.py`
- `client_backend/main.py`
- `client_backend/api/mcp.py`
- `client_backend/api/messages.py`
- `client_backend/api/proxy.py`
- `client_backend/services/local_mcp_manager.py`
- `client_backend/services/runtime_bridge.py`
- `client_backend/services/server_api.py`
- `client_backend/services/local_skills_registry.py`
- `demo.py`
- `README.md`
- `plans/AI_SDK_FE_CONTRACT.md`
- `plans/AI_SDK_FE_CONTRACT_UPDATES.md`
- `plans/SKILLS_MCP_HITL_FE_CONTRACT.md`
- `plans/CUSTOM_AGENTS_FE_CONTRACT.md`

### Migrate/deprecate

- `app/models/custom_agent.py`
- `app/alembic/versions/<new>_add_custom_agent_capability_refs_v2.py`
- `app/models/skill_setting.py`
- `app/repositories/skill_setting.py`
- `app/alembic/versions/<later>_remove_unused_skill_settings.py`

Phase 5 removal targets (deletions gated on their preconditions — see Phase 5):

- `app/ai/tool_scope.py` — remove the `client_only -> default` downgrade (T015).
- `app/ai/mcp_integration.py`, `app/ai/deferred_tool_binding.py` — retire
  `DEFAULT_SERVERS` and duplicated pin constants into policy (T016).
- `app/schemas/custom_agent.py`, `app/ai/custom_agent_runtime.py`,
  `app/services/custom_agent_service.py`,
  `app/services/custom_agent_capability_resolver.py` — remove
  `server_default`/`serverTools` shapes and `source="server"` skills (T017).
- `app/ui/rich_response.py`, `app/core/response_constants.py`,
  `app/services/event_streaming/internal_sse.py`,
  `app/services/event_streaming/ai_sdk_v6.py` — retire v1 image reads after
  backfill (T018).

Do not remove the skill settings table in the urgent image/MCP release. First prove
it has no live readers/writers in production telemetry, then remove it in the
capability migration release. Preserve the intentional resilience fallbacks listed
at the top of Phase 5; they are not cleanup targets.

## Dependency Graph

```text
T001 characterization
  |-----------+------------------+
  v           v                  v
T002 media  T006 MCP scope    T008 capability model
  |           |                  |
T003 wires  T007 endpoint/UI     +--> T009 defaults/base agents
  |                              +--> T010 Custom Agents migration
T004 resume                      +--> T011 skills/HITL cleanup
  |                                      |
T005 image E2E                           v
                                  T012 multi-device/multi-worker
                                              |
                                              v
                                      T013 rollout verification
                                              |
                                              v
                                 T014-T019 deprecation cleanup
                                 (each gated on adoption/parity/backfill)
```

T002–T005 are the urgent image workstream. T006–T007 are the urgent MCP endpoint
workstream. They can ship before T008–T013. Do not combine all changes into one
high-risk release.

---

## Phase 0 — Characterize the Real Failures

### Task 1 (T001): Add failing full-path characterization tests

**Files:**

- Create: `tests/test_image_stream_http_contract.py`
- Create: `tests/client_backend/test_image_stream_proxy.py`
- Create: `tests/test_mcp_tools_http_scope.py`
- Create: `scripts/verify_image_streaming_contract.py`
- Modify: `tests/test_image_preview_stream.py`

- [x] Build a deterministic fake image provider that emits:
  one 128 KiB partial, one final image above 4,000,000 base64 characters, then
  narrative text.
- [x] Drive it through `MessageService.create_message_stream` and the actual
  FastAPI internal SSE route. Assert partial/final ordering and terminal state.
- [x] Drive the same fixture through `/api/chat/{conversation_id}` and parse the
  AI SDK wire protocol, including exactly one `[DONE]`.
- [x] Drive both streams through `client_backend.create_app()` with dependency
  overrides. Do not call adapter helpers directly.
- [x] Add a failing test showing `GET /chat-images/{id}` is absent from the
  sidecar and the streamed reference cannot be fetched through port 8100.
- [x] Add a failing test showing protected relative image URLs are not treated as
  loose base64.
- [x] Add a failing resume-stream case in which image generation occurs after HITL.
- [x] Add mixed-catalog HTTP tests for both canonical and sidecar MCP apps:
  `brave_image_search` must never return a `widgets` tool.
- [x] Record event ordering and payload byte sizes in the test failure messages.
- [x] Keep provider-network tests opt-in; the deterministic test is the CI gate.

> **T001 status (2026-07-24):** Done — 9 RED characterization assertions
> committed (`a4c26a2`), 7 GREEN (2 regression locks + harness), 1 opt-in skip.
> See Implementation Design Decisions Log at end of doc.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_image_stream_http_contract.py `
  tests/client_backend/test_image_stream_proxy.py `
  tests/test_mcp_tools_http_scope.py -q
```

Expected before implementation: failures for oversized final preview delivery,
sidecar media read, resume preview, and any query-loss path found.

---

## Phase 1 — Reliable Image Early Delivery

### Task 2 (T002): Persist final images before narrative completion

**Files:**

- Modify: `app/services/chat_image_service.py`
- Modify: `app/ai/image_generation/emitter.py`
- Modify: `app/ai/image_generation/models.py`
- Modify: `app/ai/agents/image_generator_agent.py`
- Modify: `app/ai/graph.py`
- Modify: `app/services/message_service.py`
- Modify: `tests/test_chat_image_service.py`
- Modify: `tests/test_bot_metadata_image_externalization.py`
- Modify: `tests/test_message_service_attachment_externalization.py`

- [x] Write a failing test proving a final image is stored before the next
  narrative delta/terminal event.
- [x] Introduce an injected per-run media delivery service with
  `publish_partial(...)` and `persist_final(...)`.
- [x] Bind it with request user/conversation context at the graph boundary; do not
  let the provider or agent reach into the global DI container.
- [x] Make `persist_final` idempotent by request/item/content hash and return the
  existing `image_id/url` descriptor.
- [x] Store the descriptor in graph/message metadata so terminal persistence reuses
  it rather than decoding and writing the final bytes again.
- [x] Keep final image bytes subject to the storage byte cap, not the transient SSE
  character cap.
- [x] On storage failure, emit a typed image-delivery error and continue narrative
  only if current product policy permits a text-only answer; record the failure.
- [x] Preserve legacy metadata reads while ensuring all new generated image writes
  contain references, never base64.

> **T002 status (2026-07-24):** Done — `f107ce2`, reviewer Approved (0
> Critical/Important, 3 Minor). `MediaDeliveryService` added; final images persist
> early and idempotently; terminal reuse via `response_constants._reuse_stored_ref`.
> See Implementation Design Decisions Log at end of doc.

### Task 3 (T003): Version the event and repair both public transports

**Files:**

- Modify: `app/services/event_streaming/events.py`
- Modify: `app/services/event_streaming/graph_public_projection.py`
- Modify: `app/services/event_streaming/internal_sse.py`
- Modify: `app/services/event_streaming/ai_sdk_v6.py`
- Modify: `app/services/event_streaming/ai_sdk_projection.py`
- Modify: `client_backend/api/messages.py`
- Modify: `client_backend/services/server_api.py`
- Modify: `tests/test_image_preview_stream.py`
- Modify: `tests/test_ai_sdk_v6_stream_contract.py`

- [x] Write wire snapshots for v1 inline partial and v2 reference final events.
- [x] Emit the versioned canonical delivery union exactly as specified above.
- [x] Keep AI SDK partial/final updates as `data-image-preview`, stable by
  `item_id`, and document that transient parts require `useChat({ onData })`.
- [x] Ensure terminal AI SDK `file` parts preserve protected relative URLs as URLs;
  never reinterpret them as base64.
- [x] Preserve one upstream `[DONE]` across the canonical/sidecar proxy boundary.
- [x] Enforce a small configurable inline-partial budget at serialization time as a
  second defense, with a structured `preview_skipped` status.
- [x] Add counters for emitted, coalesced, oversized, storage-failed, proxied, and
  consumer-disconnected events.
- [x] Never include raw image base64 in logs, traces, or exception payloads.

> **T003 status (2026-07-24):** Done — `a2b1417`, reviewer Approved (0
> Critical/Important, 3 Minor). Schema-v2 `delivery` union emitted; final always by
> reference; `_normalize_image_item_to_file_part` repaired; single `[DONE]` preserved;
> configurable inline budget + `preview_skipped`; v1 read-compat kept. 7 Task-1
> characterization tests flipped GREEN. See Design Decisions Log at end of doc.

### Task 4 (T004): Add the protected sidecar media route and consumer renderers

**Files:**

- Create: `client_backend/api/chat_images.py`
- Modify: `client_backend/main.py`
- Modify: `client_backend/api/proxy.py`
- Modify: `client_backend/api/common.py`
- Modify: `demo.py`
- Modify: `tests/client_backend/test_image_stream_proxy.py`
- Modify: `tests/test_demo_image_reference_rendering.py`
- Modify: `plans/AI_SDK_FE_CONTRACT.md`
- Modify: `plans/AI_SDK_FE_CONTRACT_UPDATES.md`

- [x] Add `GET /chat-images/{image_id}` and `/api/chat-images/{image_id}` to the
  sidecar. Require the local session, attach upstream auth, stream the upstream
  body/content type, and do not buffer unbounded data.
- [x] Forward cache validators and safe content headers; add `nosniff` and a
  restrictive content security policy where applicable.
- [x] Preserve canonical `401/404/413/5xx` semantics without exposing upstream
  internal paths.
- [x] Update Streamlit preview state to replace partial -> final by item/index and
  keep the final reference visible across `complete`; do not clear first and hope a
  later gallery fetch succeeds.
- [x] Use a shared authenticated image fetch helper for live and history rendering.
- [x] Specify the AI SDK frontend handler:
  consume `data-image-preview` in `onData`, fetch reference URLs using the same
  authenticated transport, create a Blob URL, replace by `id/seq`, and revoke stale
  Blob URLs on replacement/unmount.
- [x] Specify a custom terminal `file` renderer for protected relative URLs.
- [x] Add tests for owner success, other-user 404, missing/expired token, correct
  MIME, cancellation, and a payload at the configured maximum.

> **T004 status (2026-07-24):** Done — `d6534fa`, reviewer Approved (0
> Critical/Important, 4 Minor). Sidecar `/chat-images` + `/api/chat-images` proxy
> route (auth-before-upstream, other-user→404 no oracle, streamed, 413 on over-max);
> Streamlit keeps the final reference across `complete` via one shared fetch helper;
> AI SDK handler + terminal renderer specified in the FE contract. Flips the 2 sidecar
> route characterization tests + 7 media + 3 demo tests green. See Design Decisions Log.

### Task 5 (T005): Restore resume parity and prove end-to-end behavior

**Files:**

- Modify: `app/ai/graph.py`
- Modify: `app/services/event_streaming/subagents.py`
- Modify: `app/services/message_service.py`
- Modify: `tests/test_image_stream_http_contract.py`
- Modify: `scripts/verify_image_streaming_contract.py`
- Modify: `docs/superpowers/specs/2026-07-17-image-streaming-design.md`

- [ ] Install the same request-scoped media sink for new and resumed executions.
- [ ] Preserve queue coalescing for partials but treat final references as lossless.
- [ ] On backpressure, discard stale partials for the same item before unrelated
  events; never discard the final reference.
- [ ] Verify Streamlit and AI SDK event order:
  partial (optional) -> final reference -> narrative deltas -> complete -> one done.
- [ ] Verify generation, conversation reload, and a follow-up model turn all resolve
  the same stored image.
- [ ] Run the optional real-provider smoke test once for Gemini and once for OpenAI,
  recording provider behavior without placing API keys or image data in artifacts.
- [ ] Update the old design's "oversized previews are dropped" and "resume has no
  sink" decisions to the new contract.

Image phase gate:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_image_generation_providers.py `
  tests/test_image_preview_stream.py `
  tests/test_image_stream_http_contract.py `
  tests/test_ai_sdk_v6_stream_contract.py `
  tests/test_demo_stream_rendering.py `
  tests/test_demo_image_reference_rendering.py `
  tests/client_backend/test_image_stream_proxy.py -q
```

Manual acceptance:

1. Generate one Gemini image through Streamlit; it appears before narrative finish,
   remains after completion, and remains after page reload.
2. Generate one OpenAI image; available partials replace in place and final remains.
3. Repeat through an AI SDK `useChat` client with `onData`.
4. Trigger HITL before image generation, resume, and repeat.
5. Inspect the SSE stream: no final multi-megabyte base64 frame.

---

## Phase 2 — Strict MCP Server-Scoped Catalogs

### Task 6 (T006): Move server scoping into both MCP catalog implementations

**Files:**

- Modify: `app/ai/mcp_integration.py`
- Modify: `app/services/mcp_service.py`
- Modify: `client_backend/services/local_mcp_manager.py`
- Modify: `tests/test_mcp_integration.py`
- Modify: `tests/client_backend/test_mcp_tool_execution_api.py`
- Modify: `tests/test_mcp_tools_http_scope.py`

- [ ] Add `list_tool_descriptors(server_name: str | None)` to each manager.
- [ ] For a scoped request, load/read only that server's tool collection. Do not
  call "get all" and then rely on a response-layer filter. The sidecar already
  exposes an unused `get_tools_by_server()`
  (`client_backend/services/local_mcp_manager.py:229-231`) — route the scoped path
  through it instead of `get_all_tools()`; add the equivalent scoped read on the
  backend manager.
- [ ] Consume the server provenance already stamped at load
  (`metadata["server_name"]`, `qualified_tool_id` — `app/core/mcp_adapter_utils.py:230-248`)
  rather than the fragile `id()`-keyed `_tool_server_map`
  (`app/ai/mcp_integration.py:366-368`); never re-derive ownership from a bare name.
- [ ] Make duplicate bare names legal across servers; require a server-qualified ID
  for execution when ambiguous.
- [ ] Compute a deterministic catalog hash from sorted sanitized descriptors.
- [ ] Add unit tests for known-empty, unknown, disabled, unavailable, duplicate-name,
  and same tool-name/different-server cases.

### Task 7 (T007): Add the canonical scoped route and lock proxy/UI behavior

**Files:**

- Modify: `app/api/mcp.py`
- Modify: `app/schemas/mcp.py`
- Modify: `client_backend/api/mcp.py`
- Modify: `demo.py`
- Modify: `tests/test_mcp_tools_http_scope.py`
- Modify: `tests/client_backend/test_mcp_tool_execution_api.py`
- Modify: `tests/test_demo_mcp_tools.py`

- [ ] Add `GET /mcp/servers/{server_name}/tools` to canonical and sidecar APIs.
- [ ] Make `/mcp/tools?serverName=` delegate to the same scoped service method.
- [ ] Include the applied scope and catalog version in the response.
- [ ] Use structured request parameters/URL encoding in `demo.py`, not string
  concatenation.
- [ ] Add a test using the full sidecar app and mixed real-shaped descriptors:
  `brave_image_search` response set must equal `{"brave_image_search"}`.
- [ ] Add a startup/build SHA to health diagnostics so a stale process is
  immediately distinguishable from current source.
- [ ] Add a deployment smoke check that authenticates normally and compares
  scoped endpoint output with the selected server card.
- [ ] Remove any UI-side filtering used to conceal an unscoped backend response;
  the UI may assert the response scope and fail visibly on contract violation.

MCP phase gate:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_mcp_integration.py `
  tests/test_mcp_tools_http_scope.py `
  tests/test_mcp_global_allowlist.py `
  tests/test_demo_mcp_tools.py `
  tests/client_backend/test_mcp_tool_execution_api.py -q
```

---

## Phase 3 — Unified Production Capability Model

### Task 8 (T008): Introduce shared capability descriptors and immutable resolution

**Files:**

- Create: `app/ai/capabilities/models.py`
- Create: `app/ai/capabilities/catalog.py`
- Create: `app/ai/capabilities/resolver.py`
- Create: `tests/test_capability_catalog.py`
- Create: `tests/test_effective_capability_resolver.py`
- Modify: `app/ai/tool_scope.py`
- Modify: `app/ai/client_tool_catalog.py`
- Modify: `app/ai/mcp_registry.py`

- [ ] Write tests for stable IDs, schema hashes, collision-safe call aliases, source
  scopes, and deterministic ordering.
- [ ] Adapt server MCP, client MCP, internal tools, and skill summaries into the
  shared descriptor without changing execution yet.
- [ ] Implement `EffectiveCapabilityRequest` and `EffectiveCapabilitySet`.
- [ ] Make unknown scope strings validation errors at the API boundary; do not
  coerce them to `default`.
- [ ] Preserve `CLIENT_ONLY` when device context is absent and return an explicit
  unavailable local snapshot.
- [ ] Require one consistent device snapshot identity before and after catalog read.
- [ ] Add property tests that no result contains a device binding from a non-request
  device.

### Task 9 (T009): Separate deployment availability from default agent policy

**Files:**

- Create: `app/ai/capability_policy.json`
- Create: `app/ai/capabilities/policy.py`
- Create: `tests/test_capability_policy.py`
- Modify: `app/ai/mcp_config.json`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/tool_search_tool.py`
- Modify: `app/ai/deferred_tool_state.py`
- Modify: `tests/test_mcp_global_allowlist.py`
- Modify: `tests/test_unified_tool_search.py`
- Modify: `README.md`

- [ ] Characterize the effective bound/searchable tool set for every built-in agent.
- [ ] Add a strict, versioned policy schema with exact stable IDs and startup
  validation against the deployment catalog.
- [ ] Define separately: available, default eligible, always bound, deferred
  eligible, pinned, denied, approval required.
- [ ] Change an empty allowlist to mean empty. Use an explicit `"*"`/`all_available`
  policy only where intentionally approved.
- [ ] Preserve current intended behavior through an explicit migration manifest.
- [ ] Emit policy version and effective capability IDs into sanitized run diagnostics.
- [ ] Remove hard-coded `DEFAULT_SERVERS`/agent pin duplication only after parity
  snapshots pass.

### Task 10 (T010): Migrate Custom Agents to stable capability intent

**Files:**

- Modify: `app/models/custom_agent.py`
- Create: `app/alembic/versions/<new>_add_custom_agent_capability_refs_v2.py`
- Modify: `app/schemas/custom_agent.py`
- Modify: `app/services/custom_agent_capability_resolver.py`
- Modify: `app/services/custom_agent_service.py`
- Modify: `app/ai/agents/custom_agent.py`
- Modify: `app/ai/custom_agent_runtime.py`
- Modify: `demo.py`
- Modify: `tests/test_custom_agent_capability_resolver.py`
- Modify: `tests/test_custom_agent_client_tool_resync.py`
- Modify: `tests/test_custom_agents_service.py`
- Modify: `tests/test_custom_agents_api.py`
- Modify: `tests/test_custom_agents_tools.py`
- Modify: `plans/CUSTOM_AGENTS_FE_CONTRACT.md`

- [ ] Add additive `capability_refs_v2` JSONB containing only stable
  `capability_id`, kind, and optional display metadata.
- [ ] Backfill `server_mcp`, `client`, and skill refs to v2 stable IDs. Quarantine
  invalid/ambiguous legacy refs as missing intent; never guess by bare name.
- [ ] Dual-read old rows and dual-write compatibility fields for one release.
- [ ] New create/update requests accept canonical stable refs and reject volatile
  device/session/catalog/instance fields as authorization evidence.
- [ ] Use the shared resolver for management options, availability, prompt warnings,
  model binding, and execution.
- [ ] Keep the current behavior: a missing local capability marks the agent degraded,
  skips only that capability, and does not disable model/prompt/server capabilities.
- [ ] Replace confusing `serverDefaultTools`/empty `serverTools` options with a
  canonical `capabilities` tree while retaining legacy fields through deprecation.
- [ ] Add same-account A->B rebind, A-only missing on B, offline, reconnect, and
  forged-current-binding tests.

### Task 11 (T011): Reconcile skills and HITL with the capability model

**Files:**

- Modify: `client_backend/services/local_skills_registry.py`
- Modify: `client_backend/services/runtime_bridge.py`
- Modify: `app/ai/skill_resolver.py`
- Modify: `app/ai/skills_tool.py`
- Modify: `app/ai/agents/custom_agent.py`
- Modify: `tests/test_skill_device_isolation.py`
- Modify: `tests/test_hitl_settings_device_isolation.py`
- Modify: `tests/test_hitl_turn_policy_injection.py`
- Modify: `plans/SKILLS_MCP_HITL_FE_CONTRACT.md`
- Modify: `README.md`

- [ ] Declare device-local skills as the only active skill source in schemas and
  docs; legacy `source="server"` is read-only compatibility.
- [ ] Distinguish skill instruction activation from any executable tool installed
  by the skill.
- [ ] Include source hash, descriptor schema hash, and catalog version in skill
  snapshots; never sync full secret-bearing content in catalog summaries.
- [ ] Require an exact current device/session to activate a skill.
- [ ] Route executable skill tools through the normal capability/HITL authorization
  gate with risk metadata.
- [ ] Confirm device A enablement, content, and approvals do not affect device B.
- [ ] Instrument reads/writes to `skill_settings`. After one release with zero live
  use, remove its model/repository/table in a separate migration and fix README.

### Task 12 (T012): Harden multi-worker state, authorization, and observability

**Files:**

- Modify: `app/services/client_runtime_store.py`
- Modify: `app/services/client_device_service.py`
- Modify: `app/ai/mcp_registry.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `client_backend/services/runtime_bridge.py`
- Modify: `tests/test_multi_sidecar_hardening.py`
- Modify: `tests/test_client_invocation_isolation.py`
- Create: `scripts/verify_capability_isolation.py`

- [ ] Use Redis-backed session/catalog state in production; reject process-local
  runtime stores when more than one worker is configured.
- [ ] Use deterministic catalog/config hashes, not worker-local increment values,
  for cross-worker consistency. Session-local monotonic versions may remain for
  race detection.
- [ ] Define MCP registry reload as immutable-at-startup or coordinated generation
  swap. Do not mutate a manager while requests still reference it.
- [ ] Validate exact user/device/session/catalog/instance identity again immediately
  before client dispatch.
- [ ] Audit every decision with request ID, capability ID, origin, policy version,
  device/session IDs, decision, and sanitized reason.
- [ ] Add metrics for catalog load health, resolution degradation, denied/approved
  calls, stale bindings, cross-device rejection, MCP process restart, and tool
  latency.
- [ ] Add readiness checks for required backend MCP servers and separate optional
  server degradation from application readiness.
- [ ] Run two canonical workers and two sidecars for one account in the isolation
  script; prove no cross-device lookup or dispatch.

---

## Phase 4 — Migration, Rollout, and Final Verification

### Task 13 (T013): Ship in reversible stages

1. **Release A — characterization and image transport**
   - Add sidecar media proxy and v2 reference events.
   - Keep v1 inline preview reads.
   - Enable metrics before changing caps.
2. **Release B — strict MCP scoped endpoints**
   - Add canonical server-scoped route.
   - Keep query route delegated to the same method.
   - Add build/catalog version diagnostics.
3. **Release C — capability shadow mode**
   - Compute legacy and new effective capability sets.
   - Bind/execute legacy set, compare sanitized IDs, alert on mismatch.
4. **Release D — capability enforcement**
   - Bind and authorize with the new resolver.
   - Retain old Custom Agent fields for read compatibility.
5. **Release E — cleanup** (detailed as Phase 5, Tasks T014–T019)
   - Stop dual writes after measured client adoption.
   - Remove legacy Custom Agent refs/server skill source, the `client_only -> default`
     fail-open, hard-coded default/pin duplication, v1 image reads, and unused skill
     settings — each a separately reversible change gated on its precondition.

Feature flags:

```text
IMAGE_DELIVERY_V2_ENABLED
IMAGE_INLINE_PARTIALS_ENABLED
MCP_SCOPED_CATALOG_V2_ENABLED
CAPABILITY_RESOLVER_SHADOW_ENABLED
CAPABILITY_RESOLVER_ENFORCE_ENABLED
CUSTOM_AGENT_CAPABILITY_REFS_V2_WRITE_ENABLED
```

Rollback must never make protected images public or restore fail-open
`client_only` behavior. If the capability resolver is rolled back, pin the last
known explicit legacy policy rather than reverting to inferred all-tools defaults.

---

## Phase 5 — Deprecation Cleanup (Release E)

This phase removes the legacy, fallback, and deprecated code paths the earlier phases
supersede. It is the concrete task breakdown of "Release E — cleanup". Every removal
target below was confirmed live-or-dead against the working tree on 2026-07-24, and
each task is gated on the precondition that makes deletion safe. Do not begin any
Phase 5 removal until Release D enforcement has run in production with no capability
mismatch alerts.

**Ordering:** T014 depends only on T011 telemetry; T015 depends on T008 + enforcement;
T016 depends on T009 parity; T017 depends on T010 backfill + measured client adoption;
T018 depends on image-v2 adoption + message backfill; T019 is optional and independent.
Each task ships as its own reversible change — never batch these into one release.

**Intentional fallbacks to PRESERVE (explicitly not cleanup targets):**

- Inline attachment persistence on storage failure —
  `app/services/message_service.py:2255-2284`, `app/core/response_constants.py:401-406`.
  This is resilience, not a compatibility shim. Keep it.
- LangGraph v3 stream with the `["messages","updates"]` tuple fallback —
  `app/services/event_streaming/langchain_v3.py:90-95,555-598`. Still serves
  non-v3 runnables and test doubles. Keep until those consumers are dropped.
- `client_backend/services/mcp_config_migration.py` is a one-time CLI migration
  (`client_backend mcp migrate`), not runtime dual-read (the store accepts only
  `schema_version: 2`). Retire it on a schedule keyed to user-profile migration,
  not as part of this code cleanup.

### Task 14 (T014): Remove the dead `skill_settings` model, repository, and table

**Precondition:** T011 instrumentation has recorded zero live reads/writes of
`skill_settings` across one full release window. Verified dead today: whole-repo
search for `SkillSetting`/`skill_setting` finds only the model, the repository,
`app/models/__init__.py:23,61`, Alembic, and docs — no service, DI container, API,
or agent reader.

**Files:**

- Delete: `app/models/skill_setting.py`
- Delete: `app/repositories/skill_setting.py`
- Modify: `app/models/__init__.py` (remove the import/registration at `:23` and `:61`)
- Create: `app/alembic/versions/<new>_remove_unused_skill_settings.py`
- Modify: `README.md` (remove the `skill_settings` row at `:528`)

- [ ] Confirm and record the T011 zero-use counter for the release window in the
  commit message; abort if any nonzero read/write is observed.
- [ ] Delete the model and repository files and remove their registration in
  `app/models/__init__.py`.
- [ ] Write a reversible Alembic migration dropping `skill_settings` and
  `ix_skill_settings_id`; the downgrade re-creates the table matching
  `j1k2l3m4n5o6_add_client_devices_skill_settings_and_parse_artifacts.py:188-228`.
- [ ] On a scratch DB run `upgrade head` -> `downgrade -1` -> `upgrade head` to prove
  reversibility.
- [ ] Remove the `README.md` row and run the full suite; confirm no import errors.

### Task 15 (T015): Remove the `client_only -> default` fail-open downgrade

**Precondition:** `CAPABILITY_RESOLVER_ENFORCE_ENABLED` is live and T008 returns an
explicit unavailable/empty local snapshot for a no-device `client_only` request.
This closes the Constitution "REQUIRED CHANGE: Remove `client_only -> default`".

**Files:**

- Modify: `app/ai/tool_scope.py` (remove the downgrade at `:50-51`; fix docstring
  `:11-13,34-39`)
- Modify readers: `app/ai/agents/base_agent.py:525`, `app/ai/agents/custom_agent.py:193`,
  `app/ai/tool_search_tool.py:256`, `app/ai/tool_execution.py:753,871,1000`,
  `app/ai/tool_context.py:125`
- Modify: `tests/test_client_tool_isolation.py`, `tests/test_client_invocation_isolation.py`

- [ ] Add a failing test: a `client_only` request with no `device_id` resolves to an
  empty/unavailable local scope and NEVER returns server/backend tools.
- [ ] Delete the `if candidate is ToolScope.CLIENT_ONLY and not device_id: return
  ToolScope.DEFAULT` branch; keep `CLIENT_ONLY` as the resolved scope.
- [ ] Update each reader so `CLIENT_ONLY` + no device yields empty local tools with no
  server fallback; add an explicit warning rather than a silent broadening.
- [ ] Run the isolation tests; confirm no reader path silently re-enables server tools.

### Task 16 (T016): Consolidate `DEFAULT_SERVERS` and duplicated agent pins into policy

**Precondition:** T009 parity snapshots pass — the versioned policy manifest reproduces
the exact bound/searchable tool set for every built-in agent (the plan already gates
this: "Remove hard-coded `DEFAULT_SERVERS`/agent pin duplication only after parity
snapshots pass").

**Files:**

- Modify: `app/ai/mcp_integration.py` (remove `DEFAULT_SERVERS` at `:59` and its uses
  at `:401,419`)
- Modify: `app/ai/deferred_tool_binding.py` (retire `_WIDGET_PINNED_SPECS`,
  `_SEARCH_AGENT_PINNED_SPECS`, `_IMAGE_SEARCH_PINNED_SPEC` at `:69-96`; the dead
  `_get_pinned_specs` helper at `:100-110` can go once its tests move to policy)
- Modify: `app/ai/capability_policy.json` (absorb the values as `pinned`/`deferredEligible`)
- Modify: `tests/test_mcp_global_allowlist.py`, `tests/test_widget_runtime.py`,
  `tests/test_chat_agent_image_search_binding.py`, `tests/test_search_agent_time_context.py`

- [ ] Confirm T009 parity snapshot equality per built-in agent; attach the empty diff
  to the commit.
- [ ] Move the server set and pin specs into `capability_policy.json` and read them via
  the policy loader (`app/ai/capabilities/policy.py`).
- [ ] Replace the hard-coded constants with policy reads; delete the constants.
- [ ] Point the pin/allowlist tests at the policy, not the constants; run the MCP phase
  gate and confirm identical effective sets.

### Task 17 (T017): Retire legacy Custom Agent ref shapes after the dual-write window

**Precondition:** `CUSTOM_AGENT_CAPABILITY_REFS_V2_WRITE_ENABLED` has run long enough
that measured client adoption crosses the agreed threshold, and the T010 v2 backfill
has converted or quarantined every legacy row.

**Files:**

- Modify: `app/schemas/custom_agent.py` (remove `ServerDefaultToolRef` `:35-40` and drop
  it from the `CustomAgentToolRef` union `:72-75`; collapse `CustomAgentSkillRef.source`
  `:81` to `Literal["client"]`; remove the dead `server_tools`/`serverTools` placeholder
  `:250-251`)
- Modify: `app/ai/custom_agent_runtime.py:188` (drop `server_default` handling)
- Modify: `app/services/custom_agent_service.py:312-313,488,549`
- Modify: `app/services/custom_agent_capability_resolver.py:88-90`
- Modify: `demo.py:1317-1337,2081-2089` (migrate `serverDefaultTools` consumers to the
  canonical `capabilities` tree)
- Modify: `plans/CUSTOM_AGENTS_FE_CONTRACT.md`
- Modify: `tests/test_custom_agents_service.py`, `tests/test_custom_agents_api.py`,
  `tests/test_custom_agent_capability_resolver.py`

- [ ] Confirm zero remaining `custom_agents.tool_refs`/`skill_refs` rows with
  `type="server_default"` or skill `source="server"` (report the query count).
- [ ] Coordinate the FE contract change first; then remove `ServerDefaultToolRef`,
  the `source="server"` branch, and the `server_tools`/`serverTools` placeholder.
- [ ] Stop dual-writing compatibility fields; retain dual-read one more release only
  if any old client remains, then remove.
- [ ] Update `CUSTOM_AGENTS_FE_CONTRACT.md` and the Custom Agent tests; run the full
  Custom Agent suite.

### Task 18 (T018): Retire v1 image compatibility reads after message backfill

**Precondition:** `IMAGE_DELIVERY_V2_ENABLED` is fully adopted and a one-time backfill
has converted (or policy has aged out) pre-reference assistant messages.

**Files:**

- Modify: `app/ui/rich_response.py:184-234` (remove `use_legacy_image_gallery`)
- Modify: `app/core/response_constants.py:329-372` (remove the v1 legacy-gallery branch
  of `_filter_unreferenced_images_from_metadata_images`)
- Modify: `app/services/event_streaming/internal_sse.py`,
  `app/services/event_streaming/ai_sdk_v6.py` (drop schema-v1 inline `image_preview`
  read support once no client emits it; keep schema-v2 reference delivery only)
- Modify: `tests/test_demo_image_reference_rendering.py`,
  `tests/test_image_preview_stream.py`

- [ ] Confirm no message rows still rely on `metadata["images"]` without a reference
  (report count); run the backfill first if any remain.
- [ ] Remove the legacy gallery read paths; keep the inline-on-storage-failure
  resilience fallback (`message_service.py:2255-2284`) — it is NOT a compatibility shim.
- [ ] Remove schema-v1 inline preview reads; run the image phase gate plus a
  history-render test on migrated rows.

### Task 19 (T019): Optional — unify the dual `WorkflowExecutionRequest` schemas

**Out of the image/MCP/capability critical path.** Include only if the team wants to
close the documented `_to_ai_request` drift, where the service->AI translation silently
drops any field absent from the AI schema.

**Files:**

- Modify: `app/services/ai_service.py:145-146`
- Modify: `app/schemas/workflow.py:75`, `app/ai/schemas.py:128`
- Modify: `tests/test_workflow_request_schema_parity.py`,
  `tests/test_workflow_request_conversion.py`

- [ ] Enumerate fields on the service schema absent from the AI schema (the parity test
  already lists them).
- [ ] Either merge to one shared schema or make `_to_ai_request` total — no silent drop;
  raise on an unmapped field.
- [ ] Extend the parity test to fail when a field is added to one schema and not the other.

## Acceptance Matrix

| Case | Expected |
|---|---|
| Gemini final > 4M base64, Streamlit | Reference event arrives early; image remains after complete/reload |
| OpenAI partials + final, Streamlit | Partials replace; final reference remains |
| Gemini/OpenAI, AI SDK | `onData` sees preview; authenticated file renderer shows final |
| Image after HITL resume | Same behavior as initial stream |
| Unauthorized image ID | 404 without existence leak |
| `/mcp/tools?serverName=brave_image_search` | Only brave descriptors |
| `/mcp/servers/brave_image_search/tools` | Same tool set and explicit scope |
| Duplicate tool names on two MCP servers | Distinct stable IDs/call aliases |
| Same user on A and B | Request A cannot see/execute B bindings |
| Custom Agent created on A, used on B with same logical tool | Rebound to B exact live identity |
| Custom Agent used on B without tool | Degraded warning; no fallback |
| `client_only` without device | Empty/unavailable, never backend tools |
| Empty explicit allowlist | No optional tools |
| Backend server enabled but absent from default policy | Discoverable/admin-visible, not default-bound |
| Two canonical workers | Same catalog hash and authorization outcome |

## Full Verification Commands

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_image_generation_providers.py `
  tests/test_image_preview_stream.py `
  tests/test_image_stream_http_contract.py `
  tests/test_ai_sdk_v6_stream_contract.py `
  tests/client_backend/test_image_stream_proxy.py `
  tests/test_mcp_integration.py `
  tests/test_mcp_tools_http_scope.py `
  tests/client_backend/test_mcp_tool_execution_api.py `
  tests/test_capability_catalog.py `
  tests/test_capability_policy.py `
  tests/test_effective_capability_resolver.py `
  tests/test_custom_agent_capability_resolver.py `
  tests/test_custom_agent_client_tool_resync.py `
  tests/test_custom_agents_service.py `
  tests/test_client_tool_isolation.py `
  tests/test_client_invocation_isolation.py `
  tests/test_multi_sidecar_hardening.py `
  tests/test_skill_device_isolation.py `
  tests/test_hitl_settings_device_isolation.py -q

.\.venv\Scripts\python.exe -m ruff check app client_backend tests scripts

.\.venv\Scripts\python.exe scripts/verify_image_streaming_contract.py
.\.venv\Scripts\python.exe scripts/verify_capability_isolation.py
```

Then run the full repository suite and compare failures with the pre-change baseline:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Definition of Done

- Both reported image clients pass deterministic full HTTP/proxy tests and manual
  real-provider smoke tests.
- Final image delivery uses a protected stored reference and is independent of SSE
  frame size.
- AI SDK frontend requirements are implemented, not merely documented server-side.
- The reported MCP URL returns only the requested server's tools through a restarted,
  build-identifiable sidecar.
- Base and Custom Agents resolve capabilities through one fail-closed pipeline.
- Same-account, two-device tests cover catalogs, skills, approvals, reconnect, and
  execution.
- Defaults are explicit policy, empty means empty, and collisions use stable IDs.
- Migrations are additive, reversible, observed in shadow mode, and documented.
- Deprecation cleanup (Phase 5) removes the `client_only -> default` fail-open,
  legacy Custom Agent ref shapes, hard-coded default/pin duplication, v1 image reads,
  and the dead `skill_settings` model/table — each only after its gate and each
  independently revertible. Intentional resilience fallbacks are retained.
- No secrets, image payloads, local paths, or cross-device catalog contents appear
  in logs or API diagnostics.

## Decisions to Confirm Before Phase 3

These do not block the urgent image/MCP fixes:

1. Whether deployment administrators need runtime CRUD for backend MCP servers, or
   whether production server configuration should be immutable per release.
2. Whether Custom Agent missing local capabilities should always degrade and run
   (current/recommended) or optionally support a strict "all required" mode.
3. The compatibility-window length for legacy Custom Agent refs and `skill_settings`.

---

## Implementation Design Decisions Log

Decisions made while executing the plan (append-only; newest task last).

### T001 — characterization tests (commit `a4c26a2`)

- **Fake provider seam.** The deterministic fake is injected at the
  `ai_service`/real-`MessageService` HTTP boundary, not at a graph-level provider
  injection point, because that graph seam does not exist yet — it is a T002
  deliverable. The tests still drive the real FastAPI internal SSE route, the real
  AI SDK `/api/chat` route, and the real `client_backend.create_app()` sidecar proxy
  end-to-end. **Follow-up:** re-point the resume and provider-injection tests at the
  graph boundary once T002/T004/T005 create it.
- **Two targets kept GREEN, not forced RED.** `/mcp/tools?serverName=` scoping (fixed
  in `b55d83a`) and "exactly one `[DONE]` survives the proxy" are already correct on
  current source, so they were committed as GREEN regression locks rather than forced
  to fail. The genuinely-missing `GET /mcp/servers/{server_name}/tools` route is the
  RED MCP characterization (canonical + sidecar).
- **Verify script is force-added.** `scripts/` is gitignored, but sibling scripts are
  tracked and the plan lists `scripts/verify_image_streaming_contract.py` as a tracked
  deliverable referenced by the verification commands, so it was `git add -f`'d.
- **RED committed as hard failures, not xfail.** Phase 0's intent is a failing
  baseline; T002–T007 flip these green. The branch test suite is intentionally red
  for the image/MCP contract tests until then.

### T002 — persist final images early (commit `f107ce2`)

- **New abstraction `MediaDeliveryService`** (`app/ai/image_generation/emitter.py`):
  `publish_partial(*, image_index, mime, data_b64, seq=0) -> bool` and
  `persist_final(*, image_index, mime, data_b64) -> dict|None`, plus a `.failures`
  list. Bound at the graph boundary via `use_media_delivery_service` /
  `current_media_delivery_service` (a ContextVar), mirroring the existing
  `use_image_preview_emitter` sink pattern. The image agent only READS the ContextVar
  (`image_generator_agent.py`), with a `storage=None` fallback so non-streaming/resume/
  test paths keep working without any global-container access.
- **Idempotency key** = `(run_id, item_id="image-final-<idx>", sha256(data_b64))`,
  checked before `store()`; repeat finals return the cached descriptor (no second
  write). Within-run only (FR-IMG-009). **Cross-run/resume is NOT idempotent yet** —
  a fresh service instance on resume creates a second ownership row (file bytes are
  content-addressed/deduped, but the row is not). **Deferred to T005** (resume parity).
- **Descriptor** stored at `metadata["images"][i]["stored_ref"] =
  {image_id, url, mime, name, content_hash}` (reference only, no bytes). Terminal
  persistence reuses it via `response_constants._reuse_stored_ref`, so
  `message_service.py` needed no edit (it delegates through `externalize_metadata_images`).
- **Final bytes** use the storage byte cap, never the transient SSE char cap
  (FR-IMG-003). **Storage failure** → typed `MediaDeliveryError` recorded in
  `.failures` + `logger.warning` (no base64), current narrative behavior preserved;
  the failed final keeps its inline data so terminal externalization still runs.
- **Scope boundary held:** no wire event emitted, no AI SDK projection touched — the
  T001 wire-event characterization tests correctly stayed RED (verified: 12 known
  characterization REDs, nothing else regressed; image-gen/ai_sdk_v6/demo suites 38/38).
- **Minor (final-review triage):** (1) `response_constants.py` `store is None` branch
  now `continue`s past non-dict entries (drops them) instead of returning the list
  verbatim — test-only path, no production impact; (2) no single test exercises the
  full agent→metadata→externalize chain (halves tested independently; connective
  tissue verified by reading).

### T003 — version event + repair transports (commit `a2b1417`)

- **Versioned union in one seam.** `events.py` gained `resolve_image_preview_delivery`
  + builders (`build_image_preview_inline_data` / reference / `build_image_preview_skipped_data`)
  emitting `schema_version:2` with a `delivery` union; both transports
  (`internal_sse.py`, `ai_sdk_v6.py`) share it and tolerate v1 (top-level `data_b64`)
  for read-compat. No rollout feature flag (that is T013) — v2 is the default.
- **Final always by reference.** `emitter.emit_reference` / `persist_final` emit the
  final with `delivery.kind=reference` (`image_id`+`url` from T002's stored descriptor),
  bypassing the inline char budget (FR-IMG-003). **Exception (resilience):** if
  `_store_final` returns no descriptor (storage failed), the final stays inline and a
  typed `MediaDeliveryError` is recorded (`storage_failed` counter, no base64) — the
  authoritative bytes still arrive with terminal `complete`.
- **Two-layer budget.** First defense in `ImagePreviewPublisher.publish`; second at
  serialization (`apply_inline_preview_wire_budget`) in both transports. Over-budget →
  structured `preview_skipped` status (FR-IMG-007), never a silent drop. Budget =
  existing `settings.image_stream_preview_max_b64_chars` (configurable).
- **Projection repair.** `_normalize_image_item_to_file_part` (`ai_sdk_projection.py`)
  gained a protected-URL branch (after `data:`/`http(s)`/`blob:`, before the base64
  fallback) so `/chat-images/{id}` survives as a `file` url verbatim, direct and through
  the sidecar proxy. `graph_public_projection.py` left untouched — its `dict(data)`
  passthrough already carries v2 verbatim (locked by a passthrough test).
- **Single `[DONE]`.** `server_api.stream_sse` breaks on the first upstream `[DONE]`
  and never forwards it; the sidecar appends exactly one. Dedup lock stayed green.
- **Scope held:** 7 Task-1 characterization tests flipped GREEN; resume (T005), sidecar
  `/chat-images` (T004), and MCP scope (T006/T007) stayed RED by design. Broad run 628
  passed; sidecar proxy suites 12/12; image-gen/demo 38/38; ruff clean.
- **Minor (final-review triage):** (1) `consumer-disconnected` is a debug-log event,
  not an incrementing counter (stream disconnects once; `proxied` is co-logged);
  (2) deterministic `store()` stub duplicated across two test files (extract a fixture
  if a third copy appears); (3) storage-failed inline-final fallback uses `seq=0`
  (preserved pre-existing degraded-path behavior; revisit if seq semantics tighten).

### T004 — protected sidecar media route + renderers (commit `d6534fa`)

- **New sidecar router** `client_backend/api/chat_images.py` (`GET /chat-images/{id}`),
  registered in `client_backend/main.py` `compatibility_routers` (mounted at root AND
  `/api`), placed BEFORE the catch-all `proxy_router` so the dedicated streaming route
  wins. `client_backend/api/proxy.py` was NOT modified (brief-listed) — the catch-all
  would only buffer and add no hardening; skipping it dropped nothing.
- **Security model:** `Depends(require_local_session)` gates before the handler body, so
  an unauthenticated read returns 401 WITHOUT any upstream contact (proven by asserting
  the upstream stream seam was never opened). The sidecar forwards ONLY the desktop
  user's own access token (`server_api._get_auth_headers`) — no broader credential —
  so cross-user isolation is enforced by the canonical route's per-user scoping and the
  404 is forwarded generically. `common.proxy_media_request` raises on any upstream
  `>=400` WITHOUT reading the body, so no upstream detail/path leaks. Path typed as UUID
  (removes SSRF/path-steering surface).
- **Streaming:** `StreamingResponse` over `aiter_bytes()`; `.content` never touched;
  upstream stream closed in the generator `finally` (cancellation-safe). Header
  whitelist forwards cache validators + content-disposition/length; adds `nosniff` +
  CSP. 413 forwarded, plus a proactive declared-content-length pre-check.
- **Streamlit:** partial→final replaced by image index; `finalize()` keeps
  `status=="final"` across `complete` (no clear-and-hope); one shared authenticated
  fetch primitive for live + history.
- **FE contract (doc-only, React app not in repo):** specified the `onData`
  `data-image-preview` handler (authenticated fetch → Blob URL, replace-by-id/seq,
  revoke on replace/unmount) + the terminal protected-`file` renderer.
- **Minor (final-review triage):** (1) the 25 MiB guard only fires on a DECLARED
  content-length — an undeclared/chunked oversized upstream would stream unbounded
  (canonical always declares length + upstream trusted; add a running byte-counter abort
  in `common._body()` for a hard local bound); (2) the CSP on a raw image response is
  largely inert (`nosniff` is the effective control — don't over-credit CSP);
  (3) `events.resolve_image_preview_delivery` defaults a missing `status` to `"final"`,
  which the panel trusts (emitter always sets it, so not exploitable);
  (4) panel `_entry_data_uri` calls the fetch primitive directly rather than through the
  single resolver (same auth path; cosmetic).
- ⚠️ FR-IMG-005/006 also depend on the canonical `/chat-images` per-user scoping, which
  lives upstream (pre-existing `app/api/chat_images.py`) and is stubbed in these tests.

