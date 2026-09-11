# Conversation Inputs, Outputs, and Sources Design

**Date:** 2026-09-11
**Status:** Approved architecture, written specification pending user review
**Scope:** A production-ready conversation resource system shared by Streamlit and the AI SDK path

## Context

The application can already stream tool traces, rich items, images, RAG citations, and AI SDK file parts, but those paths do not form one durable resource contract. In particular:

- the client-runtime bridge converts non-string MCP results to JSON text, so typed MCP resource links and embedded resources lose their semantics before the normal renderer sees them;
- `tool_artifacts` describe executions and previews, not durable files created by those executions;
- `artifact_ref` and tool-result blobs preserve oversized model-facing text, not user-downloadable output artifacts;
- image attachments, generated images, RAG documents, web sources, MCP resources, and device-local files have separate identities and lifecycle rules;
- neither client has a canonical Inputs/Outputs/Sources read model that survives history reload; and
- a local filesystem path mentioned in tool prose is neither proof that a file exists nor a safe, portable link.

The result is the failure mode shown in the requested experience: a conversation may say an output or image exists while the client has no verified resource to render or open.

This project follows the production web research and image grounding project. It consumes that project's canonical `SourceRecord` and source streaming contract, but it is independently testable and has its own implementation plan.

## Goals

1. Give every conversation input, output, and source a stable public identity.
2. Make eligible files and links usable inside the conversation and in an Inputs/Outputs/Sources panel.
3. Preserve typed MCP results end to end instead of reconstructing them from display text.
4. Verify device-local file outputs on the device that executed the tool.
5. Snapshot small eligible outputs into authenticated managed storage while retaining large or mutable outputs as honest device references.
6. Produce identical resource identities and lifecycle state in Streamlit, AI SDK streams, and history reloads.
7. Keep tool traces separate from durable resources.
8. Add deterministic validation and evaluation that do not require a dedicated reviewer or tester.
9. Remove or replace redundant, legacy, and dead resource paths encountered in this feature's active flow.

## Non-goals

- Clone ChatGPT's private backend or visual styling exactly.
- Treat every path-like string in tool output as a file.
- Watch the whole filesystem or infer outputs from arbitrary directory changes.
- Upload large device files automatically.
- Expose a local absolute path, bearer token, storage key, or device identifier to the model or another user.
- Turn tool-result text blobs into general file storage.
- Duplicate the web-search source database state introduced by the preceding project.
- Require a human quality gate or an additional model call for resource classification.
- Provide a visual canvas companion; the user selected a text-only design workflow.

## Considered approaches

### 1. Keep resources in message metadata

This is the smallest change, but it makes mutable availability, asynchronous snapshot completion, idempotent retries, ownership checks, and conversation-level listing difficult. It also continues to conflate tool traces with files. Rejected.

### 2. Copy every reported output to the server

This makes all links portable, but it silently transfers local data, delays tool completion, performs poorly for large files, and cannot honestly represent files that continue changing. Rejected.

### 3. Canonical public registry with hybrid storage

This is the selected approach. A common `ConversationResourceView` is projected from durable input/output records and the existing canonical web-source records. Small eligible output files can become immutable managed snapshots; large or mutable files remain device-scoped references with explicit availability. The sidecar verifies local resources and never sends raw paths to the server. This provides working links where a portable link is possible and honest actions where it is not.

## Architecture

### Canonical public view

Both clients consume one strict, versioned `ConversationResourceView` union. Its common fields are:

- `resource_id`: stable opaque public ID;
- `categories`: a non-empty ordered set containing one or more of `input`, `output`, and `source`;
- `kind`: `file`, `image`, `url`, or `artifact`;
- `name`, optional title, media type, byte size, and safe description;
- `status`: `reported`, `verified`, `snapshot_pending`, `available`, `unavailable`, `rejected`, or `deleted`;
- `access`: `managed`, `device`, or `external`;
- `message_id`, `generation_id`, tool-call provenance, and display order where applicable;
- an authenticated `content_url` only for managed resources, an HTTPS `external_url` only for external resources, or a device action descriptor only for device resources; and
- safe timestamps and integrity metadata.

The public object never contains a raw local path, object-storage key, inline file bytes, authorization token, provider payload, or private device identifier. The model receives a smaller prompt-safe descriptor with the resource token, display name, type, status, and whether it is linkable.

`resource_id` is the durable public identity and is not the model token. `R1`, `R2`, and similar tokens are compact turn-scoped aliases mapped to authorized durable IDs by the turn registry. Persisted input/output resources use their opaque resource UUID. A source that has no durable input record uses the stable namespaced identity `source:{generation_id}:{source_id}` already recoverable from message metadata. When a cited canonical URL matches an existing external input resource in the same conversation, the source relationship reuses that input resource ID and adds a `source` role instead of creating a duplicate record.

The public registry is a projection rather than a second copy of every source:

- inputs and outputs with lifecycle state use new durable conversation-resource records;
- web sources continue to use the canonical `SourceRecord` persisted in assistant message metadata by the web research project;
- existing RAG document citations are adapted to the same public source view without rewriting their ingestion storage; and
- one resource may have multiple message relationships, so a user-provided URL can be an input and later a cited source without duplicating its identity.

### Persistence boundaries

`conversation_resources` stores ownership, conversation scope, origin kind, status, access mode, display metadata, provenance, integrity fields, and lifecycle timestamps. `message_resource_links` relates a resource to a message with category role and ordinal; those relationships produce the public `categories` set. Immutable managed file bytes are represented by a dedicated content-addressed resource-blob record and storage service.

Tool-result blobs remain model-support data. Their public field is treated as `tool_result_blob_id`; legacy `artifact_ref` values remain readable only through compatibility code and are never projected into Outputs.

Managed file records are immutable. If a local file changes and is saved again, it creates a new snapshot version linked by `supersedes_resource_id`; an existing snapshot is never overwritten in place. Conversation deletion cascades links and ownership records, then removes unreferenced managed bytes according to the storage retention policy.

### MCP result preservation

Client-runtime tools return the original structured result to the central tool-execution normalizer. The bridge no longer serializes dictionaries or MCP content blocks through `_format_tool_result`. Model-facing text and runtime artifacts are derived separately:

- `model_content` is bounded, sanitized text for the `ToolMessage`;
- `render` preserves safe structured content for the execution trace; and
- `resource_candidates` carries private, typed candidates into the resource registration service.

Standard MCP `resource_link`, `resourceLink`, and embedded `resource` blocks are recognized directly. Tool-specific extractors are registered by stable qualified tool ID for tools such as Desktop Commander that return output paths in a known structured result. Extractors consume the original result and approved arguments; they do not scrape arbitrary prose.

Unknown tools still render normally. They create no durable file merely because their text resembles a path.

Embedded MCP bytes do not bypass the storage boundary. A server-executed MCP tool may hand a bounded embedded resource directly to managed resource storage; a device-executed MCP tool converts it into the same authenticated streaming snapshot flow used for a local file. Oversized or malformed inline data remains a render preview or is rejected—it is never copied into message metadata or the runtime WebSocket.

### Application-owned outputs

The same registry also covers outputs created inside the application. Generated images and server-created files register directly as managed resources at their existing validated storage boundary. A durable canvas artifact registers as an `artifact` output with its authenticated application route; exporting it to a file creates a distinct managed file resource linked to the artifact.

Transient widgets, charts, tables, and ordinary tool renders remain rich execution presentation and do not automatically become Outputs. They enter the registry only when an owning service persists a durable artifact or export and supplies an explicit resource record. This prevents the Outputs panel from becoming a duplicate of the execution trace.

## Device output discovery and verification

### Candidate evidence

A candidate is eligible for verification only when at least one of these is true:

1. the MCP result contains a typed resource block;
2. a registered qualified-tool extractor identifies the output from a versioned structured schema; or
3. an output path was deterministically present in the approved tool arguments and the registered extractor confirms that the result represents that same output.

The sidecar performs discovery because it owns the filesystem namespace. The server receives only an opaque `client_resource_id` and sanitized metadata.

### Verification rules

For each candidate, the sidecar:

1. resolves the path with platform-aware canonicalization;
2. rejects missing entries, directories, sockets, devices, named pipes, symlinks, junctions, and other reparse-point traversal;
3. opens the regular file without following links and verifies that the opened identity still matches the checked path;
4. records a post-call stat fingerprint;
5. compares a pre-call fingerprint when the destination was known before invocation;
6. classifies the change as `created`, `modified`, `unchanged`, or `unknown` without overstating evidence;
7. derives a safe base filename and media type without trusting a server-supplied extension alone; and
8. creates an opaque device-local catalog entry scoped to the authenticated user, installation, and runtime session.

Only a pre-call absence followed by a verified post-call regular file is labeled `created`. A candidate discovered only after execution may be verified and tracked, but its change kind is `unknown` unless the tool's registered contract supplies stronger evidence.

There is no global filesystem watcher and no recursive before/after scan. This avoids missed-event races, unrelated-file capture, high latency, and accidental data collection.

### Snapshot eligibility and privacy

The default managed snapshot limit is 25 MiB, configurable independently on server and sidecar. This is a byte/work bound, not a total elapsed-time deadline. Files above the limit remain device references and are not hashed in full merely to display them.

Automatic snapshotting requires all of the following:

- the output came from an approved tool call;
- a trusted typed block or registered extractor identified it;
- local verification succeeded;
- the file is at or below the configured snapshot limit;
- the file is stable across the open/stat boundary;
- its type is allowed by the snapshot policy; and
- the path was explicitly authorized by the tool request or lies in a sidecar-managed output root.

Files outside that authorization remain device-only until the user explicitly chooses **Save to conversation**. Denied types, secrets, credentials, key material, browser profiles, and application-specific sensitive paths never auto-upload. A rejection reason is represented by a safe enum, never by echoing the path.

Hashing and snapshot upload run after tool completion and do not block the answer. The initial resource can stream as `verified` or `snapshot_pending`; later events move it to `available` with `managed` or `device` access, `unavailable`, or `rejected`. Uploads use authenticated streaming, server-declared byte limits, content hashing during transfer, checksum comparison, temporary staging, atomic promotion, retry-safe upload IDs, and abandoned-upload cleanup. No base64 file payload travels in the runtime WebSocket.

## Resource lifecycle and idempotency

The central registration service creates stable IDs from an idempotency scope containing conversation, generation, tool call, candidate ordinal, origin device installation, and opaque client resource ID. A retry returns the existing resource instead of duplicating it.

Allowed state transitions are explicit:

- `reported -> verified | rejected | unavailable`;
- `verified -> snapshot_pending | available | unavailable | rejected`;
- `snapshot_pending -> available | unavailable | rejected`;
- `available -> unavailable` only for device access; managed snapshots remain available until deletion; and
- any non-deleted record may transition to `deleted` through authorized retention or conversation deletion.

Out-of-order or stale updates are ignored using a monotonically increasing resource version. A runtime reconnection cannot update an entry owned by the prior session unless it proves the stable installation identity and current catalog entry. Mutation receipts continue to protect tool execution; resource registration has its own idempotency because it can finish asynchronously after the tool receipt is complete.

## Conversation links and actions

The answer model refers to known resources with exact `[[resource:R#]]` tokens supplied in its prompt inventory. The `R#` alias is turn-scoped and never accepted as a durable API identifier. A server-owned resolver validates each token against the turn registry:

- managed and external resources become safe Markdown links with server-selected labels;
- device-only resources become existing rich-response `resource_link` markers rendered as a resource chip/action; and
- unknown, rejected, or cross-conversation IDs are removed and recorded as grounding violations.

The model never writes a storage URL or local path. The resolver—not the model—chooses the final target.

Managed content is served from an authenticated `/conversation-resources/{resource_id}/content` endpoint with ownership checks, safe `Content-Disposition`, `X-Content-Type-Options: nosniff`, and range support where appropriate. Potentially active content such as HTML and SVG downloads as an attachment rather than executing in the application origin.

A device resource exposes no ordinary download URL. Both clients render **Open on originating device**, **Refresh**, and, when eligible, **Save to conversation** actions. The action travels through the authenticated client-runtime channel, resolves the opaque local catalog entry on that device, re-verifies file identity, and only then asks the operating system to open it. If the device is disconnected or the file changed/disappeared, the resource becomes visibly unavailable; the UI does not emit a blank or dead link.

## Streaming and client projections

The internal event contract adds a canonical `resources_upsert` event with schema version, operation, and one or more `ConversationResourceView` objects. It is emitted before answer text that references a newly registered resource. Later asynchronous status changes use the same stable resource ID and a higher version.

### AI SDK path

- Public web sources project to native `source-url` parts from the web research project.
- Managed input/output files project to native AI SDK `file` parts when that representation is sufficient.
- A `data-conversation-resources` part carries the complete registry fields needed by the Inputs/Outputs/Sources panel and device actions.
- Terminal `data-assistant-message` metadata and history reload project the same IDs and states.
- Custom parts are additive; clients that understand only text, `file`, and `source-url` still receive functional managed links and sources.

### Streamlit path

Streamlit consumes the same `resources_upsert` event and the same history projection. Resource UI code moves out of the large `demo.py` rendering cluster into a focused module. The message row resolves resource tokens and rich markers, while a conversation panel groups records into Inputs, Outputs, and Sources without parsing tool trace prose.

The panel is a view, not a second store. Repeated events upsert by `resource_id` and version, and reload produces the same ordering as the live stream.

## Input resources

New uploads use a staged authenticated upload endpoint and resource ID instead of placing base64 bytes in message JSON. Finalizing a message links the staged resource to the user message in the same application transaction. Initial support includes the currently accepted image types and ordinary files allowed by upload policy.

Existing base64 image attachment requests remain a temporary compatibility input. The server immediately externalizes them using the current chat-image validation and creates a canonical input resource. New clients do not write that legacy shape. Historical `chat_images` remain readable; migration or lazy adaptation must not duplicate bytes.

User-provided HTTPS URLs become external input resources only when the client sends a typed URL part or the server's existing attachment/link parser validates an explicit URL. Merely mentioning text that resembles a URL does not automatically create a trusted source. If web research later opens and cites that URL, the source projection relates the same canonical URL to both roles.

## Sources

Sources include:

- canonical web `SourceRecord` entries;
- RAG documents and cited chunks through an adapter over existing document identities; and
- explicit MCP resource URLs whose schemes and ownership rules permit external access.

The Sources panel shows only sources actually used or explicitly attached, not every search candidate. Web and external URLs are canonicalized and deduplicated. Local device files never masquerade as URL sources. Source clicks use validated HTTPS URLs or authenticated application routes.

## Failure handling

- Resource extraction failure does not discard an otherwise valid tool result or answer.
- A verification failure creates no trusted link and exposes only a safe status/reason.
- Snapshot failure falls back to a verified device resource when safe; it does not claim a managed download exists.
- Hash or byte-count mismatch rejects the staged snapshot and deletes temporary bytes.
- Client disconnect changes device actions to unavailable while leaving managed snapshots and external sources usable.
- Unsupported clients receive ordinary text plus native file/source parts where possible; marker-bearing content is never sent without the declared rich-response capability.
- Cancellation closes open upload streams, preserves already committed resource records, and lets retry-safe updates resume without duplicates.
- History projection tolerates legacy messages and malformed old metadata by omitting invalid resource entries rather than failing the conversation.

## Security and authorization

- Every resource read, list, action, upload, and delete operation validates authenticated ownership and conversation membership.
- Device references are bound to user, stable installation, and current runtime session; public responses contain only opaque IDs.
- Raw paths are confined to a profile-scoped sidecar catalog with atomic writes and restrictive permissions.
- Snapshot endpoints stream bytes and enforce declared and observed sizes; decompression is not performed for ordinary file download storage.
- Filenames are normalized for display and `Content-Disposition`; they are never used directly as storage paths.
- Active content is attachment-only, and MIME sniffing cannot upgrade it to executable same-origin content.
- Logs and metrics contain bounded resource category combinations, status, byte buckets, safe error codes, and tool identity—not filenames, URLs with query strings, local paths, file content, or hashes usable as cross-user identifiers.

## Latency and resource use

The synchronous tool path performs only structured extraction, path verification, and bounded metadata work. Full hashing and upload occur asynchronously, so a slow snapshot cannot delay the model's next step or terminate an in-progress tool call. Device opening and explicit saving are user-initiated operations.

The system records extraction duration, verification duration, time to managed availability, uploaded bytes, status transitions, retries, and failure codes. These observations establish later service objectives from real deployments; guessed latency thresholds are not release gates.

Structural limits bound candidate count per tool call, snapshot bytes, outstanding uploads, retry attempts, storage quota, filename length, and update versions. Connection and idle-read controls remain liveness protections. No guessed end-to-end wall-clock deadline interrupts a progressing resource upload; stalled transfers can resume from an acknowledged byte position or fall back to device-only state.

## Validation and evaluation

Release validation is deterministic and does not depend on a human reviewer or an extra runtime model call.

Required automated coverage includes:

- strict schema and state-transition contract tests;
- typed MCP resource preservation through sidecar, server tool execution, stream, persistence, and history;
- Desktop Commander extractor fixtures covering create, modify, missing, malformed, and deceptive prose results;
- Windows junction/reparse and POSIX symlink rejection;
- pre/post identity race, replacement-after-check, checksum mismatch, interrupted upload, retry, and stale-version tests;
- owner, conversation, device, installation, and session isolation;
- managed download headers and active-content behavior;
- device disconnected, changed, deleted, and explicitly saved resource flows;
- AI SDK native part plus custom registry parity;
- Streamlit live/reload ordering and working action rendering;
- input upload compatibility and no-new-base64-write contracts;
- source deduplication across user URL, web source, and RAG adapters;
- resource-token grounding with invented and cross-conversation IDs;
- conversation deletion and unreferenced blob cleanup; and
- a repository inventory test forbidding raw-path exposure, prose path scraping, and new `artifact_ref` writes.

Fixtures use generic documents, images, archives, URLs, and device paths; no topic-specific question is a release condition. An optional live Desktop Commander canary may validate a configured development device, but absence of that external tool does not block the deterministic suite.

## Observability

Metrics use bounded labels:

- candidates by origin and kind;
- terminal resource status and safe failure code;
- managed versus device versus external access;
- bytes by configured bucket;
- upload retries and checksum failures;
- time from report to verified and available; and
- link/action success by client path.

Health output reports whether managed storage and the resource event projector are configured. It never enumerates filenames, paths, content hashes, source queries, or user URLs.

## Feature-scoped cleanup

Cleanup is mandatory, not optional follow-up. Before deletion, each path gets a caller inventory and a replacement test through the canonical boundary.

The implementation removes or replaces:

- `_format_tool_result` JSON stringification for client-runtime structured results;
- active writes that use `artifact_ref` to imply a user artifact;
- duplicate resource extraction in tool rendering and client-specific UI code;
- Streamlit resource rendering embedded in `demo.py` once the focused module owns it;
- new-message base64 attachment writes after the canonical upload path is available, retaining read-only compatibility only where required;
- any prose/path-regex file discovery added by earlier experiments;
- parallel AI SDK and Streamlit resource identity logic; and
- dead helpers and legacy write paths discovered by repository-wide caller searches.

Current `normalize_tool_result_for_rendering`, rich `resource_link` items, chat-image validation/storage primitives, event projectors, tool execution receipts, device isolation, and content-addressed storage patterns should be reused or generalized where their contracts fit. Tool-result text blobs remain separate rather than being stretched into binary artifact storage.

The cleanup deliverable includes a checked-in inventory naming every retained compatibility path, its caller, removal condition, and test. There is one canonical active write path for each new resource kind.

## Delivery order

1. Land canonical schemas, state transitions, persistence, and authorization.
2. Preserve structured MCP results and add the extractor/verification boundary.
3. Add asynchronous managed snapshot upload and device-resource actions.
4. Add resource-token grounding and canonical stream events.
5. Project the registry into AI SDK live/history paths.
6. Project the same registry into Streamlit live/history paths and panel.
7. Move inputs to canonical staged resources with legacy read compatibility.
8. Integrate web and RAG sources into the public projection without duplicating storage.
9. Complete deterministic evaluation, observability, cleanup inventory, and dead-path removal.

Each stage is independently testable. Managed snapshots and device actions can be disabled separately during rollout; resource identities and source projection remain stable.

## Acceptance criteria

- A typed MCP output resource survives execution without being flattened to prose.
- A registered Desktop Commander fixture produces a verified output; arbitrary path-like prose does not.
- A small eligible output becomes a managed authenticated download without delaying the answer.
- A large or mutable output remains an explicit device resource and opens only on its authorized originating device.
- A disconnected, changed, missing, rejected, or failed output is visibly represented and never produces an empty link.
- Resource references in answer text are server-validated; invented IDs cannot become links.
- Streamlit and AI SDK expose the same resource IDs, categories, status, ordering, sources, and history state.
- AI SDK clients receive native `file` and `source-url` parts where applicable.
- Inputs, Outputs, and Sources are generated from canonical records rather than tool prose.
- Local paths, private storage locations, credentials, and inline arbitrary file bytes do not enter public events, message metadata, logs, or model context.
- Deterministic tests cover general resource behavior without a human reviewer or topic-specific examples.
- Feature-scoped legacy writes, redundant renderers, and dead paths are removed or explicitly documented as read-only compatibility.
