# Model Usage Production Remediation Design

**Date:** 2026-07-22  
**Status:** Approved for implementation

## Purpose

Bring the model-usage implementation into conformance with `token_usage.md`, close the correctness and lifecycle gaps found during code review, provide a frontend-ready AI SDK usage contract, remove demonstrably obsolete code, and produce a verified sidecar bundle.

## Goals

- Keep persisted usage immutable while ensuring dashboard aggregates and conversation gauges are correct.
- Make the conversation gauge describe the user-visible assistant response rather than later helper operations.
- Correct image generation and embedding accounting without inventing unavailable token values.
- Preserve shared-context and separate input/output model limits end to end.
- Make conversation deletion semantics deterministic and compatible with reconciliation.
- Update the existing token-usage frontend contract and connect it to the main AI SDK contract.
- Remove code that is conclusively dead, redundant, legacy, deprecated, or an obsolete fallback.
- Verify the result in the supported Python environment and build the client backend sidecar artifact.

## Non-goals

- Redesigning the complete usage ledger or adding a new event-sourcing system.
- Adding frontend components or visual designs.
- Guessing token counts for image-only inputs when the provider does not return usage.
- Removing compatibility behavior whose consumers or support requirements cannot be established.
- Rewriting unrelated subsystems merely for stylistic consistency.

## Considered Approaches

### 1. Filter helper events at read time

Exclude known acknowledgement, suggestion, and helper operations when choosing the latest usage event. This is small, but every new helper operation can silently regress the gauge and the event ledger remains an indirect representation of the displayed assistant message.

### 2. Persisted assistant-message metadata as display authority

Treat the latest non-deleted assistant message containing valid `context_window` metadata as the conversation gauge. Continue recording all usage events for analytics, but merge terminal image-provider usage into the user-facing message before it is persisted. This provides a stable display boundary without expanding the usage schema.

This is the selected approach.

### 3. Add display-selection and deletion state to the ledger

Add explicit display-event markers, conversation tombstones, and reconciliation rules. This provides more historical control but introduces schema, trigger, and operational complexity that is not justified by the current requirements.

## Design

### Conversation deletion and reconciliation

The raw usage event foreign key to conversations will change from `ON DELETE SET NULL` to `ON DELETE CASCADE`, matching the existing rollup lifecycle. Deleting a conversation will therefore delete both its raw usage events and its conversation rollups. A later reconciliation cannot recreate account-level aggregates from detached raw rows.

The ORM declaration and Alembic migration must agree. The migration downgrade restores the prior `SET NULL` behavior. Existing historical rows whose conversation reference is already null cannot be safely reattributed and will not be mutated by this migration.

Operational documentation will explicitly state the deletion and reconciliation semantics.

### Authoritative conversation context gauge

The conversation endpoint will obtain its gauge from the latest non-deleted assistant message, owned by the requesting user and conversation, whose JSON metadata contains a valid `context_window` object. Ordering will use the conversation's stable message sequence, with its identifier as a deterministic tie-breaker.

The bounded lookup will live in the model-usage repository so it can reuse the existing session boundary and avoid creating a container dependency cycle. The service will validate the stored object with the existing context-window schema before returning it. Invalid or absent metadata yields no gauge rather than a fabricated value.

Raw usage events remain the authority for analytics totals and rollups. They are no longer used to infer which model call represents the visible assistant response. Helper calls remain recorded but cannot replace the UI gauge.

### Dedicated image generation

The image streaming path will return a structured outcome containing generated images, narrative content, and the terminal normalized provider usage. When the dedicated image attempt succeeds, the image agent will replace only the response metadata's `context_window` value using the image provider/model limits and terminal usage.

This merge occurs before the assistant response is persisted. Prompt, acknowledgement, and other helper usage events remain unchanged for accounting purposes.

### Inline generated-image accounting

The usage recorder will support an optional response-aware usage transformation immediately before immutable persistence. The base agent will provide an identity transformation by default. The image agent will override it to count inline generated-image output blocks in the raw provider response and return a new normalized usage value with the corrected `generated_images` count.

The transform must not mutate an existing usage value. It must preserve provider-reported fields and only supplement the generated-image count derived from the response being recorded.

### Model limit metadata

Provider context-window metadata will include `limit_type`. Normalization will distinguish shared-context keys from separate input-limit keys:

- Explicit `shared_context` retains a single shared denominator.
- Explicit `separate_io` retains independent input and output limits without synthesizing a shared context window.
- Legacy metadata that omits `limit_type` keeps the existing shared-context interpretation for compatibility.

This prevents separate-I/O models from being silently converted into shared-context models during a metadata round trip.

### Image-only embedding usage

Text embedding requests may continue to use the local estimator when provider usage is unavailable. If an embedding request contains no estimable text, including image-only provider parts, its normalized usage source will be `unavailable` and token fields will remain null. Zero is reserved for a known zero, not an unknown value.

### Cleanup policy

The implementation will include a repository-wide audit for dead, redundant, fallback, legacy, and deprecated code. Automated searches, static checks, call-site inspection, tests, and version-control history available locally will be used as evidence.

Code will be removed when it is demonstrably unreachable, unused, duplicated without a distinct contract, or superseded by the corrected implementation. Compatibility behavior will be retained when external use cannot be ruled out. Ambiguous candidates will be documented rather than removed speculatively. Cleanup changes must remain reviewable and will receive regression coverage where behavior is affected.

## Frontend Contract

The existing `plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md` will be updated rather than duplicated. It will include:

- Dashboard, conversation, and `GET /usage/capabilities` request and response contracts.
- Exact TypeScript types and nullable/unknown-state semantics.
- Capability gating and failure behavior.
- The rule that the latest qualifying assistant-message metadata is the conversation gauge authority.
- Shared-context, separate input/output, image, and unavailable examples.
- Fetching, refresh, and stale-response guidance.
- Confirmation that no new SSE event is required.
- A frontend implementation and acceptance checklist.

`plans/AI_SDK_FE_CONTRACT.md` will link to this focused contract so the frontend team has a single discoverable source for token-usage UI behavior.

## Compatibility and API Behavior

Existing endpoint paths and ledger event shapes remain stable. The changes correct the source and meaning of context-gauge data, preserve explicit model-limit semantics, and represent unknown image embedding usage accurately. Consumers must treat nullable usage values as unavailable rather than zero.

Conversation deletion becomes intentionally stronger: associated raw usage events are deleted instead of anonymized. This matches the documented user-facing deletion contract and prevents aggregate resurrection.

## Testing Strategy

Implementation will follow red-green-refactor in focused increments:

1. Add a failing deletion/reconciliation test and migration assertions, then implement cascade behavior.
2. Add failing gauge-selection tests covering later helper events and deleted/foreign messages, then switch to assistant-message metadata.
3. Add failing dedicated and inline image usage tests, then implement pre-persistence usage merging/transformation.
4. Add failing model-metadata round-trip tests for `separate_io`, then preserve `limit_type` and normalization semantics.
5. Add failing image-only embedding tests, then return unavailable usage.
6. Add contract tests for the updated Markdown examples and capability endpoint.
7. Run targeted cleanup checks and regression tests for every removed code path.

Final verification will include focused tests, the complete supported test suite, Ruff check and format validation, migration contract/full-chain checks, confirmation of one Alembic head, and live PostgreSQL tests when `TEST_DATABASE_URL` is available. Environment-dependent skips will be reported explicitly.

## Build and Deliverables

After verification, run:

```powershell
pwsh -File scripts/build-client-backend-bundle.ps1
```

Verify that both `dist/client-backend-bundle/` and `dist/client-backend-bundle.zip` exist and that any bundle-specific verification completes successfully.

Deliverables are the corrected implementation, migrations, regression tests, updated operational documentation, updated frontend contracts, cleanup results, and verified sidecar bundle.

## Risks and Mitigations

- **Historical null conversation events remain:** do not guess their origin; document that the new cascade behavior applies once references exist under the new constraint.
- **Message metadata may be malformed:** validate before response serialization and return no gauge on invalid data.
- **Cleanup could remove an implicit extension point:** require concrete call-site and compatibility evidence before deletion.
- **Provider metadata varies:** preserve legacy behavior only when `limit_type` is absent, and cover both explicit modes with tests.
- **Live database coverage may be unavailable locally:** keep offline migration assertions and report skipped live coverage without representing it as passed.

## Acceptance Criteria

- All five identified correctness defects have regression tests and are fixed.
- Conversation deletion cannot be undone by usage reconciliation.
- Helper usage events cannot replace the visible response gauge.
- Dedicated and inline image usage is recorded accurately when knowable.
- Image-only embedding usage is unknown rather than zero when not reported.
- Separate-I/O limits survive provider metadata round trips.
- The focused frontend contract is complete, linked, and consistent with backend behavior.
- Conclusively obsolete code discovered in the audit is removed without breaking supported behavior.
- Required tests and static checks pass, with environmental skips called out.
- The sidecar bundle and zip are successfully produced and verified.
