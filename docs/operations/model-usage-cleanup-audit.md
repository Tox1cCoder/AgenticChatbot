# Model-usage cleanup audit

**Audit date:** 2026-07-22  
**Scope:** repository-wide discovery, with removals limited to paths conclusively superseded by the model-usage remediation.

## Evidence

The audit used repository-wide reference searches for `deprecated`, `legacy`,
`fallback`, `compatibility`, and `obsolete`, plus exact searches for every
superseded model-usage symbol. Ruff checked all application and test modules for
unused imports, unused locals, unreachable import surfaces, and related static
issues. The affected suites were also run with warnings promoted to errors.

`vulture` is not installed in either supported local Python environment. It was
not added as a runtime or development dependency solely for this audit; Ruff,
exact call-site searches, executable contracts, and the full test suite are the
reproducible evidence used here.

## Disposition

| Candidate | Disposition | Evidence |
| --- | --- | --- |
| Raw-event latest gauge selector | Removed | `ModelUsageRepository.get_latest_conversation_event` and its event-to-gauge reconstruction had no public API contract and were superseded by the owner-bounded latest assistant-message metadata query. Exact application/test search has no remaining reference. |
| Image generation tuple result | Removed | All internal callers now consume `ImageGenerationOutcome`; no wire response used the tuple and exact application/test search has no tuple-return signature. |
| Obsolete gauge reconstruction imports and test doubles | Removed | Ruff and reference searches identified them as exclusive to the deleted raw-event selector. |
| Provider retry/fallback runtime | Retained | This is an active resilience path used by chat, RAG, planning, and vision agents. It has dedicated shared-operation usage tests and user-visible runtime metadata. |
| Image acknowledgement text fallback | Retained | The image agent calls it when a successful generated image has no provider narrative, and tests cover both success and failure responses. Removing it would make a successful image turn return prompt-engineering text or no usable acknowledgement. |
| `gemini-3-pro-image-preview` registry alias | Retained | This deprecated identifier remains necessary to render persisted message history and is covered by registry and provider tests. New configuration uses `gemini-3-pro-image`. |
| Sidecar `/api` aliases and single-upload wrapper | Retained | They are documented external compatibility contracts with active client-backend route tests. They are outside the internal model-usage replacement boundary. |
| Stream protocol compatibility adapters | Retained | The Streamlit JSON SSE and AI SDK UI Message Stream are distinct supported public protocols with dedicated projection tests. |
| Legacy provider metadata without `limit_type` | Retained | Stored/catalog metadata created before this release must keep its shared-context interpretation. Explicit `separate_io` metadata now follows the corrected independent-limit path. |

## Result

The superseded model-usage selectors, tuple response, imports, fakes, and tests
were deleted rather than wrapped. Remaining fallback and legacy references have
active callers, persisted-data requirements, public compatibility contracts, or
production failure semantics; deleting them in this release would be a breaking
change rather than dead-code cleanup.
