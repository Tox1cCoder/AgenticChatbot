# Modern Skills Architecture — Verified Proposal

**Status:** Re-audited against the codebase on 2026-08-04. Progressive
disclosure and collection installation are shipped and production-hardened.
Identity, persisted collection provenance, specification fidelity, and catalog
scaling remain follow-up work.

**Goal:** support skills as they are distributed in modern Agent Skills and
Claude Code repositories while preserving this system's stronger device-local
execution, approval, secret-isolation, and hash-bound installation model.

## What modern skill distributions look like

The checked-in reference test can inspect `superpowers-main.zip` (obra/superpowers
v6.2.0) when that local fixture is present. It contains 14 skills under one
repository and companion documents/scripts beyond each `SKILL.md`.

| Property | Common distribution shape |
|---|---|
| Frontmatter | `name` and `description`, sometimes with harness-specific optional fields |
| Body | A short router that points to supporting material |
| Companion files | Reference Markdown, examples, scripts, and other bundle-owned assets |
| Distribution | One repository/archive containing a collection of skills |
| Multi-harness metadata | `.claude-plugin`, `.codex-plugin`, `.cursor-plugin`, and similar manifests |
| Cross-references | One skill may instruct the model to activate another skill |

The load-bearing pattern is **progressive disclosure**: activation supplies the
small router and a manifest; supporting text is read only when needed.

## Verified implementation state

The findings below were rechecked against the production paths and tests, not
inferred from the reference archive alone.

| # | Finding | Current evidence | Status / consequence |
|---|---|---|---|
| G1 | Progressive disclosure | `skill_runtime/resources.py` uses one policy for listing and reading; the runtime bridge binds reads to user, device, session, skill, and source hash | **Closed.** Links, escapes, generated metadata, excluded directories, oversized files, binary data, and invalid UTF-8 are neither advertised nor readable. Listings are cached by resolved bundle root plus source hash. |
| G2 | Collection install | `collection.py` preserves each discovered bundle/member path; `transactions.py` stages every member, journals promotion, rolls back in reverse, and recovers after restart | **Closed.** One upload can install multiple skills as one logical transaction. Catalog refresh and uninstall share the mutation lock, so partial promotion is not published. |
| G3 | Provenance, version, and updates | `install.json` persists per-skill source hash/name/source; replacement already requires a previewed current hash | **Partly open.** Guarded updates exist, but collection identity/version/origin are not persisted as a lifecycle object and there is no update-discovery endpoint. |
| G4 | Flat namespace | `LocalSkillsRegistry.skills` is keyed by bare skill name | **Open.** Same-name skills cannot coexist; configured-root conflicts are rejected rather than silently overwritten. |
| G5 | Agent Skills specification fidelity | `front_matter.py` validates portable names and parses `name`, `description`, `category`, and comma-separated `tags` | **Open.** It does not preserve unknown fields or validate the complete current specification (including description bounds and name/directory agreement). `allowed-tools` is experimental pre-approval metadata in the Agent Skills specification, not a restrictive security allowlist; treating it as an execution sandbox would be incorrect. |
| G6 | Catalog scale | `get_available_skill_summaries` supplies all enabled summaries to the model | **Open, measurement required.** There is no lookup/search path. A switch threshold must come from measured prompt cost and retrieval quality, not an arbitrary skill count. |

Existing guarantees that future phases must preserve:

- execution stays device-, user-, session-, tool-instance-, and source-hash-bound;
- secrets remain encrypted and local to one profile and are injected only for the
  selected skill;
- command selection is confined, approval still applies, and no shell/global
  executable fallback is introduced;
- upload members and model-requested resources remain path-confined; and
- installs are recoverable logical transactions. No filesystem can provide one
  atomic rename across N independent destination paths, so the journal and
  rollback protocol—not the word “atomic” by itself—is the production guarantee.

## Delivered hardening (2026-08-04)

### Resource reads

`read_skill_resource(skill, path)` is routed through the selected device. The
same policy loader drives both the activation manifest and the read itself, so a
listed path cannot later bypass a stricter read rule. Only regular UTF-8 text
within the selected bundle is exposed, with a 256 KiB per-read limit and a
bounded manifest.

### Collection discovery and identity preservation

Upload preview records the exact archive-relative bundle and `SKILL.md` member
for every discovered skill. Installation revalidates those confined members
instead of rediscovering a different skill later. Older persisted uploads remain
readable through deterministic fallback discovery.

### Transactional installation and recovery

All members are prepared before mutation. Promotion runs under one profile
mutation lock with a durable journal containing the complete expected name/hash
set. A failure restores prior bundles, runtimes, and secrets in reverse order.
On restart, recovery accepts success only when the journal is committed and
every expected installed hash matches; incomplete cleanup remains retryable.
The terminal operation receipt is persisted before its journal is finalized.

### Catalog isolation

Catalog refresh and uninstall use the same mutation scope as installation. Stage
and backup directories are excluded from registry discovery, preventing a
concurrent scan from publishing partial or internal transaction state.

## Recommended follow-up phases

### Phase 2 — identity and lifecycle

1. Introduce a stable `collection:skill` identity. Allow a bare name only when it
   resolves unambiguously, with an explicit migration for existing catalogs,
   stored selections, tool names, secrets, and audit records.
2. Persist collection name, version, manifest kind, and user-visible origin in a
   collection record. Keep absolute local paths out of synchronized metadata.
3. Add update preview as a first-class operation using the existing guarded
   replacement and transaction machinery. Keep upload-only acquisition unless a
   separately reviewed network-fetch policy is introduced.

### Phase 3 — specification fidelity

1. Parse frontmatter with a bounded safe YAML implementation or an equivalently
   strict schema parser. Validate required values, length limits, portable name,
   and skill-directory agreement.
2. Preserve bounded unknown metadata for forward compatibility without allowing
   it to affect security decisions implicitly.
3. If `allowed-tools` is supported, model it as pre-approval guidance only. The
   platform's own device binding, tool policy, and HITL rules remain authoritative.
4. Define namespaced skill-to-skill activation and detect recursion/cycles.

### Phase 4 — measured catalog scaling

Instrument enabled-skill count, prompt tokens, activation success, and lookup
quality. Add indexed lookup only after defining latency and recall targets; keep
small catalogs inline when that is cheaper and more reliable.

## Risk register

| Risk | Current control | Remaining action |
|---|---|---|
| Host compromise by an approved command | Constrained argv resolution, scoped environment/secrets, HITL, audit | This is not an OS sandbox. Use a subprocess/container sandbox only if the product threat model requires untrusted code. |
| Partial multi-skill state | Pre-staging, shared mutation lock, durable journal, reverse rollback, restart recovery | Keep transaction failure-injection and recovery tests mandatory. |
| Same-name collection collision | Conflict rejection and guarded replacement | Namespaced identity and migration. |
| Stale/tampered preview | Exact member receipt plus expected source hashes revalidated at install | Preserve the receipt schema compatibility tests. |
| Prompt growth | Summary-only injection | Measure and design lookup before setting a threshold. |
| Metadata semantic drift | Known-field parser and portable-name validation | Complete specification conformance; never infer security authority from unknown metadata. |

## Decisions retained

1. **One approval and one logical transaction per collection.** Per-skill
   enable/disable remains available after install.
2. **Upload-only acquisition.** The sidecar does not fetch arbitrary URLs; that
   avoids adding SSRF and source-authentication policy to a local code installer.
3. **The HTTP contract is frontend-independent.** Streamlit and AI SDK clients
   use the same `/api/skills/*` behavior described in
   `plans/SKILL_INSTALLATION_FE_CONTRACT.md`.

## Reference-library verification boundary

When `superpowers-main.zip` is present locally, the collection test verifies its
manifest and all 14 discovered skills. Transaction tests use controlled
multi-skill fixtures to prove commit, rollback, update restoration, and restart
recovery. The reference archive is not a permanent repository fixture, so this
document intentionally does not claim that an end-to-end 14-skill installation
runs in every checkout.
