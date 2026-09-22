# Web research rollout

Canonical web research is controlled by `WEB_RESEARCH_ENABLED`. Keep the flag
enabled only after the focused web matrix, non-live suite, and deterministic
evaluation pass.

The answer model receives only MIME-validated, dimension-checked image bytes,
downscaled to a 512px preview for selection; the full-size rendition is what
gets published. Candidate metadata is private. A web image is published only
when the model selects its `I#` token; no token means no image. Web-answer text
is buffered until a citation resolves to a source admitted in that turn.

## Image discovery

Brave image search forwards the model's locale as `country` and `search_lang`,
and treats `include_domains` as a restriction: a disallowed-domain result is
dropped, never returned as a fallback. Every entry is normalized to a bare host
first, because the model routinely writes a full URL where a domain was asked
for.

When a domain-restricted search returns fewer than the requested count of
allowed results, the adapter makes **one** supplemental `site:<first allowed
domain>` probe and merges the results, deduplicated by canonical source URL
plus image URL. One probe, not one per domain: Brave's image endpoint has a
2.5s timeout (`brave_image_search_timeout_seconds`) and the free tier allows
roughly one request per second, while the whole tool call shares a 30s soft
timeout with the parallel Tavily text search. Four sequential probes would
spend up to 10s and risk HTTP 429. Worst case is now two calls and ~5s. A
failed probe is logged at warning level with provider, domain and code, and
leaves the already-returned allowed-domain results intact.

## Capacity: catalog, window, registry

Three bounds, deliberately distinct:

| Bound | Setting | Default | What it limits |
|---|---|---|---|
| Per-cohort candidate cap | `web_research_max_candidate_pool` | 8 | Candidates taken from one image search |
| Session catalog | `web_research_max_candidate_catalog` | 24 | Candidates retained across all image searches |
| Active vision window | derived from `visual_intent` | 4 (6 for `gallery`) | Images whose pixels reach the answer model |

Text sources and image pages hold **separate** quotas: a turn admits up to 5
text sources (8 agentic) plus up to `web_research_max_candidate_catalog` image
pages. Image pages are admitted first. They shared one budget until 2026-09-16,
and because text results were admitted first and a search returns its full
quota, every image was discarded before the model saw it.

The catalog is ranked on every image search and the active window recomputed
over the whole of it, so a later, better search can displace an image from a
window that was already full — and a worse later cohort cannot displace a
stronger earlier one. Retained candidates keep the `I#` the model was already
shown; a replacement gets a fresh ID, and a retired ID is never rebound to
different bytes. Evicted pending references are released only after their
replacements are prepared, so a failed fetch costs the improvement rather than
the image the model already had.

`omitted_image_count` counts candidates the session will **never** offer the
model: beyond the per-cohort cap, past catalog capacity, source page
unadmittable, fetch or validation failed, or a duplicate digest. A candidate
held in the catalog but outside the current window is *not* omitted.

## Three views of the source registry

| View | Audience | Contents |
|---|---|---|
| `source_registry.records` | Grounding and `model_context` | Every admitted `S#`, so any token the model was offered resolves |
| `bundle.operation_source_ids` | The tool result | Only what *that* operation newly contributed |
| `session.answer_sources` | The reader | Text pages, opened pages, and image pages backing an active image |

The published list is smaller than the registry on purpose. The registry holds
the whole candidate catalog's pages; most are gallery hosts whose candidate
never entered the vision window, and publishing them buries the few sources the
answer actually rests on. They stay resolvable for citations regardless.

Tool-result snippets are bounded by source status: 1,200 characters for a
search result, 3,000 for a page the model deliberately opened. The worst case
for one operation (an agentic open of 4 pages) is 12,000 characters, under
`tool_result_offload_threshold_chars` (16,000), so a deep read stays in the
transcript instead of being offloaded to a blob.

A search snippet now lives only in the tool transcript; the synthetic evidence
message carries the `S#` mapping without repeating it. A deliberately opened
page is protected by a 1,500-character excerpt in that message, but a search
snippet is not, so text evidence depends on tool messages surviving later
trimming or summarisation.

### When a search adds nothing

The text quota is **per turn**, not per search. Once it is full, a further
`web_search` cannot admit anything, so the session refuses it before spending a
provider round trip and reports
`failures: [{operation: "search", code: "source_quota_exhausted"}]` with
`status: partial`. A visual search still runs — only the text half is
exhausted, and the image cohort has its own quota.

This exists because an empty delta reported as `status: success, failures: []`
is indistinguishable from a provider that genuinely found nothing. Between
2026-09-17 and 2026-09-22 it was reported exactly that way, and the observed
result was a model that reworded and re-searched up to eight times and then
answered citing nothing, which `WebEvidencePolicy` rejects as
`missing_web_citation`. If you see that failure together with a long run of
`web_search` calls, check the quota signal first.

## Ordering is quality-only

Candidate ordering uses resolution adequacy (640px longest edge), provider
confidence, requested-domain provenance, pixel area, then provider rank.
Adequacy leads and provenance sits below it, so a small logo from the named
site cannot displace a large photo from elsewhere.

Nothing in this ordering can tell a team photo from a roster infographic, and a
larger graphic will outrank a smaller photo. Matching the requested *visual
form* is the answer model's job, asked for by the single `IMAGE TARGET` line
prepended to the image blocks with the literal `image_query`. Do not add a
visual taxonomy to the ranking to compensate;
`tests/test_web_image_relevance_regressions.py` holds a negative control that
fails if one is introduced.

There is no answer-time fallback that publishes provider links as images.
Publication still requires a validated, model-selected candidate.

## What to inspect when images look wrong

- `brave image domain probe failed` at warning level — the supplemental
  `site:` probe was refused or timed out; broad-call results still stand.
- `omitted_image_count` on the tool result — candidates permanently lost.
- `new_source_count` vs `total_source_count` — an empty `sources` list with a
  non-zero total is a search that returned only pages the turn already knew.
- `image_fetch` failures carry the rejection reason per candidate; an
  `image_search`/`image_source_capacity` failure means a whole cohort was lost.

Monitor bounded operation outcomes together with the existing rich-image,
model, and routing telemetry. Do not add
queries, URLs, titles, user IDs, tenant IDs, or exception text as metric labels.
Scrape `/metrics/web-research`. The existing ten-minute expiry cleanup worker releases
expired pending references; keep its cadence shorter than the configured TTL
and alert when it stops running.

Release check:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_web_research_contracts.py `
  tests/test_web_source_registry.py `
  tests/test_web_research_policy.py `
  tests/test_web_research_providers.py `
  tests/test_web_research_service.py `
  tests/test_web_research_images.py `
  tests/test_web_image_capacity_and_resolution.py `
  tests/test_web_research_tool_session.py `
  tests/test_web_research_model_context.py `
  tests/test_web_grounding.py `
  tests/test_web_source_streaming.py `
  tests/test_web_tool_output_privacy.py `
  tests/test_web_research_output_policy.py `
  tests/test_required_web_streaming.py `
  tests/test_web_research_continuation.py `
  tests/test_web_research_worker_remap.py `
  tests/test_web_research_active_path_inventory.py `
  tests/test_web_research_config.py `
  tests/test_web_research_metrics.py `
  tests/test_web_research_evaluation.py `
  tests/test_web_query_contract.py `
  tests/test_web_published_source_scope.py `
  tests/test_web_image_relevance_regressions.py `
  -q -p no:cacheprovider

# Supplemental deterministic contract scorer; not an end-to-end substitute.
.venv\Scripts\python.exe scripts/evaluate_web_research.py `
  --cases eval/web_research/cases.json `
  --output output/audits/web-research-eval.json
```

Rollback by disabling `WEB_RESEARCH_ENABLED`; raw Tavily and Brave tools remain
excluded from ordinary agent binding.
