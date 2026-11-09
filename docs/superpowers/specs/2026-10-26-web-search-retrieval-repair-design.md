# Web search retrieval repair

Date: 2026-10-26

Status: approved and implemented on 2026-10-26. See the implementation plan for verification evidence.

## Evidence

The supplied LangSmith run `01a17731-c2d0-7c60-800b-ebcb21ce7575` is accessible through the locally configured API. Its input is `kết quả Worlds LoL` (Worlds LoL results). Its revision is `120c6218`, matching the current checkout.

1. The first `web_search` admits five sources: a Swiss preview, two power rankings, a 2024 standings article, and patch notes. They do not establish the requested completed match results.
2. The model identifies that gap and asks a more specific question in a second `web_search`. The server returns `source_quota_exhausted`, with zero new sources and `searches_used=1`. No second search reaches the provider.
3. `web_open` reads the preview with a question about completed scores. The final response reports that scores could not be verified and asks which year the user means.
4. The trace reports four model calls, three tool calls, and no forced synthesis or exhausted execution budget. The execution budget did not cause this failure.

A local fake-provider reproduction produces the same blocked second search. The existing focused suite passes all 60 tests; `test_an_exhausted_text_quota_is_named_not_reported_as_an_empty_result` explicitly expects this failure after five sources.

Additional implementation findings:

- `create_web_search_tool` accepts `max_results` but omits it from `ResearchRequest`. The session always asks for the mode's full source capacity.
- `tavily_search_args` maps `freshness='recent'` to `topic='news'`. Temporal intent and source category are different dimensions; current results pages and reference documentation may require general search.
- Shared search guidance says "One fact is one search" and requires follow-ups to ask something different. It should instead permit a targeted reformulation of an unanswered question.
- The objective schema claims it ranks evidence, but the current adapter does not use it to rank search results. Correct that description; do not imply a reranker exists.

## Reference behavior

[OpenAI's ChatGPT search documentation](https://help.openai.com/en/articles/9237897-searching-the-web-with-chatgpt) describes rewriting a user request into targeted queries, reviewing results, and issuing additional specific searches. [OpenAI's web search API guide](https://developers.openai.com/api/docs/guides/tools-web-search) documents search, opening pages, finding content within pages, and citations. These are public behavioral references; this design does not claim access to ChatGPT's private ranking implementation.

[Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search) exposes topic separately from date bounds. [Tavily Extract](https://docs.tavily.com/documentation/api-reference/endpoint/extract) supports question-directed chunk reranking. The existing opener already sends the question and requests three chunks, so this repair retains that mechanism.

## Approaches

| Approach | Benefit | Trade-off |
| --- | --- | --- |
| Repair the existing search/session contract (recommended) | Directly addresses the traced failure, works across the current model providers, preserves streaming and citations | Search quality still depends on providers and model judgment; requires answer-quality evaluation |
| Replace retrieval with OpenAI hosted web search | Delegates orchestration to the hosted tool | Requires model/API integration, cost and compatibility decisions, and adaptation to the existing source and image contracts |
| Add a separate research planner, reranker, and evidence judge | More explicit query planning and answer coverage measurement | Adds model calls, latency, and new failure modes before the confirmed defects are repaired |

## Proposed repair

### Search and evidence capacity

Keep the existing mode-specific per-search ceilings of five results for quick mode and eight for agentic mode. Introduce a separate bounded turn-wide text-source capacity of 64. The larger registry stores citation identities; only each operation's evidence delta is returned in its tool message. Existing generation tool, model, and time budgets continue to bound work.

Honor the model's requested `max_results` end to end, clamping it to the per-search ceiling and remaining turn-wide text capacity. Admission must count distinct, valid canonical URLs; duplicates and invalid URLs must not waste available slots when later valid results are present. Stable S# IDs must never be evicted or reassigned. Preserve the independent image-source capacity and page-open headroom.

When all 64 text slots have actually been used, return an explicit capacity failure. A search requesting images may still retrieve image candidates, but it must not make a text-provider request whose results cannot be admitted. Do not add a new search-count quota.

### Search intent

Add an optional typed `topic` of `general` or `news`, defaulting to `general`, through the public search schema, research request, normalized request, provider arguments, and deduplication scope. `freshness` continues to control temporal normalization and date bounds; it no longer selects news search. A different topic must be a distinct search scope, including when research accounting is restored across epochs.

Keep date validation and existing historical-cutoff behavior. Tell the model to use the supplied runtime year explicitly when an event season matters. The server currently repairs stale years but does not add a missing year, so the prompt must not claim that it does.

### Evidence-driven search policy

Use shared guidance for chat and search agents:

- Search with a precise information need and runtime date context.
- Inspect whether the returned content answers the question. Relevant subject matter alone is insufficient.
- Open a promising page when its snippet is incomplete. Reformulate or restrict a search when the returned pages are off-topic, stale, previews, rankings, or otherwise the wrong evidence type.
- Continue with a targeted query for the missing fact; stop when adequate evidence supports the requested answer or a real execution/capacity limit is reached.
- Prefer authoritative sources where available; distinguish completed results from schedules, live states, forecasts, and commentary.
- Answer the supported portion and state specific remaining uncertainty. Never invent scores, citations, or verification.

Keep current exact/near-duplicate protection in this repair. The traced follow-up is sufficiently distinct; broad deduplication changes require separate evidence and tests.

### Tool output and compatibility

Retain operation-scoped evidence deltas, `searches_used`, stable source IDs, source counts, explicit failures, and source-addressed citations. Keep the present snippet and page-open limits; increasing text volume is not a substitute for finding the right page. Correct tool descriptions to match the new capacity behavior and the actual role of `objective`.

Preserve client-only tool policy, user/device/turn isolation, image evidence privacy, continuation remapping, citation rendering, source publication, and stream/history parity.

## Verification

Before fixes, add failing regression tests for:

1. The supplied trace sequence: five preview/ranking sources on the first search, followed by a targeted search returning completed-score evidence. Both calls reach the provider; later evidence is citable with stable IDs.
2. The public tool forwards `max_results=1`; provider and evidence output respect it. Oversized requests clamp to the mode ceiling.
3. Current general facts use `topic=general`; explicit news queries use `topic=news`. Both retain date bounds, and the topic participates in deduplication and restored accounting.
4. Duplicate/invalid results do not hide later unique admissible URLs.
5. Genuine total-capacity exhaustion is explicit, makes no wasted text-provider call, and still permits image-only contribution.

Run the web research, query, tool binding, source registry, research budget, continuation, grounding, source publication, and source streaming suites. Run lint on changed Python files. Update tests that intentionally encoded the first-search source ceiling; preserve meaningful capacity tests at the new actual ceiling.

Use deterministic fixtures for completed-match evidence rather than inventing real 2026 results. A live provider canary should separately inspect source relevance and whether the answer actually contains the requested results. Passing contract tests alone does not establish production answer quality or parity with ChatGPT.

## Scope and completion

This is a repair of the existing web retrieval path, not an implementation of OpenAI's private search stack. It does not add a new provider, research agent, frontend component, general reranker, or page-find tool. The confirmed trace blocker is resolved only when a full first result set no longer prevents a useful second search and that behavior is covered through the public tool/session path.


## Implementation notes

Search-disabled (`none`) mode retains zero total text capacity. Canonical deduplication scopes use JSON primitives; restoration migrates the old six-field format to its historical category so existing generations remain compatible. Carried sources are imported through the session and count toward total capacity. Since the snapshot does not distinguish text, image and opened-source admission classes, imported records are conservatively charged to text capacity. This may reduce headroom after Continue; it does not discard prior citable records.

Verification completed with 556 passing selected tests, clean Ruff checks, independent review and live search/open canaries. See [the implementation record](../plans/2026-10-26-web-search-retrieval-repair.md). These results establish the focused repair, not general production answer quality.
