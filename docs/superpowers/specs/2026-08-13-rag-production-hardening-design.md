# RAG Production Hardening Design

**Date:** 2026-08-13

**Status:** Revised cleanup phase pending review

## Objective

Make the document RAG pipeline reliable for complex, multimodal documents and
corpora of at least 1,000 documents without replacing its core PostgreSQL and
Qdrant architecture. The finished system must have correct embedding batches,
structure-preserving ingestion, scalable retrieval, measurable retrieval and
answer quality, machine-verifiable citations, safe abstention, bounded context,
and observable cost and latency.

## Scope

This program covers:

- Gemini text and image embedding correctness, batching, retry behavior, and
  content-hash reuse.
- MinerU/plain-text/Excel normalization and structure-aware chunking.
- Qdrant collection bootstrap, payload indexes, hybrid retrieval, candidate
  fusion, reranking, and versioned reindexing.
- Agentic RAG tool policy, evidence assembly, context-window handling, and
  prompt-injection boundaries.
- Claim-level grounding, citation validation, regeneration, and abstention.
- Caption-based and native multimodal retrieval and vision-context budgeting.
- LangSmith evaluation datasets and experiments, with selected RAGAS metrics.
- Stage-level latency/cost telemetry, safe caches, and 1,000-document load
  qualification.
- Evidence-based removal of superseded RAG paths, compatibility wrappers,
  duplicate helpers, stale settings, noisy logs, misleading comments/docstrings,
  obsolete tests, and unused dependencies after their replacements are proven.

The design does not replace Qdrant with another vector database, replace
PostgreSQL as the canonical authorized content store, or introduce TruLens.
Semantic answer caching and a graph RAG subsystem are explicitly deferred until
production measurements justify them.

## Global Constraints

- PostgreSQL remains canonical for document text, authorization-sensitive
  metadata, chunks, images, parse artifacts, and index state.
- Qdrant contains retrieval vectors and non-sensitive lookup/filter metadata.
- Every Qdrant result is re-authorized while hydrating canonical SQL records.
- New behavior is introduced behind independently reversible settings.
- Existing indexed documents remain readable while a replacement index is
  built; a failed reindex never destroys the prior active generation.
- Tests use deterministic local fakes by default. A separately marked live
  provider contract test verifies Gemini batch response shape.
- Retrieval thresholds and chunk sizes are selected from evaluation results,
  not hard-coded as universal quality claims.
- Reranking and other CPU-bound model calls must not block the async event loop.
- Document content, captions, OCR, filenames, and parser output are untrusted
  reference data and never become executable instructions.
- Tenant scope is part of every cache key and every retrieval operation.
- Cleanup may remove a path only after a repository-wide usage inventory and
  replacement test prove it is unused. Operational error/audit logs and public
  API contracts are retained unless an explicit replacement exists.

## Target Architecture

The pipeline is divided into seven independently testable units:

1. **Document normalizer** converts parser-specific output into typed
   `NormalizedBlock` values that preserve block kind, heading path, page span,
   bbox, table/image metadata, and stable provenance.
2. **Chunk builder** performs the only content-chunking pass. It uses structural
   boundaries first, token limits second, and controlled same-section overlap
   last. Tables retain their caption/header and split only on row groups.
3. **Embedding/index writer** creates one vector per separately wrapped input,
   validates vector count and dimension, reuses exact content hashes, and writes
   an inactive index generation before activation.
4. **Retriever** produces typed candidates from dense and lexical searches,
   fuses ranks, hydrates authorized SQL content, and expands adjacent chunks only
   when the evidence budget allows.
5. **Reranker** ranks a wider candidate pool off the event loop and fails open to
   fused retrieval results when unavailable.
6. **Evidence assembler** deduplicates candidates and produces a token-budgeted
   evidence pack with immutable evidence IDs and untrusted-data delimiters.
7. **Grounded answer gate** generates against that evidence, validates every
   citation and factual claim, then either accepts, regenerates once, or
   abstains.

LangSmith records the trace spanning these units. Offline experiments and
sampled online evaluators consume the same typed retrieval/evidence fields.

## Phase 0: Critical Correctness and Scalability Repairs

### Gemini Embedding 2 batches

Document inputs must be formatted as they are today, but every formatted string
is wrapped in its own `google.genai.types.Content` containing one text part.
The synchronous request may use the SDK batch endpoint, but response order must
match input order. The service rejects a response whose vector count or vector
dimension differs from the request.

The live contract test sends at least two distinct strings and asserts two
distinct embeddings. It is opt-in through a pytest marker and provider key so
normal CI remains offline. Query embedding receives the same bounded retry and
jitter policy as document embedding.

### Agent retrieval policy

`SCAN_ALL` is no longer mandatory. The agent begins with `SEARCH_CHUNKS` for
ordinary questions and uses `LIST_DOCUMENTS`, targeted previews, or bounded
corpus scans only for corpus-enumeration requests. `READ_DOCUMENT` accepts a
page/chunk window and is used for summaries or targeted deep reads, not as the
default follow-up to every relevant document.

A bounded scan accepts pagination and a maximum document count. No tool call can
construct a model-visible preview proportional to every document in a
conversation.

### Dead controls and documentation

Configuration fields that claim citation, confidence, or structured validation
behavior are either connected to the grounded answer gate described below or
removed. README claims must match executable behavior and tested settings.

## Phase 1: Evaluation Baseline and Release Gates

Before tuning chunking or retrieval, introduce a versioned LangSmith dataset and
a local evaluation target that returns:

- final answer and abstention status;
- dense, lexical, fused, and reranked candidates;
- hydrated document/chunk/page/image identifiers;
- the exact evidence pack shown to the model;
- citations and validation outcome;
- tool trajectory, iteration count, token usage, cost, and stage timings.

The first curated dataset contains 100–300 questions split across direct lookup,
exact identifiers and numbers, tables, summaries, multi-hop/cross-document
questions, conflicting sources, unanswerable questions, near-duplicate
distractors, images/charts, prompt injection, and supported languages. Gold
labels use document IDs and page/evidence spans rather than transient Qdrant
point ranks.

Deterministic metrics include document/chunk Recall@k, hit rate, MRR, nDCG,
citation validity, citation precision/recall, claim citation coverage,
abstention precision/recall, tool counts, latency, and cost. RAGAS supplies
context precision/recall, noise sensitivity, faithfulness, response relevance,
and multimodal faithfulness/relevance. LLM judges are calibrated against a
human-reviewed subset and never replace deterministic retrieval metrics.

Every later phase records a LangSmith experiment against the unchanged dataset.
A change cannot ship when it causes a material regression in a designated gate,
even if its average answer score improves.

## Phase 2: Structure-Preserving Ingestion and Chunking

MinerU content blocks are converted directly to typed normalized blocks:

- headings update and carry `section_path`;
- paragraphs retain page and bbox provenance;
- tables use `kind="table"` with caption, header, body, and footnote metadata;
- images use `kind="image"` with page, bbox, parser caption, and nearby section;
- equations retain their source representation and page location.

Plain text and Excel use the same normalized representation. The legacy
character splitter is removed from production ingestion so content is not split
twice.

The chunk builder must:

- target 400 tokens initially, cap ordinary chunks at 800 tokens, and apply
  40-token overlap only between adjacent chunks in the same section;
- never overlap independent table chunks or cross a heading boundary solely to
  satisfy overlap;
- repeat table captions and headers on row-group splits;
- verify the final rendered token count, including separators and metadata;
- use language-aware sentence boundaries with a safe token/word fallback;
- retain stable block provenance and neighboring chunk relationships;
- include filename and section path in the embedding representation while
  keeping canonical citation text unchanged.

Semantic breakpoint chunking is an experiment, not the default. It is enabled
only if it improves retrieval and answer metrics over corrected structural
chunking at acceptable ingestion cost.

For broad documents, child chunks support retrieval and optional larger parent
spans support synthesis. Parent expansion is bounded by the evidence token
budget.

## Phase 3: Qdrant, Hybrid Retrieval, and Reranking

Collection bootstrap creates payload indexes before data ingestion for
`user_id`, `conversation_id`, `document_id`, `modality`, and active index
generation. Tenant-aware indexing is used for the authenticated owner field
when compatible with the deployed Qdrant version.

Each point carries named dense retrieval data and lexical/sparse data or an
equivalent indexed text representation. Queries execute dense and lexical
prefetches under the same tenant/conversation filter. Reciprocal Rank Fusion
produces a candidate pool before SQL hydration.

Retrieval returns a typed `RetrievalCandidate` with dense rank/score, lexical
rank/score, fused score, modality, identifiers, and canonical hydrated content.
Raw scores are observability fields, not probabilities shown to the model.

The reranker evaluates a configurable pool, initially 40 candidates, and keeps
an evidence pool initially capped at 8–12 passages. Its model is selected through
multilingual/domain evaluation. Prediction runs in a bounded executor with a
timeout and concurrency limit. A model load or inference failure records a
degraded event and returns fused results instead of failing the RAG tool.

Global dense score thresholds are disabled or kept permissive before reranking.
Any post-rerank answerability threshold is calibrated on the evaluation set.

Index writes use an `index_generation` value. New SQL chunks and Qdrant points
are written inactive, verified for count/dimension/scope consistency, then
activated through a small canonical state change or Qdrant alias switch. Old
generations are deleted only after activation and a rollback window.

## Phase 4: Evidence, Grounding, Citations, and Guardrails

The evidence assembler receives authorized ranked candidates and a model input
budget. It removes duplicate/overlapping passages, balances coverage across
subquestions and documents, and optionally includes adjacent or parent context.
It returns structured evidence records with stable IDs such as `E1`, filename,
document ID, chunk ID, page range, section path, modality, and content.

Evidence is serialized as untrusted reference data with explicit start/end
markers. Tool outputs remain tool-role messages instead of being concatenated
into a synthetic current user message. The request budget can trim complete old
tool/evidence groups while preserving the current question and selected
evidence.

The answer model returns structured claim/citation data plus renderable text.
The grounded answer gate verifies:

- every cited evidence ID exists in the current evidence pack;
- citation filename/page metadata comes from the server, not model text;
- factual claims meet the configured citation-coverage requirement;
- the answer does not claim support when no evidence was retrieved;
- an optional judge/entailment pass supports high-risk claims.

Invalid citation structure or insufficient coverage triggers one constrained
regeneration. Persistent failure, low evidence sufficiency, or an unanswerable
classification returns a clear abstention that explains which information is
missing. Citation coverage is measured over answer claims, never over the number
of retrieved documents.

Prompt-injection fixtures include instructions embedded in text, tables, OCR,
captions, and filenames. The agent may quote or analyze these instructions but
must not follow them as commands.

## Phase 5: Multimodal Retrieval

Caption-augmented text remains the default image retrieval path. The captioning
prompt produces a structured description containing visible OCR, chart title,
axes, legend, values, trends, relationships, and nearby document context when
present. Captions retain the image ID, document ID, page, bbox, and section.

When native multimodal embeddings are enabled, every image is embedded as a
separate `modality="image"` point in the Gemini shared space and linked to its
canonical `DocumentImage` record. Retrieval branches by modality and hydrates
either a text chunk or an image without requiring all points to have a chunk ID.

The evidence/image selector uses query visual intent, retrieval/rerank score,
deduplication, byte limits, pixel limits, and a model-specific vision budget.
Only selected images are read and encoded. Images are resized or cropped to the
relevant region when safe, and base64 is never accumulated without a hard cap.

Multimodal activation requires image Recall@k, table/chart QA, multimodal
faithfulness, cost, and p95 latency results better than caption-only retrieval.

## Phase 6: Cost, Latency, Caching, and Scale Qualification

Stage timers cover parse, caption, embedding, Qdrant dense/lexical retrieval,
SQL hydration, reranking, evidence assembly, generation, and validation. Reports
include p50/p95/p99 latency, cost per indexed document and answered question,
tokens, cached-input ratio, tool iterations, failure rate, and queue depth.

Safe caches are introduced in this order:

1. Exact document embedding cache keyed by provider, model, dimension, prompt
   format version, and content hash.
2. Exact query embedding cache keyed by provider, model, dimension, task prefix,
   and normalized query.
3. Short-lived retrieval cache keyed by tenant, conversation, index generation,
   normalized query, and retrieval configuration.

All caches are invalidated through index generation rather than broad deletes.
Semantic answer caching is not implemented in this program. Managed-provider KV
or implicit prompt caching is observed through cached-input token telemetry;
stable prompts and tool schemas are kept at the start of requests to improve
provider cache reuse.

For non-interactive indexing, a provider Batch API experiment measures cost and
throughput against synchronous batches. Embedding dimensions 768, 1536, and
3072 are compared on the golden dataset and vector-memory budget before choosing
the production dimension.

Scale qualification uses at least 1,000 representative documents and concurrent
ingestion/query traffic. It measures total chunks, vector memory, indexing lag,
retrieval quality under distractors, tenant-filter performance, p95/p99 query
latency, queue saturation, and failure recovery. Qdrant HNSW, quantization,
sharding, and on-disk settings are changed only in response to these results.

## Phase 7: Consolidation and Dead-Code Cleanup

Cleanup runs after the replacement paths have passed their quality and rollout
gates, so it removes proven redundancy instead of deleting fallback behavior
while the new pipeline is still being validated. Each earlier implementation
task removes local superseded code when safe; this final phase performs the
cross-cutting audit that cannot be completed until every new path is active.

The phase begins with an inventory across the RAG pipeline and its directly
adjacent ingestion, workflow, configuration, tests, scripts, and documentation.
The inventory records every candidate, its callers, replacement, compatibility
reason, removal risk, and validating test. Dynamic framework registration,
Celery task names, dependency-injection providers, Pydantic settings, migrations,
and persisted tool/action names are treated as externally referenced until
runtime/configuration searches prove otherwise.

The cleanup includes:

- removing the duplicated parse/chunk helper surface from
  `DocumentProcessingService` after all callers use `DocumentParseService` and
  the new normalizer;
- deleting legacy character splitters and old rich-document chunk assembly once
  the single normalized-block path owns every supported file type;
- removing traditional/retired RAG branches, transitional adapters, stale
  feature flags, legacy setting aliases, unused multimodal stubs, and obsolete
  reindex behavior after their supported replacements ship;
- consolidating repeated retrieval, scope, image-hydration, and document
  reference logic behind the typed interfaces introduced by earlier phases;
- removing tests whose only purpose was to preserve a retired implementation,
  while retaining or rewriting behavioral regression coverage first;
- removing unused imports, dependencies, scripts, environment variables, and
  README troubleshooting instructions;
- replacing phase/task archaeology comments and inaccurate docstrings with
  concise descriptions of current contracts, invariants, and failure behavior;
- converting interpolated logging to structured lazy arguments, deduplicating
  repeated stage messages, bounding exception detail, and preventing document
  content, credentials, embeddings, or base64 media from entering logs;
- retaining warning/error/audit logs for provider failures, index activation,
  authorization mismatches, fallback operation, citation rejection, and
  degraded retrieval, with one owning layer per event to prevent duplicates.

Removal is verified through focused tests, the full RAG suite, import/static
checks, settings/documentation contract tests, and the unchanged offline quality
dataset. A compatibility shim is removed only when no supported client, queued
Celery task, stored state, deployment environment, or public API depends on it.
Database migrations remain immutable; cleanup adds new migrations when required
and never edits an already deployed migration solely for tidiness.

The result is one documented production path per responsibility: parsing,
normalization, chunking, embedding, indexing, retrieval, reranking, evidence
assembly, answer validation, and evaluation. Source files keep module/class/
public-method docstrings that describe why and contract; redundant narration,
historical phase labels, and comments that merely restate code are removed.

## Error Handling and Rollback

- Parse failures retain diagnostic artifacts and never publish partial chunks.
- Embedding count/dimension mismatches fail the inactive generation immediately.
- Provider rate limits use bounded exponential backoff with jitter and respect
  provider hints.
- Qdrant or SQL write failures leave the previous generation active.
- Reranker failures degrade to fused retrieval and emit a metric.
- Citation validation failure regenerates once, then abstains.
- Context that cannot fit after evidence reduction returns a bounded context
  error rather than silently removing the current question.
- Each feature flag can return traffic to the prior stable path without
  re-ingesting the corpus, except changes to embedding model or dimension, which
  use their own collection generation.

## Testing Strategy

Each implementation task follows test-driven development with a failing focused
test, minimal implementation, focused verification, and a small commit.

Required test layers are:

- pure unit tests for normalization, overlap, table splitting, fusion,
  deduplication, token budgeting, citation validation, and cache keys;
- repository/service tests for scoped hydration and index generation state;
- provider-contract tests for Gemini content serialization and optional live
  multi-input embedding;
- Qdrant/PostgreSQL integration tests for payload indexes, hybrid queries,
  activation, rollback, and tenant isolation;
- agent workflow tests for retrieval-first behavior, bounded scans, reranker
  fallback, untrusted evidence, regeneration, and abstention;
- offline LangSmith/RAGAS experiments for every retrieval/chunking/model change;
- load tests for 1,000 documents, concurrent users, and failure injection.
- cleanup contract tests and repository scans that prove retired names, settings,
  imports, and duplicated helpers are absent while required operational logs and
  public interfaces remain.

The existing focused RAG suite remains a regression gate. New quality gates are
reported separately so deterministic unit success cannot be mistaken for RAG
quality success.

## Rollout Sequence

1. Ship the Gemini batch fix and retrieval-first agent policy.
2. Establish the evaluation dataset, trace schema, and baseline experiment.
3. Reindex a shadow collection with normalized structural chunks.
4. Enable payload indexes and hybrid retrieval in shadow mode; compare results
   without affecting answers.
5. Enable asynchronous reranking and evidence assembly for a small traffic
   cohort.
6. Enforce structured citations, regeneration, abstention, and injection
   boundaries.
7. Run caption-only versus native multimodal experiments and enable native image
   points only when they pass gates.
8. Enable exact caches and any reduced embedding dimension proven by evaluation.
9. Complete 1,000-document load qualification before broad production rollout.
10. Remove superseded paths and finish the RAG-adjacent cleanup audit only after
    the replacement rollout and rollback window have completed.

## Completion Criteria

The remediation program is complete when:

- a live two-input Gemini embedding contract returns two vectors;
- ordinary questions never require a full corpus scan;
- rich document blocks retain tested headings, tables, page spans, images, and
  provenance through persisted chunks;
- Qdrant filtered retrieval uses payload indexes and shows no cross-user leak;
- hybrid retrieval and reranking meet the approved Recall@k and latency gates;
- every displayed citation resolves to evidence retrieved in the current turn;
- unanswerable and prompt-injection cases pass their abstention/safety gates;
- multimodal answers are measured against visual evidence when enabled;
- context input is token-bounded and reducible without losing the current
  question;
- stage-level cost and p95/p99 latency are visible in evaluation and operations;
- the 1,000-document qualification run meets the release SLOs selected from the
  baseline rather than an unmeasured target.
- repository-wide usage evidence shows no supported caller depends on removed
  RAG compatibility paths, and each responsibility has one production owner;
- RAG settings, environment examples, tests, scripts, README claims, logs,
  comments, and docstrings describe only the current supported architecture;
- the final cleanup diff passes focused and full RAG tests, static/import checks,
  documentation contracts, and the offline quality gates without regression.
