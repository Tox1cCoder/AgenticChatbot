# RAG Ingestion Pipeline: Throughput & Multi-User Concurrency

**Status:** Draft — pending review
**Date:** 2026-06-12
**Branch:** Thai-Postgre-FastAPI
**Scope:** Document ingestion only (upload → parse → embed → index). Query/chat path is out of scope.

---

## 1. Problem Statement

The ingestion pipeline is slow and does not improve when batch size or Celery worker
concurrency is increased. Root causes, verified in code:

| # | Bottleneck | Location | Impact |
|---|-----------|----------|--------|
| B1 | Embedding is one blocking Gemini API call **per chunk**, sequential. `rag_index_batch_size=16` only groups chunks; each group still makes 16 serial HTTP round-trips. | `app/services/rag_embedding_service.py:104-115`, `app/services/document_index_service.py:234-237` | A 200-chunk PDF = 200 serial calls ≈ 30–60 s of pure network latency. Batch-size tuning changes nothing. |
| B2 | MinerU cold-starts per document. With `mineru_api_url` blank (default), every CLI invocation boots a temporary MinerU service and loads models from scratch. | `app/services/document_processing_service.py:439-445`, `app/core/config.py:491-497` | ~30–90 s overhead per file; concurrent MinerU subprocesses contend for the same GPU/CPU, so raising Celery concurrency makes parses *slower*. |
| B3 | Image captioning is one sequential Gemini call per image with retry sleeps. | `app/services/document_processing_service.py:1260-1293` | Image-heavy PDFs serialize again after parsing. |
| B4 | One monolithic Celery task runs parse + caption + embed + index. GPU-bound and IO-bound work share the same worker slots; `task_time_limit=300s` covers the whole pipeline; an embed failure retries the expensive parse. | `app/workers/document_processor.py:40-216` | Concurrency can't overlap heterogeneous stages; large files hit the hard kill; retries waste GPU time. |
| B5 | `mark_indexed` issues one UPDATE per chunk. | `app/services/document_index_service.py:152-161` | Minor: N round-trips to Postgres per document. |

The surrounding architecture (FastAPI + Celery + Redis + Qdrant + Postgres) is already
correct for multi-user serving: uploads are staged to disk and enqueued without blocking
the API (`app/services/document_processing_service.py:105-146`), and chunk payloads are
scoped by `user_id`/`conversation_id` for retrieval isolation. The fixes are inside the
pipeline, not a rewrite.

## 2. Resolved Clarifications

| Question | Answer |
|----------|--------|
| Production environment | Windows machine, NVIDIA GPU available for MinerU |
| Target load | ~10 users, ~20 files in flight |
| Performance target | Throughput: 20 in-flight files finish steadily without blocking each other; no hard per-file latency number |
| Gemini API tier | Higher paid tier — rate limits are not the constraint at this scale |
| Allowed changes | Persistent MinerU service ✓, Gemini batch/concurrent embedding ✓, split pipeline into chained Celery stages ✓ |

## 3. Functional Requirements

- **FR-1** Multiple documents from multiple users MUST process concurrently; one user's large PDF MUST NOT stall another user's small upload end-to-end.
- **FR-2** Embedding MUST batch multiple chunks per Gemini API request and run a bounded number of requests concurrently.
- **FR-3** MinerU MUST run as a persistent service so model load happens once per process lifetime, not once per document.
- **FR-4** Parsing and indexing MUST be separate Celery tasks on separate queues with independent concurrency, time limits, and retry policies.
- **FR-5** A failure in the index stage MUST retry from the persisted parse artifact without re-running MinerU.
- **FR-6** Image captioning MUST run with bounded concurrency instead of sequentially.
- **FR-7** Document status reporting (PROCESSING / READY / FAILED) and existing events MUST keep working across the staged pipeline.
- **FR-8** Existing multi-user retrieval isolation (user_id/conversation_id payload scoping) MUST be preserved — covered by `tests/test_rag_multi_user_isolation.py`.

## 4. Non-Functional Requirements

- **NFR-1** Embed stage wall time for a 200-chunk document drops from ~O(200 × RTT) to ~O(⌈200/32⌉ ÷ 4 × RTT) — target ≥ 10× reduction.
- **NFR-2** Second and subsequent PDF parses show no MinerU model cold-start (verified by parse-time logs).
- **NFR-3** All concurrency limits are config settings with safe defaults; no hard-coded parallelism.
- **NFR-4** Per-stage timings (parse, caption, embed, upsert) are logged and attached to the completion event so improvements are measurable.
- **NFR-5** Works on Windows (threads pool — already the resolved default in `app/workers/start_worker.py:8-19`).

## 5. Target Architecture

```
                       ┌──────────────────────────────┐
 FastAPI upload ──────►│ stage to temp + enqueue chain │  (unchanged, non-blocking)
                       └──────────────┬───────────────┘
                                      │ chain(parse.s(), index.s())
                     ┌────────────────▼─────────────────┐
   queue: parse      │ parse_document_task              │  concurrency: celery_parse_concurrency (2)
   (GPU-bound)       │  MinerU via persistent HTTP API  │  time limit: mineru_timeout + margin
                     │  → normalized chunks + images    │
                     │  → DocumentParseArtifact (disk + │
                     │    Postgres row, JSON payload)   │
                     └────────────────┬─────────────────┘
                                      │ artifact_id only (no big payloads through Redis)
                     ┌────────────────▼─────────────────┐
   queue: index      │ index_document_task              │  concurrency: celery_index_concurrency (8)
   (IO-bound)        │  caption images (bounded conc.)  │  own time limit + retry w/o re-parse
                     │  embed: batched + concurrent     │
                     │  Qdrant upsert + bulk mark_indexed│
                     └────────────────┬─────────────────┘
                                      ▼
                          Document → READY + completion event

   Persistent MinerU service (mineru-api, GPU, started once) ◄── parse workers via mineru_api_url
```

Key decisions:

1. **Embedding concurrency lives inside `GeminiRAGEmbeddingService`** behind the existing
   sync `embed_documents()` interface (ThreadPoolExecutor over batched API calls).
   `DocumentIndexService` passes all chunk texts in one call and stays unchanged otherwise.
   No async refactor of the index path.
2. **Parse→index handoff via `DocumentParseArtifact`** (`app/models/document_parse_artifact.py`)
   — chunk JSON written to disk under `storage_path`, only the artifact id travels through
   the Celery chain. This also yields reindex-without-reparse for free.
3. **Parse logic extracted to its own service module.** `DocumentProcessingService` is
   1,650 lines mixing staging, parsing, captioning, and indexing; the split follows the
   task boundary.
4. **Document status stays the source of truth in Postgres.** The status endpoint keeps
   reading the document row; stage detail goes into event/status metadata only. No new
   status enum values.

## 6. Implementation Phases

Phases are ordered by value ÷ effort and each ships independently. Tasks marked [P]
(T016, T017) can run in parallel with Phase 3.

### Phase 1 — True batched + concurrent embedding (biggest win, no structural change)

- **T001 — Batched embedding requests.** ✅ DONE (commit 519b0f2)
  `app/services/rag_embedding_service.py`: `embed_documents()` sends up to
  `rag_embedding_batch_size` (new setting, default 32) formatted contents per
  `embed_content` call instead of one. Validate `len(response.embeddings) == len(batch)`;
  raise `RuntimeError` on mismatch (keep the no-silent-fallback invariant).
  *Design decisions:* `_API_MAX_BATCH = 100` as `ClassVar`; batch size clamped at init;
  empty-input short-circuit added; SDK accepts `list[str]` for batch contents (ContentsType).
- **T002 — Concurrent batch calls.**
  Run batches through a `ThreadPoolExecutor` bounded by `rag_embedding_max_concurrency`
  (new setting, default 4). Preserve input order in the returned vectors. Retry 429s with
  exponential backoff honoring the API's `retryDelay` hint — extract the retry-delay
  parsing already written in `document_processing_service.py:1455-1545` into a shared
  helper (`app/services/gemini_retry.py`) instead of duplicating it.
- **T003 — Index service feeds the whole document at once.**
  `app/services/document_index_service.py`: `_embed_and_upsert` passes all chunk texts in
  a single `embed_documents()` call (batching now internal to the embedding service).
  Slice the Qdrant upsert by the existing `qdrant_upsert_batch_size` setting. Remove the
  now-redundant `rag_index_batch_size` grouping (deprecate the setting; keep it parsed
  with a warning so existing `.env` files don't break).
- **T004 — Bulk `mark_indexed`.**
  `app/repositories/document_chunk.py`: add `mark_indexed_bulk(chunk_ids, …)` issuing one
  UPDATE; replace the per-chunk loop at `document_index_service.py:152-161` and the
  matching loop in `reindex_document`.
- **T005 — Tests.**
  Batch split + order preservation; response-count mismatch raises; 429 retry honors
  delay hint; concurrency cap respected (e.g., assert max in-flight via instrumented
  fake client); bulk mark_indexed marks exactly the persisted chunks. Update existing
  embedding-service tests for the new request shape.

**Acceptance:** a 200-chunk document embeds in ⌈200/32⌉ = 7 API calls across ≤ 4
concurrent workers; unit tests green; no interface change visible to callers.

### Phase 2 — Persistent MinerU service (kills the cold start)

- **T006 — Service runner script.**
  `scripts/start_mineru_service.ps1`: launch `mineru-api` bound to localhost with GPU,
  suitable for manual start and for registration as a Windows service (NSSM) — include
  the NSSM registration commands as comments. Document in README's deployment section.
- **T007 — Wire workers to the service.**
  Set `mineru_api_url` in deployment config; the CLI already forwards `--api-url`
  (`document_processing_service.py:424-425`). Decide the backend explicitly for the GPU
  host (`hybrid-http-client` or `vlm-http-client` vs `pipeline` + api-url) by
  benchmarking one representative PDF on each; record the choice and numbers in the
  README.
- **T008 — Startup health check.**
  Worker startup (`app/workers/start_worker.py` or celery signal) probes the configured
  `mineru_api_url` and logs a prominent warning when unreachable. Tasks already fail with
  a clear error and retry — no silent fallback to cold-start mode.
- **T009 — Validation.**
  Same PDF parsed twice: second run shows no model-load time in MinerU logs; parse
  wall time recorded before/after for the README.

**Acceptance:** warm parse time for the benchmark PDF; no temporary MinerU service
boot per call.

### Phase 3 — Staged pipeline: parse → index on separate queues

- **T010 — Extract parse stage into `app/services/document_parse_service.py`.**
  Move MinerU invocation, output resolution, content-list parsing, Excel/text loaders,
  and chunk-with-metadata assembly out of `DocumentProcessingService` (no behavior
  change; move + re-export). The parse service ends at "normalized chunks + extracted
  image entries".
- **T011 — Persist parse output as a `DocumentParseArtifact`.**
  Artifact type `normalized_chunks`: JSON file (chunks_with_metadata + image entries)
  under a stable artifacts dir, row via `document_parse_artifact_repository`, sha256 +
  size filled in. Extracted images moved from MinerU temp output to their permanent
  storage path at parse time (today this happens during indexing —
  `document_processing_service.py:1271-1272`).
- **T012 — Split the Celery task.**
  `app/workers/document_processor.py`: `parse_document_task` (queue `parse`) →
  returns artifact id; `index_document_task` (queue `index`) → loads artifact, captions,
  builds chunks, embeds, upserts, bulk-marks indexed, sets READY. Enqueue as
  `chain(parse.s(…), index.s(…))` from `start_processing_task`. Per-task time limits:
  parse = `mineru_timeout` + 60 s margin; index = new `celery_index_time_limit` (default
  600 s). Retry policy per stage; index retries re-read the artifact, never re-parse
  (FR-5). On terminal failure in either stage: document → FAILED, failure event, temp
  cleanup (keep artifact for diagnosis; cleanup task ages it out with the existing
  temp-file cleanup).
- **T013 — Queue routing + worker startup.**
  `app/workers/celery_app.py`: `task_routes` mapping the two tasks to their queues.
  `start_worker.py`: start two worker processes — `-Q parse --concurrency=<celery_parse_concurrency>`
  (default 2) and `-Q index --concurrency=<celery_index_concurrency>` (default 8), both
  threads pool on Windows. New settings with those names; keep single-worker
  `-Q parse,index` documented as the minimal dev mode.
- **T014 — Status and events across stages.**
  `PROCESSING_STARTED` on enqueue (exists); add stage info to event metadata
  (`stage: parse|index`) on stage completion/failure. `get_processing_status` keeps
  working: status endpoint reads the document row (source of truth); task-id based
  lookups resolve the chain's tasks. Keep the old task name
  `app.workers.document_processor.process_document_task` registered as a thin
  compatibility shim that enqueues the chain, so tasks already queued at deploy time
  still run.
- **T015 — Tests.**
  Artifact write/read roundtrip; chain wiring (parse result feeds index); index-stage
  retry does not re-invoke the parser (assert via spy); terminal failure paths set
  FAILED + emit events; integration test: N small files + 1 large file enqueued
  together, small files reach READY while the large parse is still running (FR-1);
  `tests/test_rag_multi_user_isolation.py` stays green (FR-8).

**Acceptance:** 20 mixed files in flight complete steadily; parse queue saturates the
GPU at concurrency 2 while index queue drains at concurrency 8; embed failure on one
document re-runs only its index stage.

### Phase 4 — Concurrent captioning + observability [P with Phase 3]

- **T016 — Bounded-concurrency captioning.** [P]
  `_prepare_images_for_indexing`: caption images via `asyncio.gather` over
  `asyncio.to_thread(...)` calls bounded by a semaphore — new setting
  `image_caption_max_concurrency` (default 4). Reuse the shared Gemini retry helper from
  T002. Order-independent; per-image failure still degrades to the metadata caption.
- **T017 — Per-stage timing metrics.** [P]
  Record `parse_s`, `caption_s`, `embed_s`, `upsert_s`, `total_s` and attach to the
  completion event metadata + a structured log line per document.
- **T018 — Benchmark script.**
  `scripts/benchmark_ingestion.py`: submit a corpus (mix of small txt, mid PDF, image-
  heavy PDF, xlsx) at configurable parallelism through the real API; report per-stage
  p50/p95 and total wall time. Run before Phase 1 (baseline) and after each phase;
  numbers go in this document's Verification section.

**Acceptance:** image-heavy benchmark PDF shows captioning wall time ≈ slowest-image
time × ⌈images/4⌉; benchmark report produced.

## 7. Verification & Success Criteria

| Criterion | How verified |
|-----------|--------------|
| ≥ 10× embed-stage reduction on 200-chunk doc (NFR-1) | Benchmark script, `embed_s` before/after Phase 1 |
| No MinerU cold start on warm service (NFR-2) | Parse logs + `parse_s` before/after Phase 2 |
| 20 in-flight files, no head-of-line blocking (FR-1) | Phase 3 integration test + benchmark at parallelism 20 |
| Index retry without re-parse (FR-5) | Phase 3 unit test (parser spy) |
| Multi-user isolation preserved (FR-8) | `tests/test_rag_multi_user_isolation.py` |
| Status/events unchanged for clients (FR-7) | Existing document API tests + Postman collection smoke |

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Embedding model's per-request content limit is smaller than assumed (or 1) | T001 verifies against the live API first; batch size is config-clamped; T002's concurrency alone still gives ~4× |
| GPU VRAM exhaustion with concurrent parses | `celery_parse_concurrency` default 2, documented to drop to 1; MinerU service queues internally |
| Large parse artifacts through the broker | Only artifact id travels through the chain; payload lives on disk + Postgres row |
| mineru-api instability on Windows | Health check (T008) + existing per-task retry; NSSM auto-restart in the service registration |
| In-flight tasks during deploy of Phase 3 | Old task name kept as a shim enqueueing the new chain (T014) |
| Thread-safety of per-document state | `document_processing_service` is a `providers.Factory` (`app/core/container.py:452`) — fresh instance per task; preserve this in the parse-service extraction |

## 9. Out of Scope

- Query/retrieval path performance (query embedding, reranking, top-k tuning)
- Per-user fair scheduling (unnecessary at ~10 users once queues are split)
- Horizontal multi-node scaling, Kubernetes
- Vector DB or embedding provider changes
- UI changes to upload/status flows

## 10. Settings Added / Changed

| Setting | Default | Phase |
|---------|---------|-------|
| `rag_embedding_batch_size` | 32 (clamped to API max) | 1 |
| `rag_embedding_max_concurrency` | 4 | 1 |
| `rag_index_batch_size` | deprecated (warning) | 1 |
| `mineru_api_url` | set in deployment `.env` | 2 |
| `celery_parse_concurrency` | 2 | 3 |
| `celery_index_concurrency` | 8 | 3 |
| `celery_index_time_limit` | 600 | 3 |
| `image_caption_max_concurrency` | 4 | 4 |
