# RAG scale-qualification runbook (Task 13)

## Status as of this writing: nothing in this document has been run

This runbook explains how to execute the two scale-qualification harnesses
built in Task 13 of the RAG production-hardening plan --
`scripts/benchmark_rag.py` and `scripts/experiment_embedding_dimensions.py`
-- and how to feed their output back into `eval/rag/release_gates.json`.

**No benchmark run, dimension comparison, or Qdrant tuning decision has
been made yet.** This checkout has:

- **No 1,000-document corpus.** `eval/rag/corpus_manifest.jsonl` holds 11
  fixture entries. `scripts/benchmark_rag.py --documents 1000` against this
  manifest always produces `status: "unexecuted"`.
- **No live PostgreSQL, Qdrant, or Redis.** Prior tasks in this plan
  recorded live-PostgreSQL coverage as unavailable in this environment and
  worked entirely against deterministic fakes; that has not changed.
- **No embedding-provider budget or credentials for a scale run.** The
  LangSmith account's monthly unique-trace quota is exhausted, and this
  plan skips LangSmith uploads by direction (2026-08-13).

Every number an operator sees below is a *placeholder for a real run*, not
a recorded result. `eval/rag/release_gates.json`'s Task 13 gate entries
(`p95_stage_latency_ms`, `vector_memory_mb`, `indexing_lag_p95_seconds`,
`cache_hit_ratio`, `embedding_dimension_recall_delta`, and the three
deferred cost/token gates) all carry `"status": "unmeasured"` with
`"max_regression": null` for exactly this reason. Do not hand-edit those
values to a number you have not measured -- `app/evaluation/rag/
release_gates.py`'s `compare_release_gates()` treats `status: "unmeasured"`
gates as **non-binding** (`passed: null`) specifically so an absent
measurement can never masquerade as a pass. Flip a gate to `"status":
"measured"` with a real `max_regression` only after the run below produced
the number in its `provenance.source`.

## Running the 1,000-document scale benchmark

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_rag.py `
    --documents 1000 `
    --corpus-manifest path\to\a\real\1000-document-manifest.jsonl `
    --conversation-id <uuid-to-load-the-corpus-into> `
    --auth-token <bearer-token> `
    --base-url http://your-live-server:8000 `
    --ingest-concurrency 8 `
    --query-concurrency 32 `
    --failure-injection `
    --failure-injector-hook your_ops_module:inject_failure `
    --output artifacts/rag-scale-1000.json
```

### Prerequisites (all four gate the run; any missing one produces
### `status: "unexecuted"` with the exact missing item under
### `provenance.prerequisites`)

1. **A corpus manifest with >= `--documents` entries.** Build one in the
   same shape as `eval/rag/corpus_manifest.jsonl` (id, path, media_type,
   sha256, document_id) over a representative 1,000-document set. Reuse
   `app.evaluation.rag.corpus.deterministic_document_id` for stable IDs if
   you want the manifest to be reproducible across runs.
2. **A reachable ingestion target**: `--conversation-id` plus either
   `--auth-token` or `--email`/`--password`, against a live `--base-url`.
   The ingestion phase reuses `scripts/benchmark_ingestion.py`'s
   `_process_document`/`_login`/`_get_auth_headers` helpers directly --
   see "Overlap with `benchmark_ingestion.py`" below.
3. **A reachable query target**: either `--target MODULE:FUNCTION` (a local
   callable, useful for a smoke test against an in-process app) or the
   `RAG_EVALUATION_TARGET_URL` / `RAG_EVALUATION_BEARER_TOKEN` environment
   variables consumed by `app.evaluation.rag.target.build_http_target`
   (the same mechanism `scripts/evaluate_rag.py --compare-baseline` uses
   for its online run).
4. **A reachable metrics source**: a live `/metrics/rag` endpoint
   (`--metrics-url`, default `<base-url>/metrics/rag`). This is the real
   `rag_stage_duration_seconds` histogram and `rag_cache_operations_total`
   counter exported by `app/observability/rag.py` (Task 12) -- the
   benchmark parses the actual exported buckets with the same
   bucket-interpolation math PromQL's `histogram_quantile()` uses, rather
   than re-timing anything itself.

### Optional infrastructure hooks

These stay `null` / `measured: false` unless wired, and are the correct,
honest default in an environment where the underlying infrastructure query
is deployment-specific and unverified here:

| Flag | Contract | Populates |
|---|---|---|
| `--qdrant-stats-hook MODULE:CALLABLE` | `() -> dict` | `capacity.vector_memory` |
| `--queue-inspector-hook MODULE:CALLABLE` | `() -> dict` | `capacity.queue_saturation` |
| `--cost-fetcher-hook MODULE:CALLABLE` | `() -> float` | `capacity.cost_usd` |
| `--failure-injector-hook MODULE:CALLABLE` | `(kind: str) -> dict` | `failures.results[]` |

`kind` for the failure injector is one of `"provider"`, `"qdrant"`,
`"redis"`, `"worker"`. The hook is responsible for actually inducing that
outage against your infrastructure (e.g. pausing a Docker service,
revoking a credential) and returning what it observed
(`{"injected": bool, "recovered": bool, "recovery_seconds": float}` at
minimum); the script never simulates a failure on your behalf.

### Fields that will stay `null` even on a fully executed run

- `capacity.cost_per_document_usd`, `capacity.cost_per_question_usd`,
  `capacity.cached_input_token_ratio` -- deferred pending a design decision
  from the Task 12 round-1 review on cost/token attribution. There is no
  code path that computes these yet; `validate_report()` raises if anything
  ever writes a number here instead of `null`.
- `quality.tool_iteration_metrics` -- there is no tool-iteration metric in
  this codebase yet. Its owner file is locked by another in-flight session
  as of this writing. `validate_report()` enforces this stays `null`
  rather than silently disappearing or reading as zero.

## Running the embedding-dimension / provider-Batch comparison

```powershell
.\.venv\Scripts\python.exe scripts/experiment_embedding_dimensions.py `
    --dimensions 768 1536 3072 `
    --dataset rag-golden-v1 `
    --dataset-tag v1 `
    --include-provider-batch `
    --experiment-hook your_ops_module:run_dimension_cell `
    --recall-parity-tolerance 0.02 `
    --output artifacts/rag-dimension-matrix.json
```

### Prerequisite: `--experiment-hook`

There is no built-in default hook. Standing up a same-dimension Qdrant
collection, calling Gemini's synchronous embedding endpoint, and submitting
/ polling Gemini's asynchronous embedding Batch API are real,
deployment-specific integration work this environment cannot build or
verify without live credentials and an operational decision on how
per-dimension collections are provisioned -- exactly the kind of decision
the plan's Global Constraints say must follow evaluation results, not
precede them. Without `--experiment-hook`, the script writes a schema-valid
`status: "unexecuted"` report naming this as the missing prerequisite.

Implement the hook with this contract:

```python
def run_dimension_cell(dimension: int, mode: str) -> dict:
    """mode is "synchronous" or "provider_batch"."""
    # 1. Create (or reuse) a Qdrant collection for this dimension.
    # 2. Index eval/rag/corpus_manifest.jsonl's fixtures using `mode`.
    # 3. On the "synchronous" call only, replay eval/rag/golden_v1.jsonl
    #    through app.evaluation.rag.metrics.evaluate_output() and report
    #    the resulting metric averages under "quality" -- indexing mode
    #    does not change embedding content, so measure it once.
    return {
        "documents_indexed": ...,
        "wall_time_s": ...,
        "cost_usd": ...,
        "failures": ...,
        "operational_notes": "...",
        "quality": {"document_recall_at_5": ..., "citation_validity": ...}
        if mode == "synchronous"
        else None,
    }
```

### `--recall-parity-tolerance`: an operator choice, not a shipped default

`comparison.recommendation.selected_dimension` stays `null` unless you pass
`--recall-parity-tolerance`. When you do, the script picks the *smallest*
measured dimension whose `document_recall_at_5` is within that tolerance of
the best measured dimension's, and records the exact numbers it compared
in `recommendation.rationale`. The tolerance itself is never hard-coded in
this script -- you choose it from what the run's own quality deltas show
you, consistent with the plan's constraint that "retrieval thresholds and
chunk sizes are selected from evaluation results, not hard-coded as
universal quality claims."

## Feeding a real result back into `eval/rag/release_gates.json`

Once a run has produced a real number for one of the currently-unmeasured
gates:

1. Open the gate entry (e.g. `p95_stage_latency_ms`).
2. Set `"status": "measured"`.
3. Set `"max_regression"` to a threshold chosen from the run's own results
   (not copied from another project or a general rule of thumb).
4. Update `"provenance.experiment"` to the artifact path or experiment ID
   that produced the number (e.g. `"artifacts/rag-scale-1000.json"`), and
   update or remove `"provenance.note"`.
5. Re-run `tests/test_rag_release_gates.py` to confirm the file still
   parses and validates.

Do this one gate at a time, from one real run at a time. Do not bulk-flip
every `"unmeasured"` gate to `"measured"` from a single benchmark artifact
unless that artifact actually produced every one of those numbers.

## Overlap with `scripts/benchmark_ingestion.py` (flag for Task 15)

`scripts/benchmark_rag.py`'s ingestion phase reuses
`scripts/benchmark_ingestion.py`'s `_process_document`, `_login`, and
`_get_auth_headers` helpers directly rather than re-implementing
upload/poll/percentile logic. The two scripts now substantially overlap in
purpose: `benchmark_ingestion.py` measures per-document upload+index wall
time and throughput against a running server; `benchmark_rag.py` measures
the same thing as one phase of a larger scale-qualification run (which also
covers query quality, stage latency, capacity, and failure recovery).
Task 15 (consolidation) should decide whether `benchmark_ingestion.py`
becomes a thin wrapper around `benchmark_rag.py`'s ingestion phase, or is
retired in its favor -- this task deliberately did not fold one into the
other, to avoid an architectural decision outside its scope.

## Verifying the harness itself (not a benchmark result)

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_benchmark_rag_cli.py tests/test_embedding_dimension_experiment.py tests/test_rag_release_gates.py
```

These tests exercise the CLI, the report schema, and the orchestration
logic entirely through dependency-injected fakes -- they prove the harness
behaves correctly and fails closed; they are not, and do not claim to be,
a RAG benchmark result.
