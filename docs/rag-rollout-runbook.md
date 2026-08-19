# RAG rollout runbook (Task 14)

## Nothing below is ready to enable

Every flag in this document is code-complete, tested, and default-off, and
**no evaluation evidence exists for enabling any of them.** All fourteen
gates in `eval/rag/release_gates.json` are `status: "unmeasured"` with
`max_regression: null` — see "No automated quality gate exists" below. No
1,000-document scale run, dimension comparison, or provider-Batch comparison
has been executed (`docs/rag-scale-runbook.md`).

This document's job is to state, for each flag, **what would have to become
true before it could be flipped** — not to hand anyone a green light or an
activation script. Read every "Evidence still missing" line as a hard
blocker, not a nice-to-have. If you are looking for a sequence of commands
that turns these on, that sequence does not exist yet, and writing one before
the evidence exists would be the wrong thing to do.

Do not begin Task 15 until any generation produced under a changed default in
this rollout has completed its documented rollback window.

## Verified current state of every flag

Read directly from the running `Settings` object (`app/core/config.py`), not
reconstructed from the plan text.

| Setting | Default | Live today? |
|---|---|---|
| `rag_hybrid_retrieval_enabled` | `False` | No |
| `rag_grounded_answer_gate_enabled` | `False` | No |
| `rag_multimodal_image_embeddings_enabled` | `False` | No |
| `rag_exact_cache_enabled` | `False` | No |
| `rag_semantic_chunking_enabled` | `False` | No |
| `enable_citation_verification` | **`True`** | **Yes — shadow validation only** |
| `min_citation_coverage` | `0.5` | Yes, whenever the gate above runs |
| `enable_reranking` | `True` | Yes |
| `langsmith_tracing` | `False` | No (also blocked on quota — see item 2 below) |

## Plan-defect reconciliation: `enable_citation_verification`

The plan's Task 10 Step 4 says to "enable `enable_citation_verification` only
after this path is wired and tested," treating it as a future rollout step.
**That assumption is wrong.** The setting has shipped `default=True` since
Task 10 — it was never off.

What it actually does today: with `rag_grounded_answer_gate_enabled=False`,
setting it `True` makes `RagLoop._apply_grounded_answer_gate`
(`app/ai/workflow/rag_loop.py`) run `GroundedAnswerGate.finalize_answer` in
`mode="shadow"` after every graph `rag_loop` turn. The model's answer is
**never replaced** in this mode; the gate only parses the answer's citations
against the turn's evidence and records the outcome
(`response.metadata["grounded_answer"]`) plus a Prometheus sample
(`rag_grounded_answers_total{mode="shadow", ...}`, see
`app/observability/rag.py:89-98` and `RAGMetrics.grounded_answer()` at
lines 176-181). Flipping it to `False` would delete the only
telemetry this plan has toward ever enabling the gate, so despite reading
like an inert legacy toggle, **it should stay `True`** — the reconciliation
here is documentation, not a code change.

Two caveats that matter for reading its output:

- It only covers the graph `rag_loop` path. The inline-worker RAG path
  (`app/ai/graph.py`, ~line 2090) writes the same `rag_evidence` artifact but
  never reaches this gate — see Blocker 2 below.
- `min_citation_coverage=0.5` is itself an unqualified placeholder (see next
  section), so today's shadow `citation_coverage` numbers are being compared
  against a threshold nobody selected for the quantity it currently measures.

## `min_citation_coverage`: gating the wrong quantity for its own justification

`GroundedAnswerGate.validate` (`app/services/rag_grounding.py:171-196`,
called internally by `GroundedAnswerGate.finalize`) computes
`coverage = covered_claim_count / total_claim_count` — **fraction of the
answer's claims that carry a citation.** The setting's original
justification (and its field description before this task) described a
different quantity: fraction of *retrieved documents* referenced. The number
`0.5` was never re-selected for the quantity it now gates, and it was never
selected from evaluation results in the first place. This violates the
plan's own Global Constraint that "retrieval thresholds and chunk sizes are
selected from evaluation results, not hard-coded as universal quality
claims." Do not change this number without a real coverage-distribution
measurement behind the new value — including "leave it at 0.5, now
deliberately" as a valid outcome if a measurement supports it.

## The rollout order

Nine items, in the order the plan specifies. Three are already shipped and
unconditional (no flag, nothing to roll out); the rest are the default-off
settings from the table above. Each entry states the setting (or "none"),
its status, the evidence still missing, the health signals that would
eventually confirm it, its rollback path, and whether reindexing is required.

### 1. Embedding contract fix

- **Setting:** none — unconditional code, not a flag.
- **Status:** already shipped (Task 2, `app/services/rag_embedding_service.py`,
  commit message "fix: enforce Gemini embedding batch contract"). Every
  embedding call now sends one `Content` per input and validates returned
  vector count/dimension, failing closed on a mismatch instead of silently
  misaligning chunks to vectors.
- **Evidence still missing:** none — this is a correctness fix, not a
  quality tradeoff; nothing about it is gated on evaluation.
- **Health signals:** absence of `Embedding count mismatch` /
  `Embedding dimension mismatch` errors in logs.
- **Rollback:** no flag. Reverting would mean reverting the commit, which
  would reintroduce a real data-corruption bug — do not do this.
- **Reindexing required:** no.

### 2. Evaluation tracing

- **Setting:** `langsmith_tracing` (default `False`) + `LANGSMITH_API_KEY`.
- **Status:** off, and not purely a decision — this account's monthly
  unique-trace quota is currently exhausted (HTTP 429), independent of this
  flag's value.
- **Evidence still missing:** a restored or increased LangSmith quota, or an
  alternate evaluation-tracing budget. Every later item in this order that
  says "requires an evaluation run" is transitively blocked on this one,
  because `scripts/evaluate_rag.py --compare-baseline` is how those runs are
  produced.
- **Health signals:** traces appearing in the `LANGSMITH_PROJECT` project
  without a 429; `scripts/evaluate_rag.py` completing a `--compare-baseline`
  run end to end.
- **Rollback:** `langsmith_tracing=False`. Trivial and safe — this flag only
  controls telemetry export, never RAG behavior.
- **Reindexing required:** no.

### 3. Normalized parsing/chunking for new generations

- **Setting:** `rag_semantic_chunking_enabled` (default `False`).
- **Status:** off. Structural chunking (`DocumentChunkBuilder`,
  token-target/overlap/max from `RAG_CHUNK_*`) is the shipped, unconditional
  default and is unaffected by this flag.
  This flag adds an experimental embedding-based semantic-boundary chunker
  as an alternative; its own field description says "keep disabled until the
  shadow evaluation beats structural chunking."
- **Evidence still missing:** exactly that shadow evaluation — a
  `document_recall_at_5` / `citation_validity` comparison between structural
  and semantic chunking on a real corpus. None has run (see corpus caveat
  under "No automated quality gate exists").
- **Health signals:** the comparison run's recall/citation-validity deltas
  favoring semantic chunking beyond noise, on a corpus larger than the 11
  fixture entries in `eval/rag/corpus_manifest.jsonl`.
- **Rollback:** `rag_semantic_chunking_enabled=False`.
- **Reindexing required:** not for the flag to take effect — chunking is
  decided at ingest/re-ingest time
  (`DocumentIndexService` replaces chunks idempotently by `document_id`), so
  flipping this only changes newly-ingested or re-ingested documents.
  Applying semantic boundaries to the *existing* corpus requires re-running
  ingestion per document — there is no standalone rechunk script — which is
  a real reindexing action if existing-document coverage is wanted.

### 4. Hybrid retrieval

- **Setting:** `rag_hybrid_retrieval_enabled` (default `False`).
- **Status:** off. Rank-fused dense (Qdrant) + PostgreSQL lexical retrieval
  (`rag_dense_candidate_limit=40`, RRF constant from
  `rag_rrf_smoothing_constant`) is fully implemented and tested against
  fakes.
- **Evidence still missing:** a recall/citation-validity comparison against
  dense-only retrieval on a real corpus (same corpus-size caveat as item 3).
- **Health signals:** the comparison run's quality deltas, plus
  `p95_stage_latency_ms` for the added `lexical_retrieval` stage staying
  within whatever bound that comparison run justifies (no bound exists yet
  — `p95_stage_latency_ms` is one of the fourteen unmeasured gates).
- **Rollback:** `rag_hybrid_retrieval_enabled=False`.
- **Reindexing required:** no re-embedding. A schema migration is required:
  Alembic revision `d4e5f6a7b8c9` adds a GIN full-text index
  (`idx_document_chunks_content_simple_fts`) on `document_chunks.content`.
  Confirm this specific migration is actually applied in the target
  environment before flipping the flag — this repo has previously shipped a
  migration that was not yet applied to a live database
  (`chat_images`/`d7e8f9a0b1c2`, Task 4-era); do not repeat that gap here.

### 5. Reranker

- **Setting:** `enable_reranking` (default `True`).
- **Status:** already live, predating this plan. Not a rollout item in the
  sense of the other eight — it is not gated on any of this plan's
  evaluation work and nothing here changes its state.
- **Evidence still missing:** none new. If this plan ever wants to change
  the reranker model (`rag_reranker_model`, currently
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) that would be a separate,
  evidence-gated decision, not covered here.
- **Health signals:** n/a (not a pending decision).
- **Rollback:** `enable_reranking=False` — a real, already-existing switch,
  untouched by this task.
- **Reindexing required:** no.

### 6. Evidence assembly

- **Setting:** none — `EvidenceAssembler.assemble` (`app/services/rag_evidence.py`)
  runs unconditionally on every RAG search, independent of every flag in
  this document.
- **Status:** already live, and it is the origin of Blocker 1 below:
  `_record` assigns `evidence_id=f"E{ordinal}"` per `assemble()` call
  (`app/services/rag_evidence.py:355-377`), so two search calls in one turn
  each produce their own `E1`, `E2`, ... — the ids collide across calls
  within a turn. This is already affecting today's shadow metrics (item 2's
  `enable_citation_verification=True` shadow runs parse against these
  possibly-ambiguous ids right now), not merely a future risk.
- **Evidence still missing:** none needed to observe the defect — it is
  independently verified by two reviewers (Task 14 dispatch context). What's
  missing is the fix itself; see Blocker 1.
- **Health signals:** once fixed, a turn with two or more `search_documents`
  calls should never produce a duplicate `evidence_id` in
  `state["context"]["tool_artifacts"]`.
- **Rollback:** not applicable — no dedicated flag. The practical mitigation
  today is that Blocker 1 is exactly why the grounded gate (item 7) stays
  off; evidence assembly itself cannot be turned off without turning off RAG
  search entirely.
- **Reindexing required:** no.

### 7. Grounded-answer gate

- **Setting:** `rag_grounded_answer_gate_enabled` (default `False`).
- **Status:** off, blocked on three ordered defects — see "Three ordered
  blockers" below. None has cleared.
- **Evidence still missing:** all three blockers, in order; plus, once they
  clear, a real shadow run producing a groundedness signal to threshold
  `min_citation_coverage` against (see that section above).
- **Health signals:** once unblocked, a shadow run with a non-degenerate
  distribution of `citation_coverage` values (not uniformly `0.0`), a
  `rag_grounded_answers_total{outcome="would_abstain"}` rate low enough that
  enforcement wouldn't cause routine regeneration, and zero
  `unknown_evidence_id` reason codes from turns with a single search call
  (a two-search-call turn is expected to still show some until Blocker 1 is
  fixed).
- **Rollback:** `rag_grounded_answer_gate_enabled=False`. Leaves
  `enable_citation_verification=True` shadow recording running.
- **Reindexing required:** no.

### 8. Exact caches

- **Setting:** `rag_exact_cache_enabled` (default `False`).
- **Status:** off. Ignored (cache stays disabled) whenever `redis_url` is
  blank, regardless of this flag's value.
- **Evidence still missing:** `cache_hit_ratio` under real traffic (one of
  the fourteen unmeasured release gates) and a check that cache invalidation
  on re-embedding/re-chunking (index-generation fingerprinting,
  `rag_query_embedding_cache_ttl_seconds` and friends) actually prevents
  stale hits after a document is updated — no live-Redis test of that
  exists yet.
- **Health signals:** `cache_hit_ratio` from `/metrics/rag`
  (`rag_cache_operations_total`) at a level that justifies the operational
  complexity, with zero observed stale-hit incidents after a re-index.
- **Rollback:** `rag_exact_cache_enabled=False`.
- **Reindexing required:** no re-embedding. Confirm `REDIS_URL` is
  configured in the target environment first, or the flag is a no-op.

### 9. Native image embeddings

- **Setting:** `rag_multimodal_image_embeddings_enabled` (default `False`).
- **Status:** off. Caption-augmented text chunks remain the primary
  image-retrieval path.
- **Evidence still missing:** a recall comparison showing raw image
  embeddings retrieve image-grounded questions better than caption-augmented
  text chunks alone, on a real corpus. Also unresolved: the image-slot
  reservation cap in `RagAgent._cap_candidates_with_image_reservation`
  (`app/ai/agents/rag_agent.py:357`) is unexercised at shipped defaults
  (`rag_top_k=15`, `rag_evidence_candidate_limit=10` — the reranker already
  truncates to the evidence limit, so the cap early-returns). It only
  engages if an operator raises `rag_evidence_candidate_limit` above
  `rag_top_k`, or a caller passes a smaller `top_k`. This is now pinned by
  `tests/test_rag_rollout_contract.py::test_image_slot_reservation_cap_is_inactive_at_shipped_defaults`
  — treat a change to that relationship as a signal to add real behavioral
  coverage for the cap before enabling this flag, not just a settings pin.
- **Health signals:** the recall comparison's deltas; the image-reservation
  cap actually exercised and tested at whatever `top_k` /
  `evidence_candidate_limit` combination the target deployment uses.
- **Rollback:** `rag_multimodal_image_embeddings_enabled=False`.
- **Reindexing required:** yes, for existing-document coverage. Raw image
  points are written as additional Qdrant points
  (`modality="image"` in the same collection as text points) at ingest
  time; newly-ingested documents get them automatically once the flag is on,
  but existing documents keep caption-only chunks until a backfill job
  re-embeds their images. No such backfill script exists in this repo yet.

## Three ordered blockers on the grounded-answer gate

These gate item 7 above. They are **ordered, not an unordered checklist** —
Blocker 3 cannot clear until Blocker 1 does.

**Blocker 1 — evidence ids are not turn-unique.**
`EvidenceAssembler._record` (`app/services/rag_evidence.py:355-377`) assigns
`evidence_id=f"E{ordinal}"` per `assemble()` call, not per turn. A turn with
two `search_documents` calls produces two evidence records both labelled
`E1`. The gate's only fail-closed option on an ambiguous id is to drop it,
which under enforcement means regenerate-then-abstain on routine multi-search
turns. Verified independently by two reviewers. **Nothing in this rollout
order can fix this without changing `EvidenceAssembler`'s id-assignment
scope from per-call to per-turn.**

**Blocker 2 — the inline-worker RAG path is ungated, and this task cannot
fix it.**
`app/ai/graph.py` (~line 2090) writes the same `rag_evidence` artifact as the
graph `rag_loop` path, but its final response never reaches
`RagLoop._apply_grounded_answer_gate`. Today this means shadow metrics
undercount (some RAG answers never get a shadow validation at all), and if
the gate were ever enforced, this path would bypass it entirely. **Fixing
this requires routing `app/ai/graph.py`'s inline-worker final response
through the same `RagLoop._apply_grounded_answer_gate` call** the graph
`rag_loop` path already uses — it is not optional for enabling item 7, since
an ungated path would let enforcement be bypassed simply by taking the
inline-worker route.

**Blocker 3 — shadow metrics carry no groundedness signal, and this depends
on Blocker 1.**
With enforcement off, the model is never instructed to cite its claims, so
the dominant shadow outcome is `citation_coverage=0.0` — not because answers
are ungrounded, but because nothing asked for citations. A prior fix
relabelled this outcome `would_abstain` (see `_GROUNDING_OUTCOMES` in
`app/observability/rag.py`), which stops it from being misread as "answers
are failing," but adds no positive signal. It cannot be fixed at the gate
alone: `enable_citation_verification` is already `True`, so adding a citation
instruction under it would change prompts for **all** RAG traffic today, not
just a shadow subset — and even if that were acceptable, the resulting
`citation_coverage` numbers would still be poisoned by Blocker 1's id
collisions on any multi-search turn. **Blocker 3 needs Blocker 1 resolved
first**, then a deliberate decision about how to introduce a citation
instruction (separately gated, evaluated for prompt-quality impact on
non-RAG-gate traffic) before its shadow numbers mean anything.

## No automated quality gate exists behind any of these decisions

`eval/rag/release_gates.json` marks all fourteen gates
`"status": "unmeasured"` with `"max_regression": null`, and
`app/evaluation/rag/release_gates.py`'s `compare_release_gates()` returns
`passed=None, binding=False` for every one of them — every gate is
**non-binding**. There is currently **nothing** a rollout decision could
point to as a passing quality bar.

This is deliberate, not an oversight: Task 13's round-1 review found that
five of these gates (`document_recall_at_5`, `citation_validity`,
`abstention_recall`, `latency_ms`, `cost_usd` — inherited from Task 1) had
been mislabeled `"status": "measured"` with provenance naming a
`"pre-hardening-baseline"` LangSmith experiment that was never actually run
(Task 1's own report records no LangSmith credentials were present when it
tried). Task 13 corrected all five back to `"unmeasured"` with
`max_regression: null`. Do not re-introduce that mistake by hand-editing a
gate to `"measured"` without a real experiment behind
`provenance.source` — see `docs/rag-scale-runbook.md` for the one-gate-at-a-
time procedure for doing this correctly once a real run exists.

The reason no run exists: `eval/rag/corpus_manifest.jsonl` holds 11 fixture
entries, not the 1,000-document corpus the scale harness expects; no live
PostgreSQL, Qdrant, or Redis is available in this environment; and the
LangSmith account's monthly unique-trace quota is exhausted. The harnesses
(`scripts/benchmark_rag.py`, `scripts/experiment_embedding_dimensions.py`)
exist and are tested against fakes — the numbers do not exist.

## Other inherited facts that bear on this rollout

- **Per-model telemetry attribution is incomplete.** `_STAGE_MODELS`
  (`app/observability/rag.py:42`) only enumerates
  `gemini-embedding-2` and `cross-encoder/ms-marco-minilm-l-6-v2`. The
  `generation` and `evidence_assembly` stages report `model="other"` on
  every sample until that allow-list is extended to the models this
  deployment actually uses for those stages. This is a one-line allow-list
  edit, not a code change, but it has not been done, so any health signal
  above that would want per-model breakdown for generation currently
  cannot get one.
- **Tool-iteration metrics do not exist.** Do not look for them in any
  health-signal check above; they were never built.
- **`cost_per_document_usd`, `cost_per_question_usd`, and
  `cached_input_token_ratio` are deferred**, not merely unmeasured — there
  is no code path computing them (Task 12 round-1 review deferred the
  design decision). They must stay `null` in `release_gates.json`, never a
  computed placeholder.
- **No scale qualification exists.** The 1,000-document run, the
  768/1536/3072 embedding-dimension matrix, and the provider-Batch
  comparison are all unexecuted. See `docs/rag-scale-runbook.md` for what
  each one requires before it can run.

## `.env.example` — requires a manual edit by the repository owner

A repository guard blocks `.env*` paths for both shell commands and the
file-editing tools used to write this runbook, so this change could not be
applied directly (attempted once during this task; both the read and the
write were rejected by the guard). **The same guard blocks reviewers,
not just authors: nobody in this process has read the live
`.env.example`, so the block below has not been diffed against its actual
current contents by anyone.** Whoever applies this by hand must open the
real file first and merge by hand — do not append blindly on the assumption
this block is conflict-free. Add it near the existing
`RAG_MULTIMODAL_IMAGE_EMBEDDINGS_ENABLED` line in `.env.example`:

```env
# --- Task 14: default-off RAG rollout flags -----------------------------
# Every flag below is blocked on evaluation evidence that does not exist
# yet. See docs/rag-rollout-runbook.md before changing any of these from
# their shipped defaults -- do not flip one because it "looks done."
RAG_HYBRID_RETRIEVAL_ENABLED=false
RAG_GROUNDED_ANSWER_GATE_ENABLED=false
RAG_EXACT_CACHE_ENABLED=false
RAG_SEMANTIC_CHUNKING_ENABLED=false
# RAG_MULTIMODAL_IMAGE_EMBEDDINGS_ENABLED already exists above this block.
```

Every value above matches the shipped `Settings` default — this is
documentation of the existing default, not a proposed change to it.
