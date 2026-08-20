# Multilingual RAG Model and GPU Hardening Design

**Date:** 2026-08-20

**Status:** Approved in conversation; pending written-spec review

## Objective

Correct the RAG embedding provider contract, replace the existing English-focused
MiniLM reranker with one GPU-native multilingual reranking path, and qualify that
path for English, Japanese, and Vietnamese while retaining broad but secondary
support for other languages.

This is a follow-on to the 15-task RAG production-hardening plan. It does not
reimplement the completed mechanics from that plan. It closes the model-choice,
provider-contract, language-evaluation, GPU-runtime, and remaining production
qualification gaps found during the post-plan audit.

## Scope

This work includes:

- making Gemini Embedding 2 the only supported embedding provider model;
- removing Embedding 1 compatibility code and configuration;
- replacing the current generic/MiniLM reranker implementation with a focused
  LAMAR GPU implementation;
- removing superseded reranker code, settings, tests, and documentation;
- adding explicit CUDA and BF16 runtime controls for the RTX 5060 Ti;
- creating a mixed-language evaluation dataset led by English, Japanese, and
  Vietnamese;
- comparing LAMAR with modern rerankers offline without adding production model
  compatibility paths;
- measuring the existing RAG release gates and completing outstanding live and
  scale qualification work; and
- correcting inaccurate operational documentation discovered during the audit.

This work does not change PostgreSQL/Qdrant ownership, retrieval fusion,
authorization hydration, evidence assembly, citation behavior, or the external
RAG tool contract. It also does not introduce a locally hosted embedding model,
quantized reranker, vLLM service, or automatic CPU fallback.

## Model Decision

### Embedding model

`gemini-embedding-2` remains the embedding model. It is the current stable Gemini
embedding model, supports multilingual and multimodal retrieval, and provides a
shared vector space for the text and image work already implemented by the
production-hardening plan.

The initial output dimension remains 3072. Changing dimensions requires a new
Qdrant collection generation and a complete reindex, so 1536 and 768 dimensions
are experiments rather than assumptions. A smaller dimension may become the
production value only after the existing quality, storage, and latency comparison
is measured.

Embedding dimension is not model parameter count. There is no requirement that a
reranker have more parameters than an embedding provider model. The two stages
perform different computations: the embedding model creates reusable vectors,
while the reranker jointly scores a small set of query-document pairs.

### Production reranker

`nlpai-lab/LAMAR-600m` is the selected production target. It is a 600-million
parameter multilingual cross-encoder released in July 2026 under the MIT license.
Its training languages include English, Japanese, and Vietnamese, and its design
explicitly combines semantic relevance with query-document language coherence.
It exposes the `sentence_transformers.CrossEncoder` interface already used at the
RAG service boundary.

The old BGE recommendation was a conservative compatibility choice: permissive
license, multilingual training, conventional cross-encoder architecture, and
known deployment behavior. It remains useful background evidence, but it is not
the target model and no BGE compatibility path remains in production.

### Offline comparison models

The qualification report compares LAMAR against:

- `Qwen/Qwen3-Reranker-0.6B`, the modern, broadly adopted 100+-language
  efficiency reference; and
- `Qwen/Qwen3-Reranker-4B`, the higher-quality and higher-latency reference that
  can fit the 16 GB GPU only with a conservative batch and memory budget.

These models are benchmark fixtures, not selectable production backends. The
application will not contain Qwen-specific prompts, causal-reranker adapters, or
generic model-family dispatch. If LAMAR fails a mandatory release gate, rollout
stops and the model decision returns to design review.

The offline benchmark runner receives comparison model IDs as experiment input;
it does not read or mutate production application settings.

`jina-reranker-v3.5` is excluded because its CC BY-NC license and remote custom
model code are unsuitable defaults for an on-premises commercial deployment.
`zerank-2-reranker` is excluded as the production target because its published
positioning is English-focused and its 4B size adds latency without evidence of
better Japanese or Vietnamese results for this corpus.

## Embedding Provider Contract

Gemini Embedding 2 is an instruction-in-text API. It does not accept the legacy
`task_type` field. The service uses these asymmetric retrieval formats:

- query: `task: search result | query: {content}`
- document: `title: {title-or-none} | text: {content}`

The provider request contains only the separately wrapped content inputs and
`output_dimensionality`. It does not send `task_type` or a provider-level
`title`. The document title remains part of the formatted text because that is
the Gemini Embedding 2 document contract.

The implementation removes all `gemini-embedding-001` branches, task-type
settings, task-type validation, and compatibility tests. The
`RAG_EMBEDDING_MODEL` environment contract remains and accepts only
`gemini-embedding-2`; unsupported values fail configuration validation rather
than selecting a legacy request path.

Every request continues to validate response vector count, ordering, and exact
dimension. The existing query/document prompt version remains part of cache and
index identity so a formatting change cannot silently mix incompatible vectors.

## Reranker Architecture

The existing generic/MiniLM reranker implementation is replaced rather than
extended. Superseded model defaults, generic compatibility branches, custom
local-cache retry behavior, old loader assumptions, and implementation-specific
tests are removed.

The replacement retains only the public service behavior required by retrieval:

1. receive a query and the fused, authorized candidate list;
2. construct query-document pairs for at most 40 candidates;
3. score them with LAMAR off the async event loop;
4. return candidates in descending score order with stable input-order tie
   handling; and
5. return the unchanged fused order when reranking is unavailable.

LAMAR is loaded lazily through `CrossEncoder`. Loading relies on the standard
Hugging Face cache/download behavior; the application does not implement a
second local-only/network fallback path. A single bounded GPU worker prevents
concurrent requests from oversubscribing VRAM. The first production settings
are:

```dotenv
RAG_RERANKING_ENABLED=true
RAG_RERANKER_MODEL=nlpai-lab/LAMAR-600m
RAG_RERANKER_DEVICE=cuda
RAG_RERANKER_DTYPE=bfloat16
RAG_RERANKER_BATCH_SIZE=8
RAG_RERANKER_MAX_LENGTH=1024
RAG_RERANKER_CANDIDATE_POOL=40
RAG_RERANKER_TIMEOUT_SECONDS=5
RAG_RERANKER_MAX_CONCURRENCY=1
```

These settings are deployment controls, not compatibility switches. Device must
be `cuda`, dtype must be `bfloat16`, and the model must be the selected LAMAR
model. Batch size, maximum length, pool size, timeout, and concurrency remain
bounded tunables because they directly control latency and memory use.

The 1024-token pair limit is the initial production limit for the existing
bounded chunk sizes. The benchmark records truncation rates; increasing it
requires measured evidence that the quality gain fits the latency and VRAM
budgets.

## GPU Runtime and Failure Behavior

The project environment must use the repository's CUDA-enabled PyTorch build.
The currently observed CPU-only PyTorch installation is environment drift and
must be corrected before the live GPU qualification.

When reranking is enabled, the service verifies that CUDA is available, that the
model parameters are on the configured CUDA device, and that inference uses
BF16. It never silently moves the model or request to CPU. Silent CPU fallback
would make request latency unbounded relative to the production SLO.

Model download/load failure, CUDA unavailability, timeout, CUDA OOM, malformed
scores, or inference failure produces a structured degraded reason and returns
the original fused candidates. Logs and metrics identify the failure class but
do not include document text. An OOM does not trigger a CPU retry. Disabling
reranking is the rollback path; the retired MiniLM implementation is not a
rollback dependency.

## Multilingual Evaluation Dataset

A new 300-case dataset becomes the default model and release evaluation set:

- 105 English cases (35%);
- 75 Japanese cases (25%);
- 75 Vietnamese cases (25%); and
- 45 secondary-language cases (15%), distributed across Mandarin Chinese,
  Korean, Spanish, French, and Indonesian.

Cases are stratified across direct lookup, exact identifiers and numbers,
tables, summaries, multi-hop/cross-document questions, conflicting sources,
unanswerable questions, near-duplicate distractors, visual evidence, and prompt
injection. Each primary language appears across the major question categories
rather than occupying a single category.

The dataset includes monolingual retrieval, cross-lingual query-document pairs,
and controlled groups containing semantically equivalent documents in several
languages. Language coherence is never allowed to replace relevance: a relevant
document in another language must rank above an irrelevant document in the
query language. Same-language preference is evaluated only among comparably
relevant evidence.

The prior dataset remains immutable historical evidence but is not an active
release gate. No language is inferred from a person's name, locale, or timezone.
Language is explicit dataset metadata.

## Model Release Gates

LAMAR may be enabled for production traffic only when the fixed dataset and the
target GPU show all of the following:

- primary-language macro nDCG@10 improves by at least 2.0 percentage points over
  the same retrieval pipeline with reranking disabled;
- English, Japanese, and Vietnamese nDCG@10 individually regress by no more than
  1.0 point;
- cross-lingual relevance cases pass the relevance-before-language invariant;
- reranking 40 production-sized candidates meets a five-second p95 latency
  budget;
- peak allocated GPU memory remains below 12 GB, preserving at least 4 GB of
  device headroom;
- repeated load and inference runs produce no CUDA OOM;
- timeout and injected failure tests preserve fused retrieval results; and
- all pre-existing RAG release gates are measured and pass their declared
  thresholds.

The report includes per-language Recall@k, MRR, nDCG@10, truncation rate,
reranker failure rate, p50/p95/p99 latency, throughput, and peak VRAM. Aggregate
scores never hide a failing primary language. Qwen results inform the decision
record but cannot automatically change the production model.

## Testing Strategy

Implementation follows test-driven development. Required focused tests cover:

- configuration acceptance of the single supported embedding and reranker
  models and rejection of legacy or unsupported values;
- absence of the removed task-type settings and old MiniLM defaults;
- Gemini Embedding 2 serialization with separate `Content` inputs,
  `output_dimensionality`, and no `task_type` or provider-level `title`;
- exact vector count and dimension validation;
- LAMAR loading with CUDA/BF16 constructor arguments;
- batch size, pair-length, candidate-pool, and stable-order behavior;
- off-event-loop execution and single-worker concurrency bounding;
- CUDA unavailable, timeout, OOM, malformed-score, and load-failure degradation;
- absence of the retired reranker implementation, settings, imports, and tests;
  and
- environment-example and runbook agreement with executable settings.

A separately marked live Gemini test embeds at least two distinct inputs and
asserts distinct 3072-dimensional vectors. A real-GPU test on the RTX 5060 Ti
asserts CUDA availability, BF16 model parameters, successful 40-candidate
inference, latency percentiles, and peak memory. Normal CI remains deterministic
and does not require model downloads, provider credentials, or a GPU.

The final verification includes the focused suite, complete repository pytest
run with unrelated known failures reported separately, Ruff, Alembic's single
head check, evaluation-schema validation, offline model comparison, and the live
provider/GPU qualifications.

## Relationship to the 15-Task Hardening Plan

The audit found that the 15 implementation tasks and their review fixes are
present, but the program is not production-qualified merely because its unit
tests pass. This follow-on must record evidence for the remaining completion
work:

- live Gemini multi-input provider contract;
- LangSmith baseline experiment on the new fixed dataset;
- multilingual reranker comparison and release-gate measurement;
- 768/1536/3072 embedding-dimension comparison;
- synchronous versus provider Batch embedding cost/throughput comparison;
- representative 1,000-document load qualification;
- resolution and measurement of the documented grounded-answer gate blockers;
  and
- population of every currently unmeasured release gate with evidence and a
  binding pass/fail result.

Until these items pass, the correct project status is “implementation complete,
production qualification pending.” The implementation plan must not mark the
original program complete by substituting unit-test success for these live,
quality, and scale results.

## Documentation and Cleanup

The environment example and operational runbook describe only the supported
Gemini Embedding 2 and LAMAR CUDA path. References to MiniLM, generic reranker
selection, embedding task types, Embedding 1 compatibility, or silent CPU
fallback are removed.

The rollout runbook is corrected to state that the current document reindex
command re-embeds text chunks in batches. It does not claim that the command
re-embeds images unless image rows are actually supplied through the native
multimodal indexing path.

Cleanup retains operational failure/audit telemetry and the public reranking
service contract, but it does not preserve retired implementation details for
compatibility.

## Rollout and Rollback

1. Correct the embedding request contract and remove legacy embedding settings.
2. Replace the reranker with the LAMAR-specific CUDA/BF16 implementation and
   complete deterministic failure-path tests.
3. Repair the local CUDA PyTorch environment and run the real-GPU smoke and load
   tests.
4. Build and validate the fixed multilingual dataset.
5. Run retrieval-without-reranking, LAMAR, and offline Qwen comparison
   experiments using identical candidate inputs.
6. Populate the release gates and enable LAMAR only if every mandatory gate
   passes.
7. Complete the remaining live provider, dimension, Batch API, grounded-answer,
   and 1,000-document qualification work from the original plan.
8. Remove superseded documentation and confirm repository-wide absence of the
   retired model paths.

At runtime, rollback disables reranking and restores fused retrieval ordering.
Embedding rollback is not supported because legacy embedding spaces are not
retained; a future embedding-model change requires a new index generation and
an explicit migration design.

## Completion Criteria

This follow-on is complete only when:

- production configuration exposes no legacy embedding or MiniLM behavior;
- Gemini Embedding 2 requests follow the current provider contract and pass the
  live multi-input test;
- LAMAR executes on the RTX 5060 Ti with CUDA/BF16 and passes all model gates;
- English, Japanese, and Vietnamese have individually reported passing quality
  results, with secondary-language results retained in the report;
- reranker failure preserves authorized fused retrieval without CPU fallback;
- obsolete code, configuration, tests, and documentation are absent;
- every remaining original-plan qualification item has measured evidence; and
- the repository's focused and full static/test/database checks pass, with any
  unrelated pre-existing failure explicitly separated from this work.

## References

- Google Gemini Embedding 2 model:
  <https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2>
- Google Gemini embeddings task and dimension guidance:
  <https://ai.google.dev/gemini-api/docs/embeddings>
- LAMAR model card: <https://huggingface.co/nlpai-lab/LAMAR-600m>
- LAMAR paper: <https://arxiv.org/abs/2607.22042>
- Qwen3 Reranker 0.6B:
  <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>
- Qwen3 Reranker 4B:
  <https://huggingface.co/Qwen/Qwen3-Reranker-4B>
- Jina Reranker v3.5:
  <https://huggingface.co/jinaai/jina-reranker-v3.5>
- ZeRank 2: <https://huggingface.co/zeroentropy/zerank-2-reranker>
