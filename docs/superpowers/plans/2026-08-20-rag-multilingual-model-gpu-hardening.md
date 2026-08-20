# Multilingual RAG Model and GPU Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace legacy RAG embedding/reranker paths with a Gemini Embedding 2-only provider contract and a CUDA/BF16 LAMAR reranker, then qualify English, Japanese, Vietnamese, and secondary-language behavior with measured release evidence.

**Architecture:** The application keeps one production embedding adapter and one production reranker implementation. Gemini input formatting remains inside `GeminiRAGEmbeddingService`; LAMAR loading and bounded fail-open inference remain inside `RAGReranker`; Qwen models are loaded only by an offline benchmark harness. A versioned multilingual dataset and explicit reranker gate evaluator feed the existing LangSmith, dimension, and scale qualification tools without making unmeasured gates appear to pass.

**Tech Stack:** Python 3.12, Pydantic Settings, Google Gen AI SDK, Sentence Transformers 5.6.0, Transformers 4.57.6, PyTorch 2.11.0+cu130, CUDA/BF16, pytest, Ruff, LangSmith, Qdrant, PostgreSQL, Redis.

## Global Constraints

- Execute this plan in a separate git worktree; the main worktree contains user-owned edits in `app/ai/graph.py`, `app/ai/workflow/planning_loop.py`, `app/ai/workflow/rag_loop.py`, `app/ai/workflow/tool_loop.py`, `tests/test_custom_agents_graph.py`, `tests/test_graph_planning_subagents.py`, and `tests/test_hitl_backend_regressions.py`.
- Production supports only `gemini-embedding-2` at an initially configured 3072 dimensions and `nlpai-lab/LAMAR-600m` for reranking.
- Remove `gemini-embedding-001`, local SentenceTransformer embedding, MiniLM reranking, and generic model-family compatibility paths.
- Keep `RAG_EMBEDDING_MODEL` and `RAG_RERANKER_MODEL` as strict single-value deployment contracts; unsupported values fail settings validation.
- Use `RAG_RERANKER_DEVICE=cuda`, `RAG_RERANKER_DTYPE=bfloat16`, batch size 8, maximum pair length 1024, candidate pool 40, timeout 5 seconds, and maximum concurrency 1.
- Never silently fall back from CUDA to CPU. Reranker failure returns the authorized fused order and records a bounded failure code.
- Keep the old golden dataset immutable but non-binding. The new default dataset contains exactly 300 cases: 105 English, 75 Japanese, 75 Vietnamese, and 45 secondary-language cases.
- Secondary cases are evenly distributed across Mandarin Chinese, Korean, Spanish, French, and Indonesian: 9 cases per language.
- LAMAR must improve primary-language macro document nDCG@10 by at least 0.02 versus reranking disabled, regress no primary language by more than 0.01, keep p95 reranking latency at or below 5000 ms, and keep peak allocated VRAM below 12288 MB.
- Normal CI uses deterministic fakes and performs no provider calls, model downloads, or GPU inference. Live provider and GPU tests remain explicitly marked.
- Do not mark an existing release gate measured or binding without a recorded experiment and artifact proving its value.
- Every task uses red-green TDD, focused verification, and an isolated commit. Do not stage unrelated files.

---

## File Structure

- `app/core/config.py`: the only supported embedding/reranker settings and validation.
- `app/core/container.py`: direct Gemini and LAMAR dependency wiring; no provider/model-family branching.
- `app/services/rag_embedding_service.py`: Gemini Embedding 2 formatting, calls, retries, and response validation.
- `app/services/rag_reranker.py`: LAMAR CUDA/BF16 loading, bounded inference, scoring, and fail-open behavior.
- `app/evaluation/rag/corpus.py`: version-aware dataset/manifest validation and evaluation scope seeding.
- `app/evaluation/rag/metrics.py`: deterministic aggregate and per-language retrieval metrics.
- `app/evaluation/rag/reranker_benchmark.py`: benchmark case/report contracts and pure reranker metric calculations.
- `app/evaluation/rag/reranker_gates.py`: the fixed LAMAR model-release gate evaluator.
- `scripts/evaluate_rag.py`: default v2 local/LangSmith evaluation orchestration.
- `scripts/benchmark_rerankers.py`: GPU benchmark entry point; the only place Qwen comparison models are loaded.
- `scripts/benchmark_rag.py`: 1,000-document qualification using the v2 dataset.
- `scripts/experiment_embedding_dimensions.py`: 768/1536/3072 and synchronous/provider-Batch comparison using v2.
- `eval/rag/golden_v2.jsonl`: immutable 300-case multilingual reference dataset.
- `eval/rag/corpus_manifest_v2.jsonl`: hashes and deterministic IDs for v2 fixtures.
- `eval/rag/reranker_release_gates.json`: fixed model-specific quality/latency/VRAM requirements.
- `app/services/rag_evidence.py`, `app/ai/rag_tool_actions.py`, `app/ai/workflow/rag_loop.py`, and `app/ai/graph.py`: turn-unique evidence IDs and one grounded-answer finalization path.
- `docs/rag-rollout-runbook.md`, `docs/rag-scale-runbook.md`, `docs/rag-cleanup-inventory.md`, `README.md`, and `.env.example`: current supported path and honest qualification state.
- `.superpowers/sdd/2026-08-13-rag-production-hardening/progress.md`: audit and qualification ledger.

---

### Task 1: Make Gemini Embedding 2 the Only Embedding Path

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Modify: `app/services/rag_embedding_service.py`
- Modify: `.env.example`
- Modify: `tests/test_rag_embedding_service.py`
- Modify: `tests/test_retrieval_model_selection.py`
- Modify: `tests/test_model_usage_client_retry_config.py`
- Modify: `tests/live/test_gemini_embedding_contract.py`

**Interfaces:**
- Consumes: `RAGEmbeddingService.embed_documents(texts, titles=..., usage_context=...)` and `embed_query(query, usage_context=...)` used by indexing and retrieval.
- Produces: `GeminiRAGEmbeddingService(api_key, model_name="gemini-embedding-2", dimension=3072, embedding_batch_size=32, embedding_max_concurrency=4, recorder=None)` with no provider or task-type compatibility branch.

- [ ] **Step 1: Write failing settings and container tests for a single embedding path**

Update `tests/test_retrieval_model_selection.py` so the contract is exact:

```python
def test_rag_embedding_settings_are_gemini_2_only():
    from app.core.config import Settings

    fields = Settings.model_fields
    assert fields["rag_embedding_model"].default == "gemini-embedding-2"
    assert fields["rag_embedding_dimension"].default == 3072
    assert "rag_embedding_provider" not in fields
    assert "rag_embedding_query_task" not in fields


def test_unsupported_embedding_model_fails_settings_validation(monkeypatch):
    from pydantic import ValidationError
    from app.core.config import Settings

    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
    with pytest.raises(ValidationError, match="gemini-embedding-2"):
        Settings(_env_file=None)
```

Add a source contract asserting `SentenceTransformerRAGEmbeddingService`, `SentenceTransformer(`, and `rag_embedding_provider` are absent from `app/services/rag_embedding_service.py` and `app/core/container.py`.

- [ ] **Step 2: Run the focused tests and verify the legacy fields still fail the contract**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_retrieval_model_selection.py tests/test_rag_embedding_service.py -k "embedding_settings or unsupported_embedding_model or provider"
```

Expected: failures showing that provider/query-task fields and the local SentenceTransformer branch still exist.

- [ ] **Step 3: Write failing Gemini request-payload tests**

Replace the old payload assertions in `tests/test_rag_embedding_service.py` with:

```python
config = client.models.embed_content.call_args.kwargs["config"]
assert config.output_dimensionality == 3072
assert getattr(config, "task_type", None) is None
assert getattr(config, "title", None) is None
```

Assert query text is always `task: search result | query: {query}` and document text is always `title: {title-or-none} | text: {text}`. Remove the configurable-query-task test. Update the live test to construct the service at 3072 dimensions and assert two distinct 3072-value vectors.

- [ ] **Step 4: Run the payload tests and verify they fail on serialized legacy fields**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_embedding_service.py -k "output_dimensionality or prefixes or task_type or title"
```

Expected: the current provider config still exposes `task_type` and singleton `title`.

- [ ] **Step 5: Implement the Gemini-only service and configuration**

In `app/core/config.py`, use a literal single-value contract and remove provider/query-task fields:

```python
rag_embedding_model: Literal["gemini-embedding-2"] = "gemini-embedding-2"
rag_embedding_dimension: int = Field(default=3072, ge=128, le=3072)
```

Delete `SentenceTransformerRAGEmbeddingService`. Fix the query prefix in the service rather than reading a setting. Reduce the retry call chain to:

```python
def _embed_with_retries(
    self,
    *,
    contents: Any,
    expected_count: int,
    operation: UsageOperation | None,
) -> list[list[float]]:
    response = self._run_embed_content(contents=contents, operation=operation)


def _run_embed_content(self, *, contents: Any, operation: UsageOperation | None) -> Any:
    return self.client.models.embed_content(
        model=self.model_name,
        contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=self.dimension),
    )
```

Retain the existing retry loop and usage recording around that minimal request. In the container, instantiate `GeminiRAGEmbeddingService` directly after validating `GEMINI_API_KEY`; remove provider dispatch and local embedding imports. Let `DocumentIndexService` read `provider="gemini"` from the service instead of passing a removed setting.

- [ ] **Step 6: Update environment documentation and run impacted tests**

Remove `RAG_EMBEDDING_PROVIDER` and `RAG_EMBEDDING_QUERY_TASK` from `.env.example`. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_embedding_service.py tests/test_retrieval_model_selection.py tests/test_model_usage_client_retry_config.py tests/test_document_index_service.py tests/test_rag_cache.py
.\.venv\Scripts\python.exe -m ruff check app/core/config.py app/core/container.py app/services/rag_embedding_service.py tests/test_rag_embedding_service.py tests/test_retrieval_model_selection.py tests/live/test_gemini_embedding_contract.py
```

Expected: all pass; normal tests make no external request.

- [ ] **Step 7: Commit the embedding contract cleanup**

```powershell
git add app/core/config.py app/core/container.py app/services/rag_embedding_service.py .env.example tests/test_rag_embedding_service.py tests/test_retrieval_model_selection.py tests/test_model_usage_client_retry_config.py tests/live/test_gemini_embedding_contract.py
git commit -m "fix: enforce Gemini Embedding 2 contract"
```

---

### Task 2: Replace the Reranker with a LAMAR CUDA/BF16 Service

**Files:**
- Modify: `app/services/rag_reranker.py`
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `.env.example`
- Modify: `pyproject.toml`
- Rewrite: `tests/test_rag_reranker.py`
- Modify: `tests/test_retrieval_model_selection.py`
- Modify: `tests/test_rag_agent.py`
- Create: `tests/live/test_lamar_reranker_gpu.py`

**Interfaces:**
- Consumes: `RetrievalCandidate` and existing RAG metrics methods `stage`, `stage_failure`, and `degraded`.
- Produces: `load_lamar_cross_encoder(model_name: str, device: str, dtype: str, max_length: int) -> CrossEncoder`; `RAGReranker(model_name: str, enabled: bool, candidate_pool: int, output_limit: int, timeout_seconds: float, max_concurrency: int, device: str, dtype: str, batch_size: int, max_length: int, model_loader: Callable[[], Any] | None = None, metrics: RerankerMetrics | None = None)`; `RAGReranker.model -> Any | None`; and `RAGReranker.rank(query: str, candidates: Sequence[RetrievalCandidate]) -> list[RetrievalCandidate]`.

- [ ] **Step 1: Replace old settings tests with the exact LAMAR deployment contract**

Write assertions for these defaults and retired fields:

```python
assert fields["rag_reranking_enabled"].default is True
assert fields["rag_reranker_model"].default == "nlpai-lab/LAMAR-600m"
assert fields["rag_reranker_device"].default == "cuda"
assert fields["rag_reranker_dtype"].default == "bfloat16"
assert fields["rag_reranker_batch_size"].default == 8
assert fields["rag_reranker_max_length"].default == 1024
assert fields["rag_rerank_candidate_pool"].default == 40
assert fields["rag_reranker_timeout_seconds"].default == 5.0
assert fields["rag_reranker_max_concurrency"].default == 1
assert {"enable_reranking", "reranker_model", "rerank_top_k"}.isdisjoint(fields)
```

Parametrize invalid settings for a MiniLM model, `device="cpu"`, `dtype="float32"`, zero batch size, and concurrency other than 1; require `ValidationError`.

- [ ] **Step 2: Run the settings tests and verify current MiniLM/alias defaults fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_retrieval_model_selection.py tests/test_rag_reranker.py -k "settings or defaults or invalid"
```

Expected: failures identify MiniLM, old aliases, missing device/dtype/batch/length fields, and concurrency 2.

- [ ] **Step 3: Write failing loader tests for explicit CUDA/BF16 placement**

Use fake `torch` capability and a fake `CrossEncoder` to assert this constructor call:

```python
CrossEncoder(
    "nlpai-lab/LAMAR-600m",
    device="cuda",
    max_length=1024,
    model_kwargs={"torch_dtype": torch.bfloat16},
)
```

The fake exposes `model.parameters()` with one parameter on `cuda` and in BF16. Add separate failures for `torch.cuda.is_available() is False`, a parameter left on CPU, and a non-BF16 parameter. Assert bounded codes `cuda_unavailable`, `model_device_mismatch`, and `model_dtype_mismatch`; exception text must not enter metric labels.

- [ ] **Step 4: Write failing inference tests for LAMAR-specific arguments and OOM**

The fake model must receive:

```python
model.predict(
    [[query, candidate.content], ...],
    batch_size=8,
    show_progress_bar=False,
    convert_to_numpy=True,
)
```

Retain behavioral coverage for pool/output caps, raw finite scores, stable ties, off-event-loop execution, worker-held timeout permits, singleton lazy loading, missing candidate IDs, malformed score counts, and disabled reranking. Add a fake `torch.cuda.OutOfMemoryError` and require fail-open code `cuda_oom`. Change maximum observed concurrency expectation to 1.

- [ ] **Step 5: Run the new loader/inference tests and verify the old generic loader fails them**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_reranker.py
```

Expected: failures for old cache-first loading, missing GPU validation, missing predict kwargs, and missing OOM classification.

- [ ] **Step 6: Rewrite `RAGReranker` around the LAMAR contract**

Keep score application and fail-open telemetry, but replace the loader and constructor defaults. Use explicit bounded failures:

```python
LAMAR_MODEL_ID = "nlpai-lab/LAMAR-600m"
RERANK_SCORE_SEMANTICS = "uncalibrated_model_score"


class _RerankerRuntimeFailure(RuntimeError):
    def __init__(self, failure_code: str) -> None:
        super().__init__(failure_code)
        self.failure_code = failure_code
```

`load_lamar_cross_encoder` validates CUDA before download/load, constructs the model once, then verifies device and dtype from parameters. `RAGReranker._predict` passes batch size and disables progress output. Catch `torch.cuda.OutOfMemoryError` before the generic exception handler and map loader/runtime failures without logging document content. Do not call `torch.cuda.empty_cache()` as a retry strategy and do not retry on CPU.

- [ ] **Step 7: Replace configuration and wiring**

Declare strict literals for model/device/dtype, bounded integers for batch and length, rename `enable_reranking` to `rag_reranking_enabled`, delete the two old aliases, and pass all fields from `Container.rag_reranker` and direct `RAGAgent` construction. Update `_search` to read `self.rag_reranking_enabled` or retain the instance attribute `enable_reranking` only as runtime state, never as a Settings field.

Update `.env.example` with exactly one assignment for each production value. Add a `gpu` pytest marker beside `live_provider` in `pyproject.toml`.

- [ ] **Step 8: Add the opt-in real-GPU contract test**

`tests/live/test_lamar_reranker_gpu.py` must skip unless `RUN_RAG_GPU_TESTS=1`, then assert:

```python
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0)
assert all(parameter.device.type == "cuda" for parameter in reranker.model.model.parameters())
assert all(parameter.dtype == torch.bfloat16 for parameter in reranker.model.model.parameters())
```

Warm the model, rank 40 mixed English/Japanese/Vietnamese passages, synchronize CUDA, and assert 40 scores are produced without exceeding 12288 MB peak allocated memory. Record latency but leave the p95 decision to the benchmark task.

- [ ] **Step 9: Run the focused service, agent, and container suites**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_reranker.py tests/test_rag_agent.py tests/test_retrieval_model_selection.py tests/test_ai_service_initialization.py
.\.venv\Scripts\python.exe -m ruff check app/services/rag_reranker.py app/core/config.py app/core/container.py app/ai/agents/rag_agent.py tests/test_rag_reranker.py tests/live/test_lamar_reranker_gpu.py
```

Expected: all deterministic tests pass; live GPU test is skipped unless explicitly enabled.

- [ ] **Step 10: Commit the LAMAR production replacement**

```powershell
git add app/services/rag_reranker.py app/core/config.py app/core/container.py app/ai/agents/rag_agent.py .env.example pyproject.toml tests/test_rag_reranker.py tests/test_retrieval_model_selection.py tests/test_rag_agent.py tests/live/test_lamar_reranker_gpu.py
git commit -m "feat: run LAMAR reranking on CUDA"
```

---

### Task 3: Remove Retired Model Paths and Operational Claims

**Files:**
- Delete: `scripts/download_reranker.py`
- Modify: `README.md`
- Modify: `docs/rag-cleanup-inventory.md`
- Modify: `docs/rag-rollout-runbook.md`
- Modify: `tests/test_rag_dead_code_cleanup.py`
- Modify: `tests/test_rag_rollout_contract.py`
- Modify: `tests/test_rag_multi_user_isolation.py`

**Interfaces:**
- Consumes: the Gemini-only and LAMAR-only contracts from Tasks 1 and 2.
- Produces: repository-wide absence guarantees for retired settings, scripts, imports, and documentation.

- [ ] **Step 1: Write failing repository scans for retired names**

Add a parameterized scan over `app`, `tests`, `scripts`, `.env.example`, `README.md`, and active RAG runbooks. Exclude historical specs/plans and the immutable progress ledger. Require these active-path patterns to be absent:

```python
RETIRED_PATTERNS = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "SentenceTransformerRAGEmbeddingService",
    "rag_embedding_provider",
    "rag_embedding_query_task",
    "settings.reranker_model",
    "settings.enable_reranking",
    "settings.rerank_top_k",
    "scripts/download_reranker.py",
)
```

Update `tests/test_rag_multi_user_isolation.py` to stop assigning the removed `rerank_top_k` test attribute.

- [ ] **Step 2: Run cleanup tests and verify they find the old script/docs**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_dead_code_cleanup.py tests/test_rag_rollout_contract.py tests/test_rag_multi_user_isolation.py
```

Expected: failures list the download script, MiniLM troubleshooting text, and old rollout setting names.

- [ ] **Step 3: Delete the old script and update active documentation**

Delete `scripts/download_reranker.py`. Replace README troubleshooting with CUDA diagnostics:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Document that first model load may download through standard Hugging Face behavior and that failure degrades to fused results. Rewrite the cleanup inventory row as removed, naming the replacement commit and tests. Update the rollout setting to `rag_reranking_enabled` and describe the new model/GPU gate rather than claiming reranking needs no evidence.

- [ ] **Step 4: Correct the reindex statement**

Change the runbook to state that `scripts/reindex_embeddings.py` re-embeds text chunks in provider batches. State explicitly that it receives no `image_rows`, does not backfill native image embeddings, and therefore does not incur one call per image.

- [ ] **Step 5: Run cleanup, docs contract, and Ruff checks**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_dead_code_cleanup.py tests/test_rag_rollout_contract.py tests/test_rag_multi_user_isolation.py
.\.venv\Scripts\python.exe -m ruff check tests/test_rag_dead_code_cleanup.py tests/test_rag_rollout_contract.py tests/test_rag_multi_user_isolation.py
rg -n "ms-marco-MiniLM|SentenceTransformerRAGEmbeddingService|rag_embedding_provider|rag_embedding_query_task|settings\.reranker_model|settings\.enable_reranking|settings\.rerank_top_k|download_reranker" app tests scripts README.md .env.example docs/rag-rollout-runbook.md docs/rag-cleanup-inventory.md
```

Expected: pytest and Ruff pass; `rg` returns no matches.

- [ ] **Step 6: Commit retired-path removal**

```powershell
git add README.md docs/rag-cleanup-inventory.md docs/rag-rollout-runbook.md tests/test_rag_dead_code_cleanup.py tests/test_rag_rollout_contract.py tests/test_rag_multi_user_isolation.py
git rm scripts/download_reranker.py
git commit -m "refactor: remove retired RAG model paths"
```

---

### Task 4: Add the Fixed Multilingual v2 Dataset

**Files:**
- Create: `eval/rag/fixtures/casebook_en.txt`
- Create: `eval/rag/fixtures/casebook_ja.txt`
- Create: `eval/rag/fixtures/casebook_vi.txt`
- Create: `eval/rag/fixtures/casebook_zh.txt`
- Create: `eval/rag/fixtures/casebook_ko.txt`
- Create: `eval/rag/fixtures/casebook_es.txt`
- Create: `eval/rag/fixtures/casebook_fr.txt`
- Create: `eval/rag/fixtures/casebook_id.txt`
- Create: `eval/rag/corpus_manifest_v2.jsonl`
- Create: `eval/rag/golden_v2.jsonl`
- Modify: `app/evaluation/rag/corpus.py`
- Modify: `tests/test_rag_evaluation_contracts.py`

**Interfaces:**
- Consumes: `load_golden_dataset`, deterministic fixture IDs, and the existing reference schema.
- Produces: `validate_golden_dataset(..., dataset_version="v2")` and a 300-row dataset with explicit `language` and `query_document_mode` metadata.

- [ ] **Step 1: Write failing v2 count, language, and metadata contracts**

Add an exact distribution assertion:

```python
assert Counter(row["metadata"]["language"] for row in rows) == {
    "en": 105,
    "ja": 75,
    "vi": 75,
    "zh": 9,
    "ko": 9,
    "es": 9,
    "fr": 9,
    "id": 9,
}
assert {row["metadata"]["query_document_mode"] for row in rows} == {
    "monolingual",
    "cross_lingual",
    "equivalent_multilingual",
}
```

Require every primary language to cover all ten required categories, every row to have `label_review_status`, and at least one relevance-before-language case per primary language. Keep the v1 test unchanged except for calling the v1 validator explicitly.

- [ ] **Step 2: Run the dataset contract and verify v2 is absent**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evaluation_contracts.py -k "v2 or language or distribution"
```

Expected: failure because v2 fixtures, manifest, and golden rows do not exist.

- [ ] **Step 3: Make corpus validation version-aware**

Add immutable profiles instead of one hard-coded language set:

```python
DATASET_PROFILES = {
    "v1": {"min_rows": 100, "max_rows": 300, "required_languages": {"en", "th", "vi"}},
    "v2": {"min_rows": 300, "max_rows": 300, "required_languages": {"en", "ja", "vi", "zh", "ko", "es", "fr", "id"}},
}
```

Accept `dataset_version: Literal["v1", "v2"] = "v1"` in validation and scope seeding. Derive tenant/user/conversation UUID names from the selected version. Resolve `corpus_manifest_v2.jsonl` for v2 without changing v1 hashes or IDs.

- [ ] **Step 4: Author the multilingual fixtures and manifest**

Each casebook must express the same factual domains—warranty, quarterly revenue, returns conflict, identifiers, visual/chart description, and an embedded prompt-injection sentence—in its own language. Preserve numbers and identifiers exactly across translations so relevance labels are machine-checkable. Compute each SHA-256 from the committed UTF-8 bytes and use `deterministic_document_id(path)` for every manifest ID.

- [ ] **Step 5: Author the 300-row golden dataset**

Every JSONL row has this exact shape:

```json
{"id":"ja-direct_lookup-001","inputs":{"question":"...","user_id":"...","conversation_id":"..."},"reference":{"answer":"...","relevant_document_ids":["..."],"relevant_chunk_ids":[],"relevant_spans":[{"document_id":"...","page_start":1,"page_end":1}],"should_abstain":false},"metadata":{"category":"direct_lookup","language":"ja","document_language":"ja","query_document_mode":"monolingual","fixture_grounded":true,"relevance_before_language":false,"label_review_status":"pending_human_review"}}
```

Use stable IDs, unique questions, and manifest-resolvable document IDs. Cross-lingual rows point to a relevant document in another language. Relevance-before-language rows include an irrelevant query-language distractor in the seeded corpus but label only the relevant other-language document. Do not label generated translations as human-reviewed.

- [ ] **Step 6: Run validation and freeze the dataset**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evaluation_contracts.py
.\.venv\Scripts\python.exe -c "from app.evaluation.rag.corpus import load_corpus_manifest, load_golden_dataset, validate_golden_dataset; rows=load_golden_dataset('eval/rag/golden_v2.jsonl'); manifest=load_corpus_manifest('eval/rag/corpus_manifest_v2.jsonl'); validate_golden_dataset(rows, manifest, dataset_version='v2'); print(len(rows))"
```

Expected: tests pass and the command prints `300`.

- [ ] **Step 7: Commit the v2 corpus and validation**

```powershell
git add app/evaluation/rag/corpus.py tests/test_rag_evaluation_contracts.py eval/rag/golden_v2.jsonl eval/rag/corpus_manifest_v2.jsonl eval/rag/fixtures/casebook_en.txt eval/rag/fixtures/casebook_ja.txt eval/rag/fixtures/casebook_vi.txt eval/rag/fixtures/casebook_zh.txt eval/rag/fixtures/casebook_ko.txt eval/rag/fixtures/casebook_es.txt eval/rag/fixtures/casebook_fr.txt eval/rag/fixtures/casebook_id.txt
git commit -m "test: add multilingual RAG golden dataset"
```

---

### Task 5: Add Per-Language Metrics and Fixed Reranker Gates

**Files:**
- Modify: `app/evaluation/rag/metrics.py`
- Create: `app/evaluation/rag/reranker_gates.py`
- Create: `eval/rag/reranker_release_gates.json`
- Modify: `scripts/evaluate_rag.py`
- Modify: `tests/test_rag_evaluation_metrics.py`
- Modify: `tests/test_rag_evaluation_cli.py`
- Create: `tests/test_reranker_release_gates.py`

**Interfaces:**
- Consumes: per-row deterministic metrics plus row metadata from v2.
- Produces: `summarize_metrics_by_language(rows: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, float]]) -> dict[str, float]` and `evaluate_reranker_gates(baseline: Mapping[str, float], candidate: Mapping[str, float], runtime: Mapping[str, float]) -> list[RerankerGateResult]`.

- [ ] **Step 1: Write failing grouped-metric tests**

For rows in `en`, `ja`, and `vi`, prove that aggregate metrics include:

```python
{
    "en_document_ndcg_at_10": 1.0,
    "ja_document_ndcg_at_10": 0.5,
    "vi_document_ndcg_at_10": 0.75,
    "primary_language_macro_document_ndcg_at_10": 0.75,
}
```

Also require `secondary_language_macro_document_ndcg_at_10` and a `cross_lingual_relevance_pass_rate` computed only from rows with `relevance_before_language=true`.

- [ ] **Step 2: Write failing fixed-gate tests**

Test all boundaries exactly:

```python
baseline = {"primary_language_macro_document_ndcg_at_10": 0.70, "en_document_ndcg_at_10": 0.70, "ja_document_ndcg_at_10": 0.70, "vi_document_ndcg_at_10": 0.70}
candidate = {"primary_language_macro_document_ndcg_at_10": 0.72, "en_document_ndcg_at_10": 0.69, "ja_document_ndcg_at_10": 0.70, "vi_document_ndcg_at_10": 0.73, "cross_lingual_relevance_pass_rate": 1.0}
runtime = {"reranker_p95_ms": 5000.0, "reranker_peak_vram_mb": 12287.99, "cuda_oom_count": 0.0}
assert all(result.passed for result in evaluate_reranker_gates(baseline, candidate, runtime))
```

Add one failure test each for +0.0199 macro improvement, -0.0101 primary regression, cross-lingual pass rate below 1.0, latency above 5000, VRAM at or above 12288, and any OOM.

- [ ] **Step 3: Run the new tests and verify the helpers are missing**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evaluation_metrics.py tests/test_reranker_release_gates.py
```

Expected: import failures for the new grouped metrics and gate module.

- [ ] **Step 4: Implement grouped metrics**

Make row/score association explicit and strict-length checked. Average each metric within a language, then macro-average the three primary languages and five secondary languages so English row count cannot dominate. For relevance-before-language rows, consume a boolean `relevance_before_language_passed` score emitted by the reranker benchmark.

- [ ] **Step 5: Implement a separate fixed reranker gate evaluator**

Use a small immutable result type:

```python
@dataclass(frozen=True)
class RerankerGateResult:
    name: str
    actual: float
    threshold: float
    passed: bool
```

Load exact values from `eval/rag/reranker_release_gates.json`; reject unknown keys, negative tolerances, or missing metrics. This file is separate from `release_gates.json`: its values are approved model requirements, while the original fourteen gates remain evidence-derived and unmeasured until Task 9.

- [ ] **Step 6: Make v2 the evaluation CLI default**

Set `DEFAULT_DATASET="rag-golden-v2"`, `DEFAULT_DATASET_TAG="v2"`, and v2 file paths. Refactor `run_offline` so one deterministic score dictionary stays associated with each row, then append grouped metrics to the summary. Online evaluation validates v2 references and records language/query-mode metadata. Baseline comparison remains blocked while any v2 row is pending human review.

- [ ] **Step 7: Run evaluation tests and offline validation**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evaluation_metrics.py tests/test_rag_evaluation_contracts.py tests/test_rag_evaluation_cli.py tests/test_reranker_release_gates.py
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --offline --target tests.fixtures.rag_eval_target:target
```

Expected: tests pass. If the named fixture target does not yet exist, add `tests/fixtures/rag_eval_target.py` returning a deterministic abstaining output and include it in this task's commit.

- [ ] **Step 8: Commit grouped evaluation and gates**

```powershell
git add app/evaluation/rag/metrics.py app/evaluation/rag/reranker_gates.py eval/rag/reranker_release_gates.json scripts/evaluate_rag.py tests/test_rag_evaluation_metrics.py tests/test_rag_evaluation_cli.py tests/test_reranker_release_gates.py tests/fixtures/rag_eval_target.py
git commit -m "feat: gate multilingual reranker quality"
```

---

### Task 6: Build the GPU Reranker Comparison Harness

**Files:**
- Create: `app/evaluation/rag/reranker_benchmark.py`
- Create: `scripts/benchmark_rerankers.py`
- Create: `tests/test_reranker_benchmark.py`
- Modify: `docs/rag-scale-runbook.md`

**Interfaces:**
- Consumes: an operator hook `MODULE:FUNCTION` returning a sequence of 300 candidate cases from the seeded v2 corpus.
- Produces: a schema-validated JSON report containing quality, per-language metrics, p50/p95/p99 latency, peak VRAM, OOM count, gate verdicts, git SHA, config hash, and provenance.

- [ ] **Step 1: Write failing benchmark case/report schema tests**

Define the hook result shape exactly:

```python
{
    "id": "ja-direct_lookup-001",
    "language": "ja",
    "query_document_mode": "cross_lingual",
    "relevance_before_language": True,
    "query": "...",
    "candidates": [
        {"id": "chunk-a", "text": "...", "language": "en", "relevant": True},
        {"id": "chunk-b", "text": "...", "language": "ja", "relevant": False},
    ],
}
```

Reject duplicate case/candidate IDs, empty candidates, missing primary languages, fewer than 300 cases, non-boolean relevance, or a relevance-before-language case without both a relevant other-language candidate and an irrelevant same-language candidate.

- [ ] **Step 2: Write failing no-fabrication and gate tests**

Without `--case-provider`, require `status="unexecuted"`, null quality/runtime fields, a non-empty prerequisite, and exit code 3. With fake loaders/timers/CUDA stats, require three fixed model cells:

```python
MODEL_IDS = (
    "nlpai-lab/LAMAR-600m",
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
)
```

Only the LAMAR cell receives production gate verdicts.

- [ ] **Step 3: Run tests and verify the benchmark modules are missing**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_reranker_benchmark.py
```

Expected: import or file-not-found failure.

- [ ] **Step 4: Implement pure benchmark metrics and report validation**

In `reranker_benchmark.py`, compute binary-relevance nDCG@10, Recall@10, MRR, stable ranked IDs, and the relevance-before-language invariant. Reject non-finite scores and score-count mismatches. Validate every unexecuted report field is null rather than zero.

- [ ] **Step 5: Implement the CLI with comparison-only Qwen loading**

The script resolves `--case-provider MODULE:FUNCTION`, loads each exact model through `CrossEncoder` on CUDA/BF16, warms once, synchronizes CUDA around every timed run, and records `torch.cuda.max_memory_allocated() / 1024**2`. Use batch size 8 and maximum length 1024 for all models. Catch OOM per model, record it, and continue to the next comparison model; never CPU-fallback.

The application service is not imported for Qwen. Add an explicit source-contract test that Qwen IDs occur in `scripts/benchmark_rerankers.py` and approved docs only, never under `app/services`, `app/core`, or `.env.example`.

- [ ] **Step 6: Add a latency-only 40-candidate smoke mode**

`--smoke` constructs fixed English/Japanese/Vietnamese queries and 40 production-sized passages for GPU plumbing and latency only. It must mark quality and gates unmeasured because synthetic passages cannot qualify model quality.

- [ ] **Step 7: Run deterministic tests and the honest no-hook CLI**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_reranker_benchmark.py tests/test_reranker_release_gates.py
.\.venv\Scripts\python.exe scripts/benchmark_rerankers.py --output artifacts/rag-reranker-unexecuted.json
.\.venv\Scripts\python.exe -m ruff check app/evaluation/rag/reranker_benchmark.py scripts/benchmark_rerankers.py tests/test_reranker_benchmark.py
```

Expected: pytest/Ruff pass; the no-hook command exits 3 and writes an honest unexecuted report.

- [ ] **Step 8: Document the exact live command and commit**

Add to the scale runbook:

```powershell
$env:RUN_RAG_GPU_TESTS='1'
.\.venv\Scripts\python.exe -m pytest -q -m gpu tests/live/test_lamar_reranker_gpu.py
.\.venv\Scripts\python.exe scripts/benchmark_rerankers.py --case-provider my_ops_module:load_v2_candidate_cases --output artifacts/rag-reranker-comparison.json
```

Commit:

```powershell
git add app/evaluation/rag/reranker_benchmark.py scripts/benchmark_rerankers.py tests/test_reranker_benchmark.py docs/rag-scale-runbook.md
git commit -m "feat: benchmark multilingual rerankers on GPU"
```

---

### Task 7: Resolve the Three Grounded-Answer Rollout Blockers

**Files:**
- Modify: `app/services/rag_evidence.py`
- Modify: `app/ai/rag_tool_actions.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `app/ai/workflow/rag_loop.py`
- Modify: `app/ai/graph.py`
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Modify: `tests/test_rag_evidence.py`
- Modify: `tests/test_rag_tool_actions.py`
- Modify: `tests/test_rag_tool_loop_finalization.py`
- Modify: `tests/test_rag_inline_worker_evidence.py`
- Modify: `tests/test_rag_rollout_contract.py`

**Interfaces:**
- Consumes: `EvidenceAssembler`, `execute_search_documents_action`, and `RagLoopMixin._apply_grounded_answer_gate`.
- Produces: `EvidenceAssembler.assemble(..., evidence_id_start: int = 1) -> EvidencePack`, `EvidenceAssembler.assemble_exact(..., evidence_id_start: int = 1) -> EvidencePack`, `execute_search_documents_action(..., evidence_id_start: int = 1) -> tuple[str, str, dict[str, Any]]`, grounded finalization on both graph and inline-worker paths, and a separately controlled shadow citation prompt.

- [ ] **Step 1: Write failing evidence-ID continuation tests**

Add `evidence_id_start: int = 1` to the intended `assemble`/`assemble_exact` contract. Test two packs:

```python
first = assembler.assemble("q1", candidates[:2], max_tokens=500, evidence_id_start=1)
second = assembler.assemble("q2", candidates[2:], max_tokens=500, evidence_id_start=3)
assert [record.evidence_id for record in first.records] == ["E1", "E2"]
assert [record.evidence_id for record in second.records] == ["E3", "E4"]
```

Reject `evidence_id_start < 1` and ensure truncated records use the continued ordinal.

- [ ] **Step 2: Write failing multi-search loop tests**

Drive two `search_documents` calls in one turn through both RagLoop and inline-worker paths. Assert tool-visible evidence blocks and artifacts contain `E1`, then `E2`, with `ambiguous_evidence_id_count == 0` at finalization.

- [ ] **Step 3: Write the failing inline-worker gate test**

Enable the grounded gate, produce an uncited factual inline-worker answer, and spy on `_apply_grounded_answer_gate`. Require one call using local worker messages and local accumulated artifacts; the returned response must contain grounded metadata or enforced abstention. This test must fail on the current bypass.

- [ ] **Step 4: Write failing citation-prompt flag tests**

Introduce `rag_grounded_citation_prompt_enabled=False`. Require citation instructions when either this flag or `rag_grounded_answer_gate_enabled` is true, and no instructions when both are false. Keep enforcement controlled only by `rag_grounded_answer_gate_enabled`.

- [ ] **Step 5: Run blocker tests and verify all three defects are reproduced**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evidence.py tests/test_rag_tool_actions.py tests/test_rag_tool_loop_finalization.py tests/test_rag_inline_worker_evidence.py tests/test_rag_rollout_contract.py -k "ordinal or multi_search or inline_worker or citation_prompt or ambiguous"
```

Expected: duplicate IDs, missing inline finalization, and missing independent prompt flag fail.

- [ ] **Step 6: Implement loop-local evidence ordinals**

Thread `evidence_id_start` through `assemble`, `assemble_exact`, and `execute_search_documents_action`. In each tool loop, initialize `next_evidence_ordinal = 1` at the start of the current user turn, pass it into each search action, then increment by `len(evidence.get("records", ()))`. Do not store the counter in global settings, a process-global variable, or a cross-conversation ContextVar.

- [ ] **Step 7: Route inline-worker final responses through the shared gate**

Before returning the inline RAG worker's final response, construct a local grounding state from the worker's messages and accumulated artifacts:

```python
grounding_state = dict(parent_state)
grounding_state["messages"] = [*worker_messages, *rag_tool_messages]
grounding_state["context"] = {
    **rag_context,
    "tool_artifacts": accumulated_artifacts,
}
response = await self._apply_grounded_answer_gate(
    grounding_state,
    response,
    question=task_prompt,
)
```

Use the existing mixin method rather than copying gate logic into `graph.py`.

- [ ] **Step 8: Implement the separate shadow citation prompt control**

Declare and document `rag_grounded_citation_prompt_enabled=False`. Append `GROUNDED_ANSWER_CITATION_INSTRUCTIONS` when prompt flag or enforcement flag is true. Shadow validation continues under `enable_citation_verification`; the new flag changes only prompting and can be evaluated before enforcement.

- [ ] **Step 9: Run the complete impacted workflow suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evidence.py tests/test_rag_tool_actions.py tests/test_rag_tool_loop_finalization.py tests/test_rag_inline_worker_evidence.py tests/test_rag_grounding.py tests/test_rag_agent.py tests/test_rag_rollout_contract.py tests/test_graph_planning_subagents.py tests/test_hitl_backend_regressions.py
.\.venv\Scripts\python.exe -m ruff check app/services/rag_evidence.py app/ai/rag_tool_actions.py app/ai/agents/rag_agent.py app/ai/workflow/rag_loop.py app/ai/graph.py app/core/config.py tests/test_rag_evidence.py tests/test_rag_tool_actions.py tests/test_rag_tool_loop_finalization.py tests/test_rag_inline_worker_evidence.py
```

Expected: all pass in the isolated worktree. Before integration, compare these files against the user's main-worktree edits and surface conflicts rather than overwriting them.

- [ ] **Step 10: Commit the blocker resolution**

```powershell
git add app/services/rag_evidence.py app/ai/rag_tool_actions.py app/ai/agents/rag_agent.py app/ai/workflow/rag_loop.py app/ai/graph.py app/core/config.py .env.example tests/test_rag_evidence.py tests/test_rag_tool_actions.py tests/test_rag_tool_loop_finalization.py tests/test_rag_inline_worker_evidence.py tests/test_rag_rollout_contract.py
git commit -m "fix: close grounded RAG rollout blockers"
```

---

### Task 8: Point Existing Qualification Harnesses at v2 and Preserve Honest Gates

**Files:**
- Modify: `scripts/benchmark_rag.py`
- Modify: `scripts/experiment_embedding_dimensions.py`
- Modify: `tests/test_benchmark_rag_cli.py`
- Modify: `tests/test_embedding_dimension_experiment.py`
- Modify: `eval/rag/release_gates.json`
- Modify: `tests/test_rag_release_gates.py`
- Modify: `docs/rag-scale-runbook.md`
- Modify: `docs/rag-rollout-runbook.md`

**Interfaces:**
- Consumes: v2 dataset/manifest, existing real-infrastructure hooks, and reranker benchmark artifact.
- Produces: v2-default dimension/Batch/scale reports and structurally present but non-binding original gates until real experiments run.

- [ ] **Step 1: Write failing default-path tests**

Require all three evaluation scripts to default to `golden_v2.jsonl`, `corpus_manifest_v2.jsonl`, dataset `rag-golden-v2`, and tag `v2`. Require help/docstrings to name v2 rather than v1.

- [ ] **Step 2: Write failing release-gate inventory tests**

Keep the fourteen original gates unmeasured. Add provenance links for the new artifacts without fabricating thresholds. Require every original gate to have `status="unmeasured"`, `max_regression=null`, and `provenance.experiment=null` until Task 9 records evidence.

Do not copy fixed reranker thresholds into this file; they live in `reranker_release_gates.json` and are evaluated by the reranker benchmark.

- [ ] **Step 3: Run harness tests and verify v1 defaults fail**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_benchmark_rag_cli.py tests/test_embedding_dimension_experiment.py tests/test_rag_release_gates.py
```

Expected: failures identify the v1 paths and outdated runbook commands.

- [ ] **Step 4: Update defaults and provenance**

Change constants and example commands to v2. Keep report schema anti-fabrication rules unchanged. Add the reranker report path to runbook qualification order, then dimension/Batch, LangSmith, and 1,000-document scale artifacts.

- [ ] **Step 5: Verify honest unexecuted behavior without infrastructure**

```powershell
.\.venv\Scripts\python.exe scripts/experiment_embedding_dimensions.py --include-provider-batch --output artifacts/rag-dimension-unexecuted.json
.\.venv\Scripts\python.exe scripts/benchmark_rag.py --documents 1000 --output artifacts/rag-scale-unexecuted.json
```

Expected: both commands return their documented nonzero unexecuted code, write reports with null measurements, and list missing prerequisites. They must not mutate `release_gates.json`.

- [ ] **Step 6: Run harness tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_benchmark_rag_cli.py tests/test_embedding_dimension_experiment.py tests/test_rag_release_gates.py tests/test_rag_rollout_contract.py
.\.venv\Scripts\python.exe -m ruff check scripts/benchmark_rag.py scripts/experiment_embedding_dimensions.py tests/test_benchmark_rag_cli.py tests/test_embedding_dimension_experiment.py tests/test_rag_release_gates.py
git add scripts/benchmark_rag.py scripts/experiment_embedding_dimensions.py tests/test_benchmark_rag_cli.py tests/test_embedding_dimension_experiment.py eval/rag/release_gates.json tests/test_rag_release_gates.py docs/rag-scale-runbook.md docs/rag-rollout-runbook.md
git commit -m "docs: target RAG qualification at multilingual v2"
```

---

### Task 9: Run Full Verification and Record Production Qualification

**Files:**
- Modify after measured runs only: `eval/rag/release_gates.json`
- Modify: `docs/rag-rollout-runbook.md`
- Modify: `.superpowers/sdd/2026-08-13-rag-production-hardening/progress.md`
- Create locally, normally gitignored: `artifacts/rag-reranker-comparison.json`
- Create locally, normally gitignored: `artifacts/rag-dimension-matrix.json`
- Create locally, normally gitignored: `artifacts/rag-scale-1000.json`

**Interfaces:**
- Consumes: provider credentials, LangSmith access, CUDA-enabled PyTorch, representative 1,000-document corpus, PostgreSQL, Qdrant, Redis, and operator hooks.
- Produces: reproducible experiment names/artifact hashes and a truthful final production-qualification status.

- [ ] **Step 1: Verify the CUDA environment before downloading a model**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import torch; assert torch.__version__ == '2.11.0+cu130'; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Expected: CUDA-enabled PyTorch 2.11.0+cu130 and the RTX 5060 Ti. If the assertion fails, recreate/synchronize the environment from `requirements.txt`; do not change the application to accommodate a CPU-only environment.

- [ ] **Step 2: Run deterministic repository verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_take100_api.py
.\.venv\Scripts\python.exe -m ruff check app tests scripts
.\.venv\Scripts\python.exe -m alembic heads
```

Expected: all in-scope tests pass, Ruff has no findings, and Alembic prints exactly one head `e5f6a7b8c9d0`. Also run exact `pytest -q`; if the ignored Take100 mismatch still fails independently, record its two failures separately without representing them as RAG failures.

- [ ] **Step 3: Run the live Gemini contract**

```powershell
$env:GEMINI_API_KEY='<configured outside git>'
.\.venv\Scripts\python.exe -m pytest -q -m live_provider tests/live/test_gemini_embedding_contract.py
```

Expected: two distinct vectors, each exactly 3072 values, with no unsupported request-field error.

- [ ] **Step 4: Obtain human review of v2 labels**

Export the 300 rows grouped by language/category, have qualified reviewers approve English, Japanese, and Vietnamese labels plus a sampled review of the five secondary languages, then change only approved rows from `pending_human_review` to `reviewed`. Re-run dataset validation and require zero pending rows before any binding LangSmith or reranker comparison.

- [ ] **Step 5: Run the GPU smoke and reranker comparison**

```powershell
$env:RUN_RAG_GPU_TESTS='1'
.\.venv\Scripts\python.exe -m pytest -q -m gpu tests/live/test_lamar_reranker_gpu.py
.\.venv\Scripts\python.exe scripts/benchmark_rerankers.py --case-provider my_ops_module:load_v2_candidate_cases --output artifacts/rag-reranker-comparison.json
```

Expected: the LAMAR cell passes every fixed gate. Qwen cells are comparison evidence only. If LAMAR fails, stop rollout and return to model design; do not switch production configuration automatically.

- [ ] **Step 6: Record the LangSmith v2 baseline and candidate**

```powershell
$env:LANGSMITH_API_KEY='<configured outside git>'
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v2 --dataset-tag v2 --experiment-prefix multilingual-v2-no-rerank
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v2 --dataset-tag v2 --experiment-prefix multilingual-v2-lamar --compare-baseline <recorded-no-rerank-experiment-name>
```

Expected: both experiment names exist and deterministic per-language metrics are recorded. A quota or authentication failure leaves gates unmeasured and the plan incomplete.

- [ ] **Step 7: Run embedding dimension and provider-Batch comparison**

```powershell
.\.venv\Scripts\python.exe scripts/experiment_embedding_dimensions.py --dimensions 768 1536 3072 --dataset rag-golden-v2 --dataset-tag v2 --include-provider-batch --experiment-hook my_ops_module:run_dimension_cell --recall-parity-tolerance 0.01 --output artifacts/rag-dimension-matrix.json
```

Expected: status `executed`, all three dimensions measured, both indexing modes measured, and a recommendation derived from results. If 3072 remains selected, no collection migration follows; a smaller selection requires a separate index-generation rollout before changing the default.

- [ ] **Step 8: Run representative 1,000-document qualification**

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_rag.py --documents 1000 --corpus-manifest <representative-1000-document-manifest> --golden-dataset eval/rag/golden_v2.jsonl --conversation-id <qualification-conversation-uuid> --auth-token <ephemeral-token> --ingest-concurrency 8 --query-concurrency 32 --failure-injection --qdrant-stats-hook my_ops_module:qdrant_stats --queue-inspector-hook my_ops_module:queue_stats --cost-fetcher-hook my_ops_module:cost_total --failure-injector-hook my_ops_module:inject_failure --output artifacts/rag-scale-1000.json
```

Expected: status `executed`; 1,000 documents available and indexed; quality, latency, capacity, cache, and failure sections measured rather than null.

- [ ] **Step 9: Promote only evidence-backed original release gates**

For each of the fourteen entries, copy the selected threshold and exact experiment/artifact provenance only when its source metric exists. Set `status="measured"` and non-null `max_regression` only for those entries. Leave deferred cost/cached-token gates unmeasured if their instrumentation still cannot produce the metric. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_release_gates.py
```

Update the test that currently requires all gates unmeasured so it instead verifies every measured gate names a real experiment or artifact hash and every remaining unmeasured gate stays non-binding.

- [ ] **Step 10: Update the rollout decision and progress ledger**

Record commands, timestamps, git SHA, LangSmith experiment names, artifact SHA-256 values, per-language gate verdicts, CUDA device/version, and unresolved gates. Change project status to production-qualified only if all completion criteria in the approved design are satisfied. Otherwise write the exact remaining blockers and keep status `implementation complete, production qualification pending`.

- [ ] **Step 11: Run final verification after provenance edits**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_embedding_service.py tests/test_rag_reranker.py tests/test_rag_evaluation_contracts.py tests/test_rag_evaluation_metrics.py tests/test_reranker_release_gates.py tests/test_reranker_benchmark.py tests/test_rag_release_gates.py tests/test_rag_rollout_contract.py tests/test_benchmark_rag_cli.py tests/test_embedding_dimension_experiment.py
.\.venv\Scripts\python.exe -m ruff check app tests scripts
git diff --check
```

Expected: all focused tests and Ruff pass; diff check is clean.

- [ ] **Step 12: Commit measured qualification evidence**

```powershell
git add eval/rag/golden_v2.jsonl eval/rag/release_gates.json docs/rag-rollout-runbook.md .superpowers/sdd/2026-08-13-rag-production-hardening/progress.md tests/test_rag_release_gates.py
git commit -m "docs: record multilingual RAG qualification"
```

Do not add secrets, bearer tokens, raw document content, or unredacted infrastructure output. If no gate was promoted because external prerequisites remain unavailable, commit only the honest ledger/runbook update and do not claim production completion.

---

## Final Review Checklist

- [ ] `rg` finds no active MiniLM, Embedding 1, local embedding provider, old reranker aliases, or unsupported Gemini request fields.
- [ ] Production code contains no Qwen identifiers or generic model-family dispatch.
- [ ] Gemini live test returns two distinct 3072-dimensional vectors.
- [ ] LAMAR model parameters are CUDA/BF16 on the RTX 5060 Ti.
- [ ] The 300-case v2 dataset has the exact approved language distribution and no pending label reviews before binding comparisons.
- [ ] English, Japanese, and Vietnamese metrics are reported separately and all fixed LAMAR gates pass.
- [ ] Relevant cross-language evidence outranks irrelevant same-language evidence in every invariant case.
- [ ] Reranker timeout, OOM, load, device, dtype, malformed-score, and CUDA-unavailable failures preserve fused retrieval order.
- [ ] Evidence IDs are unique across every search in a turn and inline-worker answers pass through the grounded gate.
- [ ] The reindex runbook states that the existing CLI re-embeds text chunks, not images.
- [ ] Original release gates are binding only when backed by recorded experiment/artifact provenance.
- [ ] The 1,000-document report is executed rather than synthetic before production qualification is claimed.
- [ ] Full in-scope pytest, Ruff, `git diff --check`, and the single Alembic head check pass.
