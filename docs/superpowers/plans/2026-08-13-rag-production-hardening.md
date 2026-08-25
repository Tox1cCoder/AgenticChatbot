# RAG Production Hardening Implementation Plan

> **Execution status — all fifteen tasks were implemented and reviewed.** The
> step checkboxes below were never ticked during execution; the authoritative
> record is `.superpowers/sdd/2026-08-13-rag-production-hardening/progress.md`,
> which carries every task's commit range, its review outcome, and the findings
> that were parked rather than fixed. Do not read an unticked box as unstarted
> work. The evaluation-dependent half of the Final Acceptance Gate
> (`scripts/evaluate_rag.py`, `scripts/benchmark_rag.py`) was never run: the
> corpus manifest holds 11 documents, not 1,000, and all fourteen release gates
> are recorded `status: unmeasured`. Instructions superseded after the plan was
> written are annotated inline as **PLAN CORRECTION**.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing PostgreSQL/Qdrant RAG pipeline correct, measurable, grounded, multimodal, and qualified for at least 1,000 representative documents.

**Architecture:** Parsing produces typed structural blocks once; a single chunk builder feeds versioned SQL/Qdrant index generations. A typed hybrid retriever, bounded reranker, evidence assembler, and grounded-answer gate replace dictionary-shaped retrieval and prompt-only citation behavior. LangSmith remains the experiment runner, selected RAGAS metrics run as optional evaluators, and Redis provides only exact, generation-scoped caches.

**Tech Stack:** Python 3.10+, FastAPI, Celery, SQLAlchemy 2, PostgreSQL, Alembic, Qdrant Client 1.18+, Google Gen AI SDK 2.13+, Gemini Embedding 2, LangChain/LangGraph, LangSmith, optional RAGAS Collections API, Redis, Prometheus, pytest, Ruff.

## Global Constraints

- PostgreSQL remains canonical for document text, authorization-sensitive metadata, chunks, images, parse artifacts, and index state.
- Qdrant contains retrieval vectors and non-sensitive lookup/filter metadata.
- Every Qdrant result is re-authorized while hydrating canonical SQL records.
- New behavior is introduced behind independently reversible settings.
- Existing indexed documents remain readable while a replacement index is built; a failed reindex never destroys the prior active generation.
- Tests use deterministic local fakes by default. A separately marked live provider contract test verifies Gemini batch response shape.
- Retrieval thresholds and chunk sizes are selected from evaluation results, not hard-coded as universal quality claims.
- Reranking and other CPU-bound model calls must not block the async event loop.
- Document content, captions, OCR, filenames, and parser output are untrusted reference data and never become executable instructions.
- Tenant scope is part of every cache key and every retrieval operation.
- Cleanup may remove a path only after a repository-wide usage inventory and replacement test prove it is unused. Operational error/audit logs and public API contracts are retained unless an explicit replacement exists.
- Every helper name used in a test snippet is implemented as a deterministic fixture or factory in that same test module; expected values are not calculated by the production function under test.

## File and Responsibility Map

- `app/services/document_blocks.py`: parser-independent block and built-chunk value types.
- `app/services/document_normalizer.py`: MinerU, text, and Excel output normalization only.
- `app/services/document_chunk_builder.py`: the sole structural/token chunking implementation.
- `app/services/rag_embedding_service.py`: provider calls, response validation, retries, and image embeddings.
- `app/services/document_index_service.py`: inactive generation creation, vector writes, verification, and activation.
- `app/services/rag_retrieval.py`: scoped dense/lexical retrieval, RRF, SQL hydration, and typed candidates.
- `app/services/rag_reranker.py`: bounded off-event-loop cross-encoder inference.
- `app/services/rag_evidence.py`: deduplication, context expansion, evidence IDs, and token budgeting.
- `app/services/rag_grounding.py`: structured answer validation, one regeneration, and abstention.
- `app/services/rag_image_selector.py`: bounded image hydration and vision-budget enforcement.
- `app/services/rag_cache.py`: exact Redis cache keys and generation-based invalidation.
- `app/observability/rag.py`: content-free stage, cache, degradation, latency, and cost telemetry.
- `app/evaluation/rag/`: dataset contracts, deterministic metrics, RAGAS adapters, targets, and release gates.
- `scripts/evaluate_rag.py` and `scripts/benchmark_rag.py`: offline quality and scale entry points.

---

### Task 1: Establish the evaluation contract and baseline

**Files:**
- Create: `app/evaluation/__init__.py`
- Create: `app/evaluation/rag/__init__.py`
- Create: `app/evaluation/rag/contracts.py`
- Create: `app/evaluation/rag/metrics.py`
- Create: `app/evaluation/rag/ragas_metrics.py`
- Create: `app/evaluation/rag/target.py`
- Create: `app/evaluation/rag/release_gates.py`
- Create: `app/evaluation/rag/corpus.py`
- Create: `eval/rag/corpus_manifest.jsonl`
- Create: `eval/rag/golden_v1.jsonl`
- Create: `eval/rag/release_gates.json`
- Create: `scripts/evaluate_rag.py`
- Create: `tests/test_rag_evaluation_contracts.py`
- Create: `tests/test_rag_evaluation_metrics.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: authenticated RAG target calls and gold document/chunk/page/evidence identifiers.
- Produces: `RAGEvaluationInput`, `RAGEvaluationReference`, `RAGEvaluationOutput`, deterministic evaluator functions, lazy RAGAS evaluators, and a LangSmith experiment CLI.

- [ ] **Step 1: Write failing contract and metric tests**

```python
def test_recall_at_k_uses_stable_document_ids():
    actual = [
        RetrievalTrace(document_id="doc-b", chunk_id="chunk-b", rank=1, score=0.8),
        RetrievalTrace(document_id="doc-a", chunk_id="chunk-a", rank=2, score=0.7),
    ]
    reference = RAGEvaluationReference(relevant_document_ids={"doc-a"})
    assert retrieval_metrics(actual, reference, ks=(1, 2))["document_recall_at_1"] == 0.0
    assert retrieval_metrics(actual, reference, ks=(1, 2))["document_recall_at_2"] == 1.0


def test_citation_metrics_reject_unknown_evidence_ids():
    output = RAGEvaluationOutput(
        answer="Revenue increased [E1] and margin improved [E9].",
        abstained=False,
        candidates=(),
        evidence=(EvidenceTrace("E1", "doc-a", "chunk-a", 1, 1),),
        claims=(ClaimTrace("Revenue increased.", ("E1",)), ClaimTrace("Margin improved.", ("E9",))),
        citations_valid=False,
        tool_trajectory=("search_chunks",),
        stage_ms={},
        input_tokens=100,
        output_tokens=20,
        cost_usd=None,
    )
    scores = citation_metrics(output)
    assert scores["citation_validity"] == 0.5
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evaluation_contracts.py tests/test_rag_evaluation_metrics.py
```

Expected: collection errors because the evaluation package does not exist.

- [ ] **Step 3: Implement typed outputs and deterministic metrics**

Use stable identifiers rather than Qdrant ranks:

```python
@dataclass(frozen=True)
class RAGEvaluationInput:
    question: str
    user_id: str
    conversation_id: str


@dataclass(frozen=True)
class RelevantSpan:
    document_id: str
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class RAGEvaluationReference:
    answer: str | None = None
    relevant_document_ids: frozenset[str] = frozenset()
    relevant_chunk_ids: frozenset[str] = frozenset()
    relevant_spans: tuple[RelevantSpan, ...] = ()
    should_abstain: bool = False


@dataclass(frozen=True)
class RetrievalTrace:
    document_id: str
    chunk_id: str | None
    rank: int
    score: float | None


@dataclass(frozen=True)
class EvidenceTrace:
    evidence_id: str
    document_id: str
    chunk_id: str | None
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class ClaimTrace:
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class RAGEvaluationOutput:
    answer: str
    abstained: bool
    candidates: tuple[RetrievalTrace, ...]
    evidence: tuple[EvidenceTrace, ...]
    claims: tuple[ClaimTrace, ...]
    citations_valid: bool
    tool_trajectory: tuple[str, ...]
    stage_ms: Mapping[str, float]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
```

Implement document/chunk Recall@k, hit rate, reciprocal rank, nDCG, citation validity, citation precision/recall, claim citation coverage, abstention precision/recall, tool counts, latency, tokens, and cost as pure functions.

- [ ] **Step 4: Add the LangSmith target, lazy RAGAS adapter, and release-gate comparator**

`scripts/evaluate_rag.py` must expose `--dataset`, `--dataset-tag`, `--experiment-prefix`, `--offline`, `--with-ragas`, and `--compare-baseline`. Keep RAGAS out of the server dependency set:

```toml
[project.optional-dependencies]
eval = [
    "ragas>=0.3,<1.0",
]
```

```python
results = client.evaluate(
    target,
    data=client.list_examples(dataset_name=args.dataset, as_of=args.dataset_tag),
    evaluators=deterministic_evaluators + ragas_evaluators,
    experiment_prefix=args.experiment_prefix,
    upload_results=not args.offline,
)
```

The RAGAS adapter uses `ragas.metrics.collections` and exposes context precision/recall, noise sensitivity, faithfulness, response relevance, multimodal faithfulness, and multimodal relevance. It imports RAGAS only when `--with-ragas` is present.

- [ ] **Step 5: Seed and validate the versioned dataset**

Add at least 100 labeled rows distributed across `direct_lookup`, `identifier`, `table`, `summary`, `multi_hop`, `conflict`, `unanswerable`, `distractor`, `image`, `prompt_injection`, and supported-language categories. Every row follows this shape:

```json
{"id":"direct-001","inputs":{"question":"What is the stated warranty period?","user_id":"00000000-0000-0000-0000-000000000001","conversation_id":"00000000-0000-0000-0000-000000000002"},"reference":{"answer":"The warranty period is two years.","relevant_document_ids":["00000000-0000-0000-0000-000000000003"],"relevant_spans":[{"document_id":"00000000-0000-0000-0000-000000000003","page_start":1,"page_end":1}],"should_abstain":false},"metadata":{"category":"direct_lookup","language":"en"}}
```

Add a contract test that requires 100–300 rows, all categories, unique IDs, and no transient Qdrant point IDs in gold labels.

`corpus.py` loads `corpus_manifest.jsonl` into a dedicated evaluation tenant using deterministic UUID5 document IDs, waits for the active index generation, and returns the generated tenant/conversation scope to `target.py`. The manifest contains the controlled text/table/image fixture paths and SHA-256 hashes, so every gold document ID resolves before an experiment starts.

- [ ] **Step 6: Run, record the baseline, and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evaluation_contracts.py tests/test_rag_evaluation_metrics.py
.\.venv\Scripts\python.exe -m ruff check app/evaluation scripts/evaluate_rag.py tests/test_rag_evaluation_contracts.py tests/test_rag_evaluation_metrics.py
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix pre-hardening-baseline
git add app/evaluation eval/rag scripts/evaluate_rag.py tests/test_rag_evaluation_contracts.py tests/test_rag_evaluation_metrics.py pyproject.toml
git commit -m "test: establish RAG quality baseline"
```

Expected: deterministic tests pass and LangSmith records the immutable baseline experiment. Human-review at least 25 examples before using any LLM-judge metric as a gate.

---

### Task 2: Correct and validate Gemini embedding calls

**Files:**
- Modify: `app/services/rag_embedding_service.py`
- Modify: `tests/test_rag_embedding_service.py`
- Create: `tests/live/test_gemini_embedding_contract.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: one text or image per requested vector, Gemini task type, title, dimension, and usage context.
- Produces: ordered `list[list[float]]` document vectors, a single query vector, a single image vector, bounded retries, and strict count/dimension failures.

- [ ] **Step 1: Add failing provider-shape, dimension, query-retry, and live tests**

```python
def test_each_batched_text_is_a_separate_content(monkeypatch):
    service, calls = fake_gemini_service(monkeypatch, vectors=[[1.0, 0.0], [0.0, 1.0]])
    assert service.embed_documents(["alpha", "beta"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert len(calls[0]["contents"]) == 2
    assert all(isinstance(item, types.Content) for item in calls[0]["contents"])


def test_wrong_vector_dimension_fails_closed(monkeypatch):
    service, _ = fake_gemini_service(monkeypatch, dimension=3, vectors=[[1.0, 2.0]])
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        service.embed_documents(["alpha"])
```

The live test is marked `@pytest.mark.live_provider`, skips without `GEMINI_API_KEY`, sends two distinct strings, and asserts two vectors of the configured dimension that are not identical.

Register the marker under `[tool.pytest.ini_options]` as `live_provider: calls a configured external model provider` so normal CI can exclude it with `-m "not live_provider"` without marker warnings.

- [ ] **Step 2: Run focused tests and confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_embedding_service.py -k "separate_content or wrong_vector_dimension or query_retry"
```

Expected: failures because batches currently pass strings directly, dimensions are not checked, and query embedding has no bounded retry loop.

- [ ] **Step 3: Wrap inputs and unify retry/validation**

```python
def _text_content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def _validate_vectors(vectors: list[list[float]], *, expected_count: int, dimension: int) -> None:
    if len(vectors) != expected_count:
        raise RuntimeError(f"Embedding count mismatch: expected {expected_count}, got {len(vectors)}")
    bad = [index for index, vector in enumerate(vectors) if len(vector) != dimension]
    if bad:
        raise RuntimeError(f"Embedding dimension mismatch at indices {bad}: expected {dimension}")
```

**PLAN CORRECTION (2026-08-21):** do **not** set `task_type` or a provider-level `title`. Google's Embeddings documentation states that `task_type` cannot be used with `gemini-embedding-2` and that the task must be given as an instruction in the prompt instead; the title likewise belongs in the request text. `GeminiRAGEmbeddingService` already carries both through `_format_document` (`title: {title} | text: {text}`) and `embed_query` (`task: {query_task} | query: {query}`), and now sends `output_dimensionality` as the only config field. `tests/test_rag_embedding_service.py` pins their absence. The original instruction — "Set `task_type="RETRIEVAL_DOCUMENT"` for chunks and `task_type="RETRIEVAL_QUERY"` for queries; pass title through the provider config where supported" — was implemented as written and had to be reverted; do not reinstate it.

Use one retry helper for text, query, and image calls with exponential backoff, jitter, and provider retry hints. Do not retry shape errors.

- [ ] **Step 4: Run offline and opt-in live verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_embedding_service.py
.\.venv\Scripts\python.exe -m pytest -q -m live_provider tests/live/test_gemini_embedding_contract.py
.\.venv\Scripts\python.exe -m ruff check app/services/rag_embedding_service.py tests/test_rag_embedding_service.py tests/live/test_gemini_embedding_contract.py
```

Expected: offline suite passes. The live test either passes with a configured key or reports one explicit skip.

- [ ] **Step 5: Run the baseline experiment and commit**

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix embedding-shape-fix --compare-baseline pre-hardening-baseline
git add app/services/rag_embedding_service.py tests/test_rag_embedding_service.py tests/live/test_gemini_embedding_contract.py pyproject.toml
git commit -m "fix: enforce Gemini embedding batch contract"
```

Expected: no designated retrieval gate regresses.

---

### Task 3: Bound agentic document exploration

**Files:**
- Modify: `app/ai/schemas.py`
- Modify: `app/ai/rag_tools.py`
- Modify: `app/ai/rag_tool_actions.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `app/ai/prompts.py`
- Modify: `app/repositories/document_chunk.py`
- Modify: `tests/test_rag_agent.py`
- Modify: `tests/test_rag_tool_loop_finalization.py`

**Interfaces:**
- Consumes: `SearchDocumentsInput` with bounded pagination/window fields and server-owned tenant scope.
- Produces: search-first behavior, paginated listing/scanning, and bounded chunk-window reads.

- [ ] **Step 1: Add failing schema and behavior tests**

```python
def test_read_document_schema_has_bounded_chunk_window():
    schema = SearchDocumentsInput.model_json_schema()["properties"]
    assert schema["start_chunk"]["minimum"] == 0
    assert schema["max_chunks"]["maximum"] == 20


async def test_scan_all_never_reads_more_than_requested_page(agent):
    await agent.scan_all_documents("conversation", user_id="user", page=2, page_size=5)
    assert agent.get_document_preview.await_count <= 5
```

Also assert that the system prompt tells the model to begin with `search_chunks` for ordinary questions and reserves `scan_all` for explicit corpus enumeration.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_agent.py tests/test_rag_tool_loop_finalization.py -k "bounded or scan_all or search_first"
```

- [ ] **Step 3: Add bounded input fields and repository windows**

```python
class SearchDocumentsInput(BaseModel):
    action: DocumentAction
    document_id: str | None = None
    query: str | None = None
    pattern: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=25)
    start_chunk: int = Field(default=0, ge=0)
    max_chunks: int = Field(default=8, ge=1, le=20)
    reason: str | None = None
```

Implement `DocumentChunkRepository.get_window_for_scope(document_id, user_id, conversation_id, start_chunk, max_chunks)` with the ownership join in SQL. `READ_DOCUMENT` returns the selected chunks plus `next_start_chunk`; `SCAN_ALL` and `LIST_DOCUMENTS` return one page plus total count.

- [ ] **Step 4: Replace unbounded tool execution and prompt policy**

Thread the new fields through `create_search_documents_tool`, `execute_search_documents_action`, and the agent helpers. Never call `get_document_full_content` from the default tool path. Keep `get_document_full_content` only until Phase 7 compatibility removal.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_agent.py tests/test_rag_tool_loop_finalization.py tests/test_rag_multi_user_isolation.py
.\.venv\Scripts\python.exe -m ruff check app/ai/schemas.py app/ai/rag_tools.py app/ai/rag_tool_actions.py app/ai/agents/rag_agent.py app/repositories/document_chunk.py
git add app/ai/schemas.py app/ai/rag_tools.py app/ai/rag_tool_actions.py app/ai/agents/rag_agent.py app/ai/prompts.py app/repositories/document_chunk.py tests/test_rag_agent.py tests/test_rag_tool_loop_finalization.py
git commit -m "fix: bound agentic document exploration"
```

---

### Task 4: Normalize parser output into structural blocks

**Files:**
- Create: `app/services/document_blocks.py`
- Create: `app/services/document_normalizer.py`
- Modify: `app/services/document_parse_service.py`
- Modify: `app/services/document_chunk_builder.py`
- Modify: `app/services/document_processing_service.py`
- Modify: `tests/test_unified_parse_pipeline.py`
- Create: `tests/test_document_normalizer.py`
- Modify: `tests/test_document_parse_artifact_repository.py`

**Interfaces:**
- Consumes: MinerU content-list entries, UTF-8 text, Excel rows, and parser image metadata.
- Produces: `ParseResult(blocks: list[NormalizedBlock], images_data, parse_elapsed_s, backend_used)` with stable provenance.

- [ ] **Step 1: Write failing normalizer tests for headings, tables, images, equations, and Excel**

```python
def test_mineru_heading_carries_section_path_to_later_blocks():
    blocks = normalizer.normalize_mineru([
        {"type": "text", "text_level": 1, "text": "Revenue", "page_idx": 0},
        {"type": "text", "text": "Quarterly results", "page_idx": 0, "bbox": [1, 2, 3, 4]},
    ])
    assert blocks[1].section_path == ("Revenue",)
    assert blocks[1].metadata["bbox"] == [1, 2, 3, 4]


def test_table_retains_caption_header_body_and_footnote():
    table = normalize_one_table(table_fixture())
    assert table.kind == "table"
    assert table.metadata.keys() >= {"caption", "header", "body", "footnote"}
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_document_normalizer.py tests/test_unified_parse_pipeline.py
```

- [ ] **Step 3: Introduce immutable block types and artifact serialization**

```python
@dataclass(frozen=True)
class NormalizedBlock:
    block_id: str
    kind: Literal["heading", "paragraph", "table", "image", "equation"]
    text: str
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    metadata: Mapping[str, Any]
```

Move `BuiltChunk` into the same module. Serialize blocks with explicit `schema_version=2`; keep `load_parse_result` able to read version 1 artifacts during the rollout window.

- [ ] **Step 4: Route every format through `DocumentNormalizer` once**

Plain text becomes paragraph blocks, each Excel sheet becomes a heading followed by one or more table blocks, and MinerU content entries map directly without first becoming character chunks. Markdown fallback becomes paragraph/heading blocks using the same normalizer. `DocumentProcessingService` passes `ParseResult.blocks` directly to `DocumentChunkBuilder.build`.

- [ ] **Step 5: Verify parse persistence and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_document_normalizer.py tests/test_unified_parse_pipeline.py tests/test_document_processing_service.py tests/test_document_parse_artifact_repository.py
.\.venv\Scripts\python.exe -m ruff check app/services/document_blocks.py app/services/document_normalizer.py app/services/document_parse_service.py app/services/document_processing_service.py
git add app/services/document_blocks.py app/services/document_normalizer.py app/services/document_parse_service.py app/services/document_chunk_builder.py app/services/document_processing_service.py tests/test_document_normalizer.py tests/test_unified_parse_pipeline.py tests/test_document_parse_artifact_repository.py
git commit -m "feat: preserve document structure through parsing"
```

---

### Task 5: Make chunk overlap correct and semantic chunking measurable

**Files:**
- Modify: `app/services/document_chunk_builder.py`
- Create: `app/services/semantic_breakpoints.py`
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Modify: `tests/test_document_chunk_builder.py`
- Create: `tests/test_semantic_breakpoints.py`

**Interfaces:**
- Consumes: ordered `NormalizedBlock` values, 400 target tokens, 800 hard maximum, 40 overlap tokens, and an optional semantic boundary detector.
- Produces: final-size-verified `BuiltChunk` values with same-section overlap, table row-group splits, provenance, and neighbor IDs.

- [ ] **Step 1: Add failing overlap and hard-limit tests**

```python
def test_regular_adjacent_chunks_overlap_only_inside_same_section(builder):
    chunks = builder.build(section_blocks("A", sentence_count=30))
    assert shared_tail_tokens(chunks[0].content, chunks[1].content) <= 40
    assert shared_tail_tokens(chunks[0].content, chunks[1].content) > 0


def test_overlap_does_not_cross_heading_or_table_boundary(builder):
    chunks = builder.build(mixed_section_and_table_blocks())
    assert no_cross_section_overlap(chunks)
    assert no_table_overlap(chunks)
    assert all(chunk.token_count <= 800 for chunk in chunks)
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_document_chunk_builder.py -k "overlap or hard_limit or neighbor"
```

Expected: regular buffered blocks currently split without overlap.

- [ ] **Step 3: Implement section-aware assembly and final rendered counting**

Split text into language-aware sentence units, finalize a candidate including separators and metadata, and carry at most 40 tokens only when the next block has the same `section_path`. Add `previous_chunk_index` and `next_chunk_index` after all chunks are built. Table splitting repeats caption and header but not previous rows.

```python
previous_is_table = bool(previous.metadata.get("contains_table"))
current_is_table = any(block.kind == "table" for block in current_blocks)
if previous.section_path == current.section_path and not previous_is_table and not current_is_table:
    overlap = tail_with_budget(previous.content, self.overlap_tokens, self.token_strategy)
else:
    overlap = ""
rendered = render_chunk(overlap=overlap, blocks=current_blocks)
assert self.token_strategy.count(rendered) <= self.max_tokens
```

- [ ] **Step 4: Add an opt-in semantic breakpoint detector**

```python
SemanticBoundaryDetector = Callable[
    [Sequence[NormalizedBlock]],
    frozenset[str],
]


class EmbeddingSemanticBoundaryDetector:
    def break_before(self, blocks: Sequence[NormalizedBlock]) -> frozenset[str]:
        vectors = self.embedding_service.embed_documents([block.text for block in blocks])
        distances = [1.0 - cosine(a, b) for a, b in pairwise(vectors)]
        cutoff = percentile(distances, self.breakpoint_percentile)
        return frozenset(blocks[i + 1].block_id for i, value in enumerate(distances) if value >= cutoff)
```

Implement `cosine`, adjacent-pair iteration, and percentile selection as pure local helpers in `semantic_breakpoints.py`; empty or one-block inputs return no boundaries, and zero vectors use distance `0.0`.

Add `rag_semantic_chunking_enabled=False` and `rag_semantic_breakpoint_percentile=90.0`. The builder treats detected boundaries like section boundaries. The feature stays disabled unless its LangSmith experiment beats structural chunking without exceeding the accepted ingestion-cost delta in `eval/rag/release_gates.json`.

- [ ] **Step 5: Verify structural and semantic modes, evaluate, and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_document_chunk_builder.py tests/test_semantic_breakpoints.py
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix structural-chunking --compare-baseline embedding-shape-fix
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix semantic-chunking-candidate --compare-baseline structural-chunking
git add app/services/document_chunk_builder.py app/services/semantic_breakpoints.py app/core/config.py app/core/container.py tests/test_document_chunk_builder.py tests/test_semantic_breakpoints.py
git commit -m "feat: enforce structure-aware chunk boundaries"
```

---

### Task 6: Add atomic index generations and safe activation

**Files:**
- Create: `app/models/document_index_generation.py`
- Create: `app/repositories/document_index_generation.py`
- Create: `app/alembic/versions/c3d4e5f6a7b8_add_document_index_generations.py`
- Modify: `app/models/document.py`
- Modify: `app/models/document_chunk.py`
- Modify: `app/repositories/document_chunk.py`
- Modify: `app/services/document_index_service.py`
- Modify: `scripts/reindex_embeddings.py`
- Create: `tests/test_document_index_generation_repository.py`
- Modify: `tests/test_document_index_service.py`
- Modify: `tests/test_reindex_embeddings_cli.py`
- Modify: `tests/test_database_schema_contract.py`

**Interfaces:**
- Consumes: a document, built chunks, provider/model/dimension/config version, and current active generation.
- Produces: inactive `DocumentIndexGeneration`, generation-owned chunks and points, verified activation, failed-generation state, and rollback-safe cleanup.

- [ ] **Step 1: Write failing generation lifecycle tests**

```python
def test_failed_reindex_keeps_previous_generation_active(index_service):
    old = active_generation("00000000-0000-0000-0000-000000000010")
    index_service.qdrant_client.upsert.side_effect = RuntimeError("qdrant unavailable")
    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        index_service.index_document(document=document("doc-1"), built_chunks=[built_chunk()])
    assert generation_repo.get_active(old.document_id).id == old.id
    assert generation_repo.get_latest_failed(old.document_id) is not None


def test_activation_occurs_only_after_count_dimension_and_scope_verification(index_service):
    doc = document("00000000-0000-0000-0000-000000000010")
    index_service.index_document(document=doc, built_chunks=[built_chunk()])
    assert index_service.qdrant_client.count.called
    assert generation_repo.get_active(doc.id).status == "active"
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_document_index_generation_repository.py tests/test_document_index_service.py -k "generation or activation or rollback"
```

- [ ] **Step 3: Add the schema and transactional repository**

`DocumentIndexGeneration` contains `id`, `document_id`, `status`, `embedding_provider`, `embedding_model`, `embedding_dimension`, `chunking_version`, `created_at`, `activated_at`, and bounded `failure_code`. Add a PostgreSQL partial unique index allowing only one `status='active'` row per document. Add non-null `index_generation_id` to `document_chunks`, change uniqueness to `(document_id, index_generation_id, chunk_index)`, and backfill one active legacy generation per existing document. Do not edit prior migration files.

```python
def activate(self, generation_id: UUID) -> DocumentIndexGeneration:
    with self.session_factory.begin() as db:
        target = db.execute(select(DocumentIndexGeneration).where(
            DocumentIndexGeneration.id == generation_id
        ).with_for_update()).scalar_one()
        db.execute(update(DocumentIndexGeneration).where(
            DocumentIndexGeneration.document_id == target.document_id,
            DocumentIndexGeneration.status == "active",
        ).values(status="retired"))
        target.status = "active"
        target.activated_at = func.now()
        return target
```

- [ ] **Step 4: Rewrite the index service as build, verify, activate**

Create chunks under an inactive generation, embed and upsert points carrying `index_generation` and `is_active=False`, then verify point count/vector dimension/document scope. Activation first marks the new Qdrant points `is_active=True`, then atomically activates the SQL generation, then marks prior-generation Qdrant points false. If SQL activation fails, restore the new points to false; if final old-point cleanup fails, SQL hydration still rejects retired chunks and a reconciliation job repairs payload flags. Never call delete-by-document before activation. Mark failures with a bounded code and retain the old generation. Add `reconcile_active_payloads(document_id)` and `purge_retired_generations(document_id, older_than)` for recovery and post-rollback-window cleanup.

- [ ] **Step 5: Update reindex CLI and verify the migration chain**

`scripts/reindex_embeddings.py` reports old/new generation IDs, supports `--activate` and `--keep-retired-hours`, and never marks current chunks destructively before the replacement succeeds.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_document_index_generation_repository.py tests/test_document_index_service.py tests/test_reindex_embeddings_cli.py tests/test_database_schema_contract.py tests/test_alembic_full_chain_postgres.py
.\.venv\Scripts\python.exe -m alembic heads
```

Expected: exactly one head, `c3d4e5f6a7b8`, and migration tests pass.

- [ ] **Step 6: Commit**

```powershell
git add app/models/document_index_generation.py app/repositories/document_index_generation.py app/alembic/versions/c3d4e5f6a7b8_add_document_index_generations.py app/models/document.py app/models/document_chunk.py app/repositories/document_chunk.py app/services/document_index_service.py scripts/reindex_embeddings.py tests/test_document_index_generation_repository.py tests/test_document_index_service.py tests/test_reindex_embeddings_cli.py tests/test_database_schema_contract.py
git commit -m "feat: activate RAG indexes by generation"
```

---

### Task 7: Bootstrap Qdrant indexes and implement scoped hybrid retrieval

**Files:**
- Create: `app/services/rag_retrieval.py`
- Create: `app/alembic/versions/d4e5f6a7b8c9_add_document_chunk_lexical_index.py`
- Modify: `app/repositories/document_chunk.py`
- Modify: `app/services/document_index_service.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Create: `tests/test_rag_retrieval.py`
- Modify: `tests/test_document_index_service.py`
- Modify: `tests/test_rag_multi_user_isolation.py`

**Interfaces:**
- Consumes: `RetrievalScope(user_id, conversation_id)`, query text, dense candidate limit, lexical candidate limit, final limit, and the server-resolved active-generation fingerprint.
- Produces: authorized `RetrievalCandidate` values with dense, lexical, and fused rank/score fields.

- [ ] **Step 1: Write failing bootstrap, RRF, and authorization tests**

```python
def test_collection_bootstrap_creates_filter_indexes_before_upsert(index_service):
    index_service.ensure_collection()
    fields = [call.kwargs["field_name"] for call in index_service.qdrant_client.create_payload_index.call_args_list]
    assert fields == ["user_id", "conversation_id", "document_id", "modality", "index_generation", "is_active"]


def test_rrf_combines_dense_and_lexical_ranks():
    fused = reciprocal_rank_fusion(dense=["a", "b"], lexical=["b", "c"], k=60)
    assert fused[0].chunk_id == "b"


def test_missing_sql_row_is_never_returned(retriever):
    retriever.qdrant_client.query_points.return_value = points(["missing"])
    assert retriever.search("query", scope("user", "conversation")) == []
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_retrieval.py tests/test_document_index_service.py tests/test_rag_multi_user_isolation.py -k "payload_index or rrf or missing_sql"
```

- [ ] **Step 3: Create payload and PostgreSQL lexical indexes**

Create Qdrant keyword payload indexes for owner/conversation/document/modality/generation and a boolean payload index for `is_active` before the first upsert. Use `KeywordIndexParams(is_tenant=True)` for `user_id` when supported and fall back to a normal keyword index only on the explicit unsupported-version error. Add a PostgreSQL GIN expression index over `to_tsvector('simple', content)` in the new Alembic migration.

- [ ] **Step 4: Implement typed retrieval and SQL reauthorization**

```python
@dataclass(frozen=True)
class RetrievalScope:
    user_id: str
    conversation_id: UUID


@dataclass(frozen=True)
class RetrievalCandidate:
    document_id: UUID
    chunk_id: UUID | None
    image_id: UUID | None
    modality: Literal["text", "image"]
    content: str
    filename: str
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    dense_rank: int | None
    dense_score: float | None
    lexical_rank: int | None
    lexical_score: float | None
    fused_score: float
    rerank_score: float | None = None
```

Run dense Qdrant with `is_active=True` and PostgreSQL lexical retrieval joined to active SQL generations under the same user/conversation scope. Fuse ranks with RRF, then hydrate all winners through `get_active_by_ids_for_scope`, which rejects retired or unauthorized chunks even if Qdrant payload reconciliation is pending. Compute `active_generation_fingerprint` as SHA-256 over the sorted active generation UUIDs in that conversation; use it in traces and later cache keys. Raw scores remain trace fields and are never formatted as percentages or confidence probabilities.

Add `rag_hybrid_retrieval_enabled=False`; when false, the retriever uses the same typed dense-only path so rollback does not restore dictionary-shaped or unauthorized retrieval.

Add `rag_dense_candidate_limit=40`, `rag_lexical_candidate_limit=40`, and `rag_rrf_k=60`. These are experiment inputs recorded in every trace and may change only through evaluation.

- [ ] **Step 5: Replace `RAGAgent._search` internals and verify**

Keep `_search` as a temporary adapter returning existing dictionaries, but delegate to `RAGRetriever.search`. This lets later tasks migrate tool formatting without breaking the public tool contract.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_retrieval.py tests/test_document_index_service.py tests/test_rag_multi_user_isolation.py tests/test_rag_agent.py
.\.venv\Scripts\python.exe -m alembic heads
.\.venv\Scripts\python.exe -m ruff check app/services/rag_retrieval.py app/services/document_index_service.py app/repositories/document_chunk.py
git add app/services/rag_retrieval.py app/alembic/versions/d4e5f6a7b8c9_add_document_chunk_lexical_index.py app/repositories/document_chunk.py app/services/document_index_service.py app/ai/agents/rag_agent.py app/core/config.py app/core/container.py tests/test_rag_retrieval.py tests/test_document_index_service.py tests/test_rag_multi_user_isolation.py
git commit -m "feat: add scoped hybrid RAG retrieval"
```

---

### Task 8: Isolate reranking behind a bounded fail-open service

**Files:**
- Create: `app/services/rag_reranker.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Create: `tests/test_rag_reranker.py`
- Modify: `tests/test_rag_agent.py`

**Interfaces:**
- Consumes: query plus up to `rag_rerank_candidate_pool=40` typed candidates.
- Produces: up to `rag_evidence_candidate_limit=10` candidates with `rerank_score`; timeout/load/inference failure returns fused order.

- [ ] **Step 1: Write failing timeout, concurrency, and fail-open tests**

```python
@pytest.mark.asyncio
async def test_reranker_runs_off_event_loop(monkeypatch, reranker):
    called = False
    async def fake_to_thread(fn, *args):
        nonlocal called
        called = True
        return fn(*args)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    await reranker.rank("query", candidates(4))
    assert called


@pytest.mark.asyncio
async def test_timeout_returns_fused_order(reranker):
    reranker.model.predict = blocking_prediction
    assert await reranker.rank("query", candidates(4)) == candidates(4)[:reranker.output_limit]
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_reranker.py
```

- [ ] **Step 3: Implement bounded inference and canonical settings**

```python
async def rank(self, query: str, candidates: Sequence[RetrievalCandidate]) -> list[RetrievalCandidate]:
    pool = list(candidates[: self.candidate_pool])
    try:
        async with self._semaphore:
            scores = await asyncio.wait_for(
                asyncio.to_thread(self._predict, query, pool),
                timeout=self.timeout_seconds,
            )
    except Exception as exc:
        self.metrics.degraded("reranker", classify_failure(exc))
        return pool[: self.output_limit]
    return apply_rerank_scores(pool, scores)[: self.output_limit]
```

Implement `apply_rerank_scores` by checking `len(scores) == len(pool)`, using `dataclasses.replace(candidate, rerank_score=float(score))`, and sorting descending. A score-count mismatch follows the same fail-open path and emits `failure_code="score_count_mismatch"`.

Add `rag_rerank_candidate_pool=40`, `rag_evidence_candidate_limit=10`, `rag_reranker_timeout_seconds=5.0`, and `rag_reranker_max_concurrency=2`. Use only `rag_reranker_model`; the old alias remains until cleanup.

- [ ] **Step 4: Evaluate candidate models and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_reranker.py tests/test_rag_agent.py tests/test_rag_retrieval.py
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix hybrid-reranked --compare-baseline structural-chunking
git add app/services/rag_reranker.py app/ai/agents/rag_agent.py app/core/config.py app/core/container.py tests/test_rag_reranker.py tests/test_rag_agent.py
git commit -m "feat: bound RAG reranker inference"
```

Expected: select the multilingual/domain model from measured MRR/nDCG/Recall and latency results; do not change the default solely from a model-card claim.

---

### Task 9: Assemble immutable, token-budgeted evidence

**Files:**
- Create: `app/services/rag_evidence.py`
- Modify: `app/repositories/document_chunk.py`
- Modify: `app/ai/rag_tool_actions.py`
- Modify: `app/ai/workflow/rag_loop.py`
- Modify: `app/ai/request_budget.py`
- Create: `tests/test_rag_evidence.py`
- Modify: `tests/test_request_budget.py`
- Modify: `tests/test_rag_tool_loop_finalization.py`

**Interfaces:**
- Consumes: authorized ranked candidates, current question, optional subquestions, and an exact model-input token allowance.
- Produces: `EvidencePack(records, token_count, omitted_count)` with immutable IDs `E1..En` and an untrusted-data serialization.

- [ ] **Step 1: Write failing deduplication, provenance, injection, and budget tests**

```python
def test_evidence_ids_and_metadata_are_server_owned(assembler):
    pack = assembler.assemble("question", duplicate_candidates(), max_tokens=500)
    assert [record.evidence_id for record in pack.records] == ["E1"]
    assert pack.records[0].filename == "server-filename.pdf"


def test_evidence_serialization_marks_content_untrusted(assembler):
    pack = assembler.assemble("question", [injection_candidate()], max_tokens=500)
    text = pack.to_tool_text()
    assert "BEGIN UNTRUSTED EVIDENCE E1" in text
    assert "END UNTRUSTED EVIDENCE E1" in text
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evidence.py tests/test_request_budget.py
```

- [ ] **Step 3: Implement evidence records and bounded expansion**

```python
@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    document_id: UUID
    chunk_id: UUID | None
    image_id: UUID | None
    filename: str
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    modality: Literal["text", "image"]
    content: str


@dataclass(frozen=True)
class EvidencePack:
    records: tuple[EvidenceRecord, ...]
    token_count: int
    omitted_count: int

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(record.evidence_id for record in self.records)

    def to_tool_text(self) -> str:
        parts: list[str] = []
        for record in self.records:
            parts.extend([
                f"BEGIN UNTRUSTED EVIDENCE {record.evidence_id}",
                f"source={record.filename} pages={record.page_start}-{record.page_end}",
                record.content,
                f"END UNTRUSTED EVIDENCE {record.evidence_id}",
            ])
        return "\n".join(parts)
```

Deduplicate by canonical chunk/image ID and overlapping content hashes. Balance top evidence across subquestions and documents before filling remaining score order. Fetch parent/adjacent chunks only through scoped repository methods and only while the token budget remains.

- [ ] **Step 4: Preserve tool roles and complete evidence groups**

`execute_search_documents_action` stores the full structured `EvidencePack` in the tool artifact and emits only its bounded serialization as the `ToolMessage`. Remove the synthetic `Previous Tool Results` text appended to the current user message. Extend request budgeting so complete historical assistant-tool groups can be trimmed, while the current question and current evidence pack remain fixed input.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evidence.py tests/test_request_budget.py tests/test_rag_tool_loop_finalization.py tests/test_rag_multi_user_isolation.py
.\.venv\Scripts\python.exe -m ruff check app/services/rag_evidence.py app/ai/rag_tool_actions.py app/ai/workflow/rag_loop.py app/ai/request_budget.py
git add app/services/rag_evidence.py app/repositories/document_chunk.py app/ai/rag_tool_actions.py app/ai/workflow/rag_loop.py app/ai/request_budget.py tests/test_rag_evidence.py tests/test_request_budget.py tests/test_rag_tool_loop_finalization.py
git commit -m "feat: assemble bounded RAG evidence packs"
```

---

### Task 10: Enforce grounded answers, citations, and abstention

**Files:**
- Create: `app/services/rag_grounding.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `app/ai/workflow/rag_loop.py`
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Create: `tests/test_rag_grounding.py`
- Modify: `tests/test_rag_tool_loop_finalization.py`
- Create: `tests/fixtures/rag_prompt_injection_cases.json`

**Interfaces:**
- Consumes: current question, current `EvidencePack`, and a structured-answer generator.
- Produces: accepted `GroundedAnswer`, one constrained regeneration, or an abstaining `GroundedAnswer` with bounded missing-information text.

- [ ] **Step 1: Write failing citation, claim-coverage, no-evidence, and injection tests**

```python
def test_unknown_citation_is_rejected(gate):
    answer = GroundedAnswer(claims=[GroundedClaim(text="Revenue rose.", evidence_ids=("E9",))])
    result = gate.validate(answer, evidence_pack("E1"))
    assert result.valid is False
    assert result.reason_codes == ("unknown_evidence_id",)


def test_factual_answer_without_evidence_abstains(gate):
    result = gate.finalize(question="What was revenue?", evidence=empty_pack())
    assert result.abstained is True
    assert result.reason_code == "insufficient_evidence"
```

Fixtures must place adversarial commands in paragraph text, table cells, OCR, captions, and filenames and assert none changes tool policy or system behavior.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_grounding.py tests/test_rag_tool_loop_finalization.py
```

- [ ] **Step 3: Implement structured claims and deterministic validation**

```python
class GroundedClaim(BaseModel):
    text: str
    evidence_ids: tuple[str, ...]


class GroundedAnswer(BaseModel):
    claims: tuple[GroundedClaim, ...] = ()
    abstained: bool = False
    missing_information: str | None = None
    reason_code: str | None = None


class ValidationResult(BaseModel):
    valid: bool
    reason_codes: tuple[str, ...]
    citation_coverage: float


class GroundedAnswerGate:
    def __init__(self, min_coverage: float) -> None:
        self.min_coverage = min_coverage

    def validate(self, answer: GroundedAnswer, evidence: EvidencePack) -> ValidationResult:
        known = evidence.evidence_ids
        unknown = any(eid not in known for claim in answer.claims for eid in claim.evidence_ids)
        covered = sum(bool(claim.evidence_ids) for claim in answer.claims)
        coverage = covered / len(answer.claims) if answer.claims else 1.0
        reasons = tuple(code for code, failed in (
            ("unknown_evidence_id", unknown),
            ("citation_coverage_below_minimum", coverage < self.min_coverage),
            ("answer_without_evidence", bool(answer.claims) and not known),
        ) if failed)
        return ValidationResult(valid=not reasons, reason_codes=reasons, citation_coverage=coverage)
```

Validate cited IDs against the current pack, compute coverage across factual claims, reject supported-sounding answers with empty evidence, and derive filename/page citation rendering from server records. `min_citation_coverage` becomes an active setting used here.

- [ ] **Step 4: Add one regeneration and explicit abstention**

When the agent returns a final response, the graph invokes the grounded finalizer against current structured evidence. Invalid structure or insufficient coverage triggers one generation with reason codes and the same evidence. A second failure returns an abstention with bounded missing-information text. Enable `enable_citation_verification` only after this path is wired and tested.

Add `rag_grounded_answer_gate_enabled=False`; the disabled path retains the current final response while still recording validation shadow metrics. Enable enforcement only through the rollout task.

- [ ] **Step 5: Run guardrail and quality gates, then commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_grounding.py tests/test_rag_tool_loop_finalization.py tests/test_rag_agent.py
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix grounded-answer-gate --with-ragas --compare-baseline hybrid-reranked
git add app/services/rag_grounding.py app/ai/agents/rag_agent.py app/ai/workflow/rag_loop.py app/core/config.py app/core/container.py tests/test_rag_grounding.py tests/test_rag_tool_loop_finalization.py tests/fixtures/rag_prompt_injection_cases.json
git commit -m "feat: validate grounded RAG answers"
```

Expected: citation validity is 1.0, injection cases pass, and abstention precision/recall do not regress beyond the configured baseline deltas.

---

### Task 11: Complete caption-first and native multimodal retrieval

**Files:**
- Create: `app/services/rag_image_selector.py`
- Create: `app/alembic/versions/e5f6a7b8c9d0_add_document_image_provenance.py`
- Modify: `app/models/document_image.py`
- Modify: `app/schemas/document_image.py`
- Modify: `app/repositories/document_image.py`
- Modify: `app/services/document_processing_service.py`
- Modify: `app/services/document_index_service.py`
- Modify: `app/services/rag_retrieval.py`
- Modify: `app/services/rag_embedding_service.py`
- Modify: `app/ai/agents/rag_agent.py`
- Create: `tests/test_rag_image_selector.py`
- Modify: `tests/test_document_processing_service.py`
- Modify: `tests/test_document_index_service.py`
- Modify: `tests/test_rag_multi_user_isolation.py`

**Interfaces:**
- Consumes: structured captions, canonical `DocumentImage` rows, optional native image embeddings, query visual intent, and byte/pixel/vision-token budgets.
- Produces: caption-retrievable text evidence by default and separately hydrated `modality="image"` candidates when enabled.

```python
@dataclass(frozen=True)
class SelectedImage:
    image_id: UUID
    mime_type: str
    data: bytes
    byte_count: int
    pixel_count: int
    estimated_vision_tokens: int
    page_number: int | None
    caption: str | None
```

- [ ] **Step 1: Write failing caption provenance, image point, and budget tests**

```python
def test_image_point_links_image_without_chunk_id(index_service):
    point = index_service._point_for_image(document(), document_image())
    assert point.payload["modality"] == "image"
    assert point.payload["image_id"]
    assert "chunk_id" not in point.payload


def test_selector_enforces_count_bytes_pixels_and_dedup(selector):
    selected = selector.select("compare these charts", image_candidates())
    assert len(selected) <= selector.max_images
    assert sum(item.byte_count for item in selected) <= selector.max_bytes
    assert sum(item.pixel_count for item in selected) <= selector.max_pixels
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_image_selector.py tests/test_document_index_service.py -k "image_point or selector"
```

- [ ] **Step 3: Persist structured image provenance and caption content**

Add `bbox`, `section_path`, and `content_sha256` to `DocumentImage`. Captioning must return visible OCR, chart title, axes, legend, values, trends, relationships, and nearby section context as a validated structured object. Store its rendered searchable text in `image_caption`; do not put base64 or raw OCR payloads in logs.

- [ ] **Step 4: Index and hydrate native image points behind the existing flag**

Persist prepared `DocumentImage` rows with nullable `chunk_id` before vector writes, then extend `DocumentIndexService.index_document(..., image_rows: Sequence[DocumentImage] = ())` to link images to newly persisted chunks by page. When `rag_multimodal_image_embeddings_enabled` is true, call `embed_image` once per canonical image and upsert a separate point with `document_id`, `image_id`, `modality`, owner/conversation, page, and index generation. Verify both text and image point counts before activating the generation. `RAGRetriever` branches hydration by modality and authorizes `DocumentImage` through its parent document.

- [ ] **Step 5: Select and load images only after retrieval**

`RAGImageSelector.select` deduplicates, ranks visual-intent candidates, and applies `agentic_rag_max_images`, `rag_vision_max_bytes`, `rag_vision_max_pixels`, and `rag_vision_max_tokens`. Read files with `asyncio.to_thread`, resize/crop with Pillow when over budget, and encode only the selected set.

- [ ] **Step 6: Evaluate caption-only versus native multimodal and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_image_selector.py tests/test_document_processing_service.py tests/test_document_index_service.py tests/test_rag_multi_user_isolation.py tests/test_rag_agent_image_attachments.py
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix caption-only-images --with-ragas
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix native-image-embeddings --with-ragas --compare-baseline caption-only-images
git add app/services/rag_image_selector.py app/alembic/versions/e5f6a7b8c9d0_add_document_image_provenance.py app/models/document_image.py app/schemas/document_image.py app/repositories/document_image.py app/services/document_processing_service.py app/services/document_index_service.py app/services/rag_retrieval.py app/services/rag_embedding_service.py app/ai/agents/rag_agent.py tests/test_rag_image_selector.py tests/test_document_processing_service.py tests/test_document_index_service.py tests/test_rag_multi_user_isolation.py
git commit -m "feat: add bounded multimodal RAG retrieval"
```

Keep native image embeddings disabled unless image Recall@k, chart/table QA, multimodal faithfulness, cost, and p95 latency beat caption-only gates.

---

### Task 12: Add stage telemetry and exact caches

**Files:**
- Create: `app/observability/rag.py`
- Create: `app/services/rag_cache.py`
- Modify: `app/services/rag_embedding_service.py`
- Modify: `app/services/document_index_service.py`
- Modify: `app/services/rag_retrieval.py`
- Modify: `app/services/rag_reranker.py`
- Modify: `app/services/rag_evidence.py`
- Modify: `app/services/rag_grounding.py`
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Create: `tests/test_rag_cache.py`
- Create: `tests/test_rag_observability.py`

**Interfaces:**
- Consumes: stage name, content-free labels, tenant, provider/model/dimension, prompt-format version, normalized query, retrieval configuration, and active generation.
- Produces: Prometheus stage/cost/cache/degradation metrics and optional exact Redis cache hits.

- [ ] **Step 1: Write failing cache-isolation and telemetry tests**

```python
def test_query_cache_key_changes_with_tenant_and_generation():
    a = retrieval_key(tenant="u1", conversation="c1", generation="g1",
                      normalized_query="revenue?", retrieval_config_sha256="cfg")
    b = retrieval_key(tenant="u2", conversation="c1", generation="g1",
                      normalized_query="revenue?", retrieval_config_sha256="cfg")
    c = retrieval_key(tenant="u1", conversation="c1", generation="g2",
                      normalized_query="revenue?", retrieval_config_sha256="cfg")
    assert len({a, b, c}) == 3


def test_metrics_never_receive_document_content(metrics):
    metrics.stage("retrieval", elapsed_seconds=0.1, labels={"document_text": "secret"})
    assert "secret" not in metrics.render()
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_cache.py tests/test_rag_observability.py
```

- [ ] **Step 3: Implement content-free stage telemetry**

Instrument parse, caption, embedding, dense retrieval, lexical retrieval, SQL hydration, reranking, evidence assembly, generation, and validation. Labels are bounded enums such as provider, model, modality, cache result, and failure code; document IDs, filenames, text, credentials, vectors, OCR, and base64 are forbidden. Report p50/p95/p99 from exported histograms, cost per indexed document/question, token totals, cached-input ratio, tool iterations, failure rate, and queue depth.

- [ ] **Step 4: Implement caches in the approved order**

```python
def _digest(parts: Sequence[str]) -> str:
    encoded = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def document_embedding_key(*, tenant: str, provider: str, model: str, dimension: int,
                           format_version: str, content_sha256: str) -> str:
    return "rag:doc-embedding:" + _digest([
        tenant, provider, model, str(dimension), format_version, content_sha256
    ])

def query_embedding_key(*, tenant: str, provider: str, model: str, dimension: int,
                        task_prefix: str, normalized_query: str) -> str:
    return "rag:query-embedding:" + _digest([
        tenant, provider, model, str(dimension), task_prefix, normalized_query
    ])

def retrieval_key(*, tenant: str, conversation: str, generation: str,
                  normalized_query: str, retrieval_config_sha256: str) -> str:
    return "rag:retrieval:" + _digest([
        tenant, conversation, generation, normalized_query, retrieval_config_sha256
    ])
```

Use JSON-encoded float arrays with dimension validation. Cache document embeddings by exact content hash, query embeddings by exact normalized query, and retrieval results for a short TTL. The retrieval key's `generation` argument receives the conversation's exact `active_generation_fingerprint`, providing invalidation whenever any document activates a replacement generation. Redis failure is a cache miss. Do not add semantic answer caching.

Add `rag_exact_cache_enabled=False`, `rag_query_embedding_cache_ttl_seconds=300`, and `rag_retrieval_cache_ttl_seconds=60`. Cache settings become part of the retrieval configuration hash.

- [ ] **Step 5: Preserve provider prompt-cache eligibility**

Keep system prompts and tool schemas stable at the front of model requests and record provider cached-input tokens when returned. Do not build or manage an application KV cache for the model's internal attention state.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_cache.py tests/test_rag_observability.py tests/test_rag_embedding_service.py tests/test_rag_retrieval.py tests/test_rag_reranker.py tests/test_rag_evidence.py tests/test_rag_grounding.py
.\.venv\Scripts\python.exe -m ruff check app/observability/rag.py app/services/rag_cache.py
git add app/observability/rag.py app/services/rag_cache.py app/services/rag_embedding_service.py app/services/document_index_service.py app/services/rag_retrieval.py app/services/rag_reranker.py app/services/rag_evidence.py app/services/rag_grounding.py app/core/config.py app/core/container.py tests/test_rag_cache.py tests/test_rag_observability.py
git commit -m "feat: observe and cache exact RAG stages"
```

---

### Task 13: Qualify dimensions, batch indexing, and 1,000-document scale

**Files:**
- Create: `scripts/benchmark_rag.py`
- Create: `scripts/experiment_embedding_dimensions.py`
- Create: `tests/test_benchmark_rag_cli.py`
- Create: `tests/test_embedding_dimension_experiment.py`
- Modify: `eval/rag/release_gates.json`
- Create: `docs/rag-scale-runbook.md`

**Interfaces:**
- Consumes: representative corpus manifest, concurrency levels, dimensions 768/1536/3072, synchronous/provider Batch modes, and baseline experiment IDs.
- Produces: JSON/JSONL benchmark artifacts, release-gate verdicts, and evidence-based Qdrant tuning recommendations.

- [ ] **Step 1: Write failing CLI and report-schema tests**

```python
def test_benchmark_defaults_to_required_scale():
    args = parse_args([])
    assert args.documents == 1000
    assert args.output.endswith(".json")


def test_report_contains_quality_latency_capacity_and_recovery_sections():
    report = benchmark_report_fixture()
    validate_report(report)
    assert report.keys() >= {"corpus", "quality", "latency", "capacity", "failures"}
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_benchmark_rag_cli.py tests/test_embedding_dimension_experiment.py
```

- [ ] **Step 3: Implement repeatable scale and failure qualification**

The benchmark loads at least 1,000 representative documents, runs configurable concurrent ingestion/query traffic, and records total chunks, vector memory, indexing lag, distractor quality, tenant-filter latency, p50/p95/p99 stage latency, queue saturation, cache ratios, cost, and recovery from provider/Qdrant/Redis/worker failure. It writes configuration hashes and git SHA with results.

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_rag.py --documents 1000 --ingest-concurrency 8 --query-concurrency 32 --failure-injection --output artifacts/rag-scale-1000.json
```

- [ ] **Step 4: Compare dimensions and provider Batch indexing**

Run 768, 1536, and 3072 as separate index generations against the unchanged golden dataset. For non-interactive indexing, compare synchronous request batches with Gemini's asynchronous embedding Batch API for total cost, wall time, failures, and operational complexity. Keep interactive query embeddings synchronous.

```powershell
.\.venv\Scripts\python.exe scripts/experiment_embedding_dimensions.py --dimensions 768 1536 3072 --dataset rag-golden-v1 --include-provider-batch --output artifacts/rag-dimension-matrix.json
```

- [ ] **Step 5: Tune only from captured results and verify gates**

Record selected collection dimension and any HNSW, quantization, shard, or on-disk setting in `docs/rag-scale-runbook.md` with the benchmark artifact that justified it. Populate release gates with baseline-relative deltas and selected latency/cost SLOs, then run:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix scale-qualified --compare-baseline grounded-answer-gate
.\.venv\Scripts\python.exe -m pytest -q tests/test_benchmark_rag_cli.py tests/test_embedding_dimension_experiment.py
```

- [ ] **Step 6: Commit**

```powershell
git add scripts/benchmark_rag.py scripts/experiment_embedding_dimensions.py tests/test_benchmark_rag_cli.py tests/test_embedding_dimension_experiment.py eval/rag/release_gates.json docs/rag-scale-runbook.md
git commit -m "perf: qualify RAG at thousand-document scale"
```

---

### Task 14: Roll out independently reversible features

**Files:**
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Modify: `README.md`
- Create: `docs/rag-rollout-runbook.md`
- Create: `tests/test_rag_rollout_contract.py`

**Interfaces:**
- Consumes: completed experiment and load artifacts.
- Produces: explicit default-off flags, activation order, rollback conditions, and generation-safe operational commands.

- [ ] **Step 1: Write failing settings/documentation contract tests**

```python
def test_risky_rag_features_default_off(settings):
    assert settings.rag_hybrid_retrieval_enabled is False
    assert settings.rag_grounded_answer_gate_enabled is False
    assert settings.rag_multimodal_image_embeddings_enabled is False
    assert settings.rag_exact_cache_enabled is False
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_rollout_contract.py
```

- [ ] **Step 3: Add reversible flags and the rollout sequence**

Document this order: embedding contract fix; evaluation tracing; normalized parsing/chunking for new generations; hybrid retrieval; reranker; evidence assembly; grounded-answer gate; exact caches; native image embeddings. Each entry names its setting, required experiment, health signals, rollback setting, and whether reindexing is required.

- [ ] **Step 4: Verify rollback paths and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_rollout_contract.py tests/test_config_chat_image_settings.py tests/test_document_index_service.py tests/test_rag_retrieval.py tests/test_rag_grounding.py
git add app/core/config.py .env.example README.md docs/rag-rollout-runbook.md tests/test_rag_rollout_contract.py
git commit -m "docs: define reversible RAG rollout"
```

Do not begin Task 15 until replacement generations have completed the documented rollback window.

---

### Task 15: Consolidate and remove proven dead paths

**Files:**
- Create: `docs/rag-cleanup-inventory.md`
- Modify: `app/services/document_processing_service.py`
- Modify: `app/services/document_parse_service.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `app/ai/rag_tool_actions.py`
- Modify: `app/ai/workflow/rag_loop.py`
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Modify: `tests/test_rag_dead_code_cleanup.py`
- Modify: affected RAG behavior tests from Tasks 2–14

**Interfaces:**
- Consumes: completed rollout evidence, repository-wide caller inventory, deployment environment inventory, queued Celery task names, and public API contracts.
- Produces: one production owner per RAG responsibility with behavior coverage preserved.

- [ ] **Step 1: Inventory candidates before deleting code**

Create a table with `candidate`, `all_callers`, `replacement`, `compatibility_reason`, `removal_risk`, `proof`, and `decision`. Populate it from repository/config/deployment searches:

```powershell
rg -n "_legacy_char_chunk|_create_chunks_with_page_metadata|_process_excel_workbook|_resolve_mineru_output_dir|_resolve_markdown_file" app tests scripts docs
rg -n "reranker_model|enable_citation_verification|min_citation_coverage|rag_multimodal_image_embeddings_enabled" app tests scripts docs README.md .env.example pyproject.toml requirements.txt
rg -n "SCAN_ALL|get_document_full_content|traditional.rag|set_payload|reindex" app tests scripts docs
rg -n "logger\.(debug|info|warning|error|exception)|^\s*(class|def|async def) " app/services app/ai app/workers
```

Treat Celery task names, dependency-injection providers, Pydantic settings, persisted tool/action names, API schemas, and migrations as externally referenced until runtime/deployment evidence proves otherwise.

- [ ] **Step 2: Rewrite behavior coverage before removing implementation-shaped tests**

Add contract assertions that all supported formats reach `DocumentNormalizer`, all searches use `RAGRetriever`, every accepted answer passes `GroundedAnswerGate`, and old queued task names still resolve when required. Remove a test only after its behavior is covered through the replacement public boundary.

- [ ] **Step 3: Remove duplicated and superseded production paths**

After the inventory marks them removable:

- delete parse/chunk delegation helpers from `DocumentProcessingService`;
- delete `RecursiveCharacterTextSplitter` and legacy rich-chunk assembly from `DocumentParseService`;
- delete `_search` and `_rerank_results` adapters once tool actions consume typed services;
- delete unbounded `get_document_full_content` and old scan behavior;
- delete the `reranker_model` alias after environment and deployment searches show no use;
- delete settings, multimodal stubs, reindex behavior, imports, dependencies, and scripts whose replacements are active;
- preserve all prior Alembic migrations unchanged.

- [ ] **Step 4: Consolidate logs, comments, and docstrings**

Convert interpolated logging to lazy structured arguments, assign each event one owning layer, bound exception detail, and ensure no content, filename, credential, vector, OCR, or base64 enters logs. Retain warning/error/audit events for provider failures, index activation, authorization mismatch, fallback/degraded operation, citation rejection, and rollback.

Keep module/class/public-method docstrings that document contracts or non-obvious invariants. Remove historical phase labels, comments that restate code, inaccurate promises, and private-method docstrings with no added meaning.

- [ ] **Step 5: Prove retired names are absent and behavior remains**

Extend `tests/test_rag_dead_code_cleanup.py` with AST/import assertions for removed helpers and settings. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_dead_code_cleanup.py tests/test_unified_parse_pipeline.py tests/test_document_chunk_builder.py tests/test_rag_embedding_service.py tests/test_document_index_service.py tests/test_rag_retrieval.py tests/test_rag_reranker.py tests/test_rag_evidence.py tests/test_rag_grounding.py tests/test_rag_image_selector.py tests/test_rag_multi_user_isolation.py tests/test_rag_tool_loop_finalization.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app tests scripts
.\.venv\Scripts\python.exe -m alembic heads
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix post-cleanup --compare-baseline scale-qualified
```

Expected: one Alembic head, full test and Ruff success, retired-name scans empty, citation validity unchanged, and no designated quality/latency/cost gate regresses.

- [ ] **Step 6: Update supported-architecture documentation and commit**

Update README, environment examples, and runbooks to describe only the remaining settings and paths. Record every retained compatibility shim and its removal condition in the inventory.

```powershell
git add docs/rag-cleanup-inventory.md app/services/document_processing_service.py app/services/document_parse_service.py app/ai/agents/rag_agent.py app/ai/rag_tool_actions.py app/ai/workflow/rag_loop.py app/core/config.py app/core/container.py README.md .env.example requirements.txt pyproject.toml tests/test_rag_dead_code_cleanup.py tests/test_unified_parse_pipeline.py tests/test_document_chunk_builder.py tests/test_rag_embedding_service.py tests/test_document_index_service.py tests/test_rag_retrieval.py tests/test_rag_reranker.py tests/test_rag_evidence.py tests/test_rag_grounding.py tests/test_rag_image_selector.py tests/test_rag_multi_user_isolation.py tests/test_rag_tool_loop_finalization.py
git commit -m "refactor: consolidate the production RAG pipeline"
```

---

## Final Acceptance Gate

Run the complete verification from a clean worktree and save experiment/load URLs or artifact paths in `docs/rag-rollout-runbook.md`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app tests scripts
.\.venv\Scripts\python.exe -m alembic heads
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix production-candidate --with-ragas --compare-baseline scale-qualified
.\.venv\Scripts\python.exe scripts/benchmark_rag.py --documents 1000 --ingest-concurrency 8 --query-concurrency 32 --failure-injection --output artifacts/rag-production-candidate.json
git status --short
```

Acceptance requires correct Gemini batch shape, one active generation per document, scoped SQL reauthorization, no mandatory corpus scan, structure-aware overlap, calibrated hybrid/reranked retrieval, bounded evidence and images, machine-valid citations, one-regeneration abstention, deterministic plus calibrated RAGAS metrics, visible cost/p95/p99 latency, a passing 1,000-document run, and no unapproved cleanup candidate removed.

## Primary API References

- LangSmith dataset versioning and experiments: <https://docs.langchain.com/langsmith/manage-datasets>
- LangSmith code evaluators: <https://docs.langchain.com/langsmith/code-evaluator-sdk>
- RAGAS available metrics: <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>
- Qdrant payload and tenant indexes: <https://qdrant.tech/documentation/manage-data/indexing/>
- Qdrant hybrid/RRF queries: <https://qdrant.tech/documentation/search/hybrid-queries/>
- Gemini Embedding 2 model: <https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2>
- Google Gen AI Python embedding API: <https://googleapis.github.io/python-genai/genai.html>
