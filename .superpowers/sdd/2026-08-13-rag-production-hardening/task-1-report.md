# Task 1 Report — Establish the Evaluation Contract and Baseline

## Implementation

- Added immutable typed RAG evaluation input, reference, retrieval, evidence, claim, and output contracts.
- Added pure stable-ID retrieval, citation, aggregate-safe abstention, tool-count, latency, token, and cost metrics.
- Added lazy RAGAS collection adapter, fail-closed baseline-relative release-gate comparator, authenticated HTTP target adapter, and offline-safe LangSmith CLI.
- Added a deterministic UUID5 evaluation corpus manifest, controlled text/table/chart fixtures, and 110 versioned golden examples across all required categories and supported languages.
- Added the `eval` optional dependency group containing `ragas>=0.3,<1.0`; RAGAS is not in the server dependency set.

## Files

- `app/evaluation/__init__.py`
- `app/evaluation/rag/{__init__,contracts,metrics,ragas_metrics,target,release_gates,corpus}.py`
- `eval/rag/{corpus_manifest,golden_v1}.jsonl`
- `eval/rag/release_gates.json`
- `eval/rag/fixtures/{warranty.txt,financial_table.csv,chart_description.txt}`
- `scripts/evaluate_rag.py`
- `tests/test_rag_evaluation_contracts.py`
- `tests/test_rag_evaluation_metrics.py`
- `pyproject.toml`

## TDD evidence

### RED

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evaluation_contracts.py tests/test_rag_evaluation_metrics.py
```

Exact result before implementation:

```text
ERROR tests/test_rag_evaluation_contracts.py
ERROR tests/test_rag_evaluation_metrics.py
ModuleNotFoundError: No module named 'app.evaluation'
2 errors in 0.16s
```

### GREEN

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evaluation_contracts.py tests/test_rag_evaluation_metrics.py
```

Exact output:

```text
.....                                                                    [100%]
5 passed in 0.09s
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff check app/evaluation scripts/evaluate_rag.py tests/test_rag_evaluation_contracts.py tests/test_rag_evaluation_metrics.py
```

```text
All checks passed!
```

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --offline
```

```json
{"dataset": "rag-golden-v1", "dataset_tag": "v1", "mode": "offline", "network_calls": 0, "rows_validated": 110}
```

Full suite command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Exact output:

```text
3772 passed, 77 skipped, 30 warnings in 155.56s (0:02:35)
Failed to send compressed multipart ingest: langsmith.utils.LangSmithRateLimitError: Rate limit exceeded for https://api.smith.langchain.com/runs/multipart. HTTPError('429 Client Error: Too Many Requests for url: https://api.smith.langchain.com/runs/multipart', '{"error":"Too many requests: tenant exceeded usage limits: Monthly unique traces usage limit exceeded"}')
```

The 30 warnings are the pre-existing FastAPI duplicate-operation-ID warnings; the LangSmith HTTP 429 occurred after the passing test result while background trace upload was attempted.

Review-fix focused command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_evaluation_contracts.py tests/test_rag_evaluation_metrics.py tests/test_rag_release_gates.py
```

Exact output:

```text
..........                                                               [100%]
10 passed in 0.12s
```

The additional RED evidence was three expected failures before implementing the fixes: the dataset validator did not accept/resolve the manifest, a LangSmith-style object caused `AttributeError: 'Run' object has no attribute 'get'`, and a missing release-gate baseline did not raise `ValueError`. The abstention aggregate test initially failed with `ImportError: cannot import name 'abstention_summary_metrics'`.

Online baseline command:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_rag.py --dataset rag-golden-v1 --dataset-tag v1 --experiment-prefix pre-hardening-baseline
```

Exact output:

```text
LANGSMITH_API_KEY=False RAG_EVALUATION_TARGET_URL=False RAG_EVALUATION_BEARER_TOKEN=False
LangSmith is not configured; run with --offline for local validation.
```

## Self-review

- Stable document and chunk identifiers, not Qdrant point IDs or rank values, drive retrieval metrics and gold labels.
- Dataset contract checks cardinality, required categories, unique IDs, required input/reference fields, and absence of transient point IDs.
- RAGAS imports only when `--with-ragas` is selected; normal server imports do not pull the optional dependency.
- Offline mode reads and validates the immutable local data without initializing LangSmith or making network calls.
- Corpus fixture hashes and UUID5 IDs are verified before injected ingestion callbacks are called.
- The online runner now creates the isolated evaluation scope before evaluation and binds every target call to its generated user/conversation IDs. Deployment-owned authenticated corpus ingestion/index-status endpoints are required for this online-only step.
- LangSmith-style `Run`/`Example` objects are read through their `outputs` fields, and release-gate comparisons fail closed for a missing baseline or candidate metric.
- Per-example abstention evaluation emits TP/FP/FN contributions; a LangSmith summary evaluator computes precision and recall over the full experiment, so false positives and false negatives are not conflated.
- Every answerable gold row now points only to its supporting controlled warranty, table, chart-description, or SVG image fixture. Rows are marked `label_review_status: pending_human_review`; none are represented as already human-reviewed.

## Concerns

- No LangSmith API key or authenticated RAG target credentials were present in this shell, so an immutable online baseline experiment could not be recorded. The known LangSmith monthly trace quota 429 also occurred in background trace upload after the otherwise-passing full suite.
- At least 25 human-reviewed examples are still required before any LLM-judge/RAGAS metric is made a release gate.
- The controlled fixtures establish the executable evaluation contract; the 110 labels still require the specified human review before they are treated as a calibrated production-quality gate.
