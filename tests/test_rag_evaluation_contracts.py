"""Behavior contracts for the versioned RAG evaluation corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from app.evaluation.rag.contracts import RAGEvaluationReference
from app.evaluation.rag.corpus import (
    EvaluationScope,
    load_corpus_manifest,
    load_golden_dataset,
    validate_golden_dataset,
)


def _evaluation_script_module():
    script_path = Path("scripts/evaluate_rag.py")
    spec = importlib.util.spec_from_file_location("evaluate_rag_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REQUIRED_CATEGORIES = {
    "direct_lookup",
    "identifier",
    "table",
    "summary",
    "multi_hop",
    "conflict",
    "unanswerable",
    "distractor",
    "image",
    "prompt_injection",
}


def test_reference_keeps_stable_gold_identifiers():
    reference = RAGEvaluationReference(
        relevant_document_ids={"doc-a"},
        relevant_chunk_ids={"chunk-a"},
    )

    assert reference.relevant_document_ids == frozenset({"doc-a"})
    assert reference.relevant_chunk_ids == frozenset({"chunk-a"})


def test_golden_dataset_is_versioned_complete_and_uses_stable_ids():
    dataset_path = Path("eval/rag/golden_v1.jsonl")
    rows = load_golden_dataset(dataset_path)
    manifest = load_corpus_manifest("eval/rag/corpus_manifest.jsonl")

    validate_golden_dataset(rows, manifest)

    assert 100 <= len(rows) <= 300
    assert {row["metadata"]["category"] for row in rows} >= REQUIRED_CATEGORIES
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["inputs"]["question"] for row in rows}) == len(rows)
    assert len({row["reference"]["answer"] for row in rows}) >= 200
    assert {"en", "th", "vi"} <= {row["metadata"]["language"] for row in rows}
    assert all("point_id" not in json.dumps(row["reference"]) for row in rows)


def test_online_runner_seeds_a_scope_and_binds_target_calls_to_it():
    script = _evaluation_script_module()
    scope = EvaluationScope("tenant", "scoped-user", "scoped-conversation", ("doc-a",))
    captured: dict[str, object] = {}

    class Client:
        def list_examples(self, **kwargs):
            captured["examples"] = kwargs
            return ["example"]

        def evaluate(self, target, **kwargs):
            captured["target"] = target
            captured["evaluate"] = kwargs
            return "recorded"

    client, results = script.run_online(
        argparse.Namespace(
            dataset="rag-golden-v1",
            dataset_tag="v1",
            experiment_prefix="baseline",
            offline=False,
            with_ragas=False,
        ),
        client_factory=Client,
        target_factory=lambda: lambda payload: payload,
        prepare_scope=lambda: scope,
    )

    assert client.__class__ is Client
    assert results == "recorded"
    assert captured["target"]({"question": "q", "user_id": "old", "conversation_id": "old"}) == {
        "question": "q",
        "user_id": "scoped-user",
        "conversation_id": "scoped-conversation",
    }
