"""Controlled corpus loading for reproducible, tenant-scoped evaluation."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

EVALUATION_NAMESPACE = uuid.UUID("ff0ac59d-5f65-5c41-b09e-f24ccb75dfcb")
EVALUATION_TENANT_NAMESPACE = uuid.UUID("6f9b682a-a340-54fe-b7ed-a3409fbb76f7")


@dataclass(frozen=True)
class EvaluationScope:
    tenant_id: str
    user_id: str
    conversation_id: str
    document_ids: tuple[str, ...]


def deterministic_document_id(source_path: str) -> str:
    """Derive a durable evaluation document ID from its fixture path."""
    return str(uuid.uuid5(EVALUATION_NAMESPACE, source_path.replace("\\", "/")))


def load_golden_dataset(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def validate_golden_dataset(
    rows: Iterable[Mapping[str, Any]], manifest: Iterable[Mapping[str, Any]] | None = None
) -> None:
    materialized = list(rows)
    if not 100 <= len(materialized) <= 300:
        raise ValueError("golden dataset must contain between 100 and 300 rows")
    ids = [str(row.get("id", "")) for row in materialized]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("golden dataset IDs must be non-empty and unique")
    categories = {str(row.get("metadata", {}).get("category", "")) for row in materialized}
    required_categories = {
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
    missing = required_categories - categories
    if missing:
        raise ValueError(f"golden dataset is missing categories: {', '.join(sorted(missing))}")
    manifest_document_ids = (
        {str(entry["document_id"]) for entry in manifest} if manifest is not None else None
    )
    for row in materialized:
        inputs = row.get("inputs", {})
        reference = row.get("reference", {})
        if not {"question", "user_id", "conversation_id"} <= set(inputs):
            raise ValueError(f"golden row {row.get('id')} has incomplete inputs")
        if not {"relevant_document_ids", "relevant_spans", "should_abstain"} <= set(reference):
            raise ValueError(f"golden row {row.get('id')} has incomplete reference")
        if "point_id" in json.dumps(reference).lower():
            raise ValueError(f"golden row {row.get('id')} contains a transient point ID")
        if manifest_document_ids is not None:
            labeled_ids = set(reference["relevant_document_ids"])
            labeled_ids.update(span["document_id"] for span in reference["relevant_spans"])
            unresolved = labeled_ids - manifest_document_ids
            if unresolved:
                raise ValueError(
                    f"golden row {row.get('id')} references unknown documents: "
                    f"{', '.join(sorted(unresolved))}"
                )


def load_corpus_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    entries = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for entry in entries:
        fixture = manifest_path.parents[2] / entry["path"]
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"fixture hash mismatch: {entry['path']}")
        expected_document_id = deterministic_document_id(entry["path"])
        if entry.get("document_id") != expected_document_id:
            raise ValueError(f"manifest document ID is not deterministic: {entry['path']}")
    return entries


def seed_evaluation_corpus(
    manifest_path: str | Path,
    ingest_document: Callable[..., Any],
    wait_for_active_generation: Callable[[EvaluationScope], Any],
) -> EvaluationScope:
    """Ingest the controlled fixtures under a deterministic isolated tenant.

    ``ingest_document`` and ``wait_for_active_generation`` are injected so the
    evaluator does not couple the application server to a test-only ingestion API.
    """
    manifest = load_corpus_manifest(manifest_path)
    tenant_id = str(uuid.uuid5(EVALUATION_TENANT_NAMESPACE, "rag-evaluation-v1"))
    user_id = str(uuid.uuid5(EVALUATION_TENANT_NAMESPACE, "rag-evaluation-user-v1"))
    conversation_id = str(uuid.uuid5(EVALUATION_TENANT_NAMESPACE, "rag-evaluation-conversation-v1"))
    scope = EvaluationScope(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        document_ids=tuple(entry["document_id"] for entry in manifest),
    )
    for entry in manifest:
        result = ingest_document(entry=entry, scope=scope)
        if inspect.isawaitable(result):
            raise RuntimeError("async corpus ingestion must be awaited by the caller")
    result = wait_for_active_generation(scope)
    if inspect.isawaitable(result):
        raise RuntimeError("async index wait must be awaited by the caller")
    return scope


def prepare_evaluation_scope_from_environment() -> EvaluationScope:
    """Seed and wait for the isolated corpus through deployment-owned endpoints."""
    ingest_url = os.getenv("RAG_EVALUATION_CORPUS_INGEST_URL")
    status_url = os.getenv("RAG_EVALUATION_INDEX_STATUS_URL")
    token = os.getenv("RAG_EVALUATION_BEARER_TOKEN")
    if not ingest_url or not status_url or not token:
        raise RuntimeError(
            "RAG_EVALUATION_CORPUS_INGEST_URL, RAG_EVALUATION_INDEX_STATUS_URL, and "
            "RAG_EVALUATION_BEARER_TOKEN are required for an online evaluation"
        )
    headers = {"Authorization": f"Bearer {token}"}

    def ingest_document(*, entry: Mapping[str, Any], scope: EvaluationScope) -> None:
        fixture_path = Path(__file__).resolve().parents[3] / entry["path"]
        response = httpx.post(
            ingest_url,
            headers=headers,
            json={
                "tenant_id": scope.tenant_id,
                "user_id": scope.user_id,
                "conversation_id": scope.conversation_id,
                "document_id": entry["document_id"],
                "path": entry["path"],
                "media_type": entry["media_type"],
                "sha256": entry["sha256"],
                "content": fixture_path.read_text(encoding="utf-8"),
            },
            timeout=120.0,
        )
        response.raise_for_status()

    def wait_for_active_generation(scope: EvaluationScope) -> None:
        deadline = time.monotonic() + 300.0
        while True:
            response = httpx.post(
                status_url,
                headers=headers,
                json={
                    "tenant_id": scope.tenant_id,
                    "conversation_id": scope.conversation_id,
                    "document_ids": list(scope.document_ids),
                },
                timeout=30.0,
            )
            response.raise_for_status()
            if response.json().get("active_generation"):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("evaluation corpus did not reach an active index generation")
            time.sleep(1.0)

    manifest_path = Path(__file__).resolve().parents[3] / "eval" / "rag" / "corpus_manifest.jsonl"
    return seed_evaluation_corpus(manifest_path, ingest_document, wait_for_active_generation)
