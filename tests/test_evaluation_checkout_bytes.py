"""Evaluation evidence must keep its recorded hashes on Windows checkouts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from app.evaluation.rag.corpus import load_corpus_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _windows_checkout(tmp_path: Path, files: list[str]) -> Path:
    destination = tmp_path / "checkout"
    destination.mkdir()
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "checkout-index",
            f"--prefix={destination.as_posix()}/",
            "--",
            *files,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return destination


def test_rag_fixture_evidence_survives_windows_checkout(tmp_path):
    manifest = "eval/rag/corpus_manifest.jsonl"
    entries = [json.loads(line) for line in (REPO_ROOT / manifest).read_text("utf-8").splitlines()]
    checkout = _windows_checkout(tmp_path, [manifest, *[entry["path"] for entry in entries]])

    checked_entries = load_corpus_manifest(checkout / manifest)

    assert len(checked_entries) == len(entries)


def test_routing_review_stays_bound_to_windows_checkout_dataset(tmp_path):
    dataset = "eval/routing/golden_v1.jsonl"
    review = "eval/routing/golden_v1.review.json"
    checkout = _windows_checkout(tmp_path, [dataset, review])
    recorded = json.loads((checkout / review).read_text("utf-8"))

    actual = hashlib.sha256((checkout / dataset).read_bytes()).hexdigest()
    assert actual == recorded["dataset_sha256"]
