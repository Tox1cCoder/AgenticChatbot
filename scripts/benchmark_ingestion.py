"""Benchmark ingestion script for RAG pipeline.

Uploads a corpus of files concurrently to the running server's upload API,
polls each document until READY or FAILED (or timeout), and reports per-document
wall time, total corpus throughput, and p50/p95 per file type.

Usage::

    python scripts/benchmark_ingestion.py \\
        --base-url http://localhost:8000 \\
        --email user@example.com \\
        --password mypassword \\
        --conversation-id <uuid> \\
        --corpus-dir ./benchmark_corpus \\
        --parallelism 4 \\
        --timeout 300 \\
        --output benchmark_results.json

    # Or with auth token (takes precedence over --email/--password):
    python scripts/benchmark_ingestion.py \\
        --base-url http://localhost:8000 \\
        --auth-token <bearer_token> \\
        --conversation-id <uuid> \\
        --corpus-dir ./benchmark_corpus \\
        --output benchmark_results.json

Flags:
  --base-url (str): Server base URL (default: http://localhost:8000)
  --email (str): Email for login (ignored if --auth-token provided)
  --password (str): Password for login (ignored if --auth-token provided)
  --auth-token (str): Bearer token (takes precedence over --email/--password)
  --conversation-id (str): Conversation UUID (required)
  --corpus-dir (str): Directory of files to benchmark (required)
  --parallelism (int): Concurrent upload + poll workers (default: 4)
  --timeout (int): Per-document timeout in seconds (default: 300)
  --output (str): JSON output file path (default: benchmark_results.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("benchmark_ingestion")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Supported file extensions (mirrors SUPPORTED_UPLOAD_EXTENSIONS in app/api/documents.py)
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md"}


def _percentile(values: list[float], p: int) -> float:
    """Calculate the p-th percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _get_auth_headers(auth_token: str | None = None) -> dict[str, str]:
    """Build Authorization header if token is provided."""
    if auth_token:
        return {"Authorization": f"Bearer {auth_token}"}
    return {}


def _login(requests, base_url: str, email: str, password: str) -> str:
    """Login and return Bearer token."""
    response = requests.post(
        f"{base_url}/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    token = data.get("data", {}).get("accessToken")
    if not token:
        raise ValueError("No accessToken in login response")
    return token


def _upload_document(
    requests,
    base_url: str,
    conversation_id: str,
    file_path: Path,
    auth_headers: dict[str, str],
) -> tuple[str, str | None]:
    """Upload a single file and return (filename, document_id or None)."""
    with open(file_path, "rb") as f:
        files = {"files": (file_path.name, f, "application/octet-stream")}
        data = {"conversation_id": conversation_id}
        response = requests.post(
            f"{base_url}/documents/uploads",
            files=files,
            data=data,
            headers=auth_headers,
            timeout=30,
        )
    response.raise_for_status()
    resp_data = response.json()
    api_data = resp_data.get("data", {})

    # Extract document_id from the first accepted result
    for result in api_data.get("files", []):
        if result.get("status") == "accepted" and result.get("document") is not None:
            doc = result.get("document", {})
            return file_path.name, doc.get("id")

    raise ValueError(f"No accepted documents in upload response for {file_path.name}")


def _poll_document_status(
    requests,
    base_url: str,
    document_id: str,
    timeout_s: int,
    auth_headers: dict[str, str],
) -> str:
    """Poll document status until READY, FAILED, or timeout.

    Returns the final status string: "READY", "FAILED", or "TIMEOUT".
    """
    start_time = time.time()
    poll_interval = 2  # seconds

    status_map = {1: "PROCESSING", 2: "READY", 3: "FAILED"}

    while True:
        elapsed = time.time() - start_time

        if elapsed > timeout_s:
            return "TIMEOUT"

        try:
            response = requests.get(
                f"{base_url}/documents/{document_id}",
                headers=auth_headers,
                timeout=10,
            )
            response.raise_for_status()
            resp_data = response.json()
            api_data = resp_data.get("data", {})
            status_int = api_data.get("status")

            if status_int is None:
                logger.error(f"No status in response for document {document_id}")
                return "FAILED"

            status_str = status_map.get(status_int, "UNKNOWN")

            if status_str in ("READY", "FAILED"):
                return status_str

        except Exception as e:
            logger.error(f"Error polling {document_id}: {e}")
            return "FAILED"

        time.sleep(poll_interval)


def _process_document(
    requests,
    base_url: str,
    conversation_id: str,
    file_path: Path,
    timeout_s: int,
    auth_headers: dict[str, str],
) -> dict[str, Any]:
    """Upload and poll a single document.

    Returns a dict with: filename, file_type, document_id, status, elapsed_s
    """
    start_time = time.time()

    try:
        filename, document_id = _upload_document(
            requests, base_url, conversation_id, file_path, auth_headers
        )

        if not document_id:
            return {
                "filename": file_path.name,
                "file_type": file_path.suffix[1:],
                "document_id": None,
                "status": "FAILED",
                "elapsed_s": time.time() - start_time,
            }

        # Poll until completion
        status = _poll_document_status(requests, base_url, document_id, timeout_s, auth_headers)
        elapsed_s = time.time() - start_time

        return {
            "filename": filename,
            "file_type": file_path.suffix[1:],
            "document_id": str(document_id),
            "status": status,
            "elapsed_s": elapsed_s,
        }

    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return {
            "filename": file_path.name,
            "file_type": file_path.suffix[1:],
            "document_id": None,
            "status": "FAILED",
            "elapsed_s": time.time() - start_time,
        }


def _enumerate_corpus_files(corpus_dir: Path) -> list[Path]:
    """Enumerate supported files in corpus directory."""
    if not corpus_dir.is_dir():
        raise ValueError(f"Corpus directory not found: {corpus_dir}")

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(corpus_dir.glob(f"*{ext}"))

    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    # Parse args first (before importing requests so --help works)
    parser = argparse.ArgumentParser(
        description="Benchmark document ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Server base URL (default: http://localhost:8000)",
    )
    parser.add_argument("--email", default=None, help="Email for login")
    parser.add_argument("--password", default=None, help="Password for login")
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Bearer token (takes precedence over --email/--password)",
    )
    parser.add_argument("--conversation-id", required=True, help="Conversation UUID (required)")
    parser.add_argument(
        "--corpus-dir", required=True, help="Directory of files to benchmark (required)"
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Concurrent workers (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-document timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="JSON output file path (default: benchmark_results.json)",
    )

    args = parser.parse_args(argv)

    try:
        import requests
    except ImportError:
        sys.exit("Error: requests module not found. Install it via: pip install requests")

    base_url = args.base_url.rstrip("/")
    conversation_id = args.conversation_id
    corpus_dir = Path(args.corpus_dir)
    parallelism = args.parallelism
    timeout_s = args.timeout
    output_file = args.output

    # Get auth token
    auth_token = args.auth_token
    if not auth_token:
        if not args.email or not args.password:
            parser.error("Either --auth-token or both --email and --password required")
        try:
            auth_token = _login(requests, base_url, args.email, args.password)
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return 1

    auth_headers = _get_auth_headers(auth_token)

    # Enumerate corpus files
    try:
        files = _enumerate_corpus_files(corpus_dir)
    except Exception as e:
        logger.error(f"Error reading corpus: {e}")
        return 1

    if not files:
        logger.warning(f"No supported files found in {corpus_dir}")
        return 1

    logger.info(f"Uploading {len(files)} files with parallelism={parallelism}...")

    # Process documents concurrently
    results = []
    global_start = time.time()

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {
            executor.submit(
                _process_document,
                requests,
                base_url,
                conversation_id,
                file_path,
                timeout_s,
                auth_headers,
            ): (i, file_path)
            for i, file_path in enumerate(files, 1)
        }

        for future in as_completed(futures):
            i, file_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = result["status"]
                elapsed = result["elapsed_s"]
                logger.info(f"[{i}/{len(files)}] {result['filename']} {status} in {elapsed:.1f}s")
            except Exception as e:
                logger.error(f"[{i}/{len(files)}] {file_path.name} ERROR: {e}")
                results.append(
                    {
                        "filename": file_path.name,
                        "file_type": file_path.suffix[1:],
                        "document_id": None,
                        "status": "FAILED",
                        "elapsed_s": time.time() - global_start,
                    }
                )

    total_wall_s = time.time() - global_start

    # Compute summary statistics
    completed = sum(1 for r in results if r["status"] == "READY")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    timed_out = sum(1 for r in results if r["status"] == "TIMEOUT")

    # Per-document timings for percentile calculation
    ready_times = [r["elapsed_s"] for r in results if r["status"] == "READY"]
    p50_s = _percentile(ready_times, 50)
    p95_s = _percentile(ready_times, 95)

    # By file type
    by_type: dict[str, dict[str, Any]] = {}
    for result in results:
        file_type = result["file_type"]
        if file_type not in by_type:
            by_type[file_type] = {"times": [], "count": 0}

        if result["status"] == "READY":
            by_type[file_type]["times"].append(result["elapsed_s"])
            by_type[file_type]["count"] += 1

    by_type_summary = {}
    for file_type, data in by_type.items():
        times = data["times"]
        by_type_summary[file_type] = {
            "p50_s": _percentile(times, 50),
            "p95_s": _percentile(times, 95),
            "n": len(times),
        }

    # Print summary
    logger.info("")
    logger.info("=== Benchmark Report ===")
    logger.info(f"Corpus: {len(results)} files, {parallelism} workers")
    logger.info(f"Total wall time: {total_wall_s:.1f}s (longest document)")
    logger.info("")
    logger.info("Per-file results:")
    for i, result in enumerate(results, 1):
        status = result["status"]
        elapsed = result["elapsed_s"]
        file_type = result["file_type"]
        logger.info(
            f"  [{i}/{len(results)}] {result['filename']:<30} "
            f"{status:<12} {elapsed:>7.1f}s  [{file_type}]"
        )

    if by_type_summary:
        logger.info("")
        logger.info("By file type:")
        for file_type in sorted(by_type_summary.keys()):
            stats = by_type_summary[file_type]
            logger.info(
                f"  {file_type:<6} p50={stats['p50_s']:>7.1f}s  "
                f"p95={stats['p95_s']:>7.1f}s  n={stats['n']}"
            )

    # Write JSON output
    json_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "parallelism": parallelism,
        "total_wall_s": round(total_wall_s, 1),
        "documents": results,
        "summary": {
            "completed": completed,
            "failed": failed,
            "timed_out": timed_out,
            "p50_s": round(p50_s, 1),
            "p95_s": round(p95_s, 1),
            "by_type": by_type_summary,
        },
    }

    with open(output_file, "w") as f:
        json.dump(json_output, f, indent=2)

    logger.info(f"Results written to: {output_file}")

    # Exit code: 0 if no timeouts, 1 if any timeout
    return 1 if timed_out > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
