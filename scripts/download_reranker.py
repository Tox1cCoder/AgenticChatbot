"""Pre-download the cross-encoder reranker into the local HuggingFace cache.

The RAG agent loads its reranker offline-first (``local_files_only=True``) so a
slow or unreachable ``huggingface.co`` cannot time out cold start. That only
works once the model is present locally. Run this script once on any new
machine, container image, or CI worker to warm the cache; afterwards the server
never touches the network for the reranker.

Usage::

    python scripts/download_reranker.py
    python scripts/download_reranker.py --model cross-encoder/ms-marco-MiniLM-L-6-v2
    python scripts/download_reranker.py --retries 5

The default model is read from ``settings.rag_reranker_model`` (falling back to
the legacy ``settings.reranker_model`` alias), so it always matches what the
server loads.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as ``python scripts/download_reranker.py`` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sentence_transformers import CrossEncoder  # noqa: E402

from app.core.config import settings  # noqa: E402

logger = logging.getLogger("download_reranker")


def _default_model() -> str:
    return getattr(settings, "rag_reranker_model", None) or settings.reranker_model


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download the cross-encoder reranker into the local cache"
    )
    parser.add_argument(
        "--model",
        default=_default_model(),
        help="HuggingFace cross-encoder id (default: the active reranker setting)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of download attempts before giving up (default: 3)",
    )
    return parser.parse_args(argv)


def download(model_name: str, retries: int) -> int:
    """Fetch ``model_name`` with bounded retries, then confirm an offline load."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading reranker '%s' (attempt %d/%d)", model_name, attempt, retries)
            CrossEncoder(model_name)
            break
        except Exception as exc:  # network/timeout errors vary by backend
            last_error = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                backoff = 2**attempt
                logger.info("Retrying in %ds...", backoff)
                time.sleep(backoff)
    else:
        logger.error(
            "Failed to download '%s' after %d attempts: %s",
            model_name,
            retries,
            last_error,
        )
        return 1

    # Prove the server's offline-first path will now succeed without the network.
    CrossEncoder(model_name, local_files_only=True)
    logger.info("Reranker '%s' is cached and loads offline. Done.", model_name)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    return download(args.model, args.retries)


if __name__ == "__main__":
    raise SystemExit(main())
