"""One-shot reindex utility for document chunks.

Runs inside the canonical server's DI container so it reuses the same
repositories, embedding model, and Qdrant client as normal processing.

Usage::

    python scripts/reindex_documents.py --dry-run
    python scripts/reindex_documents.py --document-id <uuid>
    python scripts/reindex_documents.py --conversation-id <uuid>
    python scripts/reindex_documents.py --all --continue-on-error

Behavior:
  * ``--dry-run`` prints the documents that would be reindexed and exits 0.
  * ``--document-id`` / ``--conversation-id`` / ``--all`` select the target
    set. Exactly one must be supplied (or ``--dry-run`` alone).
  * Rows are first marked ``index_status = 'needs_reindex'`` so a partial
    run leaves a consistent resumable state.
  * Each document is handed to ``DocumentIndexService.reindex_document``.
  * The script exits non-zero on the first failure unless
    ``--continue-on-error`` is passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from uuid import UUID

# Allow running as ``python scripts/reindex_documents.py`` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import text  # noqa: E402

logger = logging.getLogger("reindex_documents")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reindex document chunks")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--document-id", type=str, default=None)
    target.add_argument("--conversation-id", type=str, default=None)
    target.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _resolve_targets(db, args: argparse.Namespace) -> list[UUID]:
    """Return the list of document IDs we plan to touch."""
    with db.session() as session:
        if args.document_id:
            rows = session.execute(
                text("SELECT id FROM documents WHERE id = :id"),
                {"id": args.document_id},
            ).all()
        elif args.conversation_id:
            rows = session.execute(
                text("SELECT id FROM documents WHERE conversation_id = :cid"),
                {"cid": args.conversation_id},
            ).all()
        elif args.all:
            rows = session.execute(text("SELECT id FROM documents")).all()
        else:
            # Dry run without a selector — use the needs_reindex queue.
            rows = session.execute(
                text(
                    "SELECT DISTINCT document_id FROM document_chunks "
                    "WHERE index_status = 'needs_reindex'"
                )
            ).all()
        return [UUID(str(r[0])) for r in rows]


def _mark_needs_reindex(db, document_ids: list[UUID]) -> None:
    if not document_ids:
        return
    with db.session() as session:
        session.execute(
            text(
                "UPDATE document_chunks "
                "SET index_status = 'needs_reindex', "
                "    index_error = NULL, "
                "    updated_at = NOW() "
                "WHERE document_id = ANY(CAST(:ids AS uuid[]))"
            ),
            {"ids": [str(doc_id) for doc_id in document_ids]},
        )
        session.commit()


def _run(args: argparse.Namespace) -> int:
    # Late imports: the container pulls in the whole app, which we only want
    # to pay for once we've validated CLI args.
    from app.core.container import get_container

    container = get_container()
    db = container.db()
    index_service = container.document_index_service()

    selector_count = sum(bool(x) for x in (args.document_id, args.conversation_id, args.all))
    if not args.dry_run and selector_count == 0:
        logger.error(
            "Refusing to run without an explicit selector. "
            "Use --document-id, --conversation-id, --all, or --dry-run."
        )
        return 2

    targets = _resolve_targets(db, args)
    logger.info("Found %d documents in scope", len(targets))

    if args.dry_run:
        for doc_id in targets:
            print(str(doc_id))
        print(f"DRY RUN — {len(targets)} documents would be reindexed")
        return 0

    if not targets:
        logger.info("No documents to reindex. Nothing to do.")
        return 0

    _mark_needs_reindex(db, targets)

    documents_scanned = len(targets)
    documents_reindexed = 0
    documents_failed = 0
    chunks_written = 0
    qdrant_points_written = 0

    for doc_id in targets:
        try:
            chunks = index_service.reindex_document(doc_id)
            documents_reindexed += 1
            chunks_written += len(chunks)
            qdrant_points_written += len(chunks)
            logger.info("Reindexed %s (%d chunks)", doc_id, len(chunks))
        except Exception as exc:
            documents_failed += 1
            logger.error("Failed to reindex %s: %s", doc_id, exc, exc_info=True)
            if not args.continue_on_error:
                break

    print(
        f"documents_scanned={documents_scanned} "
        f"documents_reindexed={documents_reindexed} "
        f"documents_failed={documents_failed} "
        f"chunks_written={chunks_written} "
        f"qdrant_points_written={qdrant_points_written}"
    )
    return 0 if documents_failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
