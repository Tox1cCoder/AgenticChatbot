"""Re-embed existing PostgreSQL chunks into the active Qdrant collection.

This command never reparses source documents. It reuses chunk text already
stored in PostgreSQL and is typically used when the embedding provider, model,
or vector dimension changes.

Usage::

    python scripts/reindex_embeddings.py --dry-run
    python scripts/reindex_embeddings.py --document-id <uuid>
    python scripts/reindex_embeddings.py --conversation-id <uuid>
    python scripts/reindex_embeddings.py --all --continue-on-error

The script uses the same DI container as the running server, so it picks
up the active ``rag_embedding_service`` adapter (Gemini in production)
and writes points to ``settings.qdrant_collection_name``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from uuid import UUID

# Allow running as ``python scripts/reindex_embeddings.py`` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import text  # noqa: E402

logger = logging.getLogger("reindex_embeddings")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-embed existing document chunks into the active Qdrant collection"
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--document-id", type=str, default=None)
    target.add_argument("--conversation-id", type=str, default=None)
    target.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Activate each verified replacement (default: leave it ready for inspection)",
    )
    parser.add_argument(
        "--keep-retired-hours",
        type=int,
        default=168,
        help="Purge retired generations older than this after successful activation",
    )
    return parser.parse_args(argv)


def _resolve_targets(db, args: argparse.Namespace) -> list[UUID]:
    with db.session() as session:
        if args.document_id:
            rows = session.execute(
                text("SELECT id FROM documents WHERE id = :id"),
                {"id": args.document_id},
            ).all()
        elif args.conversation_id:
            rows = session.execute(
                text("SELECT DISTINCT d.id FROM documents d WHERE d.conversation_id = :cid"),
                {"cid": args.conversation_id},
            ).all()
        elif args.all:
            rows = session.execute(text("SELECT DISTINCT document_id FROM document_chunks")).all()
        else:
            # Dry run without a selector — surface the existing needs_reindex queue.
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
    selector_count = sum(bool(x) for x in (args.document_id, args.conversation_id, args.all))
    if not args.dry_run and selector_count == 0:
        logger.error(
            "Refusing to run without an explicit selector. "
            "Use --document-id, --conversation-id, --all, or --dry-run."
        )
        return 2

    from app.core.container import get_container

    container = get_container()
    db = container.db()

    targets = _resolve_targets(db, args)
    logger.info("Found %d documents in scope", len(targets))

    if args.dry_run:
        for doc_id in targets:
            print(str(doc_id))
        print(f"DRY RUN — {len(targets)} documents would be re-embedded")
        return 0

    from app.core.config import settings

    index_service = container.document_index_service()
    embedding_service = container.rag_embedding_service()

    logger.info(
        "Re-embedding with provider=%s model=%s dimension=%d collection=%s",
        getattr(embedding_service, "provider", "unknown"),
        getattr(embedding_service, "model_name", "unknown"),
        getattr(embedding_service, "dimension", 0),
        settings.qdrant_collection_name,
    )

    if not targets:
        logger.info("No documents to re-embed. Nothing to do.")
        return 0

    chunks_scanned = 0
    chunks_reembedded = 0
    chunks_failed = 0
    qdrant_points_written = 0

    for doc_id in targets:
        try:
            previous = index_service.generation_repository.get_active(doc_id)
            chunks = index_service.reindex_document(doc_id, activate=args.activate)
            new_generation_id = chunks[0].index_generation_id if chunks else None
            chunks_scanned += len(chunks)
            chunks_reembedded += len(chunks)
            qdrant_points_written += len(chunks)
            print(
                f"document_id={doc_id} old_generation_id="
                f"{getattr(previous, 'id', None)} new_generation_id={new_generation_id} "
                f"activated={args.activate}"
            )
            if args.activate and args.keep_retired_hours >= 0:
                index_service.purge_retired_after_hours(doc_id, args.keep_retired_hours)
        except Exception as exc:
            chunks_failed += 1
            logger.error("Failed to re-embed %s: %s", doc_id, exc, exc_info=True)
            if not args.continue_on_error:
                break

    print(
        f"chunks_scanned={chunks_scanned} "
        f"chunks_reembedded={chunks_reembedded} "
        f"chunks_failed={chunks_failed} "
        f"qdrant_points_written={qdrant_points_written}"
    )
    return 0 if chunks_failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
