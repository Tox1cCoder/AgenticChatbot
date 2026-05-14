from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_ACTION_TITLES = {
    "scan_all": "Scan All Documents",
    "list_documents": "List Documents",
    "search_chunks": "Search Chunks",
    "read_document": "Read Document",
    "grep_document": "Grep Document",
    "view_images": "View Images",
}


@dataclass(frozen=True)
class RAGChunkView:
    rank: int
    source: str
    score: float | None
    document_id: str | None
    chunk_id: str | None
    page_label: str | None
    content: str
    image_count: int
    image_captions: tuple[str, ...]
    has_tables: bool
    table_count: int


@dataclass(frozen=True)
class RAGDocumentListing:
    rank: int
    filename: str | None
    document_id: str | None
    chunk_count: int | None


@dataclass(frozen=True)
class RAGArtifactView:
    tool_call_id: str
    action: str
    title: str
    query: str | None
    document_id: str | None
    status: str
    output: str
    preview: str
    blob_id: str | None = None
    blob_size_bytes: int | None = None
    chunks: tuple[RAGChunkView, ...] = ()
    documents: tuple[RAGDocumentListing, ...] = ()


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _preview(value: str, max_chars: int = 1200) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def extract_rag_artifact_views(metadata: dict[str, Any] | None) -> list[RAGArtifactView]:
    if not isinstance(metadata, dict):
        return []

    raw_artifacts = metadata.get("tool_artifacts")
    if not isinstance(raw_artifacts, list):
        return []

    views: list[RAGArtifactView] = []
    for index, artifact in enumerate(raw_artifacts, start=1):
        if not isinstance(artifact, dict):
            continue
        if artifact.get("tool") != "search_documents":
            continue

        args = artifact.get("args") if isinstance(artifact.get("args"), dict) else {}
        action = _string_value(args.get("action") or "search_chunks").strip() or "search_chunks"
        output = _string_value(
            artifact.get("output") if artifact.get("output") is not None else artifact.get("result")
        )
        if not output.strip() and artifact.get("error"):
            output = _string_value(artifact.get("error"))

        blob_id_raw = artifact.get("blob_id")
        blob_id = _string_value(blob_id_raw).strip() or None if blob_id_raw else None
        blob_size_raw = artifact.get("blob_size_bytes")
        try:
            blob_size = int(blob_size_raw) if blob_size_raw is not None else None
        except (TypeError, ValueError):
            blob_size = None

        evidence = (
            artifact.get("rag_evidence") if isinstance(artifact.get("rag_evidence"), dict) else {}
        )
        chunks = tuple(
            _build_chunk_views(evidence.get("chunks") if isinstance(evidence, dict) else None)
        )
        documents = tuple(
            _build_document_views(evidence.get("documents") if isinstance(evidence, dict) else None)
        )

        views.append(
            RAGArtifactView(
                tool_call_id=_string_value(artifact.get("tool_call_id") or f"rag-artifact-{index}"),
                action=action,
                title=_ACTION_TITLES.get(action, action.replace("_", " ").title()),
                query=_string_value(args.get("query")).strip() or None,
                document_id=_string_value(args.get("document_id")).strip() or None,
                status=_string_value(artifact.get("status") or "success").strip() or "success",
                output=output,
                preview=_preview(output),
                blob_id=blob_id,
                blob_size_bytes=blob_size,
                chunks=chunks,
                documents=documents,
            )
        )

    return views


def _format_page_label(chunk: dict[str, Any]) -> str | None:
    page_number = chunk.get("page_number")
    if page_number not in (None, "", 0):
        return f"p. {page_number}"
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    if page_start or page_end:
        start = page_start if page_start is not None else "?"
        end = page_end if page_end is not None else "?"
        if start == end:
            return f"p. {start}"
        return f"pp. {start}-{end}"
    return None


def _build_chunk_views(raw_chunks: Any) -> list[RAGChunkView]:
    if not isinstance(raw_chunks, list):
        return []

    chunk_views: list[RAGChunkView] = []
    for index, chunk in enumerate(raw_chunks, start=1):
        if not isinstance(chunk, dict):
            continue

        score_raw = chunk.get("score")
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None

        image_ids = chunk.get("image_ids") if isinstance(chunk.get("image_ids"), list) else []
        image_captions_raw = (
            chunk.get("image_captions") if isinstance(chunk.get("image_captions"), list) else []
        )
        image_captions = tuple(
            _string_value(caption).strip()
            for caption in image_captions_raw
            if _string_value(caption).strip()
        )

        try:
            table_count = int(chunk.get("table_count") or 0)
        except (TypeError, ValueError):
            table_count = 0

        chunk_views.append(
            RAGChunkView(
                rank=int(chunk.get("rank") or index),
                source=_string_value(chunk.get("source") or "unknown"),
                score=score,
                document_id=_string_value(chunk.get("document_id")).strip() or None,
                chunk_id=_string_value(chunk.get("chunk_id")).strip() or None,
                page_label=_format_page_label(chunk),
                content=_string_value(chunk.get("content")),
                image_count=len(image_ids),
                image_captions=image_captions,
                has_tables=bool(chunk.get("has_tables")),
                table_count=table_count,
            )
        )
    return chunk_views


def _build_document_views(raw_documents: Any) -> list[RAGDocumentListing]:
    if not isinstance(raw_documents, list):
        return []
    listings: list[RAGDocumentListing] = []
    for index, doc in enumerate(raw_documents, start=1):
        if not isinstance(doc, dict):
            continue
        chunk_count_raw = doc.get("chunk_count")
        try:
            chunk_count = int(chunk_count_raw) if chunk_count_raw is not None else None
        except (TypeError, ValueError):
            chunk_count = None
        listings.append(
            RAGDocumentListing(
                rank=int(doc.get("rank") or index),
                filename=_string_value(doc.get("filename")).strip() or None,
                document_id=_string_value(doc.get("document_id")).strip() or None,
                chunk_count=chunk_count,
            )
        )
    return listings
