from __future__ import annotations

import logging
import re
from typing import Any
from uuid import UUID

from ..core.rich_response import (
    ALLOWED_IMAGE_MIME_TYPES,
    RichDisplayPolicy,
    RichItemType,
)
from .schemas import DocumentAction

logger = logging.getLogger(__name__)

_RAG_ACTION_TOOL_NAMES = {action.value for action in DocumentAction}


def canonicalize_rag_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Map direct RAG action tool calls onto the real search_documents tool.

    Some providers may emit enum values from the search_documents schema
    (for example ``grep_document``) as the function name. Those are not
    standalone tools; they are actions handled by search_documents.
    """
    tool_name = str(tool_call.get("name") or "").strip().lower()
    if tool_name not in _RAG_ACTION_TOOL_NAMES:
        return tool_call

    canonical = dict(tool_call)
    args = canonical.get("args")
    canonical_args = dict(args) if isinstance(args, dict) else {}
    canonical_args.setdefault("action", tool_name)
    canonical["name"] = "search_documents"
    canonical["args"] = canonical_args
    return canonical


def _coerce_uuid_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(UUID(str(value).strip()))
    except (TypeError, ValueError, AttributeError):
        return None


def _normalize_document_reference(value: Any) -> str:
    text = str(value or "").strip().strip("\"'`")
    text = re.sub(r"^\[\s*\d+\s*/\s*\d+\s*\]\s*", "", text)
    text = re.sub(r"^\[\s*(\d+)\s*\]\s*$", r"\1", text)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    lowered = text.casefold()
    for prefix in ("document id:", "id:", "filename:", "name:"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return " ".join(text.casefold().split())


def _filename_stem_reference(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if "." in text:
        text = text.rsplit(".", 1)[0]
    return _normalize_document_reference(text)


def _tool_document_reference(tool_args: dict[str, Any]) -> Any:
    for key in ("document_id", "filename", "name", "document"):
        value = tool_args.get(key)
        if value:
            return value
    return None


async def _resolve_document_reference(
    *,
    rag_agent: Any,
    document_ref: Any,
    conversation_id: str | None,
    user_id: str | None,
) -> str | None:
    uuid_text = _coerce_uuid_text(document_ref)
    if uuid_text is not None:
        return uuid_text
    if not conversation_id:
        return None

    documents = await rag_agent.list_conversation_documents(conversation_id, user_id=user_id)
    if not documents:
        return None

    requested = _normalize_document_reference(document_ref)
    requested_stem = _filename_stem_reference(document_ref)
    matches: list[str] = []

    for index, document in enumerate(documents, 1):
        doc_id = document.get("document_id")
        if not doc_id:
            continue

        filename = document.get("filename") or ""
        candidates = {
            str(index),
            _normalize_document_reference(doc_id),
            _normalize_document_reference(filename),
            _filename_stem_reference(filename),
        }
        if requested in candidates or requested_stem in candidates:
            matches.append(str(doc_id))

    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


def _build_document_image_candidate(image: dict[str, Any]) -> dict[str, Any] | None:
    """Build a public rich-item candidate from a RAG document image record.

    Returns ``None`` for entries missing a stable id or whose MIME type is not
    in the inline-allowed raster set.
    """
    image_id = image.get("id")
    if not image_id:
        return None
    mime_type = image.get("mime_type") or "image/png"
    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        return None
    payload: dict[str, Any] = {"mime_type": mime_type}
    data = image.get("data")
    url = image.get("url")
    if url:
        payload["url"] = str(url)
    elif data:
        payload["data"] = str(data)
    else:
        return None
    caption = image.get("caption")
    if caption:
        payload["description"] = str(caption)
    page = image.get("page_number")
    return {
        "id": f"image:document:{image_id}",
        "type": RichItemType.image.value,
        "source": "rag_document",
        "display_policy": RichDisplayPolicy.inline_only.value,
        "alt_text": str(caption or "Document figure"),
        "title": f"Page {page}" if page is not None else None,
        "payload": payload,
        "provenance": {
            "document_image_id": str(image_id),
            "page_number": page,
        },
    }


def register_document_image_candidates(
    *,
    context: dict[str, Any],
    images: list[dict[str, Any]],
) -> int:
    """Register RAG document images as public rich-item candidates in
    ``context["rich_item_candidates"]``. Returns the number of new entries.

    Candidate ids are derived from the document image row id so they remain
    stable across turns.
    """
    if not images:
        return 0
    existing: list[dict[str, Any]] = list(context.get("rich_item_candidates", []))
    seen_ids = {c.get("id") for c in existing if isinstance(c, dict)}
    added = 0
    for image in images:
        if not isinstance(image, dict):
            continue
        candidate = _build_document_image_candidate(image)
        if candidate is None:
            continue
        if candidate["id"] in seen_ids:
            continue
        existing.append(candidate)
        seen_ids.add(candidate["id"])
        added += 1
    if added:
        context["rich_item_candidates"] = existing
    return added


def merge_agentic_images(
    *,
    context: dict[str, Any],
    new_images: list[dict[str, Any]],
    max_agentic_images: int,
) -> int:
    """Merge images into context['agentic_images'] with dedupe + size cap."""
    if not new_images:
        return 0

    existing = context.get("agentic_images") or []
    existing_ids = {img.get("id") for img in existing if isinstance(img, dict) and img.get("id")}

    added = 0
    for img in new_images:
        if not isinstance(img, dict):
            continue
        img_id = img.get("id")
        if img_id and img_id in existing_ids:
            continue
        existing.append(img)
        if img_id:
            existing_ids.add(img_id)
        added += 1
        if len(existing) >= max_agentic_images:
            break

    if len(existing) > max_agentic_images:
        existing = existing[-max_agentic_images:]

    context["agentic_images"] = existing
    return added


async def execute_search_documents_action(
    *,
    rag_agent: Any,
    conversation_id: str | None,
    tool_args: dict[str, Any],
    context: dict[str, Any],
    max_agentic_images: int,
    user_id: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """
    Execute one search_documents action.

    Returns: (result, normalized_action, evidence) where ``evidence`` carries
    structured per-action data the UI can render directly (chunks with source
    references, document listings, image listings, etc.). Empty dict when the
    action does not produce structured evidence.
    """
    action_raw = tool_args.get("action")
    if isinstance(action_raw, str):
        action = action_raw.strip().lower()
    elif hasattr(action_raw, "value"):
        action = action_raw.value
    else:
        action = str(action_raw or "").strip().lower()

    result = ""
    evidence: dict[str, Any] = {}

    try:
        if action == DocumentAction.SCAN_ALL.value:
            if conversation_id:
                result = await rag_agent.scan_all_documents(conversation_id, user_id=user_id)
            else:
                result = "Error: No conversation_id available for scan"

        elif action == DocumentAction.READ_DOCUMENT.value:
            document_ref = _tool_document_reference(tool_args)
            if document_ref:
                document_id = await _resolve_document_reference(
                    rag_agent=rag_agent,
                    document_ref=document_ref,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                if not document_id:
                    result = f"Document {document_ref} not found or empty"
                    return result, action, evidence

                content = await rag_agent.get_document_full_content(
                    document_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                if content:
                    result = f"DOCUMENT CONTENT ({document_id}):\n\n{content}"
                    evidence["document"] = {
                        "document_id": document_id,
                        "content": content,
                    }
                    if str(document_ref) != document_id:
                        evidence["document"]["requested_reference"] = str(document_ref)
                else:
                    result = f"Document {document_id} not found or empty"
            else:
                result = "Error: document_id required for READ_DOCUMENT"

        elif action == DocumentAction.SEARCH_CHUNKS.value:
            query = tool_args.get("query")
            if query:
                search_results = await rag_agent._search(
                    query,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                if search_results:
                    attached_count = 0
                    try:
                        images_for_chunks = await rag_agent._fetch_images_for_chunks(search_results)
                        attached_count = merge_agentic_images(
                            context=context,
                            new_images=images_for_chunks,
                            max_agentic_images=max_agentic_images,
                        )
                    except Exception:
                        attached_count = 0

                    chunks: list[dict[str, Any]] = []
                    result = "SEARCH RESULTS:\n\n"
                    for i, doc in enumerate(search_results[:10], 1):
                        source = doc.get("source", "unknown")
                        score = doc.get("score", 0)
                        doc_id = doc.get("document_id") or "unknown"

                        page_number = doc.get("page_number")
                        page_start = doc.get("page_start")
                        page_end = doc.get("page_end")

                        image_ids = doc.get("image_ids") or []
                        image_captions = [cap for cap in (doc.get("image_captions") or []) if cap]

                        has_tables = bool(doc.get("has_tables", False))
                        table_count = doc.get("table_count", 0) or 0

                        meta_parts = [f"Document ID: {doc_id}"]
                        if page_number:
                            meta_parts.append(f"Page: {page_number}")
                        elif page_start or page_end:
                            start_label = page_start if page_start is not None else "?"
                            end_label = page_end if page_end is not None else "?"
                            meta_parts.append(f"Pages: {start_label}-{end_label}")

                        if image_ids:
                            meta_parts.append(f"Images: {len(image_ids)}")
                            if image_captions:
                                preview = ", ".join(image_captions[:3])
                                more = "…" if len(image_captions) > 3 else ""
                                meta_parts.append(f"Image captions: {preview}{more}")

                        if has_tables or table_count:
                            meta_parts.append(f"Tables: {int(table_count)}")

                        content_full = doc.get("content") or ""
                        content = content_full[:500]
                        result += (
                            f"[{i}] {source} (score: {score:.2%})\n"
                            f"  {' | '.join(meta_parts)}\n"
                            f"{content}\n\n"
                        )

                        chunks.append(
                            {
                                "rank": i,
                                "source": source,
                                "score": float(score) if isinstance(score, (int, float)) else None,
                                "document_id": doc_id,
                                "chunk_id": doc.get("chunk_id"),
                                "page_number": page_number,
                                "page_start": page_start,
                                "page_end": page_end,
                                "image_ids": list(image_ids),
                                "image_captions": list(image_captions),
                                "has_tables": has_tables,
                                "table_count": int(table_count) if table_count else 0,
                                "content": content_full,
                            }
                        )

                    evidence["chunks"] = chunks
                    if attached_count:
                        evidence["images_attached"] = attached_count
                        result += (
                            f"(Attached {attached_count} image(s) from matching chunks "
                            "for multimodal analysis.)\n"
                        )
                else:
                    result = "No search results found"
            else:
                result = "Error: query required for SEARCH_CHUNKS"

        elif action == DocumentAction.GREP_DOCUMENT.value:
            document_id = tool_args.get("document_id")
            pattern = tool_args.get("pattern")
            if document_id and pattern:
                result = await rag_agent.grep_document(
                    document_id,
                    pattern,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            else:
                result = "Error: document_id and pattern required for GREP_DOCUMENT"

        elif action == DocumentAction.LIST_DOCUMENTS.value:
            if conversation_id:
                documents = await rag_agent.list_conversation_documents(
                    conversation_id, user_id=user_id
                )
                if documents:
                    result = "AVAILABLE DOCUMENTS:\n\n"
                    listing: list[dict[str, Any]] = []
                    for i, doc in enumerate(documents, 1):
                        result += (
                            f"{i}. {doc['filename']} "
                            f"(ID: {doc['document_id']}, "
                            f"Chunks: {doc['chunk_count']})\n"
                        )
                        listing.append(
                            {
                                "rank": i,
                                "filename": doc.get("filename"),
                                "document_id": doc.get("document_id"),
                                "chunk_count": doc.get("chunk_count"),
                            }
                        )
                    evidence["documents"] = listing
                else:
                    result = "No documents found in this conversation"
            else:
                result = "Error: No conversation_id available"

        elif action == DocumentAction.VIEW_IMAGES.value:
            document_id = tool_args.get("document_id")
            if document_id:
                images = await rag_agent.get_document_images(
                    document_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                if images:
                    attached_count = merge_agentic_images(
                        context=context,
                        new_images=images,
                        max_agentic_images=max_agentic_images,
                    )
                    register_document_image_candidates(context=context, images=images)
                    result = f"IMAGES ({len(images)} found, {attached_count} added to context):\n\n"
                    for i, img in enumerate(images, 1):
                        page = img.get("page_number", "?")
                        caption = img.get("caption") or "No caption"
                        img_id = img.get("id")
                        rich_id = f"image:document:{img_id}" if img_id else ""
                        if rich_id:
                            result += f"[{i}] (id: {rich_id}) Page {page}: {caption}\n"
                        else:
                            result += f"[{i}] Page {page}: {caption}\n"
                else:
                    result = f"No images found for document {document_id}"
            else:
                result = "Error: document_id required for VIEW_IMAGES"

        else:
            result = f"Unknown action: {action}"

    except Exception as exc:
        result = f"Error executing {action}: {str(exc)}"

    reason = tool_args.get("reason", "")
    if reason:
        logger.debug("RAG Agentic: %s - %s", action, reason)

    return result, action, evidence
