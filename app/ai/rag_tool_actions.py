from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .schemas import DocumentAction

logger = logging.getLogger(__name__)


def merge_agentic_images(
    *,
    context: Dict[str, Any],
    new_images: List[Dict[str, Any]],
    max_agentic_images: int,
) -> int:
    """Merge images into context['agentic_images'] with dedupe + size cap."""
    if not new_images:
        return 0

    existing = context.get("agentic_images") or []
    existing_ids = {
        img.get("id") for img in existing if isinstance(img, dict) and img.get("id")
    }

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
    conversation_id: Optional[str],
    tool_args: Dict[str, Any],
    context: Dict[str, Any],
    max_agentic_images: int,
) -> Tuple[str, str]:
    """
    Execute one search_documents action.

    Returns: (result, normalized_action)
    """
    action_raw = tool_args.get("action")
    if isinstance(action_raw, str):
        action = action_raw.strip().lower()
    elif hasattr(action_raw, "value"):
        action = action_raw.value
    else:
        action = str(action_raw or "").strip().lower()

    result = ""

    try:
        if action == DocumentAction.SCAN_ALL.value:
            if conversation_id:
                result = await rag_agent.scan_all_documents(conversation_id)
            else:
                result = "Error: No conversation_id available for scan"

        elif action == DocumentAction.READ_DOCUMENT.value:
            document_id = tool_args.get("document_id")
            if document_id:
                content = await rag_agent.get_document_full_content(document_id)
                if content:
                    result = f"DOCUMENT CONTENT ({document_id}):\n\n{content}"
                else:
                    result = f"Document {document_id} not found or empty"
            else:
                result = "Error: document_id required for READ_DOCUMENT"

        elif action == DocumentAction.SEARCH_CHUNKS.value:
            query = tool_args.get("query")
            if query:
                search_results = await rag_agent._search(
                    query, conversation_id=conversation_id
                )
                if search_results:
                    attached_count = 0
                    try:
                        images_for_chunks = await rag_agent._fetch_images_for_chunks(
                            search_results
                        )
                        attached_count = merge_agentic_images(
                            context=context,
                            new_images=images_for_chunks,
                            max_agentic_images=max_agentic_images,
                        )
                    except Exception:
                        attached_count = 0

                    result = "SEARCH RESULTS:\n\n"
                    for i, doc in enumerate(search_results[:10], 1):
                        source = doc.get("source", "unknown")
                        score = doc.get("score", 0)
                        doc_id = doc.get("document_id") or "unknown"

                        page_number = doc.get("page_number")
                        page_start = doc.get("page_start")
                        page_end = doc.get("page_end")

                        image_ids = doc.get("image_ids") or []
                        image_captions = [
                            cap for cap in (doc.get("image_captions") or []) if cap
                        ]

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

                        content = (doc.get("content") or "")[:500]
                        result += (
                            f"[{i}] {source} (score: {score:.2%})\n"
                            f"  {' | '.join(meta_parts)}\n"
                            f"{content}\n\n"
                        )

                    if attached_count:
                        result += f"(Attached {attached_count} image(s) from matching chunks for multimodal analysis.)\n"
                else:
                    result = "No search results found"
            else:
                result = "Error: query required for SEARCH_CHUNKS"

        elif action == DocumentAction.GREP_DOCUMENT.value:
            document_id = tool_args.get("document_id")
            pattern = tool_args.get("pattern")
            if document_id and pattern:
                result = await rag_agent.grep_document(document_id, pattern)
            else:
                result = "Error: document_id and pattern required for GREP_DOCUMENT"

        elif action == DocumentAction.LIST_DOCUMENTS.value:
            if conversation_id:
                documents = await rag_agent.list_conversation_documents(conversation_id)
                if documents:
                    result = "AVAILABLE DOCUMENTS:\n\n"
                    for i, doc in enumerate(documents, 1):
                        result += (
                            f"{i}. {doc['filename']} "
                            f"(ID: {doc['document_id']}, "
                            f"Chunks: {doc['chunk_count']})\n"
                        )
                else:
                    result = "No documents found in this conversation"
            else:
                result = "Error: No conversation_id available"

        elif action == DocumentAction.VIEW_IMAGES.value:
            document_id = tool_args.get("document_id")
            if document_id:
                images = await rag_agent.get_document_images(document_id)
                if images:
                    attached_count = merge_agentic_images(
                        context=context,
                        new_images=images,
                        max_agentic_images=max_agentic_images,
                    )
                    result = f"IMAGES ({len(images)} found, {attached_count} added to context):\n\n"
                    for i, img in enumerate(images, 1):
                        page = img.get("page_number", "?")
                        caption = img.get("caption") or "No caption"
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

    return result, action
