from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import UUID

from langchain_core.messages import ToolMessage

from app.ai.token_counter import TokenCounter
from app.services.rag_evidence import EvidenceAssembler
from app.services.rag_retrieval import RetrievalScope

from ..core.rich_response import (
    ALLOWED_IMAGE_MIME_TYPES,
    RichDisplayPolicy,
    RichItemType,
)
from .rich_image_selection import apply_rich_image_selection
from .schemas import DocumentAction
from .tool_error_policy import ToolErrorKind, ToolErrorSummary

logger = logging.getLogger(__name__)

_RAG_ACTION_TOOL_NAMES = {action.value for action in DocumentAction}


def fit_rag_tool_message_content(
    *,
    content: Any,
    allowance: int | None,
    token_counter: Any,
    provider: str,
    model: str,
    tool_call_id: Any,
    tool_name: Any,
) -> tuple[str, int, bool]:
    """Fit one result into its already-reserved empty ToolMessage wrapper.

    Returns ``(model_visible_text, tokens_consumed, omitted)``.

    ``allowance`` is the authoritative model-input remainder in tokens. Pass
    ``None`` when no authoritative allowance was propagated at all: that is an
    *unknown* budget, not a zero-token one, and bounding against zero would
    blank every tool result and starve the loop of its own evidence. In that
    case the result is charged but never replaced.

    A result that does not fit is replaced whole with a compact, constant-size
    JSON omission marker rather than truncated. Splitting JSON, tables, or image
    descriptors yields misleading fragments, and a blank ``ToolMessage`` is
    indistinguishable from an empty-but-successful result, so the model would
    simply call the tool again and spend more budget than the marker costs. The
    caller keeps the complete result in its artifact/blob; only model-visible
    content is bounded here.
    """

    text = str(content or "")

    def content_delta(candidate: str) -> int:
        count_messages = getattr(token_counter, "count_messages", None)
        if callable(count_messages):
            empty = ToolMessage(
                content="",
                tool_call_id=str(tool_call_id or ""),
                name=str(tool_name or "search_documents"),
            )
            actual = ToolMessage(
                content=candidate,
                tool_call_id=str(tool_call_id or ""),
                name=str(tool_name or "search_documents"),
            )
            try:
                empty_count = count_messages(
                    provider=provider,
                    model=model,
                    messages=(empty,),
                )
                actual_count = count_messages(
                    provider=provider,
                    model=model,
                    messages=(actual,),
                )
                return max(0, int(actual_count.tokens) - int(empty_count.tokens))
            except Exception:
                # Counters are duck-typed; one without a usable message-level
                # entry point is handled by the plain-text count below.
                logger.debug("Message-level token delta unavailable", exc_info=True)
        counted = token_counter.count_text(
            provider=provider,
            model=model,
            text=candidate,
        )
        return max(0, int(counted.tokens))

    consumed = content_delta(text)
    if allowance is None or consumed <= max(0, int(allowance)):
        return text, consumed, False

    omission = json.dumps(
        {
            "status": "omitted",
            "reason": "context_budget",
            "tool": str(tool_name or "search_documents"),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return omission, content_delta(omission), True


def compact_rag_tool_error(
    *,
    error_type: str,
    message: str,
    hint: str,
    retryable: bool = False,
) -> str:
    """Render a compact JSON error payload for RAG document tool failures.

    Matches the model-facing contract produced by ``tool_error_policy`` so the
    model sees one consistent error shape across all tool surfaces.
    """
    summary = ToolErrorSummary(
        error_type=error_type,
        failure_retryable=retryable,
        message=message,
        hint=hint,
        attempts=1,
    )
    return json.dumps(
        summary.model_dict(policy_retry_allowed=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )


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


def _tool_document_reference(tool_args: dict[str, Any]) -> Any:
    for key in ("document_id", "filename", "name", "document"):
        value = tool_args.get(key)
        if value:
            return value
    return None


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


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

    requested = _normalize_document_reference(document_ref)
    if not requested or requested.isdigit():
        return None
    return await rag_agent.resolve_document_filename(
        requested,
        conversation_id=conversation_id,
        user_id=user_id,
    )


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
        apply_rich_image_selection(context)
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
    question: str = "",
    subquestions: tuple[str, ...] = (),
    evidence_max_tokens: int = 0,
    evidence_provider: str = "gemini",
    evidence_model: str = "gemini-2.5-flash",
    evidence_token_counter: Any | None = None,
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
    page = _bounded_int(tool_args.get("page"), default=1, minimum=1, maximum=1_000_000)
    page_size = _bounded_int(tool_args.get("page_size"), default=10, minimum=1, maximum=25)
    start_chunk = _bounded_int(
        tool_args.get("start_chunk"), default=0, minimum=0, maximum=1_000_000
    )
    max_chunks = _bounded_int(
        tool_args.get("max_chunks"), default=8, minimum=1, maximum=20
    )

    if action in _RAG_ACTION_TOOL_NAMES and (not user_id or not conversation_id):
        return (
            compact_rag_tool_error(
                error_type=ToolErrorKind.VALIDATION.value,
                message=(
                    "search_documents requires authenticated user and conversation context."
                ),
                hint="Retry within the authenticated conversation that owns the documents.",
            ),
            action,
            evidence,
        )

    try:
        if action == DocumentAction.SCAN_ALL.value:
            if conversation_id:
                result = await rag_agent.scan_all_documents(
                    conversation_id,
                    user_id=user_id,
                    page=page,
                    page_size=page_size,
                )
            else:
                result = compact_rag_tool_error(
                    error_type=ToolErrorKind.VALIDATION.value,
                    message="search_documents has no conversation context to scan.",
                    hint="Retry within an active conversation that has uploaded documents.",
                )

        elif action == DocumentAction.READ_DOCUMENT.value:
            if not conversation_id:
                result = compact_rag_tool_error(
                    error_type=ToolErrorKind.VALIDATION.value,
                    message="read_document has no conversation context.",
                    hint="Retry within the active conversation that owns the document.",
                )
                return result, action, evidence

            document_ref = _tool_document_reference(tool_args)
            if document_ref:
                document_id = await _resolve_document_reference(
                    rag_agent=rag_agent,
                    document_ref=document_ref,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                if not document_id:
                    result = compact_rag_tool_error(
                        error_type=ToolErrorKind.NOT_FOUND.value,
                        message=f"Document {document_ref} was not found.",
                        hint="List available documents first, then read one by its exact id.",
                    )
                    return result, action, evidence

                window = await rag_agent.get_document_chunk_window(
                    document_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    start_chunk=start_chunk,
                    max_chunks=max_chunks,
                )
                if window:
                    chunks = window.get("chunks", [])
                    chunk_text = "\n\n".join(
                        f"[Chunk {chunk['chunk_index']}]\n{chunk['content']}" for chunk in chunks
                    )
                    next_start = window.get("next_start_chunk")
                    result = f"DOCUMENT CHUNKS ({document_id}):\n\n{chunk_text}"
                    if next_start is not None:
                        result += f"\n\n[next_start_chunk={next_start}]"
                    evidence["document"] = {
                        **window,
                        "document_id": document_id,
                    }
                    if str(document_ref) != document_id:
                        evidence["document"]["requested_reference"] = str(document_ref)
                else:
                    result = compact_rag_tool_error(
                        error_type=ToolErrorKind.NOT_FOUND.value,
                        message=f"Document {document_id} was found but has no readable content.",
                        hint="Try another document or use search_chunks for relevant passages.",
                    )
            else:
                result = compact_rag_tool_error(
                    error_type=ToolErrorKind.ARGUMENT.value,
                    message="read_document requires a document_id.",
                    hint="Provide the document_id, or list documents first to find it.",
                )

        elif action == DocumentAction.SEARCH_CHUNKS.value:
            query = tool_args.get("query")
            if query:
                search_results = await rag_agent._search(
                    query,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    include_evidence_metadata=True,
                )
                if search_results:
                    attached_count = 0
                    try:
                        images_for_chunks = await rag_agent._fetch_images_for_chunks(
                            search_results,
                            user_id=user_id,
                            conversation_id=conversation_id,
                        )
                        attached_count = merge_agentic_images(
                            context=context,
                            new_images=images_for_chunks,
                            max_agentic_images=max_agentic_images,
                        )
                    except Exception:
                        attached_count = 0

                    repository = getattr(
                        getattr(rag_agent, "retriever", None),
                        "chunk_repository",
                        None,
                    )
                    assembler = EvidenceAssembler(
                        token_counter=evidence_token_counter or TokenCounter(),
                        provider=evidence_provider,
                        model=evidence_model,
                        repository=repository,
                    )
                    try:
                        typed_conversation_id: Any = UUID(str(conversation_id))
                    except (TypeError, ValueError, AttributeError):
                        typed_conversation_id = conversation_id
                    scope = RetrievalScope(
                        user_id=str(user_id),
                        conversation_id=typed_conversation_id,
                    )
                    pack = await assembler.assemble_exact(
                        question or str(query),
                        search_results[:10],
                        subquestions=subquestions,
                        max_tokens=evidence_max_tokens,
                        scope=scope,
                    )
                    result = pack.to_tool_text()
                    evidence.update(pack.to_dict())
                    if attached_count:
                        evidence["images_attached"] = attached_count
                else:
                    result = "No search results found"
            else:
                result = compact_rag_tool_error(
                    error_type=ToolErrorKind.ARGUMENT.value,
                    message="search_chunks requires a query.",
                    hint="Provide a non-empty query describing what to find.",
                )

        elif action == DocumentAction.GREP_DOCUMENT.value:
            document_id = tool_args.get("document_id")
            pattern = tool_args.get("pattern")
            if document_id and pattern:
                result = await rag_agent.grep_document(
                    document_id,
                    pattern,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    start_chunk=start_chunk,
                    max_chunks=max_chunks,
                )
            else:
                result = compact_rag_tool_error(
                    error_type=ToolErrorKind.ARGUMENT.value,
                    message="grep_document requires both document_id and pattern.",
                    hint="Provide the document_id and a valid regex pattern.",
                )

        elif action == DocumentAction.LIST_DOCUMENTS.value:
            if conversation_id:
                listing_page = await rag_agent.list_conversation_documents(
                    conversation_id,
                    user_id=user_id,
                    page=page,
                    page_size=page_size,
                )
                documents = listing_page["documents"]
                next_page = (
                    listing_page["page"] + 1
                    if listing_page["page"] * listing_page["page_size"]
                    < listing_page["total"]
                    else None
                )
                evidence["pagination"] = {
                    "page": listing_page["page"],
                    "page_size": listing_page["page_size"],
                    "total": listing_page["total"],
                    "next_page": next_page,
                }
                if documents:
                    result = (
                        f"AVAILABLE DOCUMENTS: Page {listing_page['page']} with "
                        f"{len(documents)} of {listing_page['total']} documents\n\n"
                    )
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
                    result = (
                        f"AVAILABLE DOCUMENTS: Page {listing_page['page']} with 0 of "
                        f"{listing_page['total']} documents\n\nNo documents found on this page"
                    )
            else:
                result = compact_rag_tool_error(
                    error_type=ToolErrorKind.VALIDATION.value,
                    message="search_documents has no conversation context.",
                    hint="Retry within an active conversation that has uploaded documents.",
                )

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
                result = compact_rag_tool_error(
                    error_type=ToolErrorKind.ARGUMENT.value,
                    message="view_images requires a document_id.",
                    hint="Provide the document_id, or list documents first to find it.",
                )

        else:
            result = compact_rag_tool_error(
                error_type=ToolErrorKind.VALIDATION.value,
                message="search_documents rejected the requested action.",
                hint="Use one of the supported document exploration actions from the tool schema.",
            )

    except Exception as exc:
        result = f"Error executing {action}: {str(exc)}"

    reason = tool_args.get("reason", "")
    if reason:
        logger.debug("RAG Agentic: %s - %s", action, reason)

    return result, action, evidence
