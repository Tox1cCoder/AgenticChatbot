import base64
import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
)
from sentence_transformers import CrossEncoder
from sqlalchemy import func

from ...core.config import Settings, settings
from ...core.runtime_modeling import ResolvedRuntimeModelConfig
from ...database.session import SessionLocal
from ...interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ...models.conversation import Conversation
from ...models.document import Document
from ...models.document_chunk import DocumentChunk
from ...observability.conversation_compaction import conversation_compaction_metrics
from ...repositories.document_chunk import DocumentChunkRepository
from ...repositories.document_image import DocumentImageRepository
from ..context_overflow import is_context_overflow_error, prepare_aggressive_context_retry
from ..image_context import build_multimodal_content, has_image_parts, image_url_part
from ..mcp_registry import get_global_mcp_manager, get_mcp_tools_generation
from ..model_factory import ModelFactory
from ..prompts import (
    AGENTIC_RAG_SYSTEM_PROMPT,
    MARKDOWN_CURRENCY_GUIDANCE,
    TOOL_EXPLORATION_SUFFIX,
)
from ..rag_tools import create_search_documents_tool
from ..request_budget import ContextBudgetExceededError
from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..token_instrumentation import compute_token_breakdown, extract_actual_usage
from ..utils import coerce_response_text
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)
_RERANKER_INIT_LOCK = threading.Lock()

if TYPE_CHECKING:
    from ...usage.recorder import ModelUsageRecorder


def _load_cross_encoder(model_name: str) -> CrossEncoder:
    """Load a cross-encoder, preferring the local cache over the network.

    sentence-transformers revalidates the cached config against huggingface.co
    on every construction unless ``local_files_only`` is set. A slow or
    unreachable hub then times out on that HEAD request (ReadTimeoutError) and
    blocks cold start even though the model is fully cached. Load offline first;
    only reach the network when the model is genuinely missing. Pre-fetch with
    ``scripts/download_reranker.py`` to avoid the one-time download at runtime.
    """
    try:
        return CrossEncoder(model_name, local_files_only=True)
    except OSError:
        logger.info(
            "Reranker '%s' not in local cache; downloading from HuggingFace "
            "(one-time). Pre-fetch with scripts/download_reranker.py to avoid "
            "this at runtime.",
            model_name,
        )
        return CrossEncoder(model_name)


class RAGAgent(BaseAgent):
    def __init__(
        self,
        settings: Settings,
        qdrant_client: QdrantClient,
        embedding_service: Any,
        collection_name: str = "documents_gemini_embedding_2_3072",
        runtime_model_resolver: IRuntimeModelResolver | None = None,
        recorder: "ModelUsageRecorder | None" = None,
    ):
        # Initialise BaseAgent (sets model_name, gemini_client, langchain_model,
        # mcp_manager, tools, skills tracking, etc.)
        super().__init__(
            agent_config_key="rag",
            runtime_model_resolver=runtime_model_resolver,
            recorder=recorder,
        )

        # RAG-specific fields
        self.settings = settings
        self.qdrant_client = qdrant_client
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self.embedding_dimension = settings.rag_embedding_dimension

        # Retrieval parameters
        self.top_k = settings.rag_top_k
        self.score_threshold = settings.rag_score_threshold
        self.enable_reranking = settings.enable_reranking
        self.reranker = None

        # Thinking support
        self._last_thinking_summary = None

        # Agentic RAG tuning — agentic is the only mode.
        self.agentic_max_iterations = settings.agentic_max_iterations
        self.agentic_preview_chars = settings.agentic_preview_chars

        # Initialize re-ranker if enabled
        if self.enable_reranking:
            self._init_reranker()

    # ------------------------------------------------------------------
    # Abstract member implementations
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.RAG

    @property
    def agent_id(self) -> str:
        return "rag_agent"

    def _get_base_system_prompt(self) -> str:
        """Return the agentic RAG system prompt (the only prompt path)."""
        return AGENTIC_RAG_SYSTEM_PROMPT

    def _init_reranker(self):
        model_name = (
            getattr(
                self.settings,
                "rag_reranker_model",
                None,
            )
            or self.settings.reranker_model
        )
        with _RERANKER_INIT_LOCK:
            self.reranker = _load_cross_encoder(model_name)
        logger.debug(f"Re-ranker initialized: {model_name}")

    def _get_full_system_prompt(
        self,
        base_prompt: str,
        *,
        user_id: str | None = None,
        device_id: str | None = None,
    ) -> str:
        """Return base prompt + tool exploration suffix + active skills suffix."""
        prompt = f"{base_prompt}{MARKDOWN_CURRENCY_GUIDANCE}{TOOL_EXPLORATION_SUFFIX}"
        suffix = self._build_skills_suffix(user_id=user_id, device_id=device_id)
        if suffix:
            return f"{prompt}{suffix}"
        return prompt

    def _coerce_temperature(self, value: Any, default: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return float(default)

    async def _init_tools(self):
        current_generation = get_mcp_tools_generation()
        needs_refresh = (
            self.mcp_manager is None or self._tools_generation_seen != current_generation
        )

        if not needs_refresh:
            return

        # Try to load MCP tools
        try:
            self.mcp_manager = await get_global_mcp_manager()
            all_tools = await self.mcp_manager.get_tools()
            unique_tools = self._deduplicate_tools(all_tools)
            self.tools = self._filter_tools_by_allowlist(unique_tools)
            self._tools_generation_seen = current_generation
        except Exception as e:
            logger.error(
                "Failed to get global MCP manager for RAGAgent: %s",
                e,
                exc_info=True,
            )
            self.tools = []
            self.mcp_manager = None  # Mark as failed but continue

        # search_documents is always bound — agentic RAG is the only path.
        search_documents_tool = create_search_documents_tool()
        tool_names = {tool.name for tool in self.tools}
        if search_documents_tool.name not in tool_names:
            self.tools.insert(0, search_documents_tool)

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: str | None = None,
        *,
        internal_tools: list[BaseTool] | None = None,
        handoff_target_descriptions: dict[str, str] | None = None,
    ) -> AgentResponse:
        """Agentic RAG is the only execution path."""
        await self._init_tools()
        return await self._process_message_agentic(
            message,
            conversation_id,
            internal_tools=internal_tools,
            handoff_target_descriptions=handoff_target_descriptions,
        )

    async def _search(
        self,
        query: str,
        top_k: int = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        # Use configured top_k if not specified
        if top_k is None:
            top_k = self.top_k

        query_embedding = list(self.embedding_service.embed_query(query))

        must_conditions: list[FieldCondition] = []
        if conversation_id:
            must_conditions.append(
                FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))
            )
        if user_id:
            must_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))

        search_filter = Filter(must=must_conditions) if must_conditions else None

        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=self.score_threshold,
            query_filter=search_filter,
        ).points

        hydrated_chunks: dict[str, Any] = {}
        chunk_ids: list[UUID] = []
        for result in search_results:
            raw_chunk_id = result.payload.get("chunk_id") if result.payload else None
            if not raw_chunk_id:
                continue
            try:
                chunk_ids.append(UUID(str(raw_chunk_id)))
            except Exception:
                logger.warning("Skipping invalid chunk_id in Qdrant payload: %s", raw_chunk_id)

        if chunk_ids:
            try:
                chunk_repo = DocumentChunkRepository(SessionLocal)
                if user_id is not None or conversation_id is not None:
                    chunks = chunk_repo.get_by_ids_for_scope(
                        chunk_ids,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                else:
                    chunks = chunk_repo.get_by_ids(chunk_ids)
                hydrated_chunks = {str(chunk.id): chunk for chunk in chunks}
            except Exception:
                logger.exception("Failed to hydrate SQL chunks for RAG search")

        image_repo = DocumentImageRepository(SessionLocal) if hydrated_chunks else None

        results = []
        for result in search_results:
            payload = result.payload or {}
            raw_chunk_id = payload.get("chunk_id")
            if not raw_chunk_id:
                logger.warning(
                    "Skipping Qdrant result without chunk_id for document_id=%s",
                    payload.get("document_id"),
                )
                continue

            chunk = hydrated_chunks.get(str(raw_chunk_id)) if raw_chunk_id else None

            if chunk is not None:
                chunk_images = []
                if image_repo is not None:
                    try:
                        chunk_images = image_repo.get_by_chunk_id(chunk.id)
                    except Exception:
                        logger.exception("Failed to hydrate images for chunk %s", chunk.id)

                document = getattr(chunk, "document", None)
                source = getattr(document, "filename", None) or payload.get("source", "unknown")
                page_start = getattr(chunk, "page_start", None)
                page_end = getattr(chunk, "page_end", None)
                page_number = (
                    page_start if page_start is not None and page_start == page_end else None
                )
                image_ids = [str(image.id) for image in chunk_images]
                image_paths = [image.image_path for image in chunk_images]
                image_captions = [image.image_caption or "" for image in chunk_images]
                content = chunk.content
                document_id = str(chunk.document_id)
                chunk_index = chunk.chunk_index
                chunk_metadata = getattr(chunk, "chunk_metadata", None) or {}
            else:
                logger.error(
                    "Qdrant returned chunk_id=%s but no SQL document_chunks row exists",
                    raw_chunk_id,
                )
                continue

            results.append(
                {
                    "content": content,
                    "source": source,
                    "score": result.score,
                    "page_number": page_number,
                    "page_start": page_start,
                    "page_end": page_end,
                    "document_id": document_id,
                    "conversation_id": payload.get("conversation_id") or None,
                    "chunk_id": str(raw_chunk_id) if raw_chunk_id else None,
                    "chunk_index": chunk_index,
                    "has_tables": bool(
                        chunk_metadata.get("has_tables")
                        or chunk_metadata.get("contains_table")
                        or payload.get("has_tables", False)
                    ),
                    "table_count": int(
                        chunk_metadata.get("table_count") or payload.get("table_count", 0) or 0
                    ),
                    "image_ids": image_ids,
                    "image_paths": image_paths,
                    "image_captions": image_captions,
                }
            )

        if self.enable_reranking and len(results) > 3:
            results = await self._rerank_results(query, results)

        return results

    async def _rerank_results(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self.reranker or not results:
            return results

        # Prepare pairs for re-ranking
        pairs = [[query, doc["content"]] for doc in results]

        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Add rerank scores to results
        for i, score in enumerate(rerank_scores):
            results[i]["rerank_score"] = float(score)

        # Sort by rerank score
        results = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Keep only top K after re-ranking
        results = results[: self.settings.rerank_top_k]

        return results

    async def _fetch_images_for_chunks(
        self, retrieved_docs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        images = []
        seen_image_ids = set()

        for doc in retrieved_docs:
            image_ids = doc.get("image_ids", [])
            if not image_ids:
                continue

            for image_id in image_ids:
                if image_id and image_id not in seen_image_ids:
                    seen_image_ids.add(image_id)

        if not seen_image_ids:
            return images

        image_repo = DocumentImageRepository(SessionLocal)

        for image_id in seen_image_ids:
            try:
                image_uuid = UUID(str(image_id))
            except Exception:
                continue

            image = image_repo.get_by_id(image_uuid)
            if not image:
                continue

            image_path = Path(image.image_path)
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path

            if not image_path.exists():
                continue

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            base64_data = base64.b64encode(image_bytes).decode("utf-8")

            images.append(
                {
                    "id": str(image.id),
                    "data": base64_data,
                    "mime_type": image.mime_type,
                    "caption": image.image_caption,
                    "page_number": image.page_number,
                    "source_path": str(image_path),
                }
            )

        return images

    async def initialize(self):
        return True

    async def cleanup(self):
        await super().cleanup()

    def get_status(self) -> dict:
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)

            collection_info = None
            if collection_exists:
                collection_info = self.qdrant_client.get_collection(self.collection_name)

            return {
                "status": "healthy",
                "collection_exists": collection_exists,
                "collection_name": self.collection_name,
                "vectors_count": (collection_info.vectors_count if collection_info else 0),
                "embedding_model": getattr(self.embedding_service, "model_name", "unknown"),
                "embedding_provider": getattr(self.embedding_service, "provider", "unknown"),
                "embedding_dimension": self.embedding_dimension,
            }
        except Exception as e:
            logger.error(f"Error getting RAG agent status: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def delete_document_vectors(self, document_id: str) -> dict:
        try:
            # Delete associated images from database and filesystem
            images_deleted = 0

            image_repo = DocumentImageRepository(SessionLocal)

            # Get image paths before deletion
            image_paths = image_repo.get_image_paths_by_document_id(UUID(document_id))

            # Delete from database
            images_deleted = image_repo.delete_by_document_id(UUID(document_id))

            # Delete image files from filesystem
            for image_path in image_paths:
                # Resolve relative paths to absolute paths
                path_obj = Path(image_path)
                if not path_obj.is_absolute():
                    path_obj = Path.cwd() / path_obj

                if path_obj.exists():
                    path_obj.unlink()

            # Delete document image folder if empty
            doc_image_folder = Path(settings.document_images_storage_path) / document_id
            if doc_image_folder.exists() and not any(doc_image_folder.iterdir()):
                doc_image_folder.rmdir()

            # Delete vectors from Qdrant
            delete_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )

            result = self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=delete_filter),
            )

            return {
                "success": True,
                "document_id": document_id,
                "images_deleted": images_deleted,
                "message": (
                    f"Vectors and {images_deleted} images deleted for document {document_id}"
                ),
                "operation_result": str(result),
            }
        except Exception as e:
            logger.error(f"Error deleting vectors for document {document_id}: {e}", exc_info=True)
            return {"success": False, "document_id": document_id, "error": str(e)}

    # === Agentic RAG Content Retrieval Methods ===

    async def get_document_full_content(
        self,
        document_id: str,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> str | None:
        try:
            chunk_repo = DocumentChunkRepository(SessionLocal)
            doc_uuid = UUID(document_id)

            if user_id is not None or conversation_id is not None:
                chunks = chunk_repo.get_by_document_for_scope(
                    doc_uuid,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            else:
                chunks = chunk_repo.get_by_document_ordered(doc_uuid)

            if not chunks:
                return None

            return "\n\n".join(chunk.content for chunk in chunks)

        except Exception as e:
            logger.error(
                f"Error fetching full content for document {document_id}: {e}",
                exc_info=True,
            )
            return None

    async def get_document_preview(
        self,
        document_id: str,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        max_chars: int | None = None,
        max_chunks: int = 8,
    ) -> str | None:
        """
        Get a preview of a document (first N characters).
        Used for agentic SCAN_ALL action.
        """
        if max_chars is None:
            max_chars = self.agentic_preview_chars

        window = await self.get_document_chunk_window(
            document_id,
            user_id=user_id,
            conversation_id=conversation_id,
            start_chunk=0,
            max_chunks=max_chunks,
        )
        if not window:
            return None

        content = "\n\n".join(chunk["content"] for chunk in window["chunks"])
        if not content:
            return None

        if len(content) > max_chars:
            preview = content[:max_chars]
            preview += "\n\n[PREVIEW - bounded chunk window; use READ_DOCUMENT to continue]"
            return preview

        return content

    async def get_document_chunk_window(
        self,
        document_id: str,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        start_chunk: int = 0,
        max_chunks: int = 8,
    ) -> dict[str, Any] | None:
        """Read one server-bounded chunk window with ownership enforced in SQL."""
        try:
            bounded_start = max(0, int(start_chunk))
            bounded_limit = min(20, max(1, int(max_chunks)))
            chunk_repo = DocumentChunkRepository(SessionLocal)
            chunks = chunk_repo.get_window_for_scope(
                UUID(document_id),
                user_id,
                conversation_id,
                bounded_start,
                bounded_limit,
            )
            if not chunks:
                return None

            selected = [
                {
                    "chunk_index": int(chunk.chunk_index),
                    "content": chunk.content,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_path": list(chunk.section_path or []),
                }
                for chunk in chunks
            ]
            candidate_next = bounded_start + len(selected)
            has_more = len(selected) == bounded_limit and chunk_repo.has_chunk_after_for_scope(
                UUID(document_id),
                user_id,
                conversation_id,
                candidate_next,
            )
            next_start_chunk = candidate_next if has_more else None
            return {
                "document_id": str(document_id),
                "start_chunk": bounded_start,
                "max_chunks": bounded_limit,
                "chunks": selected,
                "next_start_chunk": next_start_chunk,
            }
        except Exception as e:
            logger.error(
                "Error fetching chunk window for document %s: %s",
                document_id,
                e,
                exc_info=True,
            )
            return None

    async def list_conversation_documents(
        self,
        conversation_id: str,
        *,
        user_id: str | None = None,
        page: int = 1,
        page_size: int = 10,
    ) -> dict[str, Any]:
        try:
            bounded_page = max(1, int(page))
            bounded_page_size = min(25, max(1, int(page_size)))
            conversation_uuid = UUID(conversation_id)
            with SessionLocal() as db:
                query = (
                    db.query(
                        Document.id,
                        Document.filename,
                        func.count(DocumentChunk.id).label("chunk_count"),
                    )
                    .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
                    .filter(Document.conversation_id == conversation_uuid)
                )
                if user_id is not None:
                    from ...models.conversation import Conversation

                    query = query.join(
                        Conversation, Document.conversation_id == Conversation.id
                    ).filter(Conversation.owner_id == user_id)
                query = query.group_by(Document.id, Document.filename, Document.upload_time)
                total = int(query.count())
                rows = (
                    query
                    .order_by(Document.upload_time.desc(), Document.id.asc())
                    .offset((bounded_page - 1) * bounded_page_size)
                    .limit(bounded_page_size)
                    .all()
                )

            return {
                "documents": [
                    {
                        "document_id": str(row.id),
                        "filename": row.filename,
                        "chunk_count": int(row.chunk_count or 0),
                    }
                    for row in rows
                ],
                "total": total,
                "page": bounded_page,
                "page_size": bounded_page_size,
            }

        except Exception as e:
            logger.error(
                f"Error listing documents for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            return {
                "documents": [],
                "total": 0,
                "page": max(1, int(page)),
                "page_size": min(25, max(1, int(page_size))),
            }

    async def resolve_document_filename(
        self,
        filename: str,
        *,
        conversation_id: str,
        user_id: str | None = None,
    ) -> str | None:
        """Resolve an exact filename under server-owned SQL scope."""
        try:
            with SessionLocal() as db:
                query = (
                    db.query(Document.id)
                    .join(Conversation, Document.conversation_id == Conversation.id)
                    .filter(Document.conversation_id == UUID(conversation_id))
                    .filter(func.lower(Document.filename) == filename.casefold())
                )
                if user_id is not None:
                    query = query.filter(Conversation.owner_id == user_id)
                rows = query.limit(2).all()
            if len(rows) != 1:
                return None
            return str(rows[0].id)
        except Exception as e:
            logger.error(
                "Error resolving document filename %s: %s",
                filename,
                e,
                exc_info=True,
            )
            return None

    async def grep_document(
        self,
        document_id: str,
        pattern: str,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        start_chunk: int = 0,
        max_chunks: int = 8,
    ) -> str | None:
        """
        Search for a regex pattern in one bounded document chunk window.
        Used for agentic GREP_DOCUMENT action.
        """
        window = await self.get_document_chunk_window(
            document_id,
            user_id=user_id,
            conversation_id=conversation_id,
            start_chunk=start_chunk,
            max_chunks=max_chunks,
        )
        if not window:
            return f"Error: Document {document_id} not found"

        try:
            regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
            content = "\n\n".join(chunk["content"] for chunk in window["chunks"])
            matches = regex.findall(content)

            if matches:
                result = f"MATCHES for '{pattern}' in document:\n\n"
                for i, match in enumerate(matches[:50], 1):  # Limit to 50 matches
                    result += f"{i}. {match}\n"
                if len(matches) > 50:
                    result += f"\n... and {len(matches) - 50} more matches"
                if window["next_start_chunk"] is not None:
                    result += f"\n[next_start_chunk={window['next_start_chunk']}]"
                return result
            else:
                result = f"No matches found for pattern '{pattern}'"
                if window["next_start_chunk"] is not None:
                    result += f"\n[next_start_chunk={window['next_start_chunk']}]"
                return result

        except re.error as e:
            return f"Error: Invalid regex pattern - {e}"

    async def scan_all_documents(
        self,
        conversation_id: str,
        *,
        user_id: str | None = None,
        page: int = 1,
        page_size: int = 10,
    ) -> str:
        """
        Scan one page of documents in a conversation and return bounded previews.
        Used for agentic SCAN_ALL action.
        """
        listing = await self.list_conversation_documents(
            conversation_id,
            user_id=user_id,
            page=page,
            page_size=page_size,
        )
        documents = listing["documents"]

        if not documents:
            return (
                f"DOCUMENT SCAN: Page {listing['page']} with 0 of "
                f"{listing['total']} documents\nNo documents found on this page"
            )

        output = []
        output.append(
            f"DOCUMENT SCAN: Page {listing['page']} with {len(documents)} of "
            f"{listing['total']} documents"
        )

        for i, doc in enumerate(documents, 1):
            doc_id = doc["document_id"]
            filename = doc["filename"]
            chunk_count = doc["chunk_count"]

            output.append(f"[{i}/{len(documents)} on page] {filename}")
            output.append(f"Document ID: {doc_id}")
            output.append(f"Chunks: {chunk_count}")

            # Get preview
            preview = await self.get_document_preview(
                doc_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if preview:
                # Indent preview lines
                preview_lines = preview.split("\n")
                for line in preview_lines[:30]:  # Limit preview lines
                    output.append(f"{line}")
                if len(preview_lines) > 30:
                    output.append("... (preview truncated)")
            else:
                output.append("[Preview unavailable]")

            output.append("")

        output.append("  NEXT STEPS:")
        output.append("  1. Categorize documents as RELEVANT / MAYBE / SKIP")
        output.append("  2. Use bounded READ_DOCUMENT windows for RELEVANT docs")
        output.append("  3. Watch for cross-references to other documents")
        if listing["page"] * listing["page_size"] < listing["total"]:
            output.append(f"  4. Continue enumeration with page={listing['page'] + 1}")

        return "\n".join(output)

    async def get_document_images(
        self,
        document_id: str,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        images = []
        image_repo = DocumentImageRepository(SessionLocal)
        if user_id is not None or conversation_id is not None:
            db_images = image_repo.get_by_document_for_scope(
                document_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
        else:
            db_images = image_repo.get_by_document_id(UUID(document_id))

        for image in db_images:
            image_path = Path(image.image_path)
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path

            if not image_path.exists():
                continue

            with open(image_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")

            images.append(
                {
                    "id": str(image.id),
                    "data": base64_data,
                    "mime_type": image.mime_type,
                    "caption": image.image_caption,
                    "page_number": image.page_number,
                }
            )

        return images

    async def _invoke_agentic_rag_model(
        self,
        *,
        conversation_id: str | None,
        messages: list[Any],
        tools: list[BaseTool],
        disable_tools: bool,
        user_id: str | None,
        runtime_config: ResolvedRuntimeModelConfig,
        has_images: bool = False,
        agentic_images_count: int = 0,
        run_config: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Single agentic RAG model invocation.

        Resolves the runtime model, optionally binds tools through the shared
        ``ModelFactory.bind_tools_to_model`` path, invokes with retries, and
        falls back to the configured provider on errors. Returns an
        ``AgentResponse`` with the standard runtime metadata applied through
        ``_apply_runtime_metadata``.
        """
        current_runtime = runtime_config
        context_overflow_retried = False
        budget_result = None
        system_messages = [message for message in messages if isinstance(message, SystemMessage)]
        non_system_messages = [
            message for message in messages if not isinstance(message, SystemMessage)
        ]
        current_messages = non_system_messages[-1:] if non_system_messages else []
        history_messages = non_system_messages[:-1] if non_system_messages else []
        system_prompt = "\n\n".join(
            coerce_response_text(getattr(message, "content", ""))
            for message in messages
            if isinstance(message, SystemMessage)
        )
        non_system_messages = [
            message for message in messages if not isinstance(message, SystemMessage)
        ]
        token_breakdown = compute_token_breakdown(
            system_prompt=system_prompt,
            history_messages=[],
            current_turn_messages=non_system_messages,
            tools=tools if tools and not disable_tools else None,
        )

        async def _invoke_with_optional_config(
            model: Any,
            model_messages: list[Any],
        ) -> Any:
            if run_config is not None:
                return await self._ainvoke_with_retries(
                    model, model_messages, run_config=run_config
                )
            return await self._ainvoke_with_retries(model, model_messages)

        while True:
            try:
                llm, _ = self._create_langchain_model_from_runtime(
                    current_runtime,
                    user_id=user_id,
                    enable_reasoning_summary=False,
                )
                if disable_tools or not tools:
                    llm_with_tools = llm
                else:
                    llm_with_tools = ModelFactory.bind_tools_to_model(
                        llm,
                        tools,
                        tool_choice=getattr(settings, "tool_choice_mode", "auto"),
                    )
                budget_result = await self._preflight_model_request(
                    current_runtime,
                    system_messages=system_messages,
                    history_messages=history_messages,
                    current_messages=current_messages,
                    tools=[] if disable_tools else tools,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                request_messages = (
                    list(budget_result.envelope.messages) if budget_result is not None else messages
                )
                try:
                    response = await _invoke_with_optional_config(llm_with_tools, request_messages)
                except Exception as exc:
                    if not settings.context_overflow_retry_enabled or not is_context_overflow_error(
                        exc
                    ):
                        raise
                    compacted_messages = prepare_aggressive_context_retry(
                        request_messages,
                        tool_preview_chars=(settings.context_overflow_retry_tool_preview_chars),
                    )
                    try:
                        response = await _invoke_with_optional_config(
                            llm_with_tools,
                            compacted_messages,
                        )
                    except Exception as retry_exc:
                        if is_context_overflow_error(retry_exc):
                            conversation_compaction_metrics.record_provider_overflow_retry(
                                "failure"
                            )
                            raise ContextBudgetExceededError(
                                "provider_context_overflow"
                            ) from retry_exc
                        raise
                    conversation_compaction_metrics.record_provider_overflow_retry("success")
                    context_overflow_retried = True
                runtime_config = current_runtime
                break
            except Exception as exc:
                if isinstance(exc, ContextBudgetExceededError):
                    raise
                logger.warning(
                    "Agentic RAG provider call failed for %s: %s",
                    current_runtime.provider,
                    exc,
                )
                fallback_runtime = self._create_fallback_runtime_config(
                    current_runtime.fallback_config,
                    reason="provider_error",
                    from_provider=current_runtime.provider,
                    inherited_warnings=current_runtime.warnings,
                )
                if not fallback_runtime or fallback_runtime.provider == current_runtime.provider:
                    raise
                current_runtime = fallback_runtime

        response_text = coerce_response_text(response.content or "")
        tool_calls = None
        actual_usage = extract_actual_usage(response)
        if any(value is not None for value in actual_usage.values()):
            token_breakdown.actual_input_tokens = actual_usage.get("input_tokens")
            token_breakdown.actual_output_tokens = actual_usage.get("output_tokens")
            token_breakdown.actual_total_tokens = actual_usage.get("total_tokens")
            token_breakdown.actual_reasoning_tokens = actual_usage.get("reasoning_tokens")

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = response.tool_calls
            logger.debug(
                f"Agentic RAG returned {len(tool_calls)} tool calls: "
                f"{[tc.get('name', tc['name']) for tc in tool_calls]}"
            )

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT,
            content=response_text,
            tool_calls=tool_calls,
        )

        metadata: dict[str, Any] = {
            "conversation_id": conversation_id,
            "agentic_mode": True,
            "token_breakdown": token_breakdown.to_dict(),
        }
        if isinstance(actual_usage.get("reasoning_tokens"), int):
            metadata["reasoning_tokens"] = actual_usage["reasoning_tokens"]
        if agentic_images_count:
            metadata["agentic_images_count"] = agentic_images_count
        if has_images:
            metadata["has_images"] = True
        if disable_tools:
            metadata["disable_tools"] = True
        if context_overflow_retried:
            metadata["context_overflow_retry"] = True
        request_budget_metadata = self._request_budget_metadata(budget_result)
        if request_budget_metadata is not None:
            metadata["request_budget"] = request_budget_metadata
        self._apply_runtime_metadata(metadata, runtime_config)
        self._merge_context_window_usage(metadata, metadata["token_breakdown"])

        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=response_message,
            metadata=metadata,
        )

    async def _process_message_agentic(
        self,
        message: AgentMessage,
        conversation_id: str | None = None,
        *,
        internal_tools: list[BaseTool] | None = None,
        handoff_target_descriptions: dict[str, str] | None = None,
    ) -> AgentResponse:
        """
        Process message using agentic document exploration.

        Uses AGENTIC_RAG_SYSTEM_PROMPT and the search_documents tool.
        Returns AgentResponse with tool_calls if exploration is needed,
        or a final answer if the LLM decides it has enough information.
        """
        query = message.content or ""
        persona = message.metadata.get("persona")
        original_query = message.metadata.get("original_query", query)
        tool_context = message.metadata.get("tool_context", [])
        conversation_history = message.metadata.get("history", [])
        agentic_images = message.metadata.get("agentic_images", [])
        user_attachments = message.attachments or []
        model_request = message.metadata.get("model_request")
        request_user_id = message.metadata.get("user_id")
        request_device_id = message.metadata.get("device_id")
        run_config = message.metadata.get("run_config")

        # Final-synthesis flags forwarded by the graph when the tool budget
        # is exhausted. Honoured by skipping tool binding and appending a
        # budget notice to the system prompt.
        rag_force_final_response = bool(message.metadata.get("rag_force_final_response"))
        rag_tool_budget_notice = message.metadata.get("rag_tool_budget_notice")

        system_prompt = f"{AGENTIC_RAG_SYSTEM_PROMPT}{MARKDOWN_CURRENCY_GUIDANCE}"
        system_prompt = f"{system_prompt}{TOOL_EXPLORATION_SUFFIX}"
        handoff_bound = any(
            getattr(tool, "name", None) == "hand_off" for tool in internal_tools or []
        )
        if handoff_bound:
            system_prompt = (
                f"{system_prompt}{self._build_delegation_suffix(handoff_target_descriptions)}"
            )

        # Append active skills
        skills_suffix = self._build_skills_suffix(
            user_id=request_user_id,
            device_id=request_device_id,
        )
        if skills_suffix:
            system_prompt = f"{system_prompt}{skills_suffix}"

        if rag_force_final_response and rag_tool_budget_notice:
            system_prompt = (
                f"{system_prompt}\n\nTOOL BUDGET NOTICE:\n{str(rag_tool_budget_notice).strip()}"
            )

        if persona:
            system_prompt = f"Custom Persona:\n{persona}\n\n---\n\n{system_prompt}"

        # Ensure tools are initialized (including search_documents)
        if not self.tools:
            await self._init_tools()

        # Get tools for binding - supports deferred loading when enabled
        tools_to_bind = self._get_tools_for_binding(
            conversation_id=conversation_id,
            internal_tools=[create_search_documents_tool(), *(internal_tools or [])],
            user_id=request_user_id,
            device_id=request_device_id,
            include_hand_off=handoff_bound,
        )

        # Build messages list with conversation history
        messages = [SystemMessage(content=system_prompt)]

        # Reuse the base history converter so prior user image attachments stay
        # available when a later follow-up is routed to RAG.
        messages.extend(self._convert_history_to_langchain_messages(conversation_history))

        # Build context for current query
        context_parts = [f"User Question: {original_query}"]
        if tool_context:
            context_parts.append("\nPrevious Tool Results:")
            for i, result in enumerate(tool_context, 1):
                context_parts.append(f"\nTool Call {i} Output:\n{result}")
        context_parts.append(f"\n\nConversation ID: {conversation_id}")
        if not rag_force_final_response:
            context_parts.append(
                "\nUse the search_documents tool to explore documents and find information "
                "to answer the question."
            )

        # Build multimodal content: current-turn user attachments + document images.
        human_content = build_multimodal_content("\n".join(context_parts), user_attachments)

        for img in agentic_images:
            img_data = img.get("data")
            mime_type = img.get("mime_type", "image/jpeg")
            caption = img.get("caption", "")
            page = img.get("page_number", "?")

            if img_data:
                human_content.append(image_url_part(f"data:{mime_type};base64,{img_data}"))
                if caption:
                    human_content.append(
                        {
                            "type": "text",
                            "text": f"[Image from page {page}: {caption}]",
                        }
                    )

        has_prompt_images = has_image_parts(human_content)
        if has_prompt_images:
            messages.append(HumanMessage(content=human_content))
        else:
            messages.append(HumanMessage(content="\n".join(context_parts)))

        runtime_config = self._resolve_runtime_model_config(request_user_id, model_request)
        if has_prompt_images and not runtime_config.capabilities.get("supports_vision", False):
            fallback_runtime = self._create_fallback_runtime_config(
                runtime_config.fallback_config,
                reason="vision_not_supported",
                from_provider=runtime_config.provider,
                inherited_warnings=runtime_config.warnings,
            )
            if fallback_runtime:
                runtime_config = fallback_runtime

        try:
            response = await self._invoke_agentic_rag_model(
                conversation_id=conversation_id,
                messages=messages,
                tools=tools_to_bind,
                disable_tools=rag_force_final_response,
                user_id=request_user_id,
                runtime_config=runtime_config,
                has_images=has_prompt_images,
                agentic_images_count=len(agentic_images) if agentic_images else 0,
                run_config=run_config,
            )
            if rag_force_final_response:
                response.metadata["rag_force_final_response"] = True
            return response
        except Exception as e:
            logger.error(f"Error in agentic RAG processing: {e}", exc_info=True)
            error_metadata = {
                "conversation_id": conversation_id,
                "agentic_mode": True,
                "error": str(e),
            }
            self._apply_runtime_metadata(error_metadata, runtime_config)
            return AgentResponse(
                agent_type=AgentType.RAG,
                agent_id="rag_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"Error during document exploration: {e}",
                ),
                metadata=error_metadata,
                error=str(e),
            )
