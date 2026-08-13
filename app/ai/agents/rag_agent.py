import base64
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
)
from sqlalchemy import func

from ...core.config import Settings, settings
from ...core.runtime_modeling import ResolvedRuntimeModelConfig
from ...database.session import SessionLocal
from ...interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ...models.conversation import Conversation
from ...models.document import Document
from ...models.document_chunk import DocumentChunk
from ...observability.conversation_compaction import conversation_compaction_metrics
from ...observability.rag import rag_metrics
from ...repositories.document_chunk import DocumentChunkRepository
from ...repositories.document_image import DocumentImageRepository
from ...services.rag_reranker import RAGReranker
from ...services.rag_retrieval import RAGRetriever, RetrievalCandidate, RetrievalScope
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

if TYPE_CHECKING:
    from ...usage.recorder import ModelUsageRecorder


class RAGAgent(BaseAgent):
    def __init__(
        self,
        settings: Settings,
        qdrant_client: QdrantClient,
        embedding_service: Any,
        collection_name: str = "documents_gemini_embedding_2_3072",
        runtime_model_resolver: IRuntimeModelResolver | None = None,
        recorder: "ModelUsageRecorder | None" = None,
        retriever: RAGRetriever | None = None,
        reranker: RAGReranker | None = None,
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
        self.evidence_candidate_limit = settings.rag_evidence_candidate_limit
        self.reranker = reranker or RAGReranker(
            model_name=settings.rag_reranker_model,
            enabled=settings.enable_reranking,
            candidate_pool=settings.rag_rerank_candidate_pool,
            output_limit=settings.rag_evidence_candidate_limit,
            timeout_seconds=settings.rag_reranker_timeout_seconds,
            max_concurrency=settings.rag_reranker_max_concurrency,
            metrics=rag_metrics,
        )
        self.retriever = retriever or RAGRetriever(
            qdrant_client=qdrant_client,
            embedding_service=embedding_service,
            chunk_repository=DocumentChunkRepository(SessionLocal),
            collection_name=collection_name,
            hybrid_enabled=settings.rag_hybrid_retrieval_enabled,
            dense_candidate_limit=settings.rag_dense_candidate_limit,
            lexical_candidate_limit=settings.rag_lexical_candidate_limit,
            rrf_k=settings.rag_rrf_k,
            score_threshold=settings.rag_score_threshold,
        )

        # Thinking support
        self._last_thinking_summary = None

        # Agentic RAG tuning — agentic is the only mode.
        self.agentic_max_iterations = settings.agentic_max_iterations
        self.agentic_preview_chars = settings.agentic_preview_chars

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
        include_evidence_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        if not user_id or not conversation_id:
            logger.warning("RAG search rejected because authenticated server scope is incomplete")
            return []

        if top_k is None:
            top_k = self.top_k
        try:
            typed_conversation_id: Any = UUID(str(conversation_id))
            legacy_generation_ids = None
        except ValueError:
            # Compatibility for isolated legacy tests; production IDs are UUIDs.
            typed_conversation_id = conversation_id
            legacy_generation_ids = [UUID(int=0)]
        scope = RetrievalScope(
            user_id=user_id,
            conversation_id=typed_conversation_id,
        )
        retriever = getattr(self, "retriever", None)
        if retriever is None:
            retriever = RAGRetriever(
                qdrant_client=self.qdrant_client,
                embedding_service=self.embedding_service,
                chunk_repository=DocumentChunkRepository(SessionLocal),
                collection_name=self.collection_name,
                hybrid_enabled=False,
                dense_candidate_limit=max(self.top_k, top_k),
                lexical_candidate_limit=40,
                rrf_k=60,
                score_threshold=self.score_threshold,
            )
        retrieval_limit = (
            self.settings.rag_rerank_candidate_pool if self.enable_reranking else top_k
        )
        search_kwargs: dict[str, Any] = {"final_limit": retrieval_limit}
        if legacy_generation_ids is not None:
            # Construction-bypassing legacy tests use non-UUID conversation
            # identifiers; production scope always resolves real SQL generations.
            search_kwargs["active_generation_ids"] = legacy_generation_ids
        candidates = retriever.search(query, scope, **search_kwargs)
        if self.reranker is not None:
            candidates = await self.reranker.rank(query, candidates)
        evidence_limit = int(getattr(self, "evidence_candidate_limit", top_k))
        candidates = candidates[: min(top_k, evidence_limit)]

        results: list[dict[str, Any]] = []
        image_repo = DocumentImageRepository(SessionLocal) if candidates else None
        for candidate in candidates:
            chunk_images = []
            if image_repo is not None and candidate.chunk_id is not None:
                try:
                    chunk_images = image_repo.get_by_chunk_id_for_scope(
                        candidate.chunk_id,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                except Exception:
                    logger.exception("Failed to hydrate images for chunk %s", candidate.chunk_id)
            metadata = candidate.metadata or {}
            page_number = (
                candidate.page_start
                if candidate.page_start is not None and candidate.page_start == candidate.page_end
                else None
            )
            result = {
                "content": candidate.content,
                "source": candidate.filename,
                "score": (
                    candidate.dense_score
                    if candidate.dense_score is not None
                    else candidate.fused_score
                ),
                "page_number": page_number,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "document_id": str(candidate.document_id),
                "conversation_id": str(conversation_id),
                "chunk_id": str(candidate.chunk_id) if candidate.chunk_id else None,
                "chunk_index": candidate.chunk_index,
                "has_tables": bool(metadata.get("has_tables") or metadata.get("contains_table")),
                "table_count": int(metadata.get("table_count") or 0),
                "image_ids": [str(image.id) for image in chunk_images],
                "image_paths": [image.image_path for image in chunk_images],
                "image_captions": [image.image_caption or "" for image in chunk_images],
                "dense_rank": candidate.dense_rank,
                "dense_score": candidate.dense_score,
                "lexical_rank": candidate.lexical_rank,
                "lexical_score": candidate.lexical_score,
                "fused_score": candidate.fused_score,
            }
            if candidate.rerank_score is not None:
                result["rerank_score"] = candidate.rerank_score
            if include_evidence_metadata:
                result["section_path"] = list(candidate.section_path)
                result["metadata"] = dict(metadata)
            results.append(result)

        return results

    async def _rerank_results(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self.reranker or not results:
            return results

        def optional_uuid(value: Any) -> UUID | None:
            try:
                return UUID(str(value)) if value else None
            except (TypeError, ValueError, AttributeError):
                return None

        candidates: list[RetrievalCandidate] = []
        for position, result in enumerate(results):
            chunk_id = optional_uuid(result.get("chunk_id"))
            image_id = optional_uuid(result.get("image_id"))
            document_id = optional_uuid(result.get("document_id")) or UUID(int=0)
            candidate = RetrievalCandidate(
                document_id=document_id,
                chunk_id=chunk_id,
                image_id=image_id,
                modality="image" if image_id is not None else "text",
                content=str(result.get("content") or ""),
                filename=str(result.get("source") or result.get("filename") or "unknown"),
                page_start=result.get("page_start"),
                page_end=result.get("page_end"),
                section_path=tuple(result.get("section_path") or ()),
                dense_rank=result.get("dense_rank"),
                dense_score=result.get("dense_score"),
                lexical_rank=result.get("lexical_rank"),
                lexical_score=result.get("lexical_score"),
                fused_score=float(result.get("fused_score") or 0.0),
                chunk_index=result.get("chunk_index"),
                metadata={
                    **dict(result.get("metadata") or {}),
                    "_legacy_rerank_position": position,
                },
            )
            candidates.append(candidate)

        ranked = await self.reranker.rank(query, candidates)
        adapted: list[dict[str, Any]] = []
        for candidate in ranked:
            position = int((candidate.metadata or {})["_legacy_rerank_position"])
            original = results[position]
            payload = dict(original)
            if candidate.rerank_score is not None:
                payload["rerank_score"] = candidate.rerank_score
            adapted.append(payload)
        return adapted

    async def _fetch_images_for_chunks(
        self,
        retrieved_docs: list[dict[str, Any]],
        *,
        user_id: str | None,
        conversation_id: str | None,
    ) -> list[dict[str, Any]]:
        if not user_id or not conversation_id:
            return []

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

            image = image_repo.get_by_id_for_scope(
                image_uuid,
                user_id=user_id,
                conversation_id=conversation_id,
            )
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
        if not user_id or not conversation_id:
            return None

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
        bounded_page = max(1, int(page))
        bounded_page_size = min(25, max(1, int(page_size)))
        if not user_id or not conversation_id:
            return {
                "documents": [],
                "total": 0,
                "page": bounded_page,
                "page_size": bounded_page_size,
            }

        try:
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
                    query.order_by(Document.upload_time.desc(), Document.id.asc())
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
        if not user_id or not conversation_id:
            return None

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
        if not user_id or not conversation_id:
            return []

        images = []
        image_repo = DocumentImageRepository(SessionLocal)
        db_images = image_repo.get_by_document_for_scope(
            document_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

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
        current_start = next(
            (
                index
                for index in range(len(non_system_messages) - 1, -1, -1)
                if isinstance(non_system_messages[index], HumanMessage)
            ),
            max(0, len(non_system_messages) - 1),
        )
        current_messages = non_system_messages[current_start:]
        history_messages = non_system_messages[:current_start]
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
                    authoritative_allowance=True,
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
        rag_tool_messages = message.metadata.get("rag_tool_messages", [])
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
        messages.extend(
            item for item in rag_tool_messages if isinstance(item, (AIMessage, ToolMessage))
        )

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
