import logging
from typing import Optional, List, Dict, Any

from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from sentence_transformers import SentenceTransformer, CrossEncoder


from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_rag_prompt
from ...core.config import Settings

logger = logging.getLogger(__name__)


class RAGAgent:

    def __init__(
        self,
        settings: Settings,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
        collection_name: str = "documents_gemma",
    ):
        self.settings = settings
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model

        self.collection_name = collection_name
        self.embedding_dimension = settings.embedding_dimension
        self.model_name = "gemini-2.5-flash"
        self.gemini_client = None

        # Store retrieval parameters
        self.top_k = settings.rag_top_k
        self.score_threshold = settings.rag_score_threshold
        self.enable_reranking = settings.enable_reranking
        self.reranker = None

        self._init_gemini()

        # Initialize re-ranker if enabled
        if self.enable_reranking:
            self._init_reranker()

    def _init_gemini(self):
        api_key = self.settings.gemini_api_key
        if not api_key:
            logger.error("Gemini API key not configured")
            return

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        self.gemini_client = genai.Client(api_key=api_key)

    def _init_reranker(self):
        """Initialize the re-ranker model"""
        self.reranker = CrossEncoder(self.settings.reranker_model)
        logger.info(f"Re-ranker initialized: {self.settings.reranker_model}")

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:

        query = message.content
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        retrieved_docs = await self._search(query, conversation_id=conversation_id)

        prompt = build_rag_prompt(
            query, retrieved_docs, conversation_history, persona=persona
        )

        response_text = await self._generate(prompt)

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=response_text
        )

        # Expand citations to include all used chunks with more metadata
        citations = [
            {
                "source": doc.get("source", "unknown"),
                "page_number": doc.get("page_number"),
                "page_start": doc.get("page_start"),
                "page_end": doc.get("page_end"),
                "score": doc.get("score", 0.0),
                "chunk_index": doc.get("chunk_index", 0),
                "character_count": len(doc.get("content", "")),
            }
            for doc in retrieved_docs
        ]

        # Calculate retrieval statistics
        avg_score = (
            sum(doc.get("score", 0.0) for doc in retrieved_docs) / len(retrieved_docs)
            if retrieved_docs
            else 0.0
        )

        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=response_message,
            metadata={
                "model": self.model_name,
                "conversation_id": conversation_id,
                "documents_found": len(retrieved_docs),
                "citations": citations,
                "context_messages": len(conversation_history),
                "retrieval_stats": {
                    "total_retrieved": len(retrieved_docs),
                    "avg_score": avg_score,
                },
                "persona_used": persona,
            },
        )

    async def _search(
        self, query: str, top_k: int = None, conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # Use configured top_k if not specified
        if top_k is None:
            top_k = self.top_k

        query_embedding = self.embedding_model.encode(query).tolist()

        search_filter = None

        if conversation_id:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="conversation_id", match=MatchValue(value=conversation_id)
                    )
                ]
            )

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=self.score_threshold,
            query_filter=search_filter,
        )

        if not search_results and conversation_id:
            logger.info(
                f"No results found for conversation {conversation_id}, trying global search"
            )
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=self.score_threshold,
            )

        # Log retrieval results
        logger.info(
            f"Retrieved {len(search_results)} chunks with scores: {[r.score for r in search_results]}"
        )

        results = []
        for result in search_results:
            results.append(
                {
                    "content": result.payload.get("content", ""),
                    "source": result.payload.get("source", "unknown"),
                    "score": result.score,
                    "page_number": result.payload.get("page_number"),
                    "page_start": result.payload.get("page_start"),
                    "page_end": result.payload.get("page_end"),
                    "document_id": result.payload.get("document_id", ""),
                    "conversation_id": result.payload.get("conversation_id", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                }
            )

        if self.enable_reranking and len(results) > 3:
            results = await self._rerank_results(query, results)

        return results

    async def _rerank_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank search results using cross-encoder"""
        if not self.reranker or not results:
            return results

        try:
            # Prepare pairs for re-ranking
            pairs = [[query, doc["content"]] for doc in results]

            # Get re-ranking scores
            rerank_scores = self.reranker.predict(pairs)

            # Add rerank scores to results
            for i, score in enumerate(rerank_scores):
                results[i]["rerank_score"] = float(score)

            # Sort by rerank score
            results = sorted(
                results, key=lambda x: x.get("rerank_score", 0), reverse=True
            )

            # Keep only top K after re-ranking
            results = results[: self.settings.rerank_top_k]

            logger.info(f"Re-ranked results, kept top {len(results)} chunks")

            return results
        except Exception as e:
            logger.warning(f"Re-ranking failed, using original results: {e}")
            return results

    async def _generate(self, prompt: str) -> str:
        response = self.gemini_client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return response.text if hasattr(response, "text") else str(response)

    async def initialize(self):
        """Initialize the RAG agent"""
        logger.info("RAG Agent initialized")
        return True

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self.qdrant_client, "close"):
                self.qdrant_client.close()
            logger.info("RAG Agent cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during RAG Agent cleanup: {e}")

    def get_status(self) -> dict:
        """Get the current status of the RAG agent"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                c.name == self.collection_name for c in collections.collections
            )

            collection_info = None
            if collection_exists:
                collection_info = self.qdrant_client.get_collection(
                    self.collection_name
                )

            return {
                "status": "healthy",
                "collection_exists": collection_exists,
                "collection_name": self.collection_name,
                "vectors_count": (
                    collection_info.vectors_count if collection_info else 0
                ),
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dimension,
            }
        except Exception as e:
            logger.error(f"Error getting RAG agent status: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def delete_document_vectors(self, document_id: str) -> dict:
        """
        Delete all vectors associated with a document ID.
        """
        try:
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            )

            result = self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=delete_filter),
            )

            return {
                "success": True,
                "document_id": document_id,
                "message": f"Vectors deleted for document {document_id}",
                "operation_result": str(result),
            }
        except Exception as e:
            logger.error(
                f"Error deleting vectors for document {document_id}: {e}", exc_info=True
            )
            return {"success": False, "document_id": document_id, "error": str(e)}
