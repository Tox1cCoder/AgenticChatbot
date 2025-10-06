import logging
from typing import Optional, List, Dict, Any
import time
import hashlib
import uuid
from datetime import datetime

from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document as DocxDocument

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_rag_prompt
from ...core.config import settings

logger = logging.getLogger(__name__)


class RAGAgent:

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "documents",
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dimension = 384
        self.model_name = "gemini-2.5-flash"
        self.gemini_client = None
        self._init_gemini()
        self._init_collection()

    def _init_gemini(self):
        api_key = settings.gemini_api_key
        if not api_key:
            logger.error("Gemini API key not configured")
            return

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        self.gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized for RAG Agent")

    def _init_collection(self):
        try:
            collections = self.qdrant_client.get_collections()
            if not any(c.name == self.collection_name for c in collections.collections):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AgentResponse:

        query = message.content
        conversation_history = message.metadata.get("history", [])

        retrieved_docs = await self._search(query, conversation_id=conversation_id)

        prompt = build_rag_prompt(query, retrieved_docs, conversation_history)

        response_text = await self._generate(prompt)

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=response_text
        )

        citations = [
            {
                "source": doc.get("source", "unknown"),
                "page_number": doc.get("page_number"),
                "score": doc.get("score", 0.0),
            }
            for doc in retrieved_docs[:3]
        ]

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
            },
        )

    async def _search(
        self, query: str, top_k: int = 5, conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query).tolist()

        search_filter = None
        try_global_search = False
        
        if conversation_id:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

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
            score_threshold=0.3,
            query_filter=search_filter,
        )

        if not search_results and conversation_id:
            logger.info(f"No results found for conversation {conversation_id}, trying global search")
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=0.3,
            )

        results = []
        for result in search_results:
            results.append(
                {
                    "content": result.payload.get("content", ""),
                    "source": result.payload.get("source", "unknown"),
                    "score": result.score,
                    "page_number": result.payload.get("page_number"),
                    "document_id": result.payload.get("document_id", ""),
                    "conversation_id": result.payload.get("conversation_id", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                }
            )

        return results

    async def _generate(self, prompt: str) -> str:
        response = self.gemini_client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return response.text if hasattr(response, "text") else str(response)

    async def process_document(
        self,
        file_path: str,
        filename: str,
        document_id: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        start_time = time.time()

        chunks_with_metadata = []

        if filename.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            chunks = self._create_chunks(text_content)
            chunks_with_metadata = [
                {"text": chunk, "page_number": None} for chunk in chunks
            ]

        elif filename.lower().endswith(".pdf"):
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text.strip():
                        chunks = self._create_chunks(text)
                        for chunk in chunks:
                            chunks_with_metadata.append(
                                {"text": chunk, "page_number": page_num}
                            )

        elif filename.lower().endswith(".docx"):
            doc = DocxDocument(file_path)
            text_content = "\n".join([p.text for p in doc.paragraphs])
            chunks = self._create_chunks(text_content)
            chunks_with_metadata = [
                {"text": chunk, "page_number": None} for chunk in chunks
            ]

        else:
            raise ValueError(f"Unsupported file type: {filename}")

        stored_chunks = await self._store_chunks(
            chunks_with_metadata, filename, document_id, conversation_id
        )

        processing_time = time.time() - start_time

        logger.info(
            f"Processed document {filename}: {len(chunks_with_metadata)} chunks in {processing_time:.2f}s"
        )

        return {
            "chunks_created": len(chunks_with_metadata),
            "chunks_stored": stored_chunks,
            "processing_time": processing_time,
            "filename": filename,
        }

    def _create_chunks(
        self, text: str, max_chunk_size: int = 1000, overlap: int = 200
    ) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                if len(words) > overlap:
                    overlap_text = " ".join(words[-overlap:])
                    current_chunk = overlap_text + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    async def _store_chunks(
        self,
        chunks_with_metadata: List[Dict[str, Any]],
        filename: str,
        document_id: str,
        conversation_id: Optional[str] = None,
    ) -> int:

        points = []

        for i, chunk_data in enumerate(chunks_with_metadata):
            chunk_text = (
                chunk_data.get("text", chunk_data)
                if isinstance(chunk_data, dict)
                else chunk_data
            )
            page_number = (
                chunk_data.get("page_number") if isinstance(chunk_data, dict) else None
            )

            embedding = self.embedding_model.encode(chunk_text).tolist()

            safe_point_id = str(uuid.uuid4())

            point = PointStruct(
                id=safe_point_id,
                vector=embedding,
                payload={
                    "content": chunk_text,
                    "source": filename,  # Original filename preserved in payload
                    "document_id": document_id,
                    "conversation_id": conversation_id,
                    "chunk_index": i,
                    "page_number": page_number,
                    "timestamp": datetime.now().isoformat(),
                    "file_type": (
                        filename.split(".")[-1] if "." in filename else "unknown"
                    ),
                },
            )
            points.append(point)

        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"Stored {len(points)} chunks in vector database")
        return len(points)

    async def initialize(self):
        """Initialize the RAG agent (already done in __init__, but provided for compatibility)"""
        logger.info("RAG Agent initialized")
        return True

    async def cleanup(self):
        """Cleanup resources (close connections if needed)"""
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
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": self.embedding_dimension,
            }
        except Exception as e:
            logger.error(f"Error getting RAG agent status: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def delete_document_vectors(self, document_id: str) -> dict:
        """Delete all vectors associated with a document ID"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            )

            self.qdrant_client.delete(
                collection_name=self.collection_name, points_selector=delete_filter
            )

            logger.info(f"Deleted vectors for document {document_id}")
            return {
                "success": True,
                "document_id": document_id,
                "message": f"Vectors deleted for document {document_id}",
            }
        except Exception as e:
            logger.error(f"Error deleting vectors for document {document_id}: {e}")
            return {"success": False, "document_id": document_id, "error": str(e)}
