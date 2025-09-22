from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import time
import asyncio
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from ..interfaces import BaseAgent
from ..schemas import AgentMessage, AgentResponse, AgentType, MessageType

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """RAG agent for document-based question answering with optional Qdrant vector storage."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "documents",
    ):
        super().__init__(AgentType.RAG)
        self.logger = logging.getLogger("rag_agent")

        self.qdrant_client = None
        self.collection_name = collection_name
        self.vector_mode = False

        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.vector_mode = True
        self.logger.info("Qdrant client initialized successfully")

        self.embedding_model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2

        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load embedding model: {e}")
            self.vector_mode = False

        # Initialize collection
        if self.vector_mode and self.qdrant_client:
            asyncio.create_task(self._initialize_collection())

    async def _initialize_collection(self):
        """Initialize Qdrant collection if it doesn't exist."""
        try:
            collections = self.qdrant_client.get_collections()
            if not any(
                collection.name == self.collection_name
                for collection in collections.collections
            ):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension, distance=Distance.COSINE
                    ),
                )
                self.logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                self.logger.info(
                    f"Qdrant collection already exists: {self.collection_name}"
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant collection: {e}")
            self.vector_mode = False
            self.qdrant_client = None

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process a message using RAG functionality with vector search"""

        try:
            query = message.content

            # Perform search
            search_results = await self._vector_search(query)

            # Generate response based on retrieved documents
            response_content = self._generate_response(query, search_results)

            response_message = AgentMessage(
                role=message.role,
                content=response_content,
                message_type=message.message_type,
            )

            return AgentResponse(
                response_id=message.id,
                request_id=message.id,
                agent_type=AgentType.RAG,
                agent_id="rag_agent",
                message=response_message,
                confidence=0.85 if self.vector_mode else 0.75,
                processing_time_ms=250,
                metadata={
                    "agent_type": "rag",
                    "query": query,
                    "documents_found": (
                        len(search_results) if isinstance(search_results, list) else 1
                    ),
                    "vector_mode": self.vector_mode,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            self.logger.error(f"RAG processing failed: {e}")

            error_message = AgentMessage(
                role=message.role,
                content="I apologize, but I couldn't find the information you're looking for. Please try rephrasing your question.",
                message_type=MessageType.ERROR,
            )

            return AgentResponse(
                response_id=message.id,
                request_id=message.id,
                agent_type=AgentType.RAG,
                agent_id="rag_agent",
                message=error_message,
                confidence=0.0,
                processing_time_ms=100,
                error=str(e),
            )

    async def _vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector search using Qdrant."""

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0.7,
        )

        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                {
                    "content": result.payload.get("content", ""),
                    "source": result.payload.get("source", "unknown"),
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {}),
                }
            )

        return formatted_results

    def _generate_response(
        self, query: str, search_results: List[Dict[str, Any]]
    ) -> str:
        """Generate a response based on search results."""

        if not search_results:
            return f"I couldn't find specific information about '{query}' in my knowledge base. Could you please rephrase your question or provide more context?"

        if isinstance(search_results, list) and search_results:
            result = search_results[0]  # Use best result
            content = result.get("content", "")
            source = result.get("source", "unknown")
            score = result.get("score", 0.0)
            vector_mode = result.get("metadata", {}).get("vector_mode", False)

            mode_indicator = (
                "semantic vector search" if vector_mode else "keyword-based analysis"
            )

            response = f"Based on my {mode_indicator} for '{query}', I found the following information:\n\n{content}\n\n"

            if score > 0.8:
                response += (
                    "This information appears to be highly relevant to your query. "
                )
            elif score > 0.7:
                response += "This information seems relevant to your query. "
            else:
                response += "This information may be related to your query. "

            response += "How can I help you explore this topic further?"

            return response

        return f"I found some information about '{query}', but couldn't format it properly. Please try rephrasing your question."

    async def process_document(
        self, file_path: str, filename: str, user_id: str
    ) -> Dict[str, Any]:
        """
        Process an uploaded document and optionally store it in the vector database.

        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            user_id: ID of the user uploading the document

        Returns:
            Dict with processing results
        """

        start_time = time.time()

        try:
            # Read the document
            if filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            elif filename.lower().endswith(".pdf"):
                text_content = self._extract_pdf_text(file_path)
            elif filename.lower().endswith(".docx"):
                text_content = self._extract_docx_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {filename}")

            # Smart chunking
            chunks = self._create_chunks(text_content, max_chunk_size=1000, overlap=200)

            # Store chunks in vector database if available
            stored_chunks = 0
            if self.vector_mode:
                stored_chunks = await self._store_chunks_in_vector_db(
                    chunks, filename, user_id
                )

            processing_time = time.time() - start_time

            self.logger.info(
                f"Processed document {filename}: {len(chunks)} chunks, {stored_chunks} stored in {processing_time:.2f}s"
            )

            return {
                "chunks_created": len(chunks),
                "chunks_stored": stored_chunks,
                "processing_time": processing_time,
                "filename": filename,
                "vector_storage": self.vector_mode,
            }

        except Exception as e:
            self.logger.error(f"Document processing failed for {filename}: {str(e)}")
            raise

    def _create_chunks(
        self, text: str, max_chunk_size: int = 1000, overlap: int = 200
    ) -> List[str]:
        """Create overlapping text chunks for better retrieval."""

        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, finalize current chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Create overlap with previous chunk
                words = current_chunk.split()
                if len(words) > overlap:
                    overlap_text = " ".join(words[-overlap:])
                    current_chunk = overlap_text + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    async def _store_chunks_in_vector_db(
        self, chunks: List[str], filename: str, user_id: str
    ) -> int:
        """Store text chunks in Qdrant vector database."""

        if not self.vector_mode or not self.qdrant_client or not self.embedding_model:
            self.logger.warning("Vector storage not available, skipping")
            return 0

        try:
            points = []

            for i, chunk in enumerate(chunks):
                # Generate embedding for the chunk
                embedding = self.embedding_model.encode(chunk).tolist()

                # Create point for Qdrant
                point = PointStruct(
                    id=f"{user_id}_{filename}_{i}_{int(time.time())}",
                    vector=embedding,
                    payload={
                        "content": chunk,
                        "source": filename,
                        "user_id": user_id,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "file_type": (
                                filename.split(".")[-1]
                                if "." in filename
                                else "unknown"
                            )
                        },
                    },
                )
                points.append(point)

            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )

            self.logger.info(f"Stored {len(points)} chunks in vector database")
            return len(points)

        except Exception as e:
            self.logger.error(f"Failed to store chunks in vector database: {e}")
            return 0

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            import PyPDF2

            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF processing. Install with: pip install PyPDF2"
            )

    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document

            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX processing. Install with: pip install python-docx"
            )

    async def search_documents(
        self, query: str, user_id: Optional[str] = None, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Public method to search documents.

        Args:
            query: Search query
            user_id: Optional user ID to filter results
            top_k: Number of results to return

        Returns:
            List of search results
        """

        return await self._vector_search(query, top_k)

    def get_status(self) -> Dict[str, Any]:
        """Get current RAG agent status and configuration."""

        return {
            "vector_mode": self.vector_mode,
            "qdrant_available": QDRANT_AVAILABLE and self.qdrant_client is not None,
            "embeddings_available": SENTENCE_TRANSFORMERS_AVAILABLE
            and self.embedding_model is not None,
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_dimension,
        }
