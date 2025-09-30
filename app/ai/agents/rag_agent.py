from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import time
import asyncio
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document

from ..interfaces import BaseAgent
from ..schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageType,
    AgentRequest,
    AgentConfig,
    RAGAgentConfig,
    AgentCapability,
)

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "documents",
    ):

        config = RAGAgentConfig(
            agent_id="rag_agent",
            agent_type=AgentType.RAG,
            name="RAG Agent",
            description="Document-based question answering with vector search",
            capabilities=[
                AgentCapability.DOCUMENT_SEARCH,
                AgentCapability.KNOWLEDGE_RETRIEVAL,
            ],
            vector_store_path=qdrant_url,
        )
        super().__init__(config)
        self.logger = logging.getLogger("rag_agent")

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.logger.info("Qdrant client initialized successfully")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        self.logger.info("Sentence transformer model loaded successfully")

        # Initialize collection
        asyncio.create_task(self._initialize_collection())

    async def _initialize_impl(self) -> None:
        """Initialize the RAG agent and vector collection."""
        await self._initialize_collection()
        self.logger.info("RAG agent initialized successfully")

    async def _cleanup_impl(self) -> None:
        """Clean up RAG agent resources including Qdrant connections."""
        if self.qdrant_client:
            try:
                self.qdrant_client.close()
            except:
                pass
        self.logger.info("RAG agent cleaned up successfully")

    async def _health_check_impl(self) -> bool:
        """Check if the RAG agent is healthy."""
        return True

    async def can_handle_request(self, request: AgentRequest) -> float:
        """Determine if this agent can handle the given request."""
        return 0.9

    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process an agent request and return a response."""
        # Delegate to the existing process_message method
        return await self.process_message(
            message=request.message,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
        )

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
            raise

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

            # Prepare citation information for metadata
            citations = []
            for result in search_results:
                citation_info = {
                    "source": result.get("source", "unknown"),
                    "page_number": result.get("page_number"),
                    "chunk_id": result.get("chunk_id"),
                    "document_id": result.get("document_id"),
                    "chunk_index": result.get("chunk_index"),
                    "score": result.get("score", 0.0),
                }
                citations.append(citation_info)

            # Log document usage for traceability
            self.logger.info(
                f"RAG Query: '{query[:100]}...' | Documents used: {len(search_results)} | "
                f"Primary source: {citations[0]['source'] if citations else 'none'} "
                f"(page {citations[0]['page_number']}) "
                if citations and citations[0].get("page_number")
                else "" f"| Chunk ID: {citations[0]['chunk_id']}" if citations else ""
            )

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
                confidence=0.85,
                processing_time_ms=250,
                metadata={
                    "agent_type": "rag",
                    "query": query,
                    "documents_found": (
                        len(search_results) if isinstance(search_results, list) else 1
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "citations": citations,  # Include citation details for traceability
                    "primary_source": citations[0] if citations else None,
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

        # Format results with all metadata
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                {
                    "content": result.payload.get("content", ""),
                    "source": result.payload.get("source", "unknown"),
                    "score": result.score,
                    "document_id": result.payload.get("document_id", ""),
                    "conversation_id": result.payload.get("conversation_id", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "page_number": result.payload.get("page_number", None),
                    "file_type": result.payload.get("file_type", "unknown"),
                    "chunk_id": result.id,
                    "metadata": result.payload.get("metadata", {}),
                }
            )

        return formatted_results

    def _generate_response(
        self, query: str, search_results: List[Dict[str, Any]]
    ) -> str:
        """Generate a response with proper citations based on search results."""

        if not search_results:
            return f"I couldn't find specific information about '{query}' in my knowledge base. Could you please rephrase your question or provide more context?"

        if isinstance(search_results, list) and search_results:
            # Use the best result for primary response
            result = search_results[0]
            content = result.get("content", "")
            source = result.get("source", "unknown")
            page_number = result.get("page_number")
            score = result.get("score", 0.0)

            # Format citation
            citation = f"'{source}'"
            if page_number:
                citation = f"'{source}' (page {page_number})"

            # Truncate content for quote if too long
            quote = content[:300] + "..." if len(content) > 300 else content

            # Build response with citation
            response = f'According to {citation}:\n\n"{quote}"\n\n'

            # Add relevance indicator
            if score > 0.8:
                response += (
                    "This information appears to be highly relevant to your query."
                )
            elif score > 0.7:
                response += "This information seems relevant to your query."
            else:
                response += "This information may be related to your query."

            # Add additional sources if available
            if len(search_results) > 1:
                response += "\n\nAdditional relevant sources found:"
                for i, additional_result in enumerate(
                    search_results[1:4], start=2
                ):  # Show up to 3 more sources
                    add_source = additional_result.get("source", "unknown")
                    add_page = additional_result.get("page_number")
                    add_citation = f"'{add_source}'"
                    if add_page:
                        add_citation = f"'{add_source}' (page {add_page})"
                    response += f"\n  {i}. {add_citation}"

            return response

        return f"I found some information about '{query}', but couldn't format it properly. Please try rephrasing your question."

    async def process_document(
        self,
        file_path: str,
        filename: str,
        document_id: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process an uploaded document and optionally store it in the vector database.

        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            document_id: ID of the document record in database
            conversation_id: ID of the conversation this document belongs to

        Returns:
            Dict with processing results
        """

        start_time = time.time()

        try:
            chunks_with_metadata = []

            # Read the document
            if filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                # Create chunks without page numbers for text files
                chunks = self._create_chunks(
                    text_content, max_chunk_size=1000, overlap=200
                )
                chunks_with_metadata = [
                    {"text": chunk, "page_number": None} for chunk in chunks
                ]

            elif filename.lower().endswith(".pdf"):
                # Extract PDF with page information
                pages_data = self._extract_pdf_text_with_pages(file_path)
                # Create chunks with page tracking
                for page_data in pages_data:
                    page_chunks = self._create_chunks(
                        page_data["text"], max_chunk_size=1000, overlap=200
                    )
                    for chunk in page_chunks:
                        chunks_with_metadata.append(
                            {"text": chunk, "page_number": page_data["page_number"]}
                        )

            elif filename.lower().endswith(".docx"):
                text_content = self._extract_docx_text(file_path)
                chunks = self._create_chunks(
                    text_content, max_chunk_size=1000, overlap=200
                )
                chunks_with_metadata = [
                    {"text": chunk, "page_number": None} for chunk in chunks
                ]
            else:
                raise ValueError(f"Unsupported file type: {filename}")

            # Store chunks in vector database
            stored_chunks = await self._store_chunks_in_vector_db(
                chunks_with_metadata, filename, document_id, conversation_id
            )

            processing_time = time.time() - start_time

            self.logger.info(
                f"Processed document {filename}: {len(chunks_with_metadata)} chunks, {stored_chunks} stored in {processing_time:.2f}s"
            )

            return {
                "chunks_created": len(chunks_with_metadata),
                "chunks_stored": stored_chunks,
                "processing_time": processing_time,
                "filename": filename,
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
        self,
        chunks_with_metadata: List[Dict[str, Any]],
        filename: str,
        document_id: str,
        conversation_id: Optional[str] = None,
    ) -> int:
        """Store text chunks with metadata in Qdrant vector database."""

        try:
            points = []

            for i, chunk_data in enumerate(chunks_with_metadata):
                chunk_text = (
                    chunk_data.get("text", chunk_data)
                    if isinstance(chunk_data, dict)
                    else chunk_data
                )
                page_number = (
                    chunk_data.get("page_number")
                    if isinstance(chunk_data, dict)
                    else None
                )

                # Generate embedding for the chunk
                embedding = self.embedding_model.encode(chunk_text).tolist()

                # Create point for Qdrant
                point = PointStruct(
                    id=f"{document_id}_{filename}_{i}_{int(time.time())}",
                    vector=embedding,
                    payload={
                        "content": chunk_text,
                        "source": filename,
                        "document_id": document_id,
                        "conversation_id": conversation_id,
                        "chunk_index": i,
                        "page_number": page_number,
                        "timestamp": datetime.now().isoformat(),
                        "file_type": (
                            filename.split(".")[-1] if "." in filename else "unknown"
                        ),
                        "metadata": {
                            "processing_timestamp": datetime.now().isoformat(),
                            "chunk_length": len(chunk_text),
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
            raise

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text

    def _extract_pdf_text_with_pages(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF file with page numbers."""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            pages_data = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    pages_data.append({"text": text, "page_number": page_num})
            return pages_data

    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    async def search_documents(
        self, query: str, conversation_id: Optional[str] = None, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Public method to search documents.

        Args:
            query: Search query
            conversation_id: Optional conversation ID to filter results
            top_k: Number of results to return

        Returns:
            List of search results
        """

        return await self._vector_search(query, top_k)

    def get_status(self) -> Dict[str, Any]:
        """Get current RAG agent status and configuration."""

        return {
            "qdrant_connected": self.qdrant_client is not None,
            "embedding_model_loaded": self.embedding_model is not None,
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_dimension,
        }
