import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

import PyPDF2
from docx import Document as DocxDocument
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.schemas.document import DocumentCreate, DocumentStatus
from app.utils.text_processing import create_chunks, extract_page_range
from app.database.qdrant import ensure_collection

logger = logging.getLogger(__name__)


class DocumentProcessingService:

    def __init__(
        self,
        settings: Settings,
        celery_app,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
    ):
        self.settings = settings
        self.celery_app = celery_app
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = settings.qdrant_collection_name
        self.embedding_dimension = settings.embedding_dimension

    async def validate_upload_file(
        self, filename: str, file_size: int
    ) -> Dict[str, Any]:

        max_size_bytes = self.settings.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed size of {self.settings.max_file_size_mb}MB"
            )

        allowed_extensions = {".txt", ".pdf", ".docx"}
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise ValueError(
                f"Unsupported file type '{file_extension}'. Allowed: {', '.join(allowed_extensions)}"
            )

        return {
            "valid": True,
            "file_type": file_extension,
            "size_mb": round(file_size / (1024 * 1024), 2),
        }

    async def start_processing_task(
        self, document_id: str, file_content: bytes, filename: str
    ) -> Dict[str, Any]:
        try:

            validation = await self.validate_upload_file(filename, len(file_content))

            task = self.celery_app.send_task(
                "app.workers.document_processor.process_document_task",
                args=[document_id, file_content, filename],
                retry=True,
                retry_policy={
                    "max_retries": 3,
                    "interval_start": 0,
                    "interval_step": 30,
                    "interval_max": 180,
                },
            )

            logger.info(f"Started processing task {task.id} for document {document_id}")

            return {
                "success": True,
                "task_id": task.id,
                "document_id": document_id,
                "file_info": validation,
                "estimated_processing_time": self._estimate_processing_time(
                    len(file_content)
                ),
                "message": f"Document '{filename}' queued for processing",
            }

        except Exception as e:
            logger.error(
                f"Failed to start processing task for document {document_id}: {str(e)}"
            )
            raise

    def _estimate_processing_time(self, file_size_bytes: int) -> str:
        size_mb = file_size_bytes / (1024 * 1024)

        if size_mb < 1:
            return "30-60 seconds"
        elif size_mb < 5:
            return "1-2 minutes"
        elif size_mb < 10:
            return "2-5 minutes"
        else:
            return "5-10 minutes"

    async def get_processing_status(self, task_id: str) -> Dict[str, Any]:
        try:
            task_result = self.celery_app.AsyncResult(task_id)

            return {
                "task_id": task_id,
                "status": task_result.status,
                "result": task_result.result if task_result.ready() else None,
                "info": task_result.info if hasattr(task_result, "info") else None,
                "traceback": task_result.traceback if task_result.failed() else None,
            }

        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {str(e)}")
            return {"task_id": task_id, "status": "UNKNOWN", "error": str(e)}

    async def process_document(
        self,
        file_path: str,
        filename: str,
        document_id: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a document file and store its chunks in the vector database"""
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

                if self.settings.preserve_cross_page_context:
                    # Extract all pages first with page markers
                    pages_text = []
                    for page_num, page in enumerate(reader.pages, start=1):
                        text = page.extract_text()
                        if text.strip():
                            pages_text.append((page_num, text))

                    # Concatenate with page markers
                    full_text = "".join(
                        [f"\n[PAGE {num}]\n{text}" for num, text in pages_text]
                    )

                    # Chunk the full document
                    chunks = self._create_chunks(full_text)

                    # Extract page range for each chunk
                    for chunk in chunks:
                        page_start, page_end = extract_page_range(chunk)
                        chunks_with_metadata.append(
                            {
                                "text": chunk,
                                "page_start": page_start,
                                "page_end": page_end,
                            }
                        )
                else:
                    # Legacy per-page chunking
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
        self, text: str, max_chunk_size: int = None, overlap: int = None
    ) -> List[str]:
        """Create text chunks using smart chunking utility"""
        # Use configured parameters if not specified
        if max_chunk_size is None:
            max_chunk_size = self.settings.document_chunk_size
        if overlap is None:
            overlap = self.settings.document_chunk_overlap

        return create_chunks(
            text, max_chunk_size, overlap, self.settings.chunk_by_sentences
        )

    async def _store_chunks(
        self,
        chunks_with_metadata: List[Dict[str, Any]],
        filename: str,
        document_id: str,
        conversation_id: Optional[str] = None,
    ) -> int:
        """Store document chunks in the vector database"""
        ensure_collection(
            qdrant_client=self.qdrant_client,
            collection_name=self.collection_name,
            vector_size=self.embedding_dimension,
        )

        points = []

        for i, chunk_data in enumerate(chunks_with_metadata):
            chunk_text = (
                chunk_data.get("text", chunk_data)
                if isinstance(chunk_data, dict)
                else chunk_data
            )

            # Handle both single page and page ranges
            page_number = (
                chunk_data.get("page_number") if isinstance(chunk_data, dict) else None
            )
            page_start = (
                chunk_data.get("page_start") if isinstance(chunk_data, dict) else None
            )
            page_end = (
                chunk_data.get("page_end") if isinstance(chunk_data, dict) else None
            )

            embedding = self.embedding_model.encode(chunk_text).tolist()

            safe_point_id = str(uuid.uuid4())

            payload = {
                "content": chunk_text,
                "source": filename,  # Original filename preserved in payload
                "document_id": document_id,
                "conversation_id": conversation_id,
                "chunk_index": i,
                "timestamp": datetime.now().isoformat(),
                "file_type": (
                    filename.split(".")[-1] if "." in filename else "unknown"
                ),
            }

            # Add page information (support both formats)
            if page_start is not None and page_end is not None:
                payload["page_start"] = page_start
                payload["page_end"] = page_end
            elif page_number is not None:
                payload["page_number"] = page_number

            point = PointStruct(
                id=safe_point_id,
                vector=embedding,
                payload=payload,
            )
            points.append(point)

        # Batch upsert operations
        batch_size = self.settings.qdrant_upsert_batch_size
        total_points = len(points)

        for i in range(0, total_points, batch_size):
            batch = points[i : i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=batch
            )
            logger.debug(f"Upserted batch {i//batch_size + 1}: {len(batch)} points")

        logger.info(
            f"Stored {total_points} chunks in vector database using {(total_points + batch_size - 1) // batch_size} batches"
        )
        return total_points

    async def cleanup_temp_files(self, older_than_hours: int = 24) -> Dict[str, Any]:
        try:
            temp_dir = os.path.join(os.getcwd(), self.settings.temp_storage_path)
            if not os.path.exists(temp_dir):
                return {"files_removed": 0, "message": "Temp directory does not exist"}

            removed_count = 0
            cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)

            for filename in os.listdir(temp_dir):
                if filename.startswith("."):
                    continue

                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime < cutoff_time:
                        try:
                            os.unlink(file_path)
                            removed_count += 1
                            logger.info(f"Removed old temp file: {filename}")
                        except Exception as e:
                            logger.warning(
                                f"Failed to remove temp file {filename}: {str(e)}"
                            )

            return {
                "files_removed": removed_count,
                "message": f"Cleaned up {removed_count} temporary files older than {older_than_hours} hours",
            }

        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {str(e)}")
            return {
                "files_removed": 0,
                "error": str(e),
                "message": "Temp file cleanup failed",
            }
