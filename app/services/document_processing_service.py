import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from app.core.config import get_settings
from app.schemas.document import DocumentCreate, DocumentStatus
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


class DocumentProcessingService:

    def __init__(self):
        self.settings = get_settings()

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

            task = celery_app.send_task(
                "app.workers.document_processor.process_document_task",
                args=[document_id, file_content, filename],
                retry=True,
                retry_policy={
                    "max_retries": 3,
                    "interval_start": 0,
                    "interval_step": 60,
                    "interval_max": 300,
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
            task_result = celery_app.AsyncResult(task_id)

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
