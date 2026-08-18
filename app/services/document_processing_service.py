import asyncio
import contextlib
import hashlib
import io
import logging
import os
import re
import shutil
import time
import unicodedata
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image
from pydantic import ValidationError

from app.core.config import Settings
from app.core.events import DocumentEvent, DocumentEventData, get_event_bus
from app.repositories.document_image import DocumentImageRepository
from app.schemas.document_image import DocumentImageCreate, ImageCaptionSections
from app.services.document_blocks import NormalizedBlock
from app.services.document_chunk_builder import DocumentChunkBuilder
from app.services.document_parse_service import DocumentParseService, ParseResult
from app.services.gemini_retry import is_rate_limit_error, parse_retry_delay
from app.services.plain_text_loader import load_utf8_text_document
from app.usage import begin_usage_operation, bind_usage_context, current_usage_context
from app.usage.types import UsageOperation

logger = logging.getLogger(__name__)

# Untrusted document content flows into this prompt only as image bytes, and
# the response is validated against ImageCaptionSections — the model's output
# is data, never treated as instructions.
_IMAGE_CAPTION_STRUCTURED_PROMPT = (
    "Describe this image for search retrieval. Extract: any visible text "
    "rendered in the image (OCR), a chart or figure title if present, axis "
    "labels, legend entries, key values, described trends or patterns, "
    "relationships between depicted elements, and one sentence tying the "
    "image to its likely surrounding document section. Leave a field empty "
    "if it does not apply."
)


class DocumentProcessingService:
    def __init__(
        self,
        settings: Settings,
        celery_app,
        document_image_repository: DocumentImageRepository,
        document_index_service: Any | None = None,
        document_chunk_builder: DocumentChunkBuilder | None = None,
        document_parse_artifact_repository: Any | None = None,
        document_parse_service: DocumentParseService | None = None,
        recorder: Any | None = None,
    ):
        self.recorder = recorder
        self.settings = settings
        self.celery_app = celery_app
        self.document_image_repository = document_image_repository
        self.document_index_service = document_index_service
        self.document_chunk_builder = document_chunk_builder or DocumentChunkBuilder(
            target_tokens=settings.rag_chunk_target_tokens,
            overlap_tokens=settings.rag_chunk_overlap_tokens,
            max_tokens=settings.rag_chunk_max_tokens,
        )
        self.document_parse_artifact_repository = document_parse_artifact_repository
        self._parse_service = document_parse_service or DocumentParseService(
            settings=settings,
            chunk_builder=self.document_chunk_builder,
        )
        self.collection_name = settings.qdrant_collection_name
        self.embedding_dimension = settings.rag_embedding_dimension
        self._event_bus = get_event_bus()
        self._mineru_output_path = None
        self.gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        api_key = self.settings.gemini_api_key
        if not api_key:
            self.gemini_client = None
            return

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        try:
            # Application owns retries; attempts=1 disables SDK-internal retry.
            self.gemini_client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
            )
        except Exception:
            self.gemini_client = None

    async def validate_upload_file(self, filename: str, file_size: int) -> dict[str, Any]:
        max_size_bytes = self.settings.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed size of "
                f"{self.settings.max_file_size_mb}MB"
            )

        file_extension = self._validate_file_extension(filename)

        return {
            "valid": True,
            "file_type": file_extension,
            "size_mb": round(file_size / (1024 * 1024), 2),
        }

    SUPPORTED_UPLOAD_EXTENSIONS: frozenset[str] = frozenset(
        {".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md"}
    )
    EXCEL_EXTENSIONS: frozenset[str] = frozenset({".xlsx"})
    MINERU_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".pptx", ".html", ".md"})

    @classmethod
    def _validate_file_extension(cls, filename: str) -> str:
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in cls.SUPPORTED_UPLOAD_EXTENSIONS:
            allowed = ", ".join(sorted(cls.SUPPORTED_UPLOAD_EXTENSIONS))
            raise ValueError(f"Unsupported file type '{file_extension}'. Allowed: {allowed}")
        return file_extension

    async def stage_upload_file(self, upload_file: Any, filename: str) -> dict[str, Any]:
        """Stream an upload to temp storage without holding the full payload in memory."""
        self._validate_file_extension(filename)

        temp_dir = Path(os.getcwd()) / self.settings.temp_storage_path
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = self._staged_temp_path(uuid.uuid4().hex, filename, temp_dir)

        bytes_written = 0
        max_size_bytes = self.settings.max_file_size_mb * 1024 * 1024

        try:
            if hasattr(upload_file, "seek"):
                await upload_file.seek(0)

            with temp_file_path.open("wb") as staged_file:
                while True:
                    chunk = await upload_file.read(1024 * 1024)
                    if not chunk:
                        break

                    bytes_written += len(chunk)
                    if bytes_written > max_size_bytes:
                        raise ValueError(
                            f"File size ({bytes_written} bytes) exceeds maximum allowed "
                            f"size of {self.settings.max_file_size_mb}MB"
                        )

                    staged_file.write(chunk)

            validation = await self.validate_upload_file(filename, bytes_written)
            return {
                "temp_file_path": str(temp_file_path),
                "file_size": bytes_written,
                "file_info": validation,
            }
        except Exception:
            try:
                if temp_file_path.exists():
                    temp_file_path.unlink()
            except Exception:
                pass
            raise

    async def start_processing_task(
        self,
        document_id: str,
        temp_file_path: str,
        filename: str,
        file_size: int,
    ) -> dict[str, Any]:
        validation = await self.validate_upload_file(filename, file_size)
        staged_path = Path(temp_file_path)
        if not staged_path.is_file():
            raise FileNotFoundError(f"Staged upload file not found: {temp_file_path}")

        logger.debug(
            "Queueing staged upload for document %s from %s (%d bytes)",
            document_id,
            staged_path,
            file_size,
        )

        try:
            from celery import chain as celery_chain

            parse_sig = self.celery_app.signature(
                "app.workers.document_processor.parse_document_task",
                args=[document_id, str(staged_path), filename],
            )
            index_sig = self.celery_app.signature(
                "app.workers.document_processor.index_document_task",
            )
            task = celery_chain(parse_sig, index_sig).apply_async(
                retry=True,
                retry_policy={
                    "max_retries": 3,
                    "interval_start": 0,
                    "interval_step": 30,
                    "interval_max": 180,
                },
            )
        except Exception:
            try:
                if staged_path.is_file():
                    staged_path.unlink()
            except Exception:
                pass
            raise

        try:
            await self._event_bus.emit(
                DocumentEvent.PROCESSING_STARTED,
                DocumentEventData(
                    document_id=UUID(document_id),
                    filename=filename,
                    status="PROCESSING",
                    metadata={"task_id": task.id},
                ),
            )
        except Exception as e:
            logger.debug(f"Event emission failed for PROCESSING_STARTED: {e}")

        return {
            "success": True,
            "task_id": task.id,
            "document_id": document_id,
            "file_info": validation,
            "estimated_processing_time": self._estimate_processing_time(file_size),
            "message": f"Document '{filename}' queued for processing",
        }

    @staticmethod
    def _staged_temp_path(document_id: str, filename: str, temp_dir: Path) -> Path:
        """Return a deterministic, safe temp file path for a staged upload."""
        base_name, ext = os.path.splitext(filename)
        normalized = unicodedata.normalize("NFKD", base_name or "")
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "document"
        normalized = normalized[:80]
        safe_ext = ext.lower() if ext else ""
        return temp_dir / f"{document_id}_{normalized}{safe_ext}"

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

    async def get_processing_status(self, task_id: str) -> dict[str, Any]:
        try:
            task_result = self.celery_app.AsyncResult(task_id)
            response: dict[str, Any] = {
                "task_id": task_id,
                "status": task_result.status,
            }

            if task_result.ready():
                result = task_result.result
                if isinstance(result, dict):
                    if "success" in result:
                        response["success"] = bool(result.get("success"))
                    if result.get("document_id"):
                        response["document_id"] = result["document_id"]
                    if result.get("message"):
                        response["message"] = result["message"]
                elif task_result.successful():
                    response["success"] = True

            info = task_result.info if hasattr(task_result, "info") else None
            if isinstance(info, dict):
                safe_info = {}
                for key in ("current", "total", "message"):
                    if key in info:
                        safe_info[key] = info[key]
                if safe_info:
                    response["info"] = safe_info

            return response

        except Exception as e:
            return {"task_id": task_id, "status": "UNKNOWN", "error": str(e)}

    async def process_document(
        self,
        file_path: str,
        filename: str,
        document_id: str,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        start_time = time.time()

        parsed_blocks: list[NormalizedBlock] | list[dict[str, Any]] = []
        self._extracted_images = []

        # Dispatch on extension.
        # * .txt goes through the plain-text loader.
        # * .xlsx uses openpyxl because MinerU may return success without
        #   emitting markdown for spreadsheet workbooks.
        # * Other rich formats (.pdf, .docx, .pptx, .html, .md) go through
        #   the unified MinerU pipeline so they produce normalized blocks
        #   with page/section metadata.
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".txt":
            document = load_utf8_text_document(file_path)
            parsed_blocks = self._get_parse_service().document_normalizer.normalize_text(
                document.page_content,
                source=filename,
            )

        elif ext in self.EXCEL_EXTENSIONS:
            parsed_blocks = self._process_excel_workbook(file_path, filename)

        elif ext in self.MINERU_EXTENSIONS:
            parsed_blocks = await self._process_with_mineru(file_path, document_id, filename)

        else:
            raise ValueError(f"Unsupported file type: {filename}")

        index_service = getattr(self, "document_index_service", None)
        if index_service is None:
            raise RuntimeError(
                "DocumentIndexService is required for document processing; "
                "direct Qdrant chunk persistence has been removed."
            )

        blocks = self._coerce_normalized_blocks(parsed_blocks)
        parse_result = ParseResult(
            blocks=blocks,
            images_data=list(self._extracted_images),
            backend_used="processing/compatibility",
        )

        prepared_images = []
        persisted_images: list[Any] = []
        if hasattr(self, "_extracted_images") and self._extracted_images:
            prepared_images = await self._prepare_images_for_indexing(
                self._extracted_images,
                document_id,
            )
            parse_result.blocks = self._attach_prepared_images_to_blocks(
                parse_result.blocks,
                prepared_images,
            )
            # Persist DocumentImage rows (nullable chunk_id) before the index
            # service's vector writes so a native image Qdrant point is never
            # created without a matching canonical SQL row to authorize it
            # through. DocumentIndexService links chunk_id by page once the
            # generation's chunks exist.
            persisted_images = self._persist_prepared_images(prepared_images, document_id)

        built_chunks = self._build_chunks_for_indexing(parse_result.blocks)
        persisted_chunks = index_service.index_document(
            document=self._document_ref(document_id, conversation_id, user_id, filename),
            built_chunks=built_chunks,
            parse_artifact_id=None,
            image_rows=persisted_images,
        )

        images_stored = len(persisted_images)

        store_result = {
            "chunks_stored": len(persisted_chunks),
            "chunk_id_mapping": {chunk.chunk_index: str(chunk.id) for chunk in persisted_chunks},
        }
        chunks_created = len(built_chunks)

        processing_time = time.time() - start_time

        return {
            "chunks_created": chunks_created,
            "chunks_stored": store_result.get("chunks_stored", 0),
            "images_stored": images_stored,
            "processing_time": processing_time,
            "filename": filename,
        }

    # ---------------------------------------------------------------------------
    # Parse-stage delegation helpers
    # ---------------------------------------------------------------------------
    # All parse logic lives in DocumentParseService.  These wrappers preserve
    # the existing public/private API so that:
    #  * Tests that call these methods directly keep working.
    #  * Tests that mock them on a service instance keep intercepting the call
    #    (Python attribute lookup checks the instance dict first).
    #  * Tests that use object.__new__(DocumentProcessingService) to bypass
    #    __init__ get a lazy-initialised _parse_service on first access.
    # ---------------------------------------------------------------------------

    def _get_parse_service(self) -> "DocumentParseService":
        """Return the parse service, constructing one lazily if __init__ was bypassed."""
        ps = getattr(self, "_parse_service", None)
        if ps is None:
            chunk_builder = getattr(self, "document_chunk_builder", None)
            if chunk_builder is None:
                chunk_builder = DocumentChunkBuilder(
                    target_tokens=self.settings.rag_chunk_target_tokens,
                    overlap_tokens=self.settings.rag_chunk_overlap_tokens,
                    max_tokens=self.settings.rag_chunk_max_tokens,
                )
            ps = DocumentParseService(settings=self.settings, chunk_builder=chunk_builder)
            self._parse_service = ps
        return ps

    async def _process_with_mineru(
        self, file_path: str, document_id: str, original_filename: str | None = None
    ) -> list[NormalizedBlock]:
        """Delegate to DocumentParseService._process_with_mineru.

        Output directory is named ``mineru_output_{document_id}`` so that
        concurrent uploads never share temp paths.
        """
        parse_service = self._get_parse_service()
        blocks, images_data = await parse_service._process_with_mineru(
            file_path=file_path,
            document_id=document_id,
            original_filename=original_filename,
        )
        # Propagate _mineru_output_path so callers that inspect it still work.
        self._mineru_output_path = parse_service._mineru_output_path
        # Store images for the index stage.
        self._extracted_images = images_data
        return blocks

    def _legacy_char_chunk_size(self) -> int:
        """Delegate to parse service."""
        return self._get_parse_service()._legacy_char_chunk_size()

    def _legacy_char_overlap(self) -> int:
        """Delegate to parse service."""
        return self._get_parse_service()._legacy_char_overlap()

    def _create_chunks(
        self,
        documents: list,
        max_chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        """Delegate to parse service."""
        return self._get_parse_service()._create_chunks(documents, max_chunk_size, overlap)

    def _process_excel_workbook(
        self,
        file_path: str,
        original_filename: str | None = None,
    ) -> list[NormalizedBlock]:
        """Delegate Excel workbook parsing to DocumentParseService."""
        return self._get_parse_service()._process_excel_workbook(file_path, original_filename)

    def _extract_excel_rows(self, formula_sheet: Any, value_sheet: Any | None) -> list[list[str]]:
        """Delegate to parse service."""
        return self._get_parse_service()._extract_excel_rows(formula_sheet, value_sheet)

    @classmethod
    def _excel_rows_to_markdown(cls, sheet_name: str, rows: list[list[str]]) -> str:
        """Delegate to parse service."""
        return DocumentParseService._excel_rows_to_markdown(sheet_name, rows)

    @staticmethod
    def _stringify_excel_cell(value: Any) -> str:
        return DocumentParseService._stringify_excel_cell(value)

    @staticmethod
    def _escape_markdown_table_cell(value: str) -> str:
        return DocumentParseService._escape_markdown_table_cell(value)

    def _parse_content_list_json(self, content_list_path: Path) -> list[dict[str, Any]]:
        """Delegate to parse service."""
        return self._get_parse_service()._parse_content_list_json(content_list_path)

    def _create_chunks_with_page_metadata(
        self,
        content_blocks: list[dict[str, Any]],
        page_to_images: dict[int, list[dict[str, Any]]],
        max_chunk_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Delegate to parse service."""
        return self._get_parse_service()._create_chunks_with_page_metadata(
            content_blocks, page_to_images, max_chunk_size
        )

    @staticmethod
    def _normalize_filename_token(value: str) -> str:
        return DocumentParseService._normalize_filename_token(value)

    @staticmethod
    def _collapse_filename_token(value: str) -> str:
        return DocumentParseService._collapse_filename_token(value)

    def _build_filename_aliases(
        self, sanitized_stem: str, original_filename: str | None = None
    ) -> list[str]:
        """Delegate to parse service."""
        return self._get_parse_service()._build_filename_aliases(sanitized_stem, original_filename)

    def _resolve_mineru_output_dir(
        self, output_dir: Path, filename_candidates: list[str]
    ) -> Path | None:
        """Delegate to parse service."""
        return self._get_parse_service()._resolve_mineru_output_dir(output_dir, filename_candidates)

    def _resolve_markdown_file(
        self, search_roots: list[Path], filename_candidates: list[str]
    ) -> Path:
        """Delegate to parse service."""
        return self._get_parse_service()._resolve_markdown_file(search_roots, filename_candidates)

    # --- End of delegated parse methods ---

    def _coerce_normalized_blocks(
        self,
        blocks: list[NormalizedBlock] | list[dict[str, Any]],
    ) -> list[NormalizedBlock]:
        if all(isinstance(block, NormalizedBlock) for block in blocks):
            return list(blocks)
        return self._get_parse_service().document_normalizer.normalize_legacy_chunks(blocks)

    def _build_chunks_for_indexing(
        self,
        blocks: list[NormalizedBlock] | list[dict[str, Any]],
    ):
        normalized_blocks = self._coerce_normalized_blocks(blocks)
        if not normalized_blocks:
            return []

        builder = getattr(self, "document_chunk_builder", None)
        if builder is None:
            builder = DocumentChunkBuilder(
                target_tokens=self.settings.rag_chunk_target_tokens,
                overlap_tokens=self.settings.rag_chunk_overlap_tokens,
                max_tokens=self.settings.rag_chunk_max_tokens,
            )
            self.document_chunk_builder = builder

        return builder.build(normalized_blocks)

    async def _prepare_images_for_indexing(
        self,
        images_data: list[dict[str, Any]],
        document_id: str,
    ) -> list[dict[str, Any]]:
        storage_path = Path(self.settings.document_images_storage_path)
        if not storage_path.is_absolute():
            storage_path = (Path.cwd() / storage_path).resolve()

        doc_storage_path = storage_path / document_id
        doc_storage_path.mkdir(parents=True, exist_ok=True)

        # Phase 1: copy files, collect caption candidates
        candidates = []  # (img_data, dest_path, metadata_caption)
        seen_source_paths: set[str] = set()

        for img_data in images_data:
            source_path = Path(img_data["path"])
            source_key = str(source_path.resolve()) if source_path.exists() else str(source_path)
            if source_key in seen_source_paths:
                continue
            seen_source_paths.add(source_key)

            if not source_path.exists():
                logger.warning("Skipping missing extracted image %s", source_path)
                continue

            dest_path = doc_storage_path / source_path.name
            shutil.copy2(source_path, dest_path)

            caption = self._caption_from_image_metadata(img_data)
            candidates.append((img_data, dest_path, caption))

        # Phase 2: caption concurrently (bounded by semaphore)
        sem = asyncio.Semaphore(self.settings.image_caption_max_concurrency)

        async def _caption_one(img_data, dest_path, metadata_caption):
            content_sha256 = await asyncio.to_thread(self._sha256_of_file, dest_path)
            caption = metadata_caption  # fallback
            if self.gemini_client:
                async with sem:
                    try:
                        image_bytes = await asyncio.to_thread(self._encode_jpeg_bytes, dest_path)
                        generated = await self._generate_image_caption_with_retry(
                            image_bytes=image_bytes,
                            image_name=dest_path.name,
                        )
                        if generated:
                            caption = generated
                    except Exception as e:
                        logger.error(
                            "Failed to generate caption for %s: %s",
                            dest_path.name,
                            e,
                            exc_info=True,
                        )
            try:
                relative_image_path = dest_path.relative_to(Path.cwd())
            except ValueError:
                relative_image_path = dest_path

            return {
                **img_data,
                "stored_path": str(relative_image_path),
                "caption": caption,
                "content_sha256": content_sha256,
                "page_number": img_data.get("page_number"),
                "mime_type": img_data["mime_type"],
            }

        results = await asyncio.gather(*[_caption_one(*c) for c in candidates])
        return list(results)

    def _attach_prepared_images_to_chunks(
        self,
        chunks_with_metadata: list[dict[str, Any]],
        prepared_images: list[dict[str, Any]],
    ) -> None:
        """Version-1 compatibility helper for mutable chunk dictionaries."""
        if not prepared_images:
            return

        for chunk_index, chunk_data in enumerate(chunks_with_metadata):
            if not isinstance(chunk_data, dict):
                continue

            matching_images = [
                image
                for image in prepared_images
                if self._image_matches_chunk(image, chunk_data, chunk_index)
            ]
            if not matching_images:
                continue

            existing_images = list(chunk_data.get("images") or [])
            image_context_lines = []
            for image in matching_images:
                if image not in existing_images:
                    existing_images.append(image)
                caption = str(image.get("caption") or "").strip()
                if caption:
                    image_context_lines.append(f"[Image: {caption}]")
                else:
                    image_name = Path(str(image.get("stored_path") or image.get("path"))).name
                    image_context_lines.append(f"[Image: {image_name}]")

            text = str(chunk_data.get("text", "") or "").rstrip()
            for line in image_context_lines:
                if line not in text:
                    text = f"{text}\n{line}" if text else line

            chunk_data["text"] = text
            chunk_data["images"] = existing_images
            chunk_data["has_images"] = True
            chunk_data["image_count"] = len(existing_images)

    def _attach_prepared_images_to_blocks(
        self,
        blocks: list[NormalizedBlock],
        prepared_images: list[dict[str, Any]],
    ) -> list[NormalizedBlock]:
        """Return blocks with image captions attached without mutating block records."""
        if not prepared_images:
            return blocks

        explicitly_owned_images = {
            image_index
            for image_index, image in enumerate(prepared_images)
            if any(self._block_explicitly_owns_image(block, image) for block in blocks)
        }
        updated: list[NormalizedBlock] = []
        for block_index, block in enumerate(blocks):
            block_view = {
                "page_start": block.page_start,
                "page_end": block.page_end,
            }
            matching_images = [
                image
                for image_index, image in enumerate(prepared_images)
                if self._block_explicitly_owns_image(block, image)
                or (
                    image_index not in explicitly_owned_images
                    and self._image_matches_normalized_block_page(
                        image,
                        block_view,
                        block_index,
                    )
                )
            ]
            if not matching_images:
                updated.append(block)
                continue

            metadata = dict(block.metadata)
            existing_images = list(metadata.get("images") or [])
            context_lines: list[str] = []
            for image in matching_images:
                if block.kind == "image":
                    # Structural provenance (Task 11): only the owning image
                    # block's own bbox/section_path apply here, never a
                    # neighboring text block's. Mutates the shared dict so it
                    # also reaches ``prepared_images`` for SQL persistence.
                    if image.get("bbox") is None:
                        image["bbox"] = block.metadata.get("bbox")
                    if not image.get("section_path"):
                        image["section_path"] = list(block.section_path)
                if image not in existing_images:
                    existing_images.append(image)
                caption = str(image.get("caption") or "").strip()
                if caption:
                    context_lines.append(f"[Image: {caption}]")
                elif block.kind != "image":
                    name = Path(str(image.get("stored_path") or image.get("path"))).name
                    context_lines.append(f"[Image: {name}]")

            text = block.text.rstrip()
            for line in context_lines:
                if line not in text:
                    text = f"{text}\n{line}" if text else line
            metadata.update(
                {
                    "images": existing_images,
                    "has_images": True,
                    "image_count": len(existing_images),
                }
            )
            updated.append(replace(block, text=text, metadata=metadata))
        return updated

    @staticmethod
    def _block_explicitly_owns_image(
        block: NormalizedBlock,
        image: dict[str, Any],
    ) -> bool:
        block_paths = {
            str(block.metadata.get(key))
            for key in ("path", "img_path", "relative_path")
            if block.metadata.get(key)
        }
        for linked_image in block.metadata.get("images") or []:
            if not isinstance(linked_image, dict):
                continue
            block_paths.update(
                str(linked_image[key])
                for key in ("path", "img_path", "relative_path", "stored_path")
                if linked_image.get(key)
            )
        image_paths = {
            str(image.get(key))
            for key in ("path", "img_path", "relative_path", "stored_path")
            if image.get(key)
        }
        if block_paths & image_paths:
            return True
        return any(
            left.replace("\\", "/").endswith(right.replace("\\", "/"))
            or right.replace("\\", "/").endswith(left.replace("\\", "/"))
            for left in block_paths
            for right in image_paths
        )

    @staticmethod
    def _image_matches_normalized_block_page(
        image: dict[str, Any],
        block_view: dict[str, Any],
        block_index: int,
    ) -> bool:
        canonical_image = dict(image)
        parser_page = image.get("page_number")
        if parser_page is not None:
            canonical_image["page_number"] = int(parser_page) + 1
        return DocumentProcessingService._image_matches_chunk(
            canonical_image,
            block_view,
            block_index,
        )

    def _persist_prepared_images(
        self,
        prepared_images: list[dict[str, Any]],
        document_id: str,
    ) -> list[Any]:
        """Persist ``DocumentImage`` rows with a nullable ``chunk_id``.

        Called before ``DocumentIndexService.index_document`` so a native
        image Qdrant point is never created without a matching canonical SQL
        row; ``index_document`` links ``chunk_id`` once its generation's
        chunks exist.

        First deletes this document's own orphaned (``chunk_id IS NULL``)
        rows from an earlier attempt that persisted but never reached
        linking (Celery retries re-enter this whole path with no delete or
        dedup otherwise, so a document that failed twice then succeeded
        would end up with three row sets for the same images). A row that
        was already linked belongs to a generation that reached
        ``mark_ready`` and is left alone.
        """
        self.document_image_repository.delete_unlinked_by_document_id(uuid.UUID(document_id))
        persisted: list[Any] = []
        for img_data in prepared_images:
            persisted.append(
                self.document_image_repository.create(
                    self._image_create_payload(img_data, document_id, chunk_id=None)
                )
            )
        return persisted

    @staticmethod
    def _image_create_payload(
        img_data: dict[str, Any],
        document_id: str,
        *,
        chunk_id: UUID | None,
    ) -> DocumentImageCreate:
        page_number = img_data.get("page_number")
        return DocumentImageCreate(
            document_id=uuid.UUID(document_id),
            chunk_id=chunk_id,
            image_path=img_data["stored_path"],
            image_caption=img_data.get("caption"),
            page_number=page_number + 1 if page_number is not None else None,
            mime_type=img_data["mime_type"],
            bbox=img_data.get("bbox"),
            section_path=list(img_data.get("section_path") or []),
            content_sha256=img_data.get("content_sha256"),
        )

    @staticmethod
    def _caption_from_image_metadata(img_data: dict[str, Any]) -> str | None:
        raw_caption = img_data.get("caption") or img_data.get("image_caption")
        if isinstance(raw_caption, list):
            return " ".join(str(item) for item in raw_caption if item).strip() or None
        if raw_caption:
            return str(raw_caption).strip() or None
        return None

    @staticmethod
    def _image_matches_chunk(
        image: dict[str, Any],
        chunk_data: dict[str, Any],
        chunk_index: int,
    ) -> bool:
        page_number = image.get("page_number")
        if page_number is None:
            return chunk_index == 0

        page_start = chunk_data.get("page_start")
        page_end = chunk_data.get("page_end")
        if page_start is None and page_end is None:
            return chunk_index == 0

        start = page_start if page_start is not None else page_end
        end = page_end if page_end is not None else page_start
        return start <= page_number <= end

    @staticmethod
    def _display_page_number(value: Any) -> int | None:
        if value is None:
            return None
        return int(value) + 1

    @staticmethod
    def _document_ref(
        document_id: str,
        conversation_id: str | None,
        user_id: str | None,
        filename: str | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=uuid.UUID(document_id),
            conversation_id=(uuid.UUID(conversation_id) if conversation_id else None),
            user_id=(uuid.UUID(user_id) if user_id else None),
            filename=filename,
        )

    async def _generate_image_caption_with_retry(
        self, image_bytes: bytes, image_name: str
    ) -> str | None:
        if not self.gemini_client:
            return None

        max_attempts = max(1, getattr(self.settings, "image_caption_max_retry_attempts", 1))

        with self._caption_usage_scope() as operation:
            return await self._caption_with_retries(
                image_bytes, image_name, max_attempts, operation
            )

    @contextlib.contextmanager
    def _caption_usage_scope(self):
        """Bind an ``image_caption`` operation spanning a caption's retries."""
        if self.recorder is None:
            yield None
            return
        context = current_usage_context().child(operation="image_caption")
        with bind_usage_context(context), begin_usage_operation() as operation:
            yield operation

    async def _caption_with_retries(
        self,
        image_bytes: bytes,
        image_name: str,
        max_attempts: int,
        operation: UsageOperation | None,
    ) -> str | None:
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return self._request_image_caption(image_bytes, image_name, operation=operation)
            except genai_errors.ClientError as e:
                last_error = e
                if is_rate_limit_error(e):
                    delay_hint = parse_retry_delay(e)
                    if delay_hint is not None:
                        delay = max(delay_hint, 0.5)
                    else:
                        base_delay = max(
                            getattr(self.settings, "image_caption_retry_delay_seconds", 5.0),
                            0.5,
                        )
                        delay = base_delay * attempt
                    logger.warning(
                        f"Gemini rate limit while captioning {image_name} "
                        f"(attempt {attempt}/{max_attempts}). Waiting {delay:.2f}s before retry."
                    )
                    await asyncio.sleep(delay)
                    continue

                raise
            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected error while captioning {image_name}: {str(e)}",
                    exc_info=True,
                )
                break

        if last_error:
            logger.error(
                f"Exhausted caption retries for {image_name} after {max_attempts} "
                f"attempts: {last_error}"
            )
        return None

    def _request_image_caption(
        self,
        image_bytes: bytes,
        image_name: str,
        *,
        operation: UsageOperation | None = None,
    ) -> str | None:
        """Request a validated structured caption and return its rendered text.

        The structured object covers visible OCR text, chart title, axes,
        legend, values, trends, relationships, and surrounding section
        context so the flattened text is retrievable via lexical/dense
        search. Never logs the caption or OCR content — only ``image_name``.
        """
        prompt_parts = [
            types.Part.from_text(text=_IMAGE_CAPTION_STRUCTURED_PROMPT),
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ]

        model_name = self.settings.image_caption_model

        def _call() -> Any:
            return self.gemini_client.models.generate_content(
                model=model_name,
                contents=prompt_parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ImageCaptionSections,
                ),
            )

        if self.recorder is None or operation is None:
            response = _call()
        else:
            response = self.recorder.record_one_sync_attempt(
                call=_call,
                provider="gemini",
                model=model_name,
                operation=operation,
            )

        return self._render_caption_response(response, image_name)

    @staticmethod
    def _render_caption_response(response: Any, image_name: str) -> str | None:
        """Render the validated structured object, or fall back to raw text.

        The request enforces ``response_schema=ImageCaptionSections`` so a
        parsed structured object is the expected shape. A response that
        somehow isn't structured JSON is treated as a plain-text caption
        as-is (not discarded, and not mislabeled as OCR-specific content).
        """
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, ImageCaptionSections):
            rendered = parsed.render().strip()
            return rendered or None

        text = getattr(response, "text", None)
        if not text:
            logger.warning(f"No caption text in Gemini response for {image_name}")
            return None
        try:
            sections = ImageCaptionSections.model_validate_json(text)
        except (ValidationError, ValueError):
            return text.strip() or None
        rendered = sections.render().strip()
        return rendered or None

    @staticmethod
    def _sha256_of_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _encode_jpeg_bytes(path: Path) -> bytes:
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            buffer = io.BytesIO()
            rgb_img.save(buffer, format="JPEG")
            return buffer.getvalue()

    async def cleanup_temp_files(self, older_than_hours: int = 24) -> dict[str, Any]:
        try:
            temp_dir = os.path.join(os.getcwd(), self.settings.temp_storage_path)
            if not os.path.exists(temp_dir):
                return {"files_removed": 0, "message": "Temp directory does not exist"}

            removed_count = 0
            removed_folders = 0
            cutoff_time = datetime.now(timezone.utc).timestamp() - (older_than_hours * 3600)

            for filename in os.listdir(temp_dir):
                if filename.startswith("."):
                    continue

                file_path = os.path.join(temp_dir, filename)

                # Handle regular files
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime < cutoff_time:
                        try:
                            os.unlink(file_path)
                            removed_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to remove temp file {filename}: {str(e)}")

                # Handle MinerU output folders
                elif os.path.isdir(file_path) and filename.startswith("mineru_output_"):
                    dir_mtime = os.path.getmtime(file_path)
                    if dir_mtime < cutoff_time:
                        try:
                            shutil.rmtree(file_path)
                            removed_folders += 1
                        except Exception as e:
                            logger.warning(f"Failed to remove MinerU folder {filename}: {str(e)}")

            # Cleanup old document image folders
            images_dir = Path(self.settings.document_images_storage_path)
            if images_dir.exists():
                for doc_folder in images_dir.iterdir():
                    if doc_folder.is_dir():
                        folder_mtime = doc_folder.stat().st_mtime
                        if folder_mtime < cutoff_time:
                            shutil.rmtree(doc_folder)
                            removed_folders += 1

            return {
                "files_removed": removed_count,
                "folders_removed": removed_folders,
                "message": (
                    f"Cleaned up {removed_count} files and {removed_folders} folders "
                    f"older than {older_than_hours} hours"
                ),
            }

        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {str(e)}")
            return {
                "files_removed": 0,
                "folders_removed": 0,
                "error": str(e),
                "message": "Temp file cleanup failed",
            }
