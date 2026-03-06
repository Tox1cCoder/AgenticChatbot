import asyncio
import io
import json
import logging
import os
import time
import uuid
import subprocess
import shutil
import re
import mimetypes
import unicodedata
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from PIL import Image
from google import genai
from google.genai import types, errors as genai_errors
from app.repositories.document_image import DocumentImageRepository
from app.schemas.document_image import DocumentImageCreate

from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.core.events import get_event_bus, DocumentEvent, DocumentEventData

logger = logging.getLogger(__name__)


class DocumentProcessingService:

    def __init__(
        self,
        settings: Settings,
        celery_app,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
        document_image_repository: DocumentImageRepository,
    ):
        self.settings = settings
        self.celery_app = celery_app
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.document_image_repository = document_image_repository
        self.collection_name = settings.qdrant_collection_name
        self.embedding_dimension = settings.embedding_dimension
        self._event_bus = get_event_bus()
        self._mineru_output_path = None
        self.gemini_client = None
        self._init_gemini()
        self._ensure_collection_exists()

    def _init_gemini(self):
        api_key = self.settings.gemini_api_key
        if not api_key:
            self.gemini_client = None
            return

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        try:
            self.gemini_client = genai.Client(api_key=api_key)
        except Exception as e:
            self.gemini_client = None

    def _ensure_collection_exists(self):
        from qdrant_client.models import Distance, VectorParams

        try:
            collections = self.qdrant_client.get_collections()
            exists = any(
                c.name == self.collection_name for c in collections.collections
            )
        except Exception as e:
            logger.warning(
                f"Could not connect to Qdrant: {e}. Collection check skipped."
            )
            return

        if not exists:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension, distance=Distance.COSINE
                ),
            )
        else:
            info = self.qdrant_client.get_collection(self.collection_name)
            actual_size = info.config.params.vectors.size

            if actual_size != self.embedding_dimension:
                raise ValueError(
                    f"Collection '{self.collection_name}' has vector size {actual_size}, "
                    f"expected {self.embedding_dimension}"
                )

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
            "estimated_processing_time": self._estimate_processing_time(
                len(file_content)
            ),
            "message": f"Document '{filename}' queued for processing",
        }

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
            return {"task_id": task_id, "status": "UNKNOWN", "error": str(e)}

    async def process_document(
        self,
        file_path: str,
        filename: str,
        document_id: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        chunks_with_metadata = []
        self._extracted_images = []

        # Load documents
        if filename.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            chunks = self._create_chunks(documents)
            chunks_with_metadata = [{"text": chunk} for chunk in chunks]

        elif filename.lower().endswith(".pdf"):
            chunks_with_metadata = await self._process_pdf_with_mineru(
                file_path, document_id, filename
            )

        elif filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            chunks = self._create_chunks(documents)
            chunks_with_metadata = [{"text": chunk} for chunk in chunks]

        else:
            raise ValueError(f"Unsupported file type: {filename}")

        store_result = await self._store_chunks(
            chunks_with_metadata, filename, document_id, conversation_id
        )

        # Store images if extracted
        images_stored = 0
        if hasattr(self, "_extracted_images") and self._extracted_images:
            images_stored = await self._store_images(
                self._extracted_images,
                document_id,
                store_result.get("chunk_id_mapping", {}),
                chunks_with_metadata,
            )

            # Update chunks with image metadata in Qdrant
            await self._update_chunks_with_images(document_id)

        processing_time = time.time() - start_time

        return {
            "chunks_created": len(chunks_with_metadata),
            "chunks_stored": store_result.get("chunks_stored", 0),
            "images_stored": images_stored,
            "processing_time": processing_time,
            "filename": filename,
        }

    async def _process_pdf_with_mineru(
        self, file_path: str, document_id: str, original_filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            temp_dir = Path(self.settings.temp_storage_path)
            output_dir = temp_dir / f"mineru_output_{document_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._mineru_output_path = str(output_dir)

            subprocess.run(
                [
                    "mineru",
                    "-p",
                    file_path,
                    "-o",
                    str(output_dir),
                    "-f",
                    "true",
                    "-t",
                    "false",
                ],
                timeout=self.settings.mineru_timeout,
                check=True,
                capture_output=True,
                text=True,
            )

            filename_without_ext = Path(file_path).stem
            filename_aliases = self._build_filename_aliases(
                filename_without_ext, original_filename
            )
            base_output_dir = self._resolve_mineru_output_dir(
                output_dir, filename_aliases
            )

            search_roots: List[Path] = []
            if base_output_dir is not None and base_output_dir.exists():
                search_roots.append(base_output_dir)
            else:
                logger.warning(
                    "MinerU output directory missing for %s. Searching entire output tree.",
                    filename_without_ext,
                )

            if output_dir not in search_roots:
                search_roots.append(output_dir)

            markdown_file = self._resolve_markdown_file(search_roots, filename_aliases)
            images_dir = markdown_file.parent / "images"

            # Try to find content_list.json for structured metadata
            content_list_path = (
                markdown_file.parent / f"{markdown_file.stem}_content_list.json"
            )
            content_blocks = None

            if content_list_path.exists():
                try:
                    content_blocks = self._parse_content_list_json(content_list_path)
                    logger.info(
                        "Loaded %d content blocks from content_list.json for %s",
                        len(content_blocks),
                        original_filename or filename_without_ext,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to parse content_list.json for %s: %s. Falling back to markdown.",
                        original_filename or filename_without_ext,
                        str(e),
                    )
                    content_blocks = None
            else:
                logger.info(
                    "content_list.json not found for %s, using markdown fallback",
                    original_filename or filename_without_ext,
                )

            images_by_path: Dict[str, int] = {}

            if content_blocks:
                for block in content_blocks:
                    block_type = block.get("type")
                    if block_type not in ("image", "table"):
                        continue

                    img_path = block.get("img_path", "")
                    page_idx = block.get("page_idx")
                    if img_path and page_idx is not None:
                        images_by_path[img_path] = page_idx

            images_data = []
            page_to_images: Dict[int, List[Dict[str, Any]]] = {}

            if images_dir.exists():
                for img_file in images_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        relative_path = f"images/{img_file.name}"
                        page_number = images_by_path.get(relative_path)

                        mime_type, _ = mimetypes.guess_type(str(img_file))
                        if not mime_type:
                            suffix = img_file.suffix.lower()
                            if suffix in {".jpg", ".jpeg"}:
                                mime_type = "image/jpeg"
                            elif suffix == ".png":
                                mime_type = "image/png"
                            elif suffix == ".gif":
                                mime_type = "image/gif"
                        mime_type = (
                            mime_type or f"image/{img_file.suffix.lstrip('.').lower()}"
                        )

                        image_entry = {
                            "path": str(img_file),
                            "page_number": page_number,
                            "mime_type": mime_type,
                        }
                        images_data.append(image_entry)

                        if page_number is not None:
                            page_to_images.setdefault(page_number, []).append(
                                image_entry
                            )

            if content_blocks:
                chunks_with_metadata = self._create_chunks_with_page_metadata(
                    content_blocks,
                    page_to_images,
                    max_chunk_size=self.settings.document_chunk_size,
                )
            else:
                with open(markdown_file, "r", encoding="utf-8") as f:
                    markdown_content = f.read()

                documents = [
                    type(
                        "Document",
                        (),
                        {"page_content": markdown_content, "metadata": {}},
                    )()
                ]
                chunks = self._create_chunks(documents)

                unpaged_images = [
                    img for img in images_data if img["page_number"] is None
                ]

                chunks_with_metadata = []
                for chunk in chunks:
                    chunks_with_metadata.append(
                        {
                            "text": chunk,
                            "page_start": None,
                            "page_end": None,
                            "has_images": bool(unpaged_images),
                            "image_count": len(unpaged_images),
                            "has_tables": False,
                            "table_count": 0,
                        }
                    )

            # Store images data for later processing
            self._extracted_images = images_data

            return chunks_with_metadata

        except subprocess.TimeoutExpired as exc:
            logger.error(
                "MinerU timed out after %ss while processing %s",
                self.settings.mineru_timeout,
                file_path,
            )
            raise RuntimeError(
                f"MinerU timed out after {self.settings.mineru_timeout}s"
            ) from exc
        except subprocess.CalledProcessError as exc:
            logger.error("MinerU failed while processing %s: %s", file_path, exc.stderr)
            raise RuntimeError(f"MinerU failed with error: {exc.stderr}") from exc
        except Exception as exc:
            logger.error(
                "Unexpected MinerU error while processing %s: %s", file_path, exc
            )
            raise RuntimeError(
                f"Unexpected error in MinerU processing: {str(exc)}"
            ) from exc

    def _create_chunks(
        self,
        documents: List,
        max_chunk_size: int = None,
        overlap: int = None,
    ) -> List[str]:
        # Use configured parameters if not specified
        if max_chunk_size is None:
            max_chunk_size = self.settings.document_chunk_size
        if overlap is None:
            overlap = self.settings.document_chunk_overlap
        separators = ["\n\n", "\n", " ", ""]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=separators,
        )
        split_docs = text_splitter.split_documents(documents)
        return [doc.page_content for doc in split_docs]

    def _parse_content_list_json(self, content_list_path: Path) -> List[Dict[str, Any]]:
        """
        Parse MinerU's content_list.json to extract structured content with metadata.

        Returns list of content blocks with:
        - text/content: The actual content
        - page_idx: Page number (0-indexed)
        - bbox: Bounding box [x0, y0, x1, y1] (normalized to 0-1000)
        - type: text, table, image, equation
        - text_level: Heading level (0=body, 1=h1, 2=h2, etc.)
        """
        with open(content_list_path, "r", encoding="utf-8") as f:
            content_list = json.load(f)

        return content_list

    def _create_chunks_with_page_metadata(
        self,
        content_blocks: List[Dict[str, Any]],
        page_to_images: Dict[int, List[Dict[str, Any]]],
        max_chunk_size: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from content_list.json blocks while preserving page metadata.

        Strategy:
        - Accumulate text from consecutive blocks
        - Split when chunk exceeds max_chunk_size
        - Track page_start and page_end for chunks spanning multiple pages
        - Associate images and tables with their source pages
        """
        if max_chunk_size is None:
            max_chunk_size = self.settings.document_chunk_size

        chunks_with_metadata: List[Dict[str, Any]] = []
        current_chunk_text = ""
        current_page_start: Optional[int] = None
        current_page_end: Optional[int] = None
        current_images: List[Dict[str, Any]] = []
        current_tables: List[Dict[str, Any]] = []
        pages_in_current_chunk: set = set()

        def _finalize_chunk():
            """Save current accumulated chunk if it has content."""
            nonlocal current_chunk_text, current_page_start, current_page_end
            nonlocal current_images, current_tables, pages_in_current_chunk

            if not current_chunk_text.strip():
                return

            # Gather images for pages in this chunk
            chunk_images = current_images.copy()
            for page in pages_in_current_chunk:
                if page in page_to_images:
                    for img in page_to_images[page]:
                        if img not in chunk_images:
                            chunk_images.append(img)

            chunks_with_metadata.append(
                {
                    "text": current_chunk_text.strip(),
                    "page_start": current_page_start,
                    "page_end": current_page_end,
                    "has_images": bool(chunk_images),
                    "image_count": len(chunk_images),
                    "images": chunk_images,
                    "has_tables": bool(current_tables),
                    "table_count": len(current_tables),
                    "tables": current_tables,
                }
            )

            # Reset for next chunk
            current_chunk_text = ""
            current_page_start = None
            current_page_end = None
            current_images = []
            current_tables = []
            pages_in_current_chunk = set()

        for block in content_blocks:
            page_idx = block.get("page_idx", 0)
            block_type = block.get("type", "text")
            bbox = block.get("bbox")
            text_level = block.get("text_level", 0)

            # Extract text content based on block type
            text = ""
            if block_type == "text":
                text = block.get("text", "")
                # Add markdown heading prefix based on text_level
                if text_level and text_level > 0:
                    heading_prefix = "#" * text_level + " "
                    text = heading_prefix + text
            elif block_type == "table":
                # Store table metadata, include body if available
                table_entry = {
                    "page": page_idx,
                    "bbox": bbox,
                    "caption": block.get("table_caption", []),
                    "footnote": block.get("table_footnote", []),
                    "body": block.get("table_body", ""),
                }
                current_tables.append(table_entry)
                # Include caption text in chunk for searchability
                captions = block.get("table_caption", [])
                footnotes = block.get("table_footnote", [])
                if captions:
                    text = "[Table: " + " ".join(captions) + "]"
                elif footnotes:
                    text = "[Table] " + " ".join(
                        str(note) for note in footnotes if note
                    )
                else:
                    text = "[Table]"
            elif block_type == "image":
                # Store image metadata from content_list
                image_entry = {
                    "page": page_idx,
                    "bbox": bbox,
                    "path": block.get("img_path", ""),
                    "caption": block.get("image_caption", []),
                    "footnote": block.get("image_footnote", []),
                }
                current_images.append(image_entry)
                # Include caption text in chunk for searchability
                captions = block.get("image_caption", [])
                footnotes = block.get("image_footnote", [])
                if captions:
                    text = "[Image: " + " ".join(captions) + "]"
                elif footnotes:
                    text = "[Image] " + " ".join(
                        str(note) for note in footnotes if note
                    )
                else:
                    text = "[Image]"
            elif block_type == "equation":
                # Include equation text
                text = block.get("text", "")
                if not text:
                    text = "[Equation]"
            else:
                # Fallback for any other block type
                text = block.get("text", "")

            # Check if adding this block would exceed chunk size
            proposed_length = len(current_chunk_text) + len(text) + 1  # +1 for newline
            if current_chunk_text and proposed_length > max_chunk_size:
                _finalize_chunk()

            # Update page tracking
            if current_page_start is None:
                current_page_start = page_idx
            current_page_end = page_idx
            pages_in_current_chunk.add(page_idx)

            # Append text
            if text:
                current_chunk_text += text + "\n"

        # Finalize the last chunk
        _finalize_chunk()

        logger.info(
            "Created %d page-aware chunks from %d content blocks",
            len(chunks_with_metadata),
            len(content_blocks),
        )

        return chunks_with_metadata

    @staticmethod
    def _normalize_filename_token(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
        normalized = re.sub(r"-+", "-", normalized).strip("-")
        return normalized

    @staticmethod
    def _collapse_filename_token(value: str) -> str:
        normalized = DocumentProcessingService._normalize_filename_token(value)
        return normalized.replace("-", "")

    def _build_filename_aliases(
        self, sanitized_stem: str, original_filename: Optional[str] = None
    ) -> List[str]:
        aliases: List[str] = []

        def _add_alias(value: Optional[str]) -> None:
            if value is None:
                return
            candidate = value.strip()
            if candidate and candidate not in aliases:
                aliases.append(candidate)

        _add_alias(sanitized_stem)
        _add_alias(self._normalize_filename_token(sanitized_stem))

        if sanitized_stem and "_" in sanitized_stem:
            without_prefix = sanitized_stem.split("_", 1)[1]
            _add_alias(without_prefix)
            _add_alias(self._normalize_filename_token(without_prefix))

        if original_filename:
            original_stem = Path(original_filename).stem
            _add_alias(original_stem)
            _add_alias(self._normalize_filename_token(original_stem))
            original_suffix = Path(original_filename).suffix.lower()
        else:
            original_suffix = ""

        if original_suffix:
            existing_aliases = list(aliases)
            for alias in existing_aliases:
                alias_with_ext = f"{alias}{original_suffix}"
                if alias_with_ext not in aliases:
                    aliases.append(alias_with_ext)

        return aliases

    def _resolve_mineru_output_dir(
        self, output_dir: Path, filename_candidates: List[str]
    ) -> Optional[Path]:
        if not output_dir.exists():
            return None

        ordered_candidates: List[str] = []
        for candidate in filename_candidates or []:
            if candidate and candidate not in ordered_candidates:
                ordered_candidates.append(candidate)

        if not ordered_candidates:
            return None

        for candidate in ordered_candidates:
            expected_dir = output_dir / candidate
            if expected_dir.exists():
                return expected_dir

        normalized_targets = list(
            filter(
                None,
                (
                    self._normalize_filename_token(candidate)
                    for candidate in ordered_candidates
                    if candidate
                ),
            )
        )
        normalized_target_set = set(normalized_targets)
        collapsed_target_set = set(
            filter(
                None,
                (
                    self._collapse_filename_token(candidate)
                    for candidate in ordered_candidates
                    if candidate
                ),
            )
        )

        normalized_matches: List[Path] = []

        child_dirs = [child for child in output_dir.iterdir() if child.is_dir()]

        for child in child_dirs:
            normalized_child = self._normalize_filename_token(child.name)
            collapsed_child = self._collapse_filename_token(child.name)

            def _matches_target(token: Optional[str], token_set: set[str]) -> bool:
                if not token or not token_set:
                    return False
                if token in token_set:
                    return True
                return any(
                    token.endswith(target)
                    or token.startswith(target)
                    or target.endswith(token)
                    or target.startswith(token)
                    for target in token_set
                )

            if _matches_target(
                normalized_child, normalized_target_set
            ) or _matches_target(collapsed_child, collapsed_target_set):
                normalized_matches.append(child)

        if normalized_matches:
            chosen = sorted(
                normalized_matches,
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[0]
            return chosen

        if child_dirs:
            chosen = sorted(
                child_dirs,
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[0]
            return chosen

        return None

    def _resolve_markdown_file(
        self, search_roots: List[Path], filename_candidates: List[str]
    ) -> Path:
        ordered_candidates: List[str] = []
        for candidate in filename_candidates or []:
            if candidate and candidate not in ordered_candidates:
                ordered_candidates.append(candidate)

        if not ordered_candidates:
            raise RuntimeError("MinerU output missing filename hints")

        def _is_markdown_file(path: Path) -> bool:
            suffixes = [suffix.lower() for suffix in path.suffixes]
            if not suffixes:
                lowered_name = path.name.lower()
                return (
                    lowered_name.endswith(".md")
                    or lowered_name.endswith(".markdown")
                    or lowered_name.endswith(".mdx")
                )
            return any(suffix in {".md", ".markdown", ".mdx"} for suffix in suffixes)

        markdown_suffixes = {".md", ".markdown", ".mdx"}

        def _search_in_root(root: Path) -> Optional[Path]:
            if root is None or not root.exists():
                return None

            for candidate in ordered_candidates:
                exact_matches = sorted(
                    root.glob(f"**/{candidate}.md"),
                    key=lambda p: len(p.parts),
                )
                if exact_matches:
                    return exact_matches[0]

            all_markdown = sorted(
                (
                    path
                    for path in root.rglob("*")
                    if path.is_file()
                    and (
                        path.suffix.lower() in markdown_suffixes
                        or _is_markdown_file(path)
                    )
                ),
                key=lambda p: (len(p.parts), p.stat().st_mtime),
            )

            normalized_map: Dict[str, Path] = {}
            collapsed_map: Dict[str, Path] = {}
            for path in all_markdown:
                normalized_name = self._normalize_filename_token(path.stem)
                if normalized_name and normalized_name not in normalized_map:
                    normalized_map[normalized_name] = path
                collapsed_name = self._collapse_filename_token(path.stem)
                if collapsed_name and collapsed_name not in collapsed_map:
                    collapsed_map[collapsed_name] = path

            for candidate in ordered_candidates:
                normalized_candidate = self._normalize_filename_token(candidate)
                if normalized_candidate and normalized_candidate in normalized_map:
                    return normalized_map[normalized_candidate]
                collapsed_candidate = self._collapse_filename_token(candidate)
                if collapsed_candidate and collapsed_candidate in collapsed_map:
                    return collapsed_map[collapsed_candidate]

            normalized_candidates = [
                self._normalize_filename_token(candidate)
                for candidate in ordered_candidates
                if candidate
            ]
            normalized_candidates = [token for token in normalized_candidates if token]
            collapsed_candidates = [
                self._collapse_filename_token(candidate)
                for candidate in ordered_candidates
                if candidate
            ]
            collapsed_candidates = [token for token in collapsed_candidates if token]

            for path in all_markdown:
                normalized_name = self._normalize_filename_token(path.stem) or ""
                collapsed_name = self._collapse_filename_token(path.stem)
                if not normalized_name and not collapsed_name:
                    continue
                for normalized_candidate in normalized_candidates:
                    if (
                        normalized_candidate in normalized_name
                        or normalized_name in normalized_candidate
                    ):
                        return path
                for collapsed_candidate in collapsed_candidates:
                    if (
                        collapsed_candidate in collapsed_name
                        or collapsed_name in collapsed_candidate
                    ):
                        return path

            if all_markdown:
                return all_markdown[0]

            return None

        unique_roots: List[Path] = []
        for root in search_roots or []:
            if root and root not in unique_roots and root.exists():
                unique_roots.append(root)

        default_output_root = Path("output")
        if default_output_root.exists() and default_output_root not in unique_roots:
            unique_roots.append(default_output_root)

        if not unique_roots:
            raise RuntimeError(
                f"MinerU output missing markdown file for {ordered_candidates[0]}"
            )

        searched_roots: List[Path] = []
        for root in unique_roots:
            searched_roots.append(root)
            result = _search_in_root(root)
            if result:
                if root != unique_roots[0]:
                    logger.warning(
                        "MinerU markdown resolved via fallback root %s for %s",
                        root,
                        ordered_candidates[0],
                    )
                return result

        raise RuntimeError(
            f"MinerU output missing markdown file for {ordered_candidates[0]}"
        )

    async def _store_chunks(
        self,
        chunks_with_metadata: List[Dict[str, Any]],
        filename: str,
        document_id: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        points = []
        chunk_id_mapping = {}

        for i, chunk_data in enumerate(chunks_with_metadata):
            chunk_text = (
                chunk_data.get("text", chunk_data)
                if isinstance(chunk_data, dict)
                else chunk_data
            )

            embedding = self.embedding_model.encode(chunk_text).tolist()

            safe_point_id = str(uuid.uuid4())
            chunk_id_mapping[i] = safe_point_id  # Store mapping

            payload = {
                "content": chunk_text,
                "source": filename,
                "document_id": document_id,
                "conversation_id": conversation_id,
                "chunk_index": i,
                "timestamp": datetime.now().isoformat(),
                "file_type": (
                    filename.split(".")[-1] if "." in filename else "unknown"
                ),
            }

            if isinstance(chunk_data, dict):
                if "has_images" in chunk_data:
                    payload["has_images"] = bool(chunk_data.get("has_images", False))
                if (
                    "image_count" in chunk_data
                    and chunk_data["image_count"] is not None
                ):
                    payload["image_count"] = int(chunk_data["image_count"])
                if chunk_data.get("image_prompts"):
                    payload["image_prompts"] = chunk_data["image_prompts"]

                # Add table metadata
                if "has_tables" in chunk_data:
                    payload["has_tables"] = bool(chunk_data.get("has_tables", False))
                if (
                    "table_count" in chunk_data
                    and chunk_data["table_count"] is not None
                ):
                    payload["table_count"] = int(chunk_data["table_count"])

                # Add page metadata for citations
                if "page_start" in chunk_data and chunk_data["page_start"] is not None:
                    # Convert 0-indexed page_idx to 1-indexed page number for user display
                    payload["page_start"] = int(chunk_data["page_start"]) + 1
                if "page_end" in chunk_data and chunk_data["page_end"] is not None:
                    payload["page_end"] = int(chunk_data["page_end"]) + 1
                # Also store single page_number for compatibility
                if "page_start" in chunk_data and chunk_data["page_start"] is not None:
                    if chunk_data.get("page_start") == chunk_data.get("page_end"):
                        payload["page_number"] = int(chunk_data["page_start"]) + 1

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

        return {
            "chunks_stored": total_points,
            "chunk_id_mapping": chunk_id_mapping,
        }

    async def _store_images(
        self,
        images_data: List[Dict[str, Any]],
        document_id: str,
        chunk_id_mapping: Dict[int, str],
        chunks_with_metadata: List[Dict[str, Any]] = None,
    ) -> int:
        stored_count = 0

        try:
            # Create permanent storage directory
            storage_path = Path(self.settings.document_images_storage_path)
            if not storage_path.is_absolute():
                storage_path = (Path.cwd() / storage_path).resolve()

            doc_storage_path = storage_path / document_id
            doc_storage_path.mkdir(parents=True, exist_ok=True)

            for img_data in images_data:
                source_path = Path(img_data["path"])
                dest_path = doc_storage_path / source_path.name
                shutil.copy2(source_path, dest_path)

                caption = None
                if self.gemini_client:
                    try:
                        with Image.open(dest_path) as img:
                            rgb_img = img.convert("RGB")
                            buffer = io.BytesIO()
                            rgb_img.save(buffer, format="JPEG")
                        image_bytes = buffer.getvalue()

                        caption = await self._generate_image_caption_with_retry(
                            image_bytes=image_bytes,
                            image_name=dest_path.name,
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to generate caption for {dest_path.name}: {str(e)}",
                            exc_info=True,
                        )
                        caption = None

                chunk_id = None
                page_number = img_data.get("page_number")

                if (
                    page_number is not None
                    and chunk_id_mapping
                    and chunks_with_metadata
                ):
                    for chunk_idx, chunk_data in enumerate(chunks_with_metadata):
                        page_start = chunk_data.get("page_start")
                        page_end = chunk_data.get("page_end")

                        if (
                            page_start is not None
                            and page_end is not None
                            and page_start <= page_number <= page_end
                        ):
                            chunk_id = chunk_id_mapping.get(chunk_idx)
                            break

                    if chunk_id is None and chunk_id_mapping:
                        chunk_id = chunk_id_mapping.get(0)
                elif chunk_id_mapping:
                    chunk_id = chunk_id_mapping.get(0)

                try:
                    relative_image_path = dest_path.relative_to(Path.cwd())
                except ValueError:
                    relative_image_path = dest_path

                image_record_data = DocumentImageCreate(
                    document_id=uuid.UUID(document_id),
                    chunk_id=uuid.UUID(chunk_id) if chunk_id else None,
                    image_path=str(relative_image_path),
                    image_caption=caption,
                    page_number=page_number + 1 if page_number is not None else None,
                    mime_type=img_data["mime_type"],
                )
                self.document_image_repository.create(image_record_data)
                stored_count += 1
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store images: {str(e)}")
            return 0

    async def _generate_image_caption_with_retry(
        self, image_bytes: bytes, image_name: str
    ) -> Optional[str]:
        if not self.gemini_client:
            return None

        max_attempts = max(
            1, getattr(self.settings, "image_caption_max_retry_attempts", 1)
        )
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                return self._request_image_caption(image_bytes, image_name)
            except genai_errors.ClientError as e:
                last_error = e
                status = (e.status or "").upper() if isinstance(e.status, str) else ""
                if e.code == 429 or status == "RESOURCE_EXHAUSTED":
                    delay_hint = None
                    details = getattr(e, "details", None)
                    detail_entries = []
                    if isinstance(details, dict):
                        error_block = details.get("error")
                        if isinstance(error_block, dict):
                            detail_entries = error_block.get("details") or []
                        if not detail_entries:
                            detail_entries = details.get("details") or []
                    elif isinstance(details, list):
                        detail_entries = details

                    for entry in detail_entries or []:
                        if not isinstance(entry, dict):
                            continue
                        retry_value = entry.get("retryDelay") or entry.get(
                            "retry_delay"
                        )
                        if retry_value is None:
                            continue

                        parsed_value = None
                        if isinstance(retry_value, (int, float)):
                            parsed_value = float(retry_value)
                        elif isinstance(retry_value, str):
                            match = re.match(
                                r"([\d\.]+)\s*([a-zA-Z]*)", retry_value.strip()
                            )
                            if match:
                                amount_str, unit = match.groups()
                                try:
                                    amount = float(amount_str)
                                except ValueError:
                                    amount = None
                                if amount is not None:
                                    unit = unit.lower()
                                    if unit in (
                                        "",
                                        "s",
                                        "sec",
                                        "secs",
                                        "second",
                                        "seconds",
                                    ):
                                        parsed_value = amount
                                    elif unit in (
                                        "ms",
                                        "millisecond",
                                        "milliseconds",
                                    ):
                                        parsed_value = amount / 1000.0
                                    elif unit in (
                                        "m",
                                        "min",
                                        "mins",
                                        "minute",
                                        "minutes",
                                    ):
                                        parsed_value = amount * 60.0
                        elif isinstance(retry_value, dict):
                            seconds = retry_value.get("seconds")
                            nanos = retry_value.get("nanos", 0)
                            if seconds is not None or nanos:
                                seconds = float(seconds or 0)
                                parsed_value = seconds + float(nanos) / 1_000_000_000

                        if parsed_value is not None:
                            delay_hint = parsed_value
                            break

                    if delay_hint is None:
                        message = getattr(e, "message", "")
                        if isinstance(message, str):
                            match = re.search(
                                r"retry in\s+([\d\.]+)s", message, re.IGNORECASE
                            )
                            if match:
                                try:
                                    delay_hint = float(match.group(1))
                                except ValueError:
                                    delay_hint = None

                    if delay_hint is not None:
                        delay = max(delay_hint, 0.5)
                    else:
                        base_delay = max(
                            getattr(
                                self.settings, "image_caption_retry_delay_seconds", 5.0
                            ),
                            0.5,
                        )
                        delay = base_delay * attempt
                    logger.warning(
                        f"Gemini rate limit while captioning {image_name} "
                        f"(attempt {attempt}/{max_attempts}). Waiting {delay:.2f}s before retry."
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error(
                    f"Gemini client error while captioning {image_name}: {str(e)}",
                    exc_info=True,
                )
                return None
            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected error while captioning {image_name}: {str(e)}",
                    exc_info=True,
                )
                break

        if last_error:
            logger.error(
                f"Exhausted caption retries for {image_name} after {max_attempts} attempts: {last_error}"
            )
        return None

    def _request_image_caption(
        self, image_bytes: bytes, image_name: str
    ) -> Optional[str]:
        prompt_parts = [
            types.Part.from_text(text="Describe this image concisely in one sentence."),
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ]

        model_name = getattr(self.settings, "image_caption_model")
        response = self.gemini_client.models.generate_content(
            model=model_name,
            contents=prompt_parts,
        )

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        logger.warning(f"No caption text in Gemini response for {image_name}")
        return None

    async def _update_chunks_with_images(self, document_id: str) -> None:
        try:
            images = self.document_image_repository.get_by_document_id(
                UUID(document_id)
            )

            if not images:
                return

            images_by_chunk = {}
            for image in images:
                if image.chunk_id:
                    chunk_id_str = str(image.chunk_id)
                    if chunk_id_str not in images_by_chunk:
                        images_by_chunk[chunk_id_str] = []
                    images_by_chunk[chunk_id_str].append(image)

            updated_count = 0
            for chunk_id_str, chunk_images in images_by_chunk.items():
                image_ids = [str(img.id) for img in chunk_images]
                image_paths = [img.image_path for img in chunk_images]
                image_captions = [img.image_caption or "" for img in chunk_images]

                self.qdrant_client.set_payload(
                    collection_name=self.collection_name,
                    payload={
                        "image_ids": image_ids,
                        "image_paths": image_paths,
                        "image_captions": image_captions,
                    },
                    points=[chunk_id_str],
                )
                updated_count += 1

        except Exception as e:
            logger.error(
                f"Failed to update chunks with image metadata for document {document_id}: {e}"
            )

    async def cleanup_temp_files(self, older_than_hours: int = 24) -> Dict[str, Any]:
        try:
            temp_dir = os.path.join(os.getcwd(), self.settings.temp_storage_path)
            if not os.path.exists(temp_dir):
                return {"files_removed": 0, "message": "Temp directory does not exist"}

            removed_count = 0
            removed_folders = 0
            cutoff_time = datetime.now(timezone.utc).timestamp() - (
                older_than_hours * 3600
            )

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
                            logger.warning(
                                f"Failed to remove temp file {filename}: {str(e)}"
                            )

                # Handle MinerU output folders
                elif os.path.isdir(file_path) and filename.startswith("mineru_output_"):
                    dir_mtime = os.path.getmtime(file_path)
                    if dir_mtime < cutoff_time:
                        try:
                            shutil.rmtree(file_path)
                            removed_folders += 1
                        except Exception as e:
                            logger.warning(
                                f"Failed to remove MinerU folder {filename}: {str(e)}"
                            )

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
                "message": f"Cleaned up {removed_count} files and {removed_folders} folders older than {older_than_hours} hours",
            }

        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {str(e)}")
            return {
                "files_removed": 0,
                "folders_removed": 0,
                "error": str(e),
                "message": "Temp file cleanup failed",
            }
