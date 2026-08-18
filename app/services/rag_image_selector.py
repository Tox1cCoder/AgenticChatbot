"""Bounded, post-retrieval selection and loading of document images.

Retrieval and reranking decide which chunks are relevant. This module decides
which of the images already linked to those chunks are worth spending vision
budget on: it deduplicates by canonical image id, ranks by simple visual-intent
overlap with the query, and enforces hard ceilings on image count, total bytes,
total decoded pixels, and an estimated vision-token cost before any bytes are
read from disk.

Selection (:meth:`RAGImageSelector.select`) does blocking file IO and Pillow
CPU work and is therefore a plain synchronous method. Callers running inside
an asyncio event loop must offload it, e.g. ``await asyncio.to_thread(
selector.select, query, candidates)`` — the same pattern used for other
blocking image work in this codebase (see ``web_image_service.py``).

Never logs image bytes, base64, or raw OCR/caption payloads — only ids and
byte/pixel counts.
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Conservative, provider-agnostic heuristic for budgeting vision tokens from
# decoded pixel count. This is not an exact accounting of any single
# provider's tokenizer — it exists only to keep a turn's total image payload
# within a configured budget before the request is ever sent.
_PIXELS_PER_ESTIMATED_TOKEN = 700

# Simple keyword set used to detect "visual intent" in a query so that images
# actually relevant to a chart/diagram/table question outrank incidental
# figures (e.g. a cover photo) when the image budget forces a choice.
_VISUAL_INTENT_KEYWORDS = frozenset(
    {
        "chart",
        "charts",
        "graph",
        "graphs",
        "diagram",
        "diagrams",
        "figure",
        "figures",
        "image",
        "images",
        "photo",
        "photos",
        "picture",
        "pictures",
        "screenshot",
        "screenshots",
        "plot",
        "plots",
        "table",
        "tables",
        "visual",
        "visualize",
        "visualization",
    }
)


@dataclass(frozen=True)
class ImageCandidate:
    """A caption-linked document image before its bytes are loaded."""

    image_id: UUID
    image_path: str
    mime_type: str
    page_number: int | None = None
    caption: str | None = None


@dataclass(frozen=True)
class SelectedImage:
    image_id: UUID
    mime_type: str
    data: bytes
    byte_count: int
    pixel_count: int
    estimated_vision_tokens: int
    page_number: int | None
    caption: str | None


class RAGImageSelector:
    """Select and load a budget-bounded set of images for one model turn."""

    def __init__(
        self,
        *,
        max_images: int,
        max_bytes: int,
        max_pixels: int,
        max_vision_tokens: int,
        base_dir: Path | None = None,
    ) -> None:
        self.max_images = max(1, int(max_images))
        self.max_bytes = max(1, int(max_bytes))
        self.max_pixels = max(1, int(max_pixels))
        self.max_vision_tokens = max(1, int(max_vision_tokens))
        self.base_dir = base_dir or Path.cwd()

    def select(self, query: str, candidates: list[ImageCandidate]) -> list[SelectedImage]:
        """Deduplicate, rank, load, and budget-bound ``candidates``.

        Blocking: reads files from disk and may run Pillow resize/re-encode
        work. Call via ``asyncio.to_thread`` from async code.
        """
        deduped = self._deduplicate(candidates)
        ranked = self._rank(query, deduped)

        selected: list[SelectedImage] = []
        total_bytes = 0
        total_pixels = 0
        total_tokens = 0
        for candidate in ranked:
            if len(selected) >= self.max_images:
                break
            loaded = self._load_and_bound(candidate)
            if loaded is None:
                continue
            if total_bytes + loaded.byte_count > self.max_bytes:
                continue
            if total_pixels + loaded.pixel_count > self.max_pixels:
                continue
            if total_tokens + loaded.estimated_vision_tokens > self.max_vision_tokens:
                continue
            selected.append(loaded)
            total_bytes += loaded.byte_count
            total_pixels += loaded.pixel_count
            total_tokens += loaded.estimated_vision_tokens
        return selected

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate(candidates: list[ImageCandidate]) -> list[ImageCandidate]:
        seen: set[UUID] = set()
        deduped: list[ImageCandidate] = []
        for candidate in candidates:
            if candidate.image_id in seen:
                continue
            seen.add(candidate.image_id)
            deduped.append(candidate)
        return deduped

    @classmethod
    def _rank(cls, query: str, candidates: list[ImageCandidate]) -> list[ImageCandidate]:
        query_terms = cls._terms(query)
        if not query_terms:
            return list(candidates)
        scored = [
            (-cls._visual_intent_score(query_terms, candidate), index, candidate)
            for index, candidate in enumerate(candidates)
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        return [candidate for _score, _index, candidate in scored]

    @classmethod
    def _visual_intent_score(cls, query_terms: set[str], candidate: ImageCandidate) -> int:
        caption_terms = cls._terms(candidate.caption or "")
        overlap = len(query_terms & caption_terms)
        visual_hit = 1 if caption_terms & _VISUAL_INTENT_KEYWORDS else 0
        return overlap + visual_hit

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {token for token in text.casefold().split() if token}

    # ------------------------------------------------------------------
    # Loading + bounding
    # ------------------------------------------------------------------
    def _load_and_bound(self, candidate: ImageCandidate) -> SelectedImage | None:
        path = self._resolve_path(candidate.image_path)
        try:
            data = path.read_bytes()
        except OSError:
            logger.warning("Selected image is missing or unreadable: id=%s", candidate.image_id)
            return None

        try:
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
        except (UnidentifiedImageError, OSError, ValueError):
            logger.warning("Selected image could not be decoded: id=%s", candidate.image_id)
            return None

        mime_type = candidate.mime_type
        pixel_count = width * height
        if len(data) > self.max_bytes or pixel_count > self.max_pixels:
            resized = self._resize_to_budget(data, width, height)
            if resized is None:
                logger.warning(
                    "Selected image could not be resized under budget: id=%s",
                    candidate.image_id,
                )
                return None
            data, width, height = resized
            mime_type = "image/jpeg"
            pixel_count = width * height

        return SelectedImage(
            image_id=candidate.image_id,
            mime_type=mime_type,
            data=data,
            byte_count=len(data),
            pixel_count=pixel_count,
            estimated_vision_tokens=self._estimate_vision_tokens(pixel_count),
            page_number=candidate.page_number,
            caption=candidate.caption,
        )

    def _resolve_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def _resize_to_budget(
        self, data: bytes, width: int, height: int
    ) -> tuple[bytes, int, int] | None:
        try:
            with Image.open(io.BytesIO(data)) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            return None

        if width * height > self.max_pixels:
            scale = math.sqrt(self.max_pixels / float(width * height))
            width = max(1, int(width * scale))
            height = max(1, int(height * scale))
            image = image.resize((width, height), Image.LANCZOS)

        quality = 85
        encoded = self._encode_jpeg(image, quality)
        while len(encoded) > self.max_bytes and quality > 20:
            quality -= 15
            encoded = self._encode_jpeg(image, quality)
        return encoded, width, height

    @staticmethod
    def _encode_jpeg(image: Any, quality: int) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()

    @staticmethod
    def _estimate_vision_tokens(pixel_count: int) -> int:
        return max(1, pixel_count // _PIXELS_PER_ESTIMATED_TOKEN)
