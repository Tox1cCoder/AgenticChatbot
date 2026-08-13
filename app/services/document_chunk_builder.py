"""Build bounded, structure-aware chunks from normalized document blocks."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol

from app.ai.token_counter import TokenCounter
from app.services.document_blocks import BuiltChunk, NormalizedBlock

_MIN_ORPHAN_TOKENS = 20
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])(?:\s+|(?=\S))")


class _TextCounter(Protocol):
    def count_text(self, *, provider: str, model: str, text: str): ...


class _BoundaryDetector(Protocol):
    def break_before(
        self,
        blocks: Sequence[NormalizedBlock],
    ) -> frozenset[str]: ...


@dataclass(frozen=True)
class DocumentTokenStrategy:
    counter: _TextCounter
    provider: str
    model: str

    def count(self, text: str) -> int:
        return int(
            self.counter.count_text(
                provider=self.provider,
                model=self.model,
                text=text,
            ).tokens
        )


@dataclass(frozen=True)
class _ChunkDraft:
    content: str
    blocks: tuple[NormalizedBlock, ...]
    section_path: tuple[str, ...]
    allow_overlap_from_previous: bool = False
    is_table: bool = False
    legacy_prechunked: bool = False

    @property
    def ends_with_heading(self) -> bool:
        return bool(self.blocks and self.blocks[-1].kind == "heading")


def _is_table(block: NormalizedBlock) -> bool:
    return block.kind == "table" or bool(block.metadata.get("is_table"))


def _render_blocks(blocks: Sequence[NormalizedBlock]) -> str:
    return "\n\n".join(block.text for block in blocks)


def _unique_blocks(blocks: Sequence[NormalizedBlock]) -> list[NormalizedBlock]:
    seen: set[str] = set()
    unique: list[NormalizedBlock] = []
    for block in blocks:
        if block.block_id not in seen:
            seen.add(block.block_id)
            unique.append(block)
    return unique


def _finalize_chunk(
    *,
    chunk_index: int,
    buffered_blocks: Sequence[NormalizedBlock],
    token_strategy: DocumentTokenStrategy,
    text: str,
) -> BuiltChunk:
    blocks = _unique_blocks(buffered_blocks)
    page_starts = [block.page_start for block in blocks if block.page_start is not None]
    page_ends = [block.page_end for block in blocks if block.page_end is not None]
    section_path: list[str] = []
    for block in blocks:
        if block.section_path:
            section_path = list(block.section_path)

    provenance = [
        {
            "block_id": block.block_id,
            "kind": block.kind,
            "page": block.page_start,
            "page_start": block.page_start,
            "page_end": block.page_end,
            "section_path": list(block.section_path),
            "metadata": dict(block.metadata),
        }
        for block in blocks
    ]

    metadata: dict[str, Any] = {}
    image_count = 0
    table_count = 0
    source_chunk_indices: list[Any] = []
    for block in blocks:
        block_metadata = block.metadata
        if block_metadata.get("image_count") is not None:
            image_count += int(block_metadata.get("image_count") or 0)
        elif block.kind == "image":
            image_count += 1
        if block_metadata.get("table_count") is not None:
            table_count += int(block_metadata.get("table_count") or 0)
        elif _is_table(block):
            table_count += 1
        if block_metadata.get("source_chunk_index") is not None:
            source_chunk_indices.append(block_metadata["source_chunk_index"])

    if image_count or any(block.metadata.get("has_images") for block in blocks):
        metadata.update(has_images=True, image_count=image_count)
    if table_count or any(block.metadata.get("has_tables") for block in blocks):
        metadata.update(
            has_tables=True,
            table_count=table_count or 1,
            contains_table=True,
        )
    if source_chunk_indices:
        metadata["source_chunk_indices"] = source_chunk_indices

    return BuiltChunk(
        chunk_index=chunk_index,
        content=text,
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
        char_count=len(text),
        token_count=token_strategy.count(text),
        page_start=min(page_starts) if page_starts else None,
        page_end=max(page_ends) if page_ends else None,
        section_path=section_path,
        block_provenance=provenance,
        metadata=metadata,
    )


def _split_sentence_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units.extend(unit.strip() for unit in _SENTENCE_BOUNDARY.split(paragraph) if unit.strip())
    return units or [text]


def _largest_prefix(
    units: Sequence[str],
    *,
    separator: str,
    budget: int,
    token_strategy: DocumentTokenStrategy,
) -> int:
    low, high = 1, len(units)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        candidate = separator.join(units[:middle])
        if token_strategy.count(candidate) <= budget:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _split_atomic_unit(
    text: str,
    *,
    budget: int,
    max_tokens: int,
    token_strategy: DocumentTokenStrategy,
) -> list[str]:
    words = text.split()
    units = words if len(words) > 1 else list(text)
    separator = " " if len(words) > 1 else ""
    pieces: list[str] = []
    remaining = units
    while remaining:
        count = _largest_prefix(
            remaining,
            separator=separator,
            budget=budget,
            token_strategy=token_strategy,
        )
        if count == 0:
            if token_strategy.count(remaining[0]) > max_tokens:
                raise ValueError("a single text unit exceeds the chunk hard limit")
            count = 1
        pieces.append(separator.join(remaining[:count]))
        remaining = remaining[count:]
    return pieces


def _split_long_text(
    text: str,
    *,
    target_tokens: int,
    max_tokens: int,
    token_strategy: DocumentTokenStrategy,
) -> list[str]:
    """Split at Latin/CJK sentence boundaries with word/character fallback."""
    pieces: list[str] = []
    current: list[str] = []
    for unit in _split_sentence_units(text):
        if token_strategy.count(unit) > target_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.extend(
                _split_atomic_unit(
                    unit,
                    budget=target_tokens,
                    max_tokens=max_tokens,
                    token_strategy=token_strategy,
                )
            )
            continue
        candidate = " ".join([*current, unit])
        if current and token_strategy.count(candidate) > target_tokens:
            pieces.append(" ".join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _table_parts(block: NormalizedBlock) -> tuple[list[str], list[str], list[str]]:
    lines = [line.strip() for line in block.text.splitlines() if line.strip()]
    if not lines:
        return [], [], []

    prefix: list[str] = []
    cursor = 0
    while cursor < len(lines) and lines[cursor].startswith("[Table:"):
        prefix.append(lines[cursor])
        cursor += 1
    if cursor < len(lines) and lines[cursor].startswith("|"):
        prefix.append(lines[cursor])
        cursor += 1
        if cursor < len(lines) and re.match(r"^\|?[\s:|-]+\|?$", lines[cursor]):
            prefix.append(lines[cursor])
            cursor += 1

    suffix: list[str] = []
    while len(lines) > cursor and lines[-1].startswith("[Table footnote:"):
        suffix.insert(0, lines.pop())
    return prefix, lines[cursor:], suffix


def _split_large_table(
    block: NormalizedBlock,
    *,
    target_tokens: int,
    max_tokens: int,
    token_strategy: DocumentTokenStrategy,
) -> list[str]:
    """Split a table into row groups while repeating its caption and header."""
    prefix, rows, suffix = _table_parts(block)
    if not rows:
        return [block.text]
    pieces: list[str] = []
    current_rows: list[str] = []

    def render(group: Sequence[str], trailing: Sequence[str] = ()) -> str:
        return "\n".join([*prefix, *group, *trailing])

    for row in rows:
        candidate = render([*current_rows, row])
        if current_rows and token_strategy.count(candidate) > target_tokens:
            pieces.append(render(current_rows))
            current_rows = []
            candidate = render([row])
        if token_strategy.count(candidate) > max_tokens:
            raise ValueError(
                f"table row in block {block.block_id!r} exceeds the chunk hard limit"
            )
        current_rows.append(row)
    if current_rows:
        final = render(current_rows, suffix)
        if token_strategy.count(final) <= max_tokens:
            pieces.append(final)
        else:
            pieces.append(render(current_rows))
            suffix_piece = render([], suffix)
            if token_strategy.count(suffix_piece) > max_tokens:
                raise ValueError(
                    f"table footnote in block {block.block_id!r} exceeds the chunk hard limit"
                )
            pieces.append(suffix_piece)
    return pieces


def _tail_with_budget(
    text: str,
    *,
    budget: int,
    token_strategy: DocumentTokenStrategy,
) -> str:
    if budget <= 0 or not text:
        return ""
    words = text.split()
    units = words if len(words) > 1 else list(text)
    separator = " " if len(words) > 1 else ""
    low, high = 1, len(units)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = separator.join(units[-middle:])
        if token_strategy.count(candidate) <= budget:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _overlap_source_blocks(
    blocks: Sequence[NormalizedBlock],
    overlap: str,
) -> list[NormalizedBlock]:
    """Return the smallest source-block suffix that contributed overlap text."""
    words = overlap.split()
    remaining_units = len(words) if len(words) > 1 else len(overlap)
    sources: list[NormalizedBlock] = []
    for block in reversed(blocks):
        block_words = block.text.split()
        block_units = len(block_words) if len(block_words) > 1 else len(block.text)
        if block_units <= 0:
            continue
        sources.append(block)
        remaining_units -= block_units
        if remaining_units <= 0:
            break
    sources.reverse()
    return sources


class DocumentChunkBuilder:
    """Build bounded chunks from parser-neutral structural blocks."""

    def __init__(
        self,
        *,
        target_tokens: int = 400,
        overlap_tokens: int = 40,
        max_tokens: int = 800,
        token_counter: _TextCounter | None = None,
        token_provider: str = "openai",
        token_model: str = "gpt-4o",
        semantic_boundary_detector: _BoundaryDetector
        | Callable[[Sequence[NormalizedBlock]], frozenset[str]]
        | None = None,
    ):
        if target_tokens <= 0 or max_tokens < target_tokens:
            raise ValueError(
                "target_tokens must be positive and max_tokens must be >= target_tokens"
            )
        if overlap_tokens < 0 or overlap_tokens > max_tokens:
            raise ValueError("overlap_tokens must be between zero and max_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens
        self.semantic_boundary_detector = semantic_boundary_detector
        self.token_strategy = DocumentTokenStrategy(
            counter=token_counter or TokenCounter(),
            provider=token_provider,
            model=token_model,
        )

    def _semantic_boundaries(
        self,
        blocks: Sequence[NormalizedBlock],
    ) -> frozenset[str]:
        detector = self.semantic_boundary_detector
        if detector is None:
            return frozenset()
        method = getattr(detector, "break_before", None)
        if method is not None:
            return frozenset(method(blocks))
        return frozenset(detector(blocks))

    def build(self, blocks: list[NormalizedBlock]) -> list[BuiltChunk]:
        semantic_boundaries = self._semantic_boundaries(blocks)
        drafts: list[_ChunkDraft] = []
        buffered: list[NormalizedBlock] = []
        buffered_tokens = 0
        buffered_allows_overlap = False

        def previous_allows_overlap(block: NormalizedBlock) -> bool:
            if (
                not drafts
                or drafts[-1].is_table
                or drafts[-1].legacy_prechunked
                or drafts[-1].ends_with_heading
            ):
                return False
            return (
                drafts[-1].section_path == tuple(block.section_path)
                and block.kind != "heading"
                and block.block_id not in semantic_boundaries
            )

        def emit_buffered() -> None:
            nonlocal buffered, buffered_tokens, buffered_allows_overlap
            if not buffered:
                return
            drafts.append(
                _ChunkDraft(
                    content=_render_blocks(buffered),
                    blocks=tuple(buffered),
                    section_path=tuple(buffered[-1].section_path),
                    allow_overlap_from_previous=buffered_allows_overlap,
                )
            )
            buffered = []
            buffered_tokens = 0
            buffered_allows_overlap = False

        def append_text_piece(
            block: NormalizedBlock,
            text: str,
            *,
            split_piece_index: int,
            allow_overlap: bool,
        ) -> None:
            piece = NormalizedBlock(
                block_id=block.block_id,
                kind=block.kind,
                text=text,
                page_start=block.page_start,
                page_end=block.page_end,
                section_path=block.section_path,
                metadata={**block.metadata, "split_piece_index": split_piece_index},
            )
            drafts.append(
                _ChunkDraft(
                    content=text,
                    blocks=(piece,),
                    section_path=tuple(block.section_path),
                    allow_overlap_from_previous=allow_overlap,
                )
            )

        for block in blocks:
            block_tokens = self.token_strategy.count(block.text)

            if block.metadata.get("legacy_prechunked"):
                emit_buffered()
                drafts.append(
                    _ChunkDraft(
                        content=block.text,
                        blocks=(block,),
                        section_path=tuple(block.section_path),
                        legacy_prechunked=True,
                    )
                )
                continue

            is_boundary = (
                block.block_id in semantic_boundaries
                or block.kind == "heading"
                or bool(buffered and buffered[-1].section_path != block.section_path)
            )
            if is_boundary:
                emit_buffered()

            if _is_table(block):
                emit_buffered()
                pieces = (
                    [block.text]
                    if block_tokens <= self.max_tokens
                    else _split_large_table(
                        block,
                        target_tokens=self.target_tokens,
                        max_tokens=self.max_tokens,
                        token_strategy=self.token_strategy,
                    )
                )
                for piece_index, piece_text in enumerate(pieces):
                    piece_block = NormalizedBlock(
                        block_id=block.block_id,
                        kind=block.kind,
                        text=piece_text,
                        page_start=block.page_start,
                        page_end=block.page_end,
                        section_path=block.section_path,
                        metadata={**block.metadata, "table_split_piece_index": piece_index},
                    )
                    drafts.append(
                        _ChunkDraft(
                            content=piece_text,
                            blocks=(piece_block,),
                            section_path=tuple(block.section_path),
                            is_table=True,
                        )
                    )
                continue

            if block_tokens > self.target_tokens:
                emit_buffered()
                pieces = _split_long_text(
                    block.text,
                    target_tokens=self.target_tokens,
                    max_tokens=self.max_tokens,
                    token_strategy=self.token_strategy,
                )
                for piece_index, piece_text in enumerate(pieces):
                    append_text_piece(
                        block,
                        piece_text,
                        split_piece_index=piece_index,
                        allow_overlap=(
                            previous_allows_overlap(block)
                            if piece_index == 0
                            else block.kind != "heading"
                        ),
                    )
                continue

            if buffered:
                candidate = _render_blocks([*buffered, block])
                candidate_tokens = self.token_strategy.count(candidate)
                if candidate_tokens > self.target_tokens:
                    if block_tokens <= _MIN_ORPHAN_TOKENS and candidate_tokens <= self.max_tokens:
                        buffered.append(block)
                        buffered_tokens = candidate_tokens
                        continue
                    emit_buffered()

            if not buffered:
                buffered_allows_overlap = previous_allows_overlap(block) and not is_boundary
                buffered_tokens = block_tokens
            buffered.append(block)

        emit_buffered()

        chunks: list[BuiltChunk] = []
        previous_draft: _ChunkDraft | None = None
        for chunk_index, draft in enumerate(drafts):
            content = draft.content
            provenance_blocks: list[NormalizedBlock] = list(draft.blocks)
            if (
                previous_draft is not None
                and draft.allow_overlap_from_previous
                and not draft.is_table
                and not previous_draft.is_table
                and self.overlap_tokens > 0
            ):
                for budget in range(self.overlap_tokens, 0, -1):
                    overlap = _tail_with_budget(
                        previous_draft.content,
                        budget=budget,
                        token_strategy=self.token_strategy,
                    )
                    if not overlap:
                        continue
                    candidate = f"{overlap}\n\n{content}"
                    if self.token_strategy.count(candidate) <= self.max_tokens:
                        content = candidate
                        provenance_blocks = [
                            *_overlap_source_blocks(previous_draft.blocks, overlap),
                            *provenance_blocks,
                        ]
                        break

            chunk = _finalize_chunk(
                chunk_index=chunk_index,
                buffered_blocks=provenance_blocks,
                token_strategy=self.token_strategy,
                text=content,
            )
            if not draft.legacy_prechunked and chunk.token_count > self.max_tokens:
                raise AssertionError(
                    f"chunk {chunk_index} exceeds hard limit: "
                    f"{chunk.token_count} > {self.max_tokens}"
                )
            chunks.append(chunk)
            previous_draft = draft

        for index, chunk in enumerate(chunks):
            chunk.metadata["previous_chunk_index"] = index - 1 if index else None
            chunk.metadata["next_chunk_index"] = index + 1 if index + 1 < len(chunks) else None
        return chunks
