"""Structure-aware chunk builder.

Pure library: takes a stream of ``NormalizedBlock`` records (the post-parse
representation from MinerU or a plain-text loader) and emits
``BuiltChunk`` records that the index service can persist.

The builder keeps tables atomic when they fit, splits large tables on
row-group boundaries, carries heading context through ``section_path``,
merges tiny orphan paragraphs into their neighbors, and preserves page
spans across merged blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from app.utils.text_processing import estimate_tokens

_MIN_ORPHAN_TOKENS = 20


@dataclass(frozen=True)
class NormalizedBlock:
    block_id: str
    kind: str
    text: str
    page: int | None = None
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuiltChunk:
    chunk_index: int
    content: str
    content_sha256: str
    char_count: int
    token_count: int
    page_start: int | None
    page_end: int | None
    section_path: list[str]
    block_provenance: list[dict[str, Any]]
    metadata: dict[str, Any]


def _is_table(block: NormalizedBlock) -> bool:
    if block.kind and block.kind.lower() == "table":
        return True
    return bool(block.metadata and block.metadata.get("is_table"))


def _finalize_chunk(
    *,
    chunk_index: int,
    buffered_blocks: list[NormalizedBlock],
    text: str | None = None,
) -> BuiltChunk:
    if text is None:
        text = "\n\n".join(b.text for b in buffered_blocks)
    char_count = len(text)
    token_count = estimate_tokens(text)
    page_starts = [b.page for b in buffered_blocks if b.page is not None]
    page_ends = [
        b.metadata.get("page_end", b.page)
        for b in buffered_blocks
        if b.metadata.get("page_end", b.page) is not None
    ]
    page_start = min(page_starts) if page_starts else None
    page_end = max(page_ends) if page_ends else None

    # Section path is taken from the most recent block's section_path.
    section_path: list[str] = []
    for b in buffered_blocks:
        if b.section_path:
            section_path = list(b.section_path)

    provenance = [
        {"block_id": b.block_id, "kind": b.kind, "page": b.page}
        for b in buffered_blocks
    ]

    metadata: dict[str, Any] = {}
    if any(_is_table(b) for b in buffered_blocks):
        metadata["contains_table"] = True

    return BuiltChunk(
        chunk_index=chunk_index,
        content=text,
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
        char_count=char_count,
        token_count=token_count,
        page_start=page_start,
        page_end=page_end,
        section_path=section_path,
        block_provenance=provenance,
        metadata=metadata,
    )


def _split_large_table(block: NormalizedBlock, *, target_tokens: int) -> list[str]:
    """Split a big table on row-group boundaries, keeping the header row."""
    lines = [line for line in block.text.splitlines() if line.strip()]
    if not lines:
        return [block.text]

    # The first two lines are header + separator for a markdown-style table.
    header_lines = lines[:2] if len(lines) >= 2 and "---" in lines[1] else lines[:1]
    row_lines = lines[len(header_lines):]

    header_tokens = estimate_tokens("\n".join(header_lines))
    out: list[str] = []
    current_rows: list[str] = []
    current_tokens = header_tokens

    for row in row_lines:
        row_tokens = estimate_tokens(row)
        if current_rows and current_tokens + row_tokens > target_tokens:
            out.append("\n".join(header_lines + current_rows))
            current_rows = []
            current_tokens = header_tokens
        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        out.append("\n".join(header_lines + current_rows))

    return out


def _split_long_text(text: str, *, target_tokens: int, overlap_tokens: int) -> list[str]:
    """Split a very long plain-text block into token-bounded pieces with overlap.

    Uses whitespace segmentation as a cheap approximation — word count
    correlates with token count closely enough for the first implementation.
    """
    words = text.split()
    if not words:
        return [text]

    # Estimate how many words roughly fit in target_tokens.
    sample_text = " ".join(words[: min(len(words), 256)])
    sample_tokens = max(1, estimate_tokens(sample_text))
    words_per_token = len(words[: min(len(words), 256)]) / sample_tokens
    target_word_count = max(1, int(target_tokens * words_per_token))
    overlap_word_count = max(0, int(overlap_tokens * words_per_token))

    chunks: list[str] = []
    idx = 0
    while idx < len(words):
        end = min(len(words), idx + target_word_count)
        chunks.append(" ".join(words[idx:end]))
        if end >= len(words):
            break
        idx = max(idx + 1, end - overlap_word_count)
    return chunks


class DocumentChunkBuilder:
    """Build chunks out of a sequence of normalized blocks."""

    def __init__(
        self,
        *,
        target_tokens: int = 400,
        overlap_tokens: int = 40,
        max_tokens: int = 800,
    ):
        if target_tokens <= 0 or max_tokens < target_tokens:
            raise ValueError(
                "target_tokens must be positive and max_tokens must be >= target_tokens"
            )
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens

    def build(self, blocks: list[NormalizedBlock]) -> list[BuiltChunk]:
        chunks: list[BuiltChunk] = []
        buffered: list[NormalizedBlock] = []
        buffered_tokens = 0
        chunk_index = 0

        def emit_buffered():
            nonlocal buffered, buffered_tokens, chunk_index
            if not buffered:
                return
            # Merge tiny orphan trailing blocks into the previous chunk if possible.
            chunks.append(
                _finalize_chunk(chunk_index=chunk_index, buffered_blocks=list(buffered))
            )
            chunk_index += 1
            buffered = []
            buffered_tokens = 0

        for block in blocks:
            block_tokens = estimate_tokens(block.text)

            # --- Atomic tables ----------------------------------------
            if _is_table(block):
                # Flush any pending text blocks first so the table stays on its own.
                emit_buffered()

                if block_tokens <= self.max_tokens:
                    chunks.append(
                        _finalize_chunk(chunk_index=chunk_index, buffered_blocks=[block])
                    )
                    chunk_index += 1
                    continue

                # Oversized table: split on row groups but keep rows intact.
                for piece in _split_large_table(block, target_tokens=self.target_tokens):
                    synthetic = NormalizedBlock(
                        block_id=f"{block.block_id}::chunk-{chunk_index}",
                        kind=block.kind,
                        text=piece,
                        page=block.page,
                        section_path=block.section_path,
                        metadata={**block.metadata, "is_table_split_piece": True},
                    )
                    chunks.append(
                        _finalize_chunk(
                            chunk_index=chunk_index, buffered_blocks=[synthetic]
                        )
                    )
                    chunk_index += 1
                continue

            # --- Long paragraphs that blow past max_tokens on their own ------
            if block_tokens > self.max_tokens:
                emit_buffered()
                for piece in _split_long_text(
                    block.text,
                    target_tokens=self.target_tokens,
                    overlap_tokens=self.overlap_tokens,
                ):
                    synthetic = NormalizedBlock(
                        block_id=f"{block.block_id}::chunk-{chunk_index}",
                        kind=block.kind,
                        text=piece,
                        page=block.page,
                        section_path=block.section_path,
                        metadata=block.metadata,
                    )
                    chunks.append(
                        _finalize_chunk(
                            chunk_index=chunk_index, buffered_blocks=[synthetic]
                        )
                    )
                    chunk_index += 1
                continue

            # --- Regular text blocks: buffer until we hit target -------------
            if buffered and buffered_tokens + block_tokens > self.target_tokens:
                # Respect max_tokens as a hard ceiling unless the current
                # block is small enough to slip in as an orphan merge.
                if (
                    block_tokens <= _MIN_ORPHAN_TOKENS
                    and buffered_tokens + block_tokens <= self.max_tokens
                ):
                    buffered.append(block)
                    buffered_tokens += block_tokens
                    continue
                emit_buffered()

            buffered.append(block)
            buffered_tokens += block_tokens

        emit_buffered()
        return chunks
