"""Typed structural records shared by document parsing and chunk building."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

BlockKind = Literal["heading", "paragraph", "table", "image", "equation"]


@dataclass(frozen=True, init=False)
class NormalizedBlock:
    """Immutable parser-neutral document block with stable source provenance."""

    block_id: str
    kind: BlockKind
    text: str
    page_start: int | None
    page_end: int | None
    page: int | None
    section_path: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        block_id: str,
        kind: BlockKind,
        text: str,
        page_start: int | None = None,
        page_end: int | None = None,
        section_path: tuple[str, ...] | list[str] = (),
        metadata: Mapping[str, Any] | None = None,
        *,
        page: int | None = None,
    ) -> None:
        # ``page`` is a rollout compatibility alias for callers of the original
        # chunk-builder-local block type.
        if page_start is None:
            page_start = page
        if page_end is None:
            page_end = page_start
        object.__setattr__(self, "block_id", str(block_id))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "text", str(text))
        object.__setattr__(self, "page_start", page_start)
        object.__setattr__(self, "page_end", page_end)
        object.__setattr__(self, "page", page_start)
        object.__setattr__(self, "section_path", tuple(section_path))
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata or {})))


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


def serialize_block(block: NormalizedBlock) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "kind": block.kind,
        "text": block.text,
        "page_start": block.page_start,
        "page_end": block.page_end,
        "section_path": list(block.section_path),
        "metadata": dict(block.metadata),
    }


def deserialize_block(data: Mapping[str, Any]) -> NormalizedBlock:
    return NormalizedBlock(
        block_id=str(data["block_id"]),
        kind=data.get("kind", "paragraph"),
        text=str(data.get("text", "") or ""),
        page_start=data.get("page_start"),
        page_end=data.get("page_end"),
        section_path=tuple(data.get("section_path") or ()),
        metadata=data.get("metadata") or {},
    )
