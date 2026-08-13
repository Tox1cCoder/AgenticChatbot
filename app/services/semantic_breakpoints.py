"""Optional semantic boundaries for normalized document blocks.

The numeric helpers are intentionally dependency-free so the candidate mode can
be measured without adding a scientific-computing dependency to ingestion.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from app.services.document_blocks import NormalizedBlock

_T = TypeVar("_T")


def pairwise(values: Sequence[_T]) -> Iterator[tuple[_T, _T]]:
    """Yield adjacent pairs without materializing a second collection."""
    return zip(values, values[1:], strict=False)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, treating a zero vector as no distance.

    A missing vector direction must not manufacture a semantic breakpoint, so
    either zero norm maps to similarity ``1.0`` (distance ``0.0``).
    """
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def percentile(values: Sequence[float], rank: float) -> float:
    """Select a percentile with linear interpolation between adjacent values."""
    if not 0.0 <= rank <= 100.0:
        raise ValueError("percentile rank must be between 0 and 100")
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * rank / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True)
class EmbeddingSemanticBoundaryDetector:
    """Detect high-distance adjacent normalized blocks using existing embeddings."""

    embedding_service: Any
    breakpoint_percentile: float = 90.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.breakpoint_percentile <= 100.0:
            raise ValueError("breakpoint_percentile must be between 0 and 100")

    def break_before(
        self,
        blocks: Sequence[NormalizedBlock],
    ) -> frozenset[str]:
        if len(blocks) < 2:
            return frozenset()
        vectors = self.embedding_service.embed_documents([block.text for block in blocks])
        if len(vectors) != len(blocks):
            raise ValueError(
                "semantic embedding count does not match normalized block count"
            )
        distances = [1.0 - cosine(left, right) for left, right in pairwise(vectors)]
        if not distances or max(distances) <= 0.0:
            return frozenset()
        cutoff = percentile(distances, self.breakpoint_percentile)
        return frozenset(
            blocks[index + 1].block_id
            for index, distance in enumerate(distances)
            if distance >= cutoff
        )
