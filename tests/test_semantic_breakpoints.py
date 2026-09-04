from __future__ import annotations

import pytest

from app.services.document_blocks import NormalizedBlock


def _block(block_id: str, text: str) -> NormalizedBlock:
    return NormalizedBlock(
        block_id=block_id,
        kind="paragraph",
        text=text,
        section_path=("A",),
    )


def test_cosine_and_pairwise_helpers_have_deterministic_edge_semantics():
    from app.services.semantic_breakpoints import cosine, pairwise

    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert list(pairwise(["a", "b", "c"])) == [("a", "b"), ("b", "c")]


@pytest.mark.parametrize(
    ("values", "rank", "expected"),
    [
        ([], 90.0, 0.0),
        ([0.5], 90.0, 0.5),
        ([0.0, 1.0], 50.0, 0.5),
        ([0.0, 1.0], 90.0, 0.9),
    ],
)
def test_percentile_uses_linear_interpolation(values, rank, expected):
    from app.services.semantic_breakpoints import percentile

    assert percentile(values, rank) == pytest.approx(expected)


def test_embedding_detector_breaks_before_high_distance_block():
    from app.services.semantic_breakpoints import EmbeddingSemanticBoundaryDetector

    class Embeddings:
        def embed_documents(self, texts):
            assert texts == ["alpha", "near alpha", "different"]
            return [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]

    detector = EmbeddingSemanticBoundaryDetector(
        embedding_service=Embeddings(),
        breakpoint_percentile=90.0,
    )

    assert detector.break_before(
        [_block("a", "alpha"), _block("b", "near alpha"), _block("c", "different")]
    ) == frozenset({"c"})


def test_embedding_detector_skips_embedding_for_empty_or_single_block():
    from app.services.semantic_breakpoints import EmbeddingSemanticBoundaryDetector

    class Embeddings:
        def embed_documents(self, texts):
            raise AssertionError(f"must not embed {texts!r}")

    detector = EmbeddingSemanticBoundaryDetector(embedding_service=Embeddings())

    assert detector.break_before([]) == frozenset()
    assert detector.break_before([_block("a", "alpha")]) == frozenset()


def test_embedding_detector_does_not_break_between_zero_vectors():
    from app.services.semantic_breakpoints import EmbeddingSemanticBoundaryDetector

    class Embeddings:
        def embed_documents(self, texts):
            return [[0.0, 0.0] for _ in texts]

    detector = EmbeddingSemanticBoundaryDetector(embedding_service=Embeddings())

    assert detector.break_before([_block("a", "a"), _block("b", "b")]) == frozenset()


def test_zero_percentile_does_not_promote_zero_distance_pairs_to_boundaries():
    from app.services.semantic_breakpoints import EmbeddingSemanticBoundaryDetector

    class Embeddings:
        def embed_documents(self, texts):
            assert len(texts) == 3
            return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    detector = EmbeddingSemanticBoundaryDetector(
        embedding_service=Embeddings(),
        breakpoint_percentile=0.0,
    )

    assert detector.break_before(
        [_block("a", "a"), _block("b", "b"), _block("c", "c")]
    ) == frozenset({"c"})


def test_default_percentile_ignores_zero_distance_population():
    from app.services.semantic_breakpoints import EmbeddingSemanticBoundaryDetector

    class Embeddings:
        def embed_documents(self, texts):
            assert len(texts) == 12
            return [[1.0, 0.0], [0.0, 1.0], *([[0.0, 0.0]] * 10)]

    blocks = [_block(f"block-{index}", str(index)) for index in range(12)]

    assert EmbeddingSemanticBoundaryDetector(embedding_service=Embeddings()).break_before(
        blocks
    ) == frozenset({"block-1"})


def test_semantic_config_defaults_to_disabled():
    from app.core.config import Settings

    fields = Settings.model_fields
    assert fields["rag_semantic_chunking_enabled"].default is False
    assert fields["rag_semantic_breakpoint_percentile"].default == 90.0


def test_disabled_builder_wiring_does_not_construct_embedding_detector():
    from app.core.container import _build_document_chunk_builder

    calls = []

    def detector_factory():
        calls.append(True)
        raise AssertionError("disabled semantic chunking instantiated embeddings")

    builder = _build_document_chunk_builder(
        target_tokens=400,
        overlap_tokens=40,
        max_tokens=800,
        semantic_chunking_enabled=False,
        semantic_detector_factory=detector_factory,
    )

    assert builder.semantic_boundary_detector is None
    assert calls == []
