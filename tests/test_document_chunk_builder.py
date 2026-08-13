"""Phase 5 guards: DocumentChunkBuilder.

The builder is a pure library — no DB sessions, no Qdrant, no request
state. It takes a stream of ``NormalizedBlock`` records plus token limits
and emits ``BuiltChunk`` records ready for the index service to persist.
"""

from __future__ import annotations

import inspect
from hashlib import sha256

from app.ai.token_counter import TokenCount


class _WhitespaceCounter:
    def count_text(self, *, provider, model, text):
        del provider, model
        return TokenCount(tokens=len(text.split()), strategy="test")


class _SeparatorAwareCounter:
    def count_text(self, *, provider, model, text):
        del provider, model
        return TokenCount(
            tokens=len(text.split()) + text.count("\n\n"),
            strategy="test",
        )


def _block(
    *,
    block_id: str,
    kind: str,
    text: str,
    page: int | None = None,
    section_path: list[str] | None = None,
    metadata: dict | None = None,
):
    from app.services.document_chunk_builder import NormalizedBlock

    return NormalizedBlock(
        block_id=block_id,
        kind=kind,
        text=text,
        page=page,
        section_path=list(section_path or []),
        metadata=dict(metadata or {}),
    )


def _build(blocks, *, target=200, overlap=20, max_tokens=400):
    from app.services.document_chunk_builder import DocumentChunkBuilder

    builder = DocumentChunkBuilder(
        target_tokens=target,
        overlap_tokens=overlap,
        max_tokens=max_tokens,
    )
    return builder.build(blocks)


def _build_with_counter(
    blocks,
    *,
    target,
    overlap,
    max_tokens,
    counter=None,
    semantic_boundary_detector=None,
):
    from app.services.document_chunk_builder import DocumentChunkBuilder

    return DocumentChunkBuilder(
        target_tokens=target,
        overlap_tokens=overlap,
        max_tokens=max_tokens,
        token_counter=counter or _WhitespaceCounter(),
        semantic_boundary_detector=semantic_boundary_detector,
    ).build(blocks)


def test_chunk_builder_is_pure_library():
    """The builder must not pull in DB sessions, Qdrant, or request state."""
    import app.services.document_chunk_builder as mod

    source = inspect.getsource(mod)
    forbidden = [
        "qdrant_client",
        "SessionLocal",
        "session_factory",
        "from app.database",
        "container",
    ]
    for token in forbidden:
        assert token not in source, f"Chunk builder must not import {token!r}"


def test_chunk_content_sha256_is_deterministic():
    blocks = [_block(block_id="b1", kind="paragraph", text="Hello world.", page=1)]
    chunks = _build(blocks)
    assert len(chunks) == 1
    expected = sha256(b"Hello world.").hexdigest()
    assert chunks[0].content_sha256 == expected


def test_chunk_builder_uses_explicit_token_counter_strategy():
    from app.services.document_chunk_builder import DocumentChunkBuilder

    calls = []

    class Counter:
        def count_text(self, *, provider, model, text):
            calls.append((provider, model, text))
            return TokenCount(tokens=7, strategy="test")

    builder = DocumentChunkBuilder(
        target_tokens=20,
        overlap_tokens=0,
        max_tokens=40,
        token_counter=Counter(),
        token_provider="openai",
        token_model="gpt-test",
    )

    chunks = builder.build([_block(block_id="b1", kind="paragraph", text="hello")])

    assert chunks[0].token_count == 7
    assert calls == [("openai", "gpt-test", "hello"), ("openai", "gpt-test", "hello")]


def test_section_path_carries_through_from_blocks():
    blocks = [
        _block(
            block_id="h1",
            kind="heading",
            text="Chapter 1",
            page=1,
            section_path=["Chapter 1"],
        ),
        _block(
            block_id="p1",
            kind="paragraph",
            text="Body paragraph for chapter 1.",
            page=1,
            section_path=["Chapter 1"],
        ),
    ]
    chunks = _build(blocks)
    assert chunks, "Expected at least one chunk"
    assert chunks[0].section_path == ["Chapter 1"]


def test_tables_kept_atomic_when_under_max_tokens():
    """A table block below max_tokens must not be merged with neighboring text."""
    table_md = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    blocks = [
        _block(block_id="p1", kind="paragraph", text="Intro paragraph."),
        _block(
            block_id="t1",
            kind="table",
            text=table_md,
            metadata={"is_table": True},
        ),
        _block(block_id="p2", kind="paragraph", text="Trailing paragraph."),
    ]
    chunks = _build(blocks, target=200, overlap=0, max_tokens=400)

    table_chunks = [c for c in chunks if "| a |" in c.content]
    assert len(table_chunks) == 1, f"Table should be atomic; got {table_chunks}"
    # The table chunk must not have been concatenated with the surrounding paragraphs.
    assert "Intro paragraph." not in table_chunks[0].content
    assert "Trailing paragraph." not in table_chunks[0].content


def test_large_table_splits_by_row_groups_not_arbitrary_chars():
    """Oversized tables must split at row boundaries, preserving header rows."""
    header = "| col1 | col2 |\n|---|---|"
    rows = [f"| r{i}a | r{i}b |" for i in range(200)]
    table_md = "\n".join([header, *rows])
    blocks = [
        _block(
            block_id="t-big",
            kind="table",
            text=table_md,
            metadata={"is_table": True},
        )
    ]
    chunks = _build(blocks, target=80, overlap=0, max_tokens=120)

    # More than one chunk produced.
    assert len(chunks) > 1
    # Every chunk still begins with the table header pipes (full row, not half).
    for chunk in chunks:
        non_empty_lines = [line for line in chunk.content.splitlines() if line.strip()]
        assert non_empty_lines, f"empty chunk: {chunk.content!r}"
        # No row is truncated mid-line.
        for line in non_empty_lines:
            assert line.strip().startswith("|"), f"Table row appears split mid-line: {line!r}"


def test_page_spans_are_preserved_across_multi_page_chunks():
    blocks = [
        _block(block_id=f"p{i}", kind="paragraph", text=f"Paragraph on page {i}.", page=i)
        for i in (1, 2, 3)
    ]
    chunks = _build(blocks, target=1000, overlap=0, max_tokens=2000)
    # All three short paragraphs fit in one chunk.
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.page_start == 1
    assert chunk.page_end == 3


def test_small_orphan_blocks_merge_with_neighbors():
    blocks = [
        _block(
            block_id="p1", kind="paragraph", text="This is a longer paragraph that stands alone."
        ),
        _block(block_id="p2", kind="paragraph", text="tiny"),
        _block(
            block_id="p3", kind="paragraph", text="Another moderate sentence follows the orphan."
        ),
    ]
    chunks = _build(blocks, target=400, overlap=0, max_tokens=800)
    # Orphan merges — single chunk, all content present.
    assert len(chunks) == 1
    assert "tiny" in chunks[0].content
    assert "This is a longer paragraph" in chunks[0].content


def test_block_provenance_records_contributing_block_ids():
    blocks = [
        _block(block_id="b-alpha", kind="paragraph", text="Alpha content."),
        _block(block_id="b-beta", kind="paragraph", text="Beta content."),
    ]
    chunks = _build(blocks, target=1000, overlap=0, max_tokens=2000)
    assert chunks
    ids = [p.get("block_id") for p in chunks[0].block_provenance]
    assert ids == ["b-alpha", "b-beta"]


def test_builder_respects_target_tokens_for_splitting():
    # A long paragraph above target should produce more than one chunk.
    long_text = " ".join([f"word{i}" for i in range(600)])
    blocks = [_block(block_id="long", kind="paragraph", text=long_text)]
    chunks = _build(blocks, target=100, overlap=10, max_tokens=200)
    assert len(chunks) >= 2, f"Expected splitting of a long paragraph; got {len(chunks)}"


def test_long_text_prefers_sentence_boundaries():
    sentences = [
        f"Sentence {i} keeps a complete thought with enough words for chunking."
        for i in range(1, 30)
    ]
    long_text = " ".join(sentences)

    chunks = _build(
        [_block(block_id="sentences", kind="paragraph", text=long_text)],
        target=35,
        overlap=0,
        max_tokens=55,
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.content.endswith("."), f"Chunk split mid-sentence: {chunk.content!r}"
        assert chunk.token_count <= 55


def test_built_chunk_exposes_required_attributes():
    from app.services.document_chunk_builder import BuiltChunk, NormalizedBlock

    expected_block_fields = {"block_id", "kind", "text", "page", "section_path", "metadata"}
    assert set(NormalizedBlock.__dataclass_fields__) >= expected_block_fields

    expected_chunk_fields = {
        "chunk_index",
        "content",
        "content_sha256",
        "char_count",
        "token_count",
        "page_start",
        "page_end",
        "section_path",
        "block_provenance",
        "metadata",
    }
    assert set(BuiltChunk.__dataclass_fields__) >= expected_chunk_fields


def test_regular_adjacent_chunks_overlap_only_inside_same_section():
    blocks = [
        _block(
            block_id=f"p-{index}",
            kind="paragraph",
            text=f"sentence{index} alpha{index} beta{index} gamma{index} delta{index}.",
            section_path=["A"],
        )
        for index in range(4)
    ]

    chunks = _build_with_counter(
        blocks,
        target=9,
        overlap=3,
        max_tokens=14,
    )

    assert len(chunks) >= 2
    first_words = chunks[0].content.split()
    second_words = chunks[1].content.split()
    shared_tail = 0
    for size in range(1, min(len(first_words), len(second_words)) + 1):
        if first_words[-size:] == second_words[:size]:
            shared_tail = size
    assert 0 < shared_tail <= 3


def test_overlap_provenance_includes_only_blocks_that_supply_the_tail():
    chunks = _build_with_counter(
        [
            _block(
                block_id="page-1",
                kind="paragraph",
                text="old1 old2 old3 old4",
                page=1,
                section_path=["A"],
            ),
            _block(
                block_id="page-2",
                kind="paragraph",
                text="near1 near2 near3 near4",
                page=2,
                section_path=["A"],
            ),
            _block(
                block_id="page-3",
                kind="paragraph",
                text=" ".join(f"current{index}" for index in range(9)),
                page=3,
                section_path=["A"],
            ),
        ],
        target=8,
        overlap=2,
        max_tokens=12,
    )

    overlapped = chunks[1]
    assert overlapped.content.startswith("near3 near4\n\ncurrent0")
    assert [item["block_id"] for item in overlapped.block_provenance] == [
        "page-2",
        "page-3",
    ]
    assert overlapped.page_start == 2
    assert overlapped.page_end == 3


def test_overlap_does_not_cross_heading_or_table_boundary():
    blocks = [
        _block(
            block_id="a-1",
            kind="paragraph",
            text="a1 a2 a3 a4 a5 a6",
            section_path=["A"],
        ),
        _block(
            block_id="heading-b",
            kind="heading",
            text="Heading B",
            section_path=["B"],
        ),
        _block(
            block_id="b-1",
            kind="paragraph",
            text="b1 b2 b3 b4 b5 b6",
            section_path=["B"],
        ),
        _block(
            block_id="table-b",
            kind="table",
            text="| h |\n|---|\n| row |",
            section_path=["B"],
            metadata={"is_table": True},
        ),
        _block(
            block_id="b-2",
            kind="paragraph",
            text="tail1 tail2 tail3 tail4 tail5 tail6",
            section_path=["B"],
        ),
    ]

    chunks = _build_with_counter(
        blocks,
        target=7,
        overlap=3,
        max_tokens=12,
    )

    heading_chunk = next(chunk for chunk in chunks if "Heading B" in chunk.content)
    table_index = next(i for i, chunk in enumerate(chunks) if "| row |" in chunk.content)
    assert "a4 a5 a6" not in heading_chunk.content
    assert table_index > 0
    assert "b4 b5 b6" not in chunks[table_index].content
    assert table_index + 1 < len(chunks)
    assert "| row |" not in chunks[table_index + 1].content
    assert all(chunk.token_count <= 12 for chunk in chunks)


def test_heading_only_chunk_is_not_used_as_overlap_for_its_section_body():
    chunks = _build_with_counter(
        [
            _block(
                block_id="heading",
                kind="heading",
                text="one two three four five",
                section_path=["Section"],
            ),
            _block(
                block_id="body",
                kind="paragraph",
                text="body six seven eight nine ten",
                section_path=["Section"],
            ),
        ],
        target=5,
        overlap=3,
        max_tokens=10,
    )

    assert len(chunks) == 3
    assert chunks[1].content == "body six seven eight nine"


def test_final_rendered_count_includes_block_separators():
    blocks = [
        _block(block_id="p-1", kind="paragraph", text="one two three four five"),
        _block(block_id="p-2", kind="paragraph", text="six seven eight nine ten"),
    ]

    chunks = _build_with_counter(
        blocks,
        target=10,
        overlap=0,
        max_tokens=10,
        counter=_SeparatorAwareCounter(),
    )

    assert len(chunks) == 2
    assert all(chunk.token_count <= 10 for chunk in chunks)


def test_large_table_repeats_caption_and_header_without_repeating_rows():
    caption = "[Table: Revenue]"
    header = "| Region | Revenue |"
    separator = "|---|---|"
    rows = [f"| region-{index} | value-{index} |" for index in range(12)]
    block = _block(
        block_id="table",
        kind="table",
        text="\n".join([caption, header, separator, *rows]),
        section_path=["Results"],
        metadata={
            "is_table": True,
            "caption": ["Revenue"],
            "header": ["Region", "Revenue"],
            "body": [[f"region-{i}", f"value-{i}"] for i in range(12)],
        },
    )

    chunks = _build_with_counter(
        [block],
        target=12,
        overlap=4,
        max_tokens=16,
    )

    assert len(chunks) > 1
    assert all(caption in chunk.content and header in chunk.content for chunk in chunks)
    for row in rows:
        assert sum(row in chunk.content for chunk in chunks) == 1
    assert all(chunk.token_count <= 16 for chunk in chunks)


def test_neighbor_indices_are_assigned_after_final_build():
    blocks = [
        _block(
            block_id=f"p-{index}",
            kind="paragraph",
            text=" ".join(f"word{index}-{part}" for part in range(21)),
            section_path=["A"],
        )
        for index in range(3)
    ]

    chunks = _build_with_counter(blocks, target=21, overlap=1, max_tokens=25)

    assert [chunk.metadata["previous_chunk_index"] for chunk in chunks] == [None, 0, 1]
    assert [chunk.metadata["next_chunk_index"] for chunk in chunks] == [1, 2, None]


def test_semantic_boundary_flushes_without_overlap_inside_same_section():
    class Detector:
        def break_before(self, blocks):
            del blocks
            return frozenset({"p-2"})

    blocks = [
        _block(
            block_id="p-1",
            kind="paragraph",
            text="first one two three",
            section_path=["A"],
        ),
        _block(
            block_id="p-2",
            kind="paragraph",
            text="second four five six",
            section_path=["A"],
        ),
    ]

    chunks = _build_with_counter(
        blocks,
        target=20,
        overlap=3,
        max_tokens=24,
        semantic_boundary_detector=Detector(),
    )

    assert [chunk.content for chunk in chunks] == [
        "first one two three",
        "second four five six",
    ]


def test_legacy_prechunked_blocks_keep_one_to_one_boundaries():
    blocks = [
        _block(
            block_id=f"legacy-{index}",
            kind="paragraph",
            text=f"legacy chunk {index} stays unchanged",
            metadata={"legacy_prechunked": True, "source_chunk_index": index},
        )
        for index in range(2)
    ]

    chunks = _build_with_counter(blocks, target=3, overlap=2, max_tokens=4)

    assert [chunk.content for chunk in chunks] == [block.text for block in blocks]
    assert [p["block_id"] for p in chunks[0].block_provenance] == ["legacy-0"]
