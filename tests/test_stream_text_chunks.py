"""Text chunk normalization keeps every character a model produced.

The previous normalizer guessed whether a chunk was cumulative from its text,
so a model that legitimately repeated itself (``ha``, ``ha``) lost text. Chunk
semantics are now declared by the caller: ``delta`` passes through unchanged,
``snapshot`` emits only the suffix that extends the same stream's text.
"""

from __future__ import annotations

import pytest

from app.services.event_streaming.text_chunks import (
    TextChunkAccumulator,
    TextNormalizationError,
)

KEY = ("run-1", "message-1", "block-0")


def _deltas(chunks: list[str], key: tuple[str, ...] = KEY) -> str:
    acc = TextChunkAccumulator()
    return "".join(acc.consume(chunk, mode="delta", stream_key=key) for chunk in chunks)


def test_explicit_chunk_modes_preserve_text():
    acc = TextChunkAccumulator()
    key = ("run-1", "message-1", "block-0")
    assert "".join(acc.consume(t, mode="delta", stream_key=key) for t in ["ha", "ha", "!"]) == (
        "haha!"
    )
    other = ("run-2", "message-2", "block-0")
    assert acc.consume("Hello", mode="snapshot", stream_key=other) == "Hello"
    assert acc.consume("Hello!", mode="snapshot", stream_key=other) == "!"
    assert acc.consume("Hello!", mode="snapshot", stream_key=other) == ""


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param(["ha", "ha", "!"], id="repeated-word"),
        pytest.param(["Wait", ".", ".", "."], id="repeated-punctuation"),
        pytest.param(["def f():\n", "    ", "    ", "return 1\n"], id="code-indentation"),
        pytest.param([" ", " ", "\n", "\n", "\t"], id="whitespace-only"),
        pytest.param(["héllo ", "héllo ", "🙂", "🙂", "中文", "中文"], id="unicode"),
        pytest.param(["abc", "bc", "c"], id="suffix-of-previous"),
        pytest.param(["abc", "abcd", "ab"], id="prefix-of-previous"),
        pytest.param(["The answer", "The answer"], id="exact-repeat"),
    ],
)
def test_deltas_are_never_deduplicated_by_content(chunks):
    assert _deltas(chunks) == "".join(chunks)


def test_empty_delta_emits_nothing():
    assert TextChunkAccumulator().consume("", mode="delta", stream_key=KEY) == ""


def test_separate_stream_keys_do_not_contaminate_each_other():
    acc = TextChunkAccumulator()
    draft = ("run-draft",)
    repair = ("run-repair",)
    assert acc.consume("Hello", mode="snapshot", stream_key=draft) == "Hello"
    # A repair run that starts with the same text is its own stream.
    assert acc.consume("Hello", mode="snapshot", stream_key=repair) == "Hello"
    assert acc.consume("Hello there", mode="snapshot", stream_key=repair) == " there"
    assert acc.consume("Hello world", mode="snapshot", stream_key=("run-3",)) == "Hello world"


def test_snapshot_after_deltas_of_the_same_stream_emits_only_new_text():
    acc = TextChunkAccumulator()
    for chunk in ["ha", " ", "ha!"]:
        acc.consume(chunk, mode="delta", stream_key=KEY)
    assert acc.consume("ha ha!", mode="snapshot", stream_key=KEY) == ""
    assert acc.consume("ha ha! More.", mode="snapshot", stream_key=KEY) == " More."


def test_rewriting_snapshot_is_an_error_not_a_silent_splice():
    acc = TextChunkAccumulator()
    acc.consume("Hello world", mode="snapshot", stream_key=KEY)
    with pytest.raises(TextNormalizationError, match="does not extend"):
        acc.consume("Goodbye", mode="snapshot", stream_key=KEY)


def test_discard_drops_state_for_one_stream_only():
    acc = TextChunkAccumulator()
    acc.consume("Hello", mode="snapshot", stream_key=("run-1", "m"))
    acc.consume("Other", mode="snapshot", stream_key=("run-2", "m"))
    acc.discard(("run-1", "m"))
    assert acc.consume("Hello", mode="snapshot", stream_key=("run-1", "m")) == "Hello"
    assert acc.consume("Other", mode="snapshot", stream_key=("run-2", "m")) == ""


def test_clear_releases_all_state():
    acc = TextChunkAccumulator()
    acc.consume("Hello", mode="snapshot", stream_key=KEY)
    acc.clear()
    assert acc.active_streams == 0
    assert acc.consume("Hello", mode="snapshot", stream_key=KEY) == "Hello"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        TextChunkAccumulator().consume("x", mode="cumulative", stream_key=KEY)  # type: ignore[arg-type]
