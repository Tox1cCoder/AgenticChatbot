"""A rich marker that can never resolve must never reach the reader.

Every marker consumer -- the reference parser, the stripper, the authorization
sweep in ``finalize_article_content`` -- matches ids against
``[A-Za-z0-9_\\-.:]+``. That is correct for *resolution* and wrong for
*removal*: a marker the grammar rejects is invisible to all of them, so nothing
deletes it and it renders to the user as literal HTML-comment text.

The observed case was a model building an id out of a title, which puts spaces
in it:

    <!--rich:image:Franz Kafka Die Verwandlung 1915 Kurt Wolff cover:f317...-->

The prompt already forbids inventing ids. This is the backstop for when the
model does it anyway.
"""

from __future__ import annotations

import pytest

from app.core.rich_response import (
    RichMarkerStreamFilter,
    parse_inline_rich_references,
    strip_inline_rich_markers,
    strip_malformed_rich_markers,
)

MALFORMED = (
    "<!--rich:image:Franz Kafka Die Verwandlung 1915 Kurt Wolff cover:"
    "f3170e70-65ba-411a-8c59-bfad1e8fa9f7-->"
)
WELL_FORMED = "<!--rich:image:tool:call_1:0-->"


def test_the_strict_parser_cannot_see_it():
    """Pins why it leaked: it is not a reference, so nothing removes it."""
    assert parse_inline_rich_references(f"a\n\n{MALFORMED}\n\nb") == []
    assert parse_inline_rich_references(f"a\n\n{WELL_FORMED}\n\nb") == ["image:tool:call_1:0"]


def test_it_is_stripped_from_rendered_markdown():
    content = f"Here is the cover.\n\n{MALFORMED}\n\nPublished in 1915."

    cleaned = strip_malformed_rich_markers(content)

    assert "<!--rich:" not in cleaned
    assert "Franz Kafka" not in cleaned
    assert "Here is the cover." in cleaned
    assert "Published in 1915." in cleaned


def test_a_well_formed_marker_is_left_alone():
    """Authorization is the finalizer's decision, not this rule's."""
    content = f"A\n\n{WELL_FORMED}\n\n{MALFORMED}\n\nB"

    cleaned = strip_malformed_rich_markers(content)

    assert WELL_FORMED in cleaned
    assert MALFORMED not in cleaned


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(f"```\n{MALFORMED}\n```", id="fenced"),
        pytest.param(f"Write `{MALFORMED}` to embed.", id="inline_code"),
        pytest.param(f"    {MALFORMED}", id="indented_code"),
    ],
)
def test_documented_examples_survive(content):
    assert strip_malformed_rich_markers(content) == content


def test_the_non_rich_client_stripper_removes_both():
    """A client that renders no markers must be left with none of either kind."""
    content = f"A\n\n{WELL_FORMED}\n\n{MALFORMED}\n\nB"

    assert "<!--rich:" not in strip_inline_rich_markers(content)


# ---------------------------------------------------------------------------
# The stream must apply the same rule, or the reader watches one answer arrive
# and a different one is stored.
# ---------------------------------------------------------------------------


def _stream(chunks: list[str]) -> str:
    stream_filter = RichMarkerStreamFilter()
    return "".join(stream_filter.feed(chunk) for chunk in chunks) + stream_filter.flush()


def test_the_stream_never_publishes_it_whole():
    assert "Franz Kafka" not in _stream(["Cover.\n\n", MALFORMED, "\n\nEnd."])


def test_the_stream_never_publishes_it_split_character_by_character():
    """The realistic case: providers chunk wherever they like."""
    published = _stream(list(f"Cover.\n\n{MALFORMED}\n\nEnd."))

    assert "<!--rich:" not in published
    assert "Franz Kafka" not in published
    assert "Cover." in published
    assert "End." in published


def test_the_stream_passes_a_well_formed_marker_through():
    assert WELL_FORMED in _stream(list(f"A\n\n{WELL_FORMED}\n\nB"))


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param(["a < b and 3<4 ", "done"], id="bare_angle_brackets"),
        pytest.param(["x <!-- an ordinary comment --> y"], id="other_html_comment"),
        pytest.param(["1 < 2", " and ", "2 > 1"], id="comparisons"),
    ],
)
def test_ordinary_prose_is_not_held_back(chunks):
    assert _stream(chunks) == "".join(chunks)


def test_an_unclosed_marker_is_released_at_flush():
    """Held text must never be swallowed, however the stream ends."""
    assert _stream(["text <!--rich:image:oops"]) == "text <!--rich:image:oops"


def test_nothing_is_lost_across_arbitrary_chunk_boundaries():
    source = f"Intro text.\n\n{WELL_FORMED}\n\nMiddle.\n\n{MALFORMED}\n\nTail."
    expected = _stream([source])

    for size in (1, 2, 3, 7, 13, 64):
        chunks = [source[i : i + size] for i in range(0, len(source), size)]
        assert _stream(chunks) == expected, f"chunk size {size} changed the output"
