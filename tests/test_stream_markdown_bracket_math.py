r"""LaTeX's own delimiters must survive into Streamlit's Markdown.

Streamlit reads only ``$`` and ``$$``. Models emit ``\[...\]`` and
``\(...\)`` constantly, because that is standard LaTeX -- and CommonMark
treats ``\[`` as an *escaped literal bracket*, so the backslash is eaten and
the reader is shown the body as prose:

    [ r_s=\frac{2GM}{c^2} ]

Not a KaTeX parse error, which would at least be visible as a failure. Just
quiet nonsense. These pin the rewrite, and pin the cases it must leave alone.
"""

from __future__ import annotations

import pytest

from app.ui.stream_markdown import (
    normalize_bracket_math,
    normalize_display_markdown_text,
)


def test_the_reported_case_renders_as_math():
    content = (
        "The Schwarzschild radius is\n"
        r"\[ r_s=\frac{2GM}{c^2} \]"
        "\n"
        r"\[ r_s \approx 3\ \text{km}\times\frac{M}{M_\odot} \]"
    )

    rendered = normalize_display_markdown_text(content)

    assert r"$$r_s=\frac{2GM}{c^2}$$" in rendered
    assert r"$$r_s \approx 3\ \text{km}\times\frac{M}{M_\odot}$$" in rendered
    assert "\\[" not in rendered


def test_display_math_split_across_lines_is_joined():
    assert normalize_bracket_math("\\[\nE = mc^2\n\\]") == "$$E = mc^2$$"


def test_inline_parens_become_single_dollars():
    assert normalize_bracket_math(r"where \(v \ll c\) holds.") == r"where $v \ll c$ holds."


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(r"an escaped \[ bracket with no close", id="unclosed_display"),
        pytest.param(r"an escaped \( paren with no close", id="unclosed_inline"),
    ],
)
def test_an_unpaired_delimiter_is_left_alone(content):
    """A lone ``\\[`` is a literal bracket, not the start of math."""
    assert normalize_bracket_math(content) == content


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("```\n\\[ x \\]\n```", id="fenced"),
        pytest.param(r"write `\[ x \]` for display math", id="inline_code"),
    ],
)
def test_documented_examples_in_code_survive(content):
    assert normalize_bracket_math(content) == content


def test_dollar_math_is_untouched():
    content = "$$E = mc^2$$ and $x$."
    assert normalize_bracket_math(content) == content


def test_currency_guards_still_apply_afterwards():
    """The rewrite runs *before* the dollar guards, not instead of them."""
    rendered = normalize_display_markdown_text("costs $5 and $10 total")

    assert r"\$5" in rendered and r"\$10" in rendered


def test_converted_math_faces_the_same_guards_as_authored_math():
    """A rewritten span is ordinary ``$$`` math and must be guarded like it."""
    authored = normalize_display_markdown_text("$$E = mc^2$$")
    converted = normalize_display_markdown_text("\\[E = mc^2\\]")

    assert authored == converted


def test_text_without_any_bracket_delimiter_is_returned_unchanged():
    content = "No math here at all."
    assert normalize_bracket_math(content) is content
