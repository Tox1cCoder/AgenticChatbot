"""Live Streamlit markdown rendering safety guards.

These tests cover the narrow entity-normalization helper used by the live
streaming path (``response_placeholder.markdown(...)`` in ``demo.py``).
The persisted message path goes through ``sanitize_message_content`` and
is exercised separately.

``demo.py`` re-exports ``normalize_stream_markdown_text`` from
``app.ui.stream_markdown`` so the helper stays importable without
``streamlit`` installed.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from tests.test_demo_plan_widget import _import_demo_with_ui_stubs


def test_escape_markdown_currency_escapes_price_ranges_and_lists():
    """Currency runs must not be mistaken for Streamlit's inline LaTeX."""
    from app.ui.stream_markdown import escape_markdown_currency

    raw = "Budget: $150–$160; options: $5, $10, and $20."

    assert escape_markdown_currency(raw) == (
        r"Budget: \$150–\$160; options: \$5, \$10, and \$20."
    )


def test_escape_markdown_currency_preserves_protected_markdown_and_latex():
    """Code, prior escapes, and intentionally delimited math stay verbatim."""
    from app.ui.stream_markdown import escape_markdown_currency

    raw = (
        "`$150–$160`\n"
        "```text\n$150–$160\n```\n"
        r"\$150\n"
        "$150 + 20$\n"
        "$$\nx + y\n$$"
    )

    assert escape_markdown_currency(raw) == raw


def test_persisted_message_rendering_escapes_currency_before_streamlit(monkeypatch):
    """Stored content stays raw; only the Streamlit render call is escaped."""
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    rendered: list[str] = []
    streamlit_stub.chat_message = lambda _avatar: nullcontext()
    streamlit_stub.markdown = lambda text, **_kwargs: rendered.append(text)
    monkeypatch.setattr(
        demo,
        "_build_rich_response_view_for_msg",
        lambda *_args: SimpleNamespace(is_v1=False, use_legacy_image_gallery=False),
    )

    message = {"content": "$150–$160", "createdAt": "2026-07-30T00:00:00Z"}
    demo.render_message_bubble(message, is_user=True)

    assert message["content"] == "$150–$160"
    assert rendered == [r"\$150–\$160"]


def test_persisted_message_rendering_escapes_unparseable_math(monkeypatch):
    """The post-stream rerun must not resurrect the KaTeX error.

    The live placeholder and the persisted bubble render the same text, so a
    guard on the streaming path alone would only move the error one frame
    later.
    """
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    rendered: list[str] = []
    streamlit_stub.chat_message = lambda _avatar: nullcontext()
    streamlit_stub.markdown = lambda text, **_kwargs: rendered.append(text)
    monkeypatch.setattr(
        demo,
        "_build_rich_response_view_for_msg",
        lambda *_args: SimpleNamespace(is_v1=False, use_legacy_image_gallery=False),
    )

    message = {
        "content": "We spent $500 on R&D and $200 on ops.",
        "createdAt": "2026-08-10T00:00:00Z",
    }
    demo.render_message_bubble(message, is_user=True)

    assert message["content"] == "We spent $500 on R&D and $200 on ops."
    assert rendered == [r"We spent \$500 on R&D and \$200 on ops."]


def test_escape_unterminated_math_fence_escapes_dangling_opener():
    """A ``$$`` block still in flight must not reach KaTeX.

    ``micromark-extension-math`` treats end-of-input as a successful close for
    a ``$$`` *flow* fence, so every partial prefix streamed while the block is
    open is parsed as display math and KaTeX reports a parse error until the
    closing fence arrives.
    """
    from app.ui.stream_markdown import escape_unterminated_math_fence

    partial = "Derivation:\n\n$$\n\\begin{aligned}\nx &= 1\n"

    assert escape_unterminated_math_fence(partial) == (
        "Derivation:\n\n\\$\\$\n\\begin{aligned}\nx &= 1\n"
    )


def test_escape_unterminated_math_fence_keeps_closed_block_verbatim():
    from app.ui.stream_markdown import escape_unterminated_math_fence

    raw = "$$\n\\begin{aligned}\nx &= 1\n\\end{aligned}\n$$\n"

    assert escape_unterminated_math_fence(raw) == raw


def test_escape_unterminated_math_fence_ignores_fenced_code():
    """``$$`` inside a code fence is sample text, not an open math block."""
    from app.ui.stream_markdown import escape_unterminated_math_fence

    raw = "```text\n$$\nnot math\n```\n"

    assert escape_unterminated_math_fence(raw) == raw


def test_escape_unparseable_math_escapes_prose_captured_between_dollars():
    """Prose KaTeX cannot parse was never math — render it literally.

    ``$500 on R&D and $200`` is not an adjacent price run, so the currency
    escaper leaves it alone and Streamlit renders the span as inline math.
    The ``&`` at top level makes KaTeX fail with ``Expected 'EOF', got '&'``.
    """
    from app.ui.stream_markdown import escape_unparseable_math

    raw = "We spent $500 on R&D and $200 on ops."

    assert escape_unparseable_math(raw) == r"We spent \$500 on R&D and \$200 on ops."


def test_escape_unparseable_math_escapes_span_with_unbalanced_brace():
    from app.ui.stream_markdown import escape_unparseable_math

    raw = "Set $x = {a$ before the run."

    assert escape_unparseable_math(raw) == r"Set \$x = {a\$ before the run."


def test_tool_text_payload_rendering_escapes_dollar_markers(monkeypatch):
    """Tool output is model/server text, not Markdown authored for Streamlit."""
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    rendered: list[str] = []
    streamlit_stub.markdown = lambda text, **_kwargs: rendered.append(text)
    streamlit_stub.expander = lambda *_args, **_kwargs: nullcontext()

    demo.render_tool_render_payload({"type": "text", "text": "Quote #7: $500 R&D, $200 ops."})

    assert rendered[0] == r"Quote #7: \$500 R&D, \$200 ops."


def test_image_tool_text_block_rendering_escapes_dollar_markers(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    rendered: list[str] = []
    streamlit_stub.markdown = lambda text, **_kwargs: rendered.append(text)

    demo._render_image_tool_result(
        {"content": [{"type": "text", "text": "Render #3 cost $5 for R&D and $9 total."}]}
    )

    assert rendered == [r"Render #3 cost \$5 for R&D and \$9 total."]


def test_rag_chunk_rendering_escapes_dollar_markers(monkeypatch):
    """Retrieved document text is arbitrary source material."""
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    rendered: list[str] = []
    streamlit_stub.markdown = lambda text, **_kwargs: rendered.append(text)

    chunk = SimpleNamespace(
        rank=1,
        source="invoice.pdf",
        score=None,
        page_label="",
        document_id="",
        chunk_id="",
        image_count=0,
        table_count=None,
        has_tables=False,
        image_captions=[],
        content="Line #4 lists $500 for R&D and $200 for ops.",
    )

    demo._render_rag_chunk_card(SimpleNamespace(), chunk)

    assert rendered[-1] == r"> Line #4 lists \$500 for R&D and \$200 for ops."


def test_escape_currency_prose_math_escapes_amounts_separated_by_prose():
    """Prices joined by ordinary prose are not a recognised price run.

    ``$500 for express and $200`` parses as math *successfully*, so there is
    no error to see — the markers are swallowed and the sentence renders as
    run-together italics instead.
    """
    from app.ui.stream_markdown import escape_currency_prose_math

    raw = "Shipping is $500 for express and $200 for standard."

    assert escape_currency_prose_math(raw) == (
        r"Shipping is \$500 for express and \$200 for standard."
    )


def test_escape_currency_prose_math_preserves_numeric_math():
    """A numeric span with no prose word is math, not a price pair."""
    from app.ui.stream_markdown import escape_currency_prose_math

    raw = "$1 + 2 = 3$ and $2 \\cdot 3$"

    assert escape_currency_prose_math(raw) == raw


def test_escape_currency_prose_math_preserves_symbolic_and_latex_math():
    from app.ui.stream_markdown import escape_currency_prose_math

    raw = "$x + y$ costs $z$, and $1 \\text{ apple} + 2$ stays math."

    assert escape_currency_prose_math(raw) == raw


def test_escape_currency_prose_math_preserves_code_spans():
    from app.ui.stream_markdown import escape_currency_prose_math

    raw = "`$500 for express and $200`"

    assert escape_currency_prose_math(raw) == raw


def test_normalize_display_markdown_text_escapes_prose_separated_prices():
    from app.ui.stream_markdown import normalize_display_markdown_text

    raw = "Shipping is $500 for express and $200 for standard."

    assert normalize_display_markdown_text(raw) == (
        r"Shipping is \$500 for express and \$200 for standard."
    )


def test_escape_unparseable_math_escapes_span_with_stray_group_command():
    r"""``\endgroup`` is in KaTeX's ``endOfExpression`` set like ``}``.

    ``Parser.endOfExpression = new Set(["}", "\\endgroup", "\\end",
    "\\right", "&"])`` — an unmatched member of that set is what produces the
    ``Expected 'EOF', got ...`` error, so all five are treated alike.
    """
    from app.ui.stream_markdown import escape_unparseable_math

    raw = "Before $a \\endgroup b$ after."

    assert escape_unparseable_math(raw) == r"Before \$a \endgroup b\$ after."


def test_escape_unparseable_math_preserves_balanced_group_commands():
    from app.ui.stream_markdown import escape_unparseable_math

    raw = "$\\begingroup a + b \\endgroup$"

    assert escape_unparseable_math(raw) == raw


def test_escape_unparseable_math_preserves_valid_math():
    """Only spans KaTeX would certainly reject may be rewritten."""
    from app.ui.stream_markdown import escape_unparseable_math

    raw = (
        "$x^2 + y^2 = z^2$ and $\\text{cost}_{i}$\n\n"
        "$$\n\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}\n$$\n"
    )

    assert escape_unparseable_math(raw) == raw


def test_escape_unparseable_math_preserves_code_spans():
    from app.ui.stream_markdown import escape_unparseable_math

    raw = "```text\n$500 on R&D and $200\n```\n`$5 & $6`"

    assert escape_unparseable_math(raw) == raw


def test_normalize_stream_markdown_text_guards_partial_display_math():
    """The live path must never hand KaTeX an in-flight math block."""
    from app.ui.stream_markdown import normalize_stream_markdown_text

    partial = "Here it is:\n\n$$\n\\frac{a}{b"

    assert normalize_stream_markdown_text(partial) == "Here it is:\n\n\\$\\$\n\\frac{a}{b"


def test_every_streamed_prefix_keeps_display_math_fence_paired():
    """The live path renders every prefix, so no frame may leave ``$$`` open.

    An escaped fence (``\\$\\$``) no longer starts a line with a dollar
    marker, so the count of *unescaped* fence lines staying even proves no
    frame was handed a half-written display-math block.
    """
    import re

    from app.ui.stream_markdown import normalize_stream_markdown_text

    message = (
        "Derivation:\n\n$$\n\\begin{aligned}\nx &= 1 \\\\\ny &= 2\n"
        "\\end{aligned}\n$$\n\nThat costs $500 for R&D and $200 for ops."
    )
    fence_line_re = re.compile(r"^[ ]{0,3}\${2,}[^$\n]*$", re.MULTILINE)

    for end in range(1, len(message) + 1):
        rendered = normalize_stream_markdown_text(message[:end])
        fences = fence_line_re.findall(rendered)
        assert len(fences) % 2 == 0, f"unpaired $$ fence at prefix {end}: {rendered!r}"


def test_normalize_stream_markdown_text_unescapes_quotes_only():
    """Quote entities (``&quot;``, ``&#34;``, ``&#x22;``, ``&#39;``,
    ``&#x27;``) must be unescaped. Angle-bracket entities must be left
    as-is — broad ``html.unescape`` in the live path would let escaped
    HTML render and is the wrong tradeoff."""
    from app.ui.stream_markdown import normalize_stream_markdown_text

    raw = (
        "model: &quot;gemini-3.1-pro&quot;, provider: &#34;gemini&#34;: "
        "&lt;b&gt;safe text&lt;/b&gt;"
    )

    assert normalize_stream_markdown_text(raw) == (
        'model: "gemini-3.1-pro", provider: "gemini": &lt;b&gt;safe text&lt;/b&gt;'
    )


def test_normalize_stream_markdown_text_handles_apostrophe_entities():
    from app.ui.stream_markdown import normalize_stream_markdown_text

    raw = "It&#39;s &#x27;safe&#x27; for the user&apos;s prompt"

    # &apos; is NOT in the narrow allowlist — only the common quote forms
    # are. The renderer keeps it as-is so we don't expand the surface
    # area beyond what the reported bug actually needed.
    assert normalize_stream_markdown_text(raw) == "It's 'safe' for the user&apos;s prompt"


def test_normalize_stream_markdown_text_handles_empty_and_non_string():
    from app.ui.stream_markdown import normalize_stream_markdown_text

    assert normalize_stream_markdown_text("") == ""
    assert normalize_stream_markdown_text(None) == ""  # type: ignore[arg-type]
    assert normalize_stream_markdown_text(123) == ""  # type: ignore[arg-type]


def test_normalize_stream_markdown_text_strips_inline_rich_markers():
    """Standalone ``<!--rich:<id>-->`` lines are stripped from the live
    placeholder so they do not appear as visible text while the stream is
    in flight. The post-stream rerun resolves the markers and renders the
    referenced rich item inline via ``build_rich_response_view``.
    """
    from app.ui.stream_markdown import normalize_stream_markdown_text

    raw = "Intro paragraph.\n\n<!--rich:widget:abc-123-->\n\nClosing paragraph."

    assert normalize_stream_markdown_text(raw) == ("Intro paragraph.\n\n\n\nClosing paragraph.")


def test_normalize_stream_markdown_text_keeps_inline_marker_in_code():
    """A marker that appears inside an inline-code span is part of prose,
    not a block-level marker line, and must be preserved verbatim."""
    from app.ui.stream_markdown import normalize_stream_markdown_text

    raw = "See `<!--rich:foo-->` for the contract grammar."

    assert normalize_stream_markdown_text(raw) == raw


def test_group_segment_renders_group_html():
    """Characterization test: the view model already routes an
    ``image_group`` rich item as a ``rich`` segment via plain id lookup, with
    no special-casing by type. This pins existing behavior in
    ``build_rich_response_view`` — it is not new behavior added by this test
    — so that a later reader does not mistake it for a regression guard on
    the Streamlit rendering branch (which is exercised separately)."""
    from app.ui.rich_response import build_rich_response_view

    metadata = {
        "rich_items_version": 1,
        "rich_items": [
            {
                "id": "imagegroup:tool:c1",
                "type": "image_group",
                "alt_text": "Images of red panda",
                "payload": {
                    "items": [
                        {"url": "/web-images/1", "mime_type": "image/jpeg"},
                        {"url": "/web-images/2", "mime_type": "image/jpeg"},
                    ]
                },
            }
        ],
    }
    view = build_rich_response_view("Body\n\n<!--rich:imagegroup:tool:c1-->\n", metadata)
    rich_segments = [s for s in view.segments if s.kind == "rich"]
    assert len(rich_segments) == 1
    assert rich_segments[0].item["type"] == "image_group"
