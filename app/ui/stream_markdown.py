"""Live-stream markdown helpers shared between ``demo.py`` and tests.

These helpers must be importable without ``streamlit`` installed so the
test suite can exercise them in headless environments.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_STREAM_MARKDOWN_ENTITY_MAP = {
    "&quot;": '"',
    "&#34;": '"',
    "&#x22;": '"',
    "&#39;": "'",
    "&#x27;": "'",
}

# Standalone inline rich-response marker line. Matches the contract defined in
# ``app.core.rich_response`` (block-level HTML comment, optional leading up to
# three spaces, optional trailing whitespace) but is duplicated here to keep
# this helper free of any heavy imports.
_RICH_MARKER_LINE_RE = re.compile(
    r"^[ ]{0,3}<!--rich:[A-Za-z0-9_\-.:]+-->[ \t]*$",
    re.MULTILINE,
)

# Markdown spans whose contents must never be rewritten. The unclosed fenced
# block is matched last so a closed block always wins; it exists because the
# live stream renders partial text, where a code fence is still open and
# CommonMark already treats the remainder as code.
_FENCED_CODE_PATTERN = (
    r"(?:^[ ]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ ]{0,3}(?P=fence)[ \t]*$)"
)
_INLINE_CODE_PATTERN = r"(?:(?P<inline_ticks>`+).*?(?P=inline_ticks))"
_UNCLOSED_FENCED_CODE_PATTERN = r"(?:^[ ]{0,3}(?P<open_fence>`{3,}|~{3,})[^\n]*\n.*\Z)"
_DISPLAY_MATH_PATTERN = r"(?:\$\$.*?\$\$)"

_MARKDOWN_PROTECTED_SPAN_RE = re.compile(
    "(?ms)"
    + "|".join(
        (
            _FENCED_CODE_PATTERN,
            _DISPLAY_MATH_PATTERN,
            _INLINE_CODE_PATTERN,
            _UNCLOSED_FENCED_CODE_PATTERN,
        )
    )
)

# Same spans, minus display math: the math guards below need to inspect
# ``$$...$$`` bodies rather than skip them.
_CODE_PROTECTED_SPAN_RE = re.compile(
    "(?ms)"
    + "|".join(
        (
            _FENCED_CODE_PATTERN,
            _INLINE_CODE_PATTERN,
            _UNCLOSED_FENCED_CODE_PATTERN,
        )
    )
)
_CURRENCY_AMOUNT_RE = r"\d+(?:,\d{3})*(?:\.\d+)?"
_CURRENCY_SEPARATOR_RE = r"(?:\s*[–—-]\s*|\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)"
_CURRENCY_PRICE_RUN_RE = re.compile(
    rf"(?<!\\)\${_CURRENCY_AMOUNT_RE}(?!\d|\.\d|,\d)"
    rf"(?:{_CURRENCY_SEPARATOR_RE}(?<!\\)\${_CURRENCY_AMOUNT_RE}(?!\d|\.\d|,\d))+"
)


# Block fences, matched per line. ``$$`` opens a math *flow* block only at the
# start of a line, and its meta segment cannot contain a dollar marker — a
# single-line ``$$x + y$$`` is inline math instead, which needs no fence guard.
_CODE_FENCE_LINE_RE = re.compile(r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_MATH_FENCE_OPEN_LINE_RE = re.compile(r"^[ ]{0,3}(?P<fence>\${2,})[^$\n]*$")
_MATH_FENCE_CLOSE_LINE_RE = re.compile(r"^[ ]{0,3}(?P<fence>\${2,})[ \t]*$")

# Candidate math spans. A delimiter run is neither opened nor closed by a
# marker that is itself escaped or part of a longer run, mirroring the
# ``previous`` guard in ``micromark-extension-math``. Inline bodies may cross a
# single newline but never a blank line, which ends the block.
_DISPLAY_MATH_SPAN_RE = re.compile(
    r"(?<![\\$])(\$\$)(?!\$)(.*?)(?<![\\$])(\$\$)(?!\$)",
    re.DOTALL,
)
_INLINE_MATH_SPAN_RE = re.compile(
    r"(?<![\\$])(\$)(?!\$)((?:[^$\n]|\n(?![ \t]*\n))*?)(?<![\\$])(\$)(?!\$)"
)

_MATH_COMMAND_RE = re.compile(r"\\([a-zA-Z]+|.)")
_MATH_MACRO_DEFINITION_RE = re.compile(
    r"\\(?:def|gdef|edef|xdef|newcommand|renewcommand|providecommand|let)\b"
)


def _escape_currency_run(match: re.Match[str]) -> str:
    """Escape every unescaped dollar marker in one recognised price run."""
    return re.sub(r"(?<!\\)\$", r"\\$", match.group(0))


def escape_markdown_currency(text: str) -> str:
    """Escape unambiguous dollar-price runs outside protected Markdown spans.

    Streamlit renders ``$...$`` as inline LaTeX. This display-only transform
    protects price ranges and lists while leaving code and intentional math
    untouched. Lone dollar amounts are intentionally not guessed because they
    are ambiguous with numeric LaTeX.
    """
    if not isinstance(text, str) or not text:
        return ""

    pieces: list[str] = []
    cursor = 0
    for protected_span in _MARKDOWN_PROTECTED_SPAN_RE.finditer(text):
        renderable_span = text[cursor : protected_span.start()]
        pieces.append(_CURRENCY_PRICE_RUN_RE.sub(_escape_currency_run, renderable_span))
        pieces.append(protected_span.group(0))
        cursor = protected_span.end()
    pieces.append(_CURRENCY_PRICE_RUN_RE.sub(_escape_currency_run, text[cursor:]))
    return "".join(pieces)


def _escape_dollars(delimiter: str) -> str:
    """Escape every dollar marker in one math delimiter run."""
    return delimiter.replace("$", r"\$")


def _apply_outside_code(text: str, transform: Callable[[str], str]) -> str:
    """Run ``transform`` over ``text``, leaving code spans byte-for-byte."""
    pieces: list[str] = []
    cursor = 0
    for protected_span in _CODE_PROTECTED_SPAN_RE.finditer(text):
        pieces.append(transform(text[cursor : protected_span.start()]))
        pieces.append(protected_span.group(0))
        cursor = protected_span.end()
    pieces.append(transform(text[cursor:]))
    return "".join(pieces)


def escape_unterminated_math_fence(text: str) -> str:
    r"""Escape a ``$$`` block fence that has no closing fence yet.

    ``micromark-extension-math`` registers ``$$`` as a *flow* construct whose
    tokenizer reports success on end-of-input (``t === null`` routes to the ok
    branch). An in-flight stream therefore parses every partial prefix of an
    open block as display math, so KaTeX is handed half-written LaTeX and
    renders ``KaTeX parse error: ...`` until the closing fence arrives.
    Escaping the opener shows the raw source for those few frames instead, and
    the block renders as math as soon as it is closed.

    Inline ``$$...$$`` needs no guard: ``mathText`` routes end-of-input to its
    *failure* branch, so an unclosed inline run stays literal text.
    """
    if not isinstance(text, str) or not text:
        return ""
    if "$$" not in text:
        return text

    lines = text.split("\n")
    open_code_fence: str | None = None
    open_math_index: int | None = None
    open_math_size = 0

    for index, line in enumerate(lines):
        if open_code_fence is not None:
            if _closes_code_fence(line, open_code_fence):
                open_code_fence = None
            continue
        if open_math_index is not None:
            close_match = _MATH_FENCE_CLOSE_LINE_RE.match(line)
            if close_match and len(close_match.group("fence")) >= open_math_size:
                open_math_index = None
            continue
        code_match = _CODE_FENCE_LINE_RE.match(line)
        if code_match:
            open_code_fence = code_match.group("fence")
            continue
        open_match = _MATH_FENCE_OPEN_LINE_RE.match(line)
        if open_match:
            open_math_index = index
            open_math_size = len(open_match.group("fence"))

    if open_math_index is None:
        return text
    # The meta segment cannot contain ``$``, so the first ``open_math_size``
    # markers on that line are exactly the fence.
    lines[open_math_index] = lines[open_math_index].replace("$", r"\$", open_math_size)
    return "\n".join(lines)


def _closes_code_fence(line: str, open_fence: str) -> bool:
    match = _CODE_FENCE_LINE_RE.match(line)
    if match is None:
        return False
    fence = match.group("fence")
    return (
        fence[0] == open_fence[0]
        and len(fence) >= len(open_fence)
        and not match.group("info").strip()
    )


def _math_span_is_unparseable(body: str) -> bool:
    r"""Report whether KaTeX would certainly reject ``body`` as math.

    Only definite failures count. KaTeX finishes ``parse()`` with
    ``expect("EOF")``, so a token that stops the top-level expression without
    being consumed — an unmatched ``}``, an alignment ``&`` outside an
    environment, a stray ``\end`` or ``\right`` — produces exactly the
    ``Expected 'EOF', got ...`` error being guarded against. Unbalanced groups
    and environments fail just as reliably. Anything else is left alone so
    working math is never rewritten.
    """
    brace_depth = 0
    environment_depth = 0
    left_depth = 0
    group_depth = 0
    allows_parameter_marker = bool(_MATH_MACRO_DEFINITION_RE.search(body))
    index = 0
    while index < len(body):
        character = body[index]
        if character == "\\":
            command_match = _MATH_COMMAND_RE.match(body, index)
            if command_match is None:
                return True  # Trailing backslash: "Expected group after ..."
            command = command_match.group(1)
            if command == "begin":
                environment_depth += 1
            elif command == "end":
                if environment_depth == 0:
                    return True
                environment_depth -= 1
            elif command == "begingroup":
                group_depth += 1
            elif command == "endgroup":
                if group_depth == 0:
                    return True
                group_depth -= 1
            elif command == "left":
                left_depth += 1
            elif command == "right":
                if left_depth == 0:
                    return True
                left_depth -= 1
            index = command_match.end()
            continue
        if character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
            if brace_depth < 0:
                return True
        elif (character == "&" and environment_depth == 0) or (
            character == "#" and not allows_parameter_marker
        ):
            return True
        index += 1
    return bool(brace_depth or environment_depth or left_depth or group_depth)


def _escape_math_span_if_unparseable(match: re.Match[str]) -> str:
    opener, body, closer = match.group(1), match.group(2), match.group(3)
    if not _math_span_is_unparseable(body):
        return match.group(0)
    return f"{_escape_dollars(opener)}{body}{_escape_dollars(closer)}"


def _escape_unparseable_math_spans(chunk: str) -> str:
    chunk = _DISPLAY_MATH_SPAN_RE.sub(_escape_math_span_if_unparseable, chunk)
    return _INLINE_MATH_SPAN_RE.sub(_escape_math_span_if_unparseable, chunk)


def escape_unparseable_math(text: str) -> str:
    r"""Render ``$...$`` spans KaTeX cannot parse as literal text.

    Streamlit reads any two unescaped dollar markers in one block as inline
    math, so ordinary prose gets captured — ``$500 on R&D and $200`` is not an
    adjacent price run, so :func:`escape_markdown_currency` leaves it alone and
    KaTeX fails with ``Expected 'EOF', got '&'``. A span KaTeX rejects was
    never math, so escaping its delimiters cannot break working math: the only
    alternative rendering is an error message.
    """
    if not isinstance(text, str) or not text:
        return ""
    if "$" not in text:
        return text
    return _apply_outside_code(text, _escape_unparseable_math_spans)


def normalize_display_markdown_text(content: str) -> str:
    """Prepare stored Markdown for ``st.markdown`` without mutating storage.

    Applies the display-only dollar-marker guards: recognised price runs are
    escaped, an unterminated ``$$`` block is neutralised, and spans KaTeX
    would reject are rendered literally instead of as a parse error.
    """
    if not isinstance(content, str) or not content:
        return ""
    normalized = escape_unterminated_math_fence(content)
    normalized = escape_markdown_currency(normalized)
    return escape_unparseable_math(normalized)


def normalize_stream_markdown_text(content: str) -> str:
    """Unescape quote entities and hide raw inline rich-item markers.

    Live token rendering calls ``st.markdown(...)`` directly without going
    through the HTML sanitizer used for persisted messages, so models that
    emit ``&quot;`` end up showing the literal entity. We unescape quote
    entities only — broad ``html.unescape`` would turn ``&lt;script&gt;``
    into actual HTML in the streamed output, which is the wrong tradeoff
    for the live path.

    We also strip standalone ``<!--rich:<id>-->`` marker lines from the live
    placeholder. They are the inline rich-response contract markers; they're
    invisible in a CommonMark renderer but Streamlit's safe-mode markdown
    leaves them as literal text. The post-stream rerun resolves the markers
    via ``app.ui.rich_response.build_rich_response_view`` and renders the
    referenced rich item (widget, image, tool view, etc.) at its inline
    position, so suppressing the raw token here only affects what the user
    sees while the stream is still in flight.

    Dollar markers then go through the shared display guards in
    :func:`normalize_display_markdown_text` so no partial prefix reaches
    Streamlit as LaTeX that KaTeX cannot parse.
    """
    if not isinstance(content, str) or not content:
        return ""
    normalized = content
    for entity, replacement in _STREAM_MARKDOWN_ENTITY_MAP.items():
        normalized = normalized.replace(entity, replacement)
    if "<!--rich:" in normalized:
        normalized = _RICH_MARKER_LINE_RE.sub("", normalized)
    return normalize_display_markdown_text(normalized)
