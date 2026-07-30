# Markdown currency rendering design

## Problem

Streamlit treats `$...$` as inline LaTeX. A response such as `$150–$160`
therefore renders the first amount as mathematics instead of two dollar
prices. The behaviour appears in streamed text, persisted chat bubbles, and
Markdown segments interleaved with rich responses.

## Decision

Add a display-only Markdown normalizer that escapes dollar signs in recognised
currency expressions before each `st.markdown` call. It must leave the model
response, AI SDK payloads, and persisted database content unchanged.

The normalizer will scan Markdown while preserving protected spans:

- fenced code blocks;
- inline-code spans;
- already escaped dollar signs;
- valid inline and display LaTeX delimiters.

Outside protected spans it will recognise only unambiguous dollar-price
constructs: ranges (`$150–$160` and `$150-$160`) and comma/conjunction price
lists (for example, `$10, $20, and $30`). Recognised currency dollar signs are
escaped as `\$`; inline or display LaTeX — including numeric expressions such
as `$150 + 20$` — remains untouched. A lone or otherwise ambiguous `$150` is
governed by the output convention below rather than guessed.

## Prompt convention

Add one shared, concise output instruction at final system-prompt assembly:

> In Markdown, write dollar prices as `\$150`; reserve `$...$` for LaTeX.

This is not copied into every prompt literal. It is appended once by
`BaseAgent` and explicitly on Agentic RAG's separate prompt-construction path.
It covers built-in and custom agents without changing user-authored custom
prompts. The UI normalizer remains the compatibility and defence-in-depth
layer for pre-existing messages and imperfect model output.

## Rendering integration

Expose the new pure normalizer from `app.ui.stream_markdown` and use it for:

1. persisted non-rich assistant and user chat content;
2. persisted Markdown segments in rich-response views;
3. incremental Markdown slots in `_StreamingRichResponseRenderer`.

The streaming path normalizes the accumulated segment on every refresh, so a
price range split across token chunks is corrected once its full pattern is
available. No AI SDK endpoint or message schema changes are required.

## Tests and verification

Add focused tests for:

- en-dash and hyphen currency ranges, plus comma/conjunction price lists;
- a range delivered incrementally through the existing streaming normalizer;
- already escaped prices and fenced/inline code remaining byte-for-byte
  unchanged;
- inline and display LaTeX remaining unchanged;
- prompt construction containing exactly the one shared currency convention
  for the base and Agentic RAG paths.

Run the focused test modules, then the related Streamlit rendering and prompt
tests. The fix is accepted when each render path receives escaped prices while
the raw persisted/API message remains unchanged.
