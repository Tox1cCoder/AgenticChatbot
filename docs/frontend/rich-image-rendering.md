# Superseded rich-item image contract

This document is retained only as a compatibility pointer. The sole normative
contract for rich items, assistant images, protected media loading, Streamlit,
and AI SDK rendering is
[AI_SDK_FE_RICH_ITEM_CONTRACT.md](../../plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md).

Do not implement behavior from an older revision of this file.
# Web-research images

Web-research images arrive through the same `rich_items` renderer as other
typed images, but are selected differently: the answer model must emit a valid
private `I#` selection after inspecting validated pixels. The server converts
that selection to the public `<!--rich:image:web:...-->` marker and protected
`/web-images/...` URL. Clients must never infer or auto-place an image from a
web tool result.

# Composed figures and answer visuals

Labelled source figures (`sourceFiguresV1`) and static answer visuals
(`answerVisualsV1`) are specified in section 2 and section 4 of the normative
contract above. `tests/fixtures/rich_answer_visuals/` holds one JSON file per
supplied example, each with the stored answer, its rich items, and the answer as
every capability combination receives it. Render those files before advertising
either flag. Streamlit advertises neither and renders the shared fallback.
