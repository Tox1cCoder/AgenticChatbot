# Web Image Publication Contract Repair

**Date:** 2026-09-16

## Problem

The canonical web-research pipeline prepares a selected image as an unchecked
dictionary. That dictionary does not satisfy the existing `ImageRichItem`
contract, which requires accessible alternative text. The mismatch survives
grounding and is detected only during message metadata finalization. The
finalizer correctly rejects the invalid item, and persistence removes its now
unresolvable marker. The source citation remains, so the user sees a link but
no image.

This is producer/consumer contract drift, not a rendering fallback problem.

## Design

Web research will construct each prepared public image through the canonical
`ImageRichItem` model at the point where validated bytes and the admitted
source record are available. The producer will provide:

- the protected `/web-images/{id}` delivery URL;
- MIME type and validated dimensions;
- bounded alternative text derived from the provider description, then title,
  then the existing generic image alternative-text constant;
- the admitted source URL as `payload.source_url`; and
- existing provider and source-ID provenance.

The validated model will be serialized to the dictionary shape already used by
grounding, outcome provenance, streaming, and persistence. No new transport,
selection stage, retry, repair call, or automatic provider-rank fallback will
be introduced.

## Data Flow

1. Provider candidates are downloaded, byte-validated, deduplicated, and
   registered exactly as today.
2. `WebResearchSession` constructs and validates the canonical rich image.
3. `GroundingParser` publishes only a model-selected candidate by converting
   its private `I#` token to a server-authored rich marker.
4. `build_bot_metadata()` validates the same canonical shape and retains the
   referenced image.
5. Existing Streamlit and AI SDK projections deliver the protected reference.

An invalid producer record fails at construction instead of being carried to a
late, silent publication drop.

## Explicit Image Requests

This repair does not force an image into the answer. The existing grounding
rule remains: only an image the answer model inspected and selected may be
published. Automatically choosing a candidate or adding a second model repair
call would conceal retrieval or relevance failures and weaken the approved
grounding policy.

## Error Handling

Existing image download, safety, lifecycle, and persistence failure behavior
is unchanged. Optional image failure must not replace or fail the text answer.
Late finalization remains fail-closed as defense in depth, but valid
web-research images should no longer reach that failure path.

## Testing

Add one regression at the producer boundary proving a prepared web image is a
valid public `ImageRichItem` with alternative text and source attribution. Add
one full-path regression that selects a prepared image, resolves grounding,
runs `build_bot_metadata()`, and asserts:

- the marker remains in assistant content;
- `metadata.rich_items` contains exactly the selected image;
- its URL is the protected `/web-images/{id}` reference;
- its alternative text and source URL survive; and
- no `invalid_rich_item` or `unknown_rich_item` warning is emitted.

Existing no-token/no-image, unselected-reference release, and client projection
tests remain authoritative for non-selection and delivery behavior.

## Non-goals

- No automatic image anchoring or provider-rank fallback.
- No additional model call when an image is not selected.
- No weakening of public rich-item validation.
- No frontend contract changes.
- No unrelated web-search or routing refactor.
