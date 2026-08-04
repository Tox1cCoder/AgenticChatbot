# Remove Public Original Image URL Design

**Date:** 2026-08-04

**Status:** Approved

## Goal

Remove `provenance.original_image_url` from every public rich-item payload sent
to Streamlit and AI SDK clients. The frontend must receive only the protected
media reference needed for rendering and the page URL needed for attribution.

## Public contract

An image rich item may expose:

- `payload.url`: the protected `/web-images/{id}` media reference;
- `payload.source_url`: the publisher page used for attribution; and
- non-sensitive provenance such as provider, query, rank, tool, and source
  metadata already allowed by the rich-item contract.

It must not expose `provenance.original_image_url`. This applies equally to
stream terminal messages, message history, and Streamlit metadata because all
three consume the same finalized rich registry.

## Internal data flow

The original upstream asset URL remains in the private
`WebImageReference.upstream_url` record. `/web-images/{id}` requires this value
to retrieve and validate the selected image. Removing the public provenance
field must not change protected-route behavior, ownership checks, SSRF
protection, media validation, or caching.

## Implementation boundary

Stop adding `original_image_url` to the image candidate's public provenance at
the source. Do not add a late response scrubber when the field can be prevented
from entering the public rich-item registry. Existing persisted messages that
already contain the field must also be sanitized by the public rich-item
serialization/finalization boundary so history cannot continue exposing it.

## Contract and compatibility

Update the normative frontend contract to state that the field is not present
on the wire. `payload.url` remains the only renderable media source, and
`payload.source_url` remains attribution-only. Compatibility AI SDK file parts
continue to use the selected protected URL.

This is a subtractive public-contract change. Frontends following the current
contract do not depend on `original_image_url`, so no fallback or migration is
required.

## Verification

Regression tests must prove:

1. newly normalized tool image candidates omit `original_image_url`;
2. final/public rich-item metadata strips the field from older persisted input;
3. protected `/web-images/{id}` rendering still uses the private upstream
   record; and
4. the frontend contract no longer advertises the field.

