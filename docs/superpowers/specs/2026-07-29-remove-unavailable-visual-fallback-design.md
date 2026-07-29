# Remove unavailable-visual fallback design

## Goal

Remove the rich-image UI that converts a failed fetch attempt into the claim
`Visual unavailable`. A fetch failure proves only that one request failed; it
does not prove that the image is permanently unavailable.

## Scope

Remove from the Streamlit renderer:

- `build_inline_image_unavailable_html` and its private fallback helper;
- the embedded fallback `<template>` and `replaceChildren` error transition;
- the fallback-only `Open source` link;
- the branch that renders fallback HTML when protected image resolution returns
  no displayable source; and
- tests and current frontend documentation that require this fallback state.

Keep unchanged:

- image discovery, selection, persistence, and `/web-images/{id}` references;
- authenticated sidecar and canonical media routes;
- secure bounded upstream retrieval and its HTTP error responses;
- successful-image alt text, structured caption, source attribution, sizing,
  and lightbox behavior; and
- metrics for individual fetch attempts.

Historical implementation plans remain historical records and are not rewritten.

## Rendering behavior

When protected image resolution returns no source, the Streamlit renderer emits
no image component at that marker. It does not claim that the image is
unavailable and does not show a fallback source link.

For a source that reaches the browser but later fails to load or decode, the
renderer silently removes the failed figure. It does not create a permanent
failure state. A later message render may attempt the protected fetch again,
subject to the existing Streamlit request cache behavior.

Successful images retain exactly one renderer-owned footer. `alt_text` remains
accessibility-only, and `payload.source_url` remains normal source attribution.

## Testing

Tests will prove that:

- generated image HTML contains no unavailable fallback template, label, or
  fallback `Open source` action;
- a failed protected resolution produces no Streamlit markdown call;
- successful images still render alt text and one source footer; and
- the existing rich-image, sidecar proxy, and API regression suites remain
  green.
