# Automatic Visual Enrichment Design

**Date:** 2026-08-07

**Status:** Approved

## Goal

Make web-researched answers proactively include useful, verified images without
depending on the answering model to opt into the image path. Keep text-only
fallback reliable, preserve the protected-image contract, and remove obsolete
deadline machinery and explanatory code bloat from the image subsystem.

## Product behavior

`web_research` considers images by default when the request supports rich
responses. The answering model may provide a short `image_query` to improve the
visual search, but omitting it uses the factual `query`; omission no longer
disables images. A `skip_images` boolean, defaulting to `false`, is the explicit
opt-out for research where a visual cannot help.

An approved image is made available to the answering inventory and remains
eligible for deterministic final placement. Search, download, verification, or
placement failure yields a normal text-only answer without a retry.

## Tool contract

The model-facing input is:

```python
class WebResearchInput:
    query: str
    image_query: str | None = None
    image_intent: Literal["figure", "gallery"] | None = None
    skip_images: bool = False
    max_results: int | None = None
    search_depth: str | None = None
```

`image_query` refines the visual subject. When it is blank, image discovery uses
`query`. `skip_images=true` prevents Brave discovery and visual verification.
The server still enforces the inline-rich capability, feature flag, and
turn-local one-image-search budget.

The media prompt describes this default behavior without enumerating topic
categories. It tells the model to refine the visual query when useful, opt out
only when a visual cannot support the answer, and use `gallery` only for an
explicit comparison or request to see several instances.

## Orchestration

Tavily factual research and Brave image discovery start concurrently. The image
path performs these stages once:

1. Search Brave with `image_query` or `query`.
2. Build at most three private candidates.
3. Fetch and validate thumbnails concurrently through `WebImageService`.
4. Submit the surviving batch to one low-thinking Gemini verifier call.
5. Admit only candidates that depict the subject, materially support the
   answer, meet the confidence threshold, and pass structural selection.
6. Offer approved candidates through the existing verified-image sink for
   inventory injection, byte caching, protected registration, and placement.

The image subsystem no longer owns an end-to-end four- or nine-second deadline.
Brave keeps its 2.5-second timeout, the concurrent thumbnail batch keeps a
2-second timeout, and a new `image_verification_timeout_seconds` setting gives
the Gemini call 10 seconds. The existing 30-second default internal-tool policy
remains the outer safety bound for the whole `web_research` call. This removes
nested cancellation clocks and remaining-time arithmetic while retaining
bounded external calls.

## Failure model and observability

Every image-path decision ends in exactly one internal outcome:

- `approved`
- `skipped`
- `unavailable`
- `search_failure`
- `fetch_failure`
- `verifier_timeout`
- `verifier_failure`
- `malformed`
- `no_match`

Expected `skipped` and `no_match` outcomes do not produce warnings. Operational
failures produce one warning at the image-orchestration boundary and one
aggregate metric. Lower layers return or raise enough information to classify
the outcome; they do not duplicate terminal logging. No outcome details or
rejected candidates enter model-visible, persisted, or public metadata.

## Simplification boundary

Remove from the web-image/research subsystem:

- `image_verification_deadline_seconds` and its environment/config contract;
- remaining-deadline calculations and nested overall timeouts;
- the startup warning that compares stage settings with the removed deadline;
- tests whose only purpose is deadline arithmetic;
- duplicate terminal loggers and broad exception swallowing that erases the
  failure category;
- stale comments and long docstrings that describe incident history, review
  history, or superseded timing measurements instead of current behavior; and
- stale four-second claims in the prior vision-injection specification.

Keep comments that explain a non-obvious invariant, security boundary, public
contract, or concurrency requirement. Do not perform an unrelated repository-wide
cleanup under this change.

## Compatibility and security

The following behavior is unchanged:

- Tavily remains text-only.
- Brave remains the only remote web-image source for answer enrichment.
- User uploads, RAG images, generated images, and explicit image-producing tools
  bypass remote visual relevance verification.
- HTTPS, DNS, redirect, SSRF, byte-size, MIME, decode, dimension, aspect-ratio,
  and deduplication checks remain enforced.
- Public payloads expose the protected `/web-images/{id}` URL and publisher
  `source_url`, never the upstream original asset URL.
- Verified bytes remain eligible for the existing bounded hand-off cache.
- A failed optional image path never converts successful factual research into
  a tool error.

## Verification

Tests must prove:

1. `web_research` without `image_query` starts image discovery using `query`.
2. The Vietnamese prompt flow represented by `cho t thông tin về T1` can reach
   Brave without an explicit image request or image query.
3. `skip_images=true` prevents Brave and verifier calls.
4. Missing capability, a disabled rollout flag, or a spent image budget prevents
   image work.
5. Tavily and Brave still begin concurrently.
6. Each operational failure maps to the correct single outcome and text-only
   fallback; verifier timeout is not reported as malformed.
7. An approved image reaches the answering inventory, protected registration,
   and final rich metadata.
8. Rejected image details and upstream URLs remain private.
9. Configuration, prompts, tests, and documentation contain no obsolete
   image-path deadline contract.
