# Rich image rendering contract

This is the production frontend contract for inline image items in rich-response v1. A visual is optional enhancement: loading or displaying it must never block, replace, or fail the assistant's text answer.

## Authoritative data and placement

When `message.metadata.rich_items_version === 1`, resolve standalone `<!--rich:<id>-->` markers against `message.metadata.rich_items` and render items in marker order. The final persisted `rich_items` list is authoritative. Replace transient registry entries with it when the terminal assistant message arrives.

Image items use `display_policy: "inline_only"`. Never append an unreferenced v1 image as a gallery. AI SDK terminal messages can also contain a `file` part for the same selected image; deduplicate it when its `url` matches an image item's `payload.url`. Do not render both.

```typescript
type RichImageItem = {
  id: string;
  type: "image";
  display_policy: "inline_only";
  alt_text?: string;
  payload: {
    url: string;
    mime_type: string;
    source_url?: string;
    caption?: string;
    width?: number;
    height?: number;
  };
};

function selectedRichImageUrls(items: RichImageItem[]): Set<string> {
  return new Set(items.map((item) => item.payload.url));
}

function withoutDuplicateFiles(
  parts: Array<Record<string, unknown>>,
  selectedUrls: Set<string>,
): Array<Record<string, unknown>> {
  return parts.filter(
    (part) => part.type !== "file" || !selectedUrls.has(String(part.url ?? "")),
  );
}
```

## Loading protected images

Newly selected web visuals persist as user-owned `/web-images/{id}` references. Chat/document images use `/chat-images/{id}`. `/api/web-images/` and `/api/chat-images/` are compatibility forms. Never place any of these protected URLs directly in `<img src>`: the browser cannot attach the app's Bearer token to a normal image request.

Fetch the bytes with authentication, validate that the response is an image, then use a temporary object URL:

```typescript
const protectedImagePrefixes = [
  "/web-images/",
  "/api/web-images/",
  "/chat-images/",
  "/api/chat-images/",
];

function isProtectedImageUrl(url: string): boolean {
  return protectedImagePrefixes.some((prefix) => url.startsWith(prefix));
}

async function loadProtectedImage(
  url: string,
  token: string,
  signal?: AbortSignal,
): Promise<string> {
  if (!isProtectedImageUrl(url)) throw new Error("visual_unavailable");
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
    credentials: "same-origin",
    signal,
  });
  if (!response.ok) throw new Error("visual_unavailable");
  const blob = await response.blob();
  if (!blob.type.startsWith("image/")) throw new Error("visual_unavailable");
  return URL.createObjectURL(blob);
}
```

Keep the object URL in component state. On URL/token change or component unmount, abort an active request and call `URL.revokeObjectURL(objectUrl)`. Do not put tokens in query strings, logs, persistent browser storage, or shared cache keys.

## Figure states

Render the whole figure as one stateful component:

- `loading`: reserve `aspect-ratio: width / height` when both values are positive; otherwise use a modest skeleton placeholder. Keep the text answer interactive.
- `loaded`: show the image at natural size, capped by the answer column; do not upscale small images.
- A failed fetch or decode attempt does not establish that the visual is permanently unavailable. Remove the current figure without rendering an unavailable label or fallback source action. A later message render may make another attempt according to the client's request-cache policy.

The image request is independent of message completion. Network, authentication, publisher, MIME, size, decoding, and DNS failures are transport outcomes for one request. They must not replace or fail the assistant text and must not be presented as a permanent property of the image.

## Accessibility and footer ownership

`alt_text` belongs only in the image's `alt` attribute; do not automatically display it. Do not promote the rich-item `title`, provider description, or model prose into a caption.

The renderer owns exactly one visible `<figcaption>`. It may contain:

1. `payload.caption`, only when supplied as trusted structured caption data; and
2. one source link derived from `payload.source_url` (display the hostname, open in a new tab with `rel="noopener noreferrer"`).

If no structured caption exists, show source attribution alone. The assistant body must not add a second Markdown caption after the marker.

## Backend/frontend responsibility boundary

The backend selects candidates, persists only selected images, converts remote URLs to owned opaque references, and performs bounded SSRF-safe retrieval when the media route is called. It does not fetch remote bytes while generating or persisting the answer. The frontend owns authenticated loading, object-URL lifetime, per-attempt display state, and deduplication.

Before release, verify: one rendered visual per selected marker; no candidate dump; no duplicate AI SDK file part; one footer; alt text is not visible; a failed attempt produces no unavailable label or fallback source action; failed media leaves the answer intact; and all object URLs are revoked.

## Image placement

When an image item lacks a model-authored marker, the backend applies automatic placement rules anchored on query data. Images are collected into `image_group` items (multiple cells per marker) and placed deterministically, with labeled failure outcomes for rollout monitoring.

### image_group shape and cell states

An `image_group` collapsible rich item contains multiple image cells under one marker:

```typescript
type ImageGroupCell = {
  url: string;
  mime_type: string;
  source_url?: string;
  caption?: string;
  width?: number;
  height?: number;
};

type RichImageGroupItem = {
  id: string;
  type: "image_group";
  display_policy: "inline_only";
  payload: {
    items: ImageGroupCell[];  // 1–3 cells per group
  };
};
```

Each cell may fail independently during render (network timeout, invalid MIME, decode error). Per-cell failures do not cascade: a successful cell is displayed even if others fail. Treat failed cells like individual image load failures — remove without placeholder or unavailable state.

### Anchor origins and fallback rules

The backend attempts to anchor an unreferenced image group on the query that produced it. Four origins have different fallback behavior:

1. **Image search** (deliberate model-run search)
   - Anchors after the paragraph matching the most keywords from the search query.
   - Falls back to after the first substantial prose block if no paragraph reaches the `rich_image_anchor_min_score` threshold.
   - Signal: the model explicitly ran an image search or selected an image from a search result.

2. **Tool-produced image** (chart, rendered diagram, other non-search tool output)
   - Anchors only when the candidate carries a genuine signal: an image-search query (usually absent), or real descriptive text (a title, a non-generic `alt_text`, or `payload.description`). A candidate with neither — for example a description-less crawler/SEO thumbnail carrying only the generic alt-text placeholder — is never anchored: nothing to relevance-match or caption means nothing to place.
   - When it does carry a signal, it falls back to after the first substantial prose block, the same as image search, since it usually has no query to score a keyword match against: the tool call itself implies display.
   - Signal: `source: "tool_image"` — an image returned directly by an MCP tool rather than harvested from a search result. This also covers a single eligible Brave image-search candidate, which is not grouped into an `image_group` (grouping needs two or more) and so carries this source instead of `image_search`.

3. **Web-search source-bound** (Tavily result image)
   - Anchors only after a paragraph matching the image query; does NOT fall back.
   - If no paragraph qualifies, the image is unplaced and does not append to the body.
   - Signal: an image harvested from a web-search result and bound to that source.

4. **Web-search query-level** (user query image)
   - Never auto-anchored; remains unplaced unless the model writes its marker explicitly.
   - Signal: an image matching the user's original message query (not the model's derived search).

### Placement watch list

Query anchoring is the only image-placement path; there is no rollback flag. Monitor these signals:

- **`rich_image_anchor_outcomes_total{outcome="unplaced"}` rising** — the primary signal that placement is miscalibrated. Unplaced images indicate that query scoring is too strict or the content lacks qualifying paragraphs. A sustained rise suggests miscalibration of `rich_image_anchor_min_score` or that source-bound images lack relevance anchors. A tool-produced image with no query and no descriptive text is intentionally counted here too — it is never anchored by design, not a scoring failure, so this baseline rate should be checked before treating a rise as a regression signal.
- **`rich_image_anchor_outcomes_total{outcome="query_anchored"}`** — images successfully placed on query match; should trend higher than `fallback_anchored` under normal conditions.
- **`rich_image_anchor_outcomes_total{outcome="fallback_anchored"}`** — images placed on fallback (first prose block); a spike suggests low query-paragraph overlap.
- **`rich_image_anchor_outcomes_total{outcome="marker"}`** — images placed on model-authored marker; should remain steady as model writing behavior is stable.

Eligibility and delivery are counted by the stage-specific
`rich_image_candidates_total` (eligibility outcome by reason),
`rich_image_presented_total` (items offered to the model), and
`rich_image_final_selection_total` (images persisted with the message).
