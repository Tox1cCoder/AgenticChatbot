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

## Deprecated metrics

`rich_image_selections_total` is superseded by the stage-specific
`rich_image_candidates_total` (eligibility outcome by reason) and
`rich_image_final_selection_total` (images persisted with the message). The old
counter still emits during the compatibility window but is removed in Task 17.
Dashboards and alerts must migrate to the new counters before then.
