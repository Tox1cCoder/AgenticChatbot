# Answer visuals and source figures: frontend contract

Audience: the developer of the AI SDK v6 (`useChat`) frontend.
Backend branch: `feat/rich-answer-visuals`. Both features are switched off on the
server until the frontend below is built and checked.

This document adds two item families to the existing rich-item contract
(`plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md`). Everything in that contract still
applies: `<!--rich:<id>-->` markers in the assistant text, the `rich_items`
registry in message metadata, protected image loading, and terminal
reconciliation.

| Family | What the reader sees | Item shape |
|---|---|---|
| Answer visual | A compact static block: numbered steps, proportional bars, or a table with status badges | New type `answer_visual` |
| Source figure | One web image or a 2-4 image comparison, each image labelled, with one caption | Existing `image` / `image_group` with extra fields |

## 1. Opting in

Send one flag per family, next to `inlineRichResponseV1`. A flag has no effect
without `inlineRichResponseV1: true`. The two flags are independent: support for
answer visuals says nothing about figures.

| Request | Where the flags go |
|---|---|
| `POST /api/chat/{conversationId}` | JSON body: `"answerVisualsV1": true`, `"sourceFiguresV1": true` |
| `POST /ai/resume-interrupt` | JSON body, same names |
| `POST /ai/continue` | JSON body, same names |
| `GET /ai/conversations/{conversationId}/messages` | Query: `?inlineRichResponseV1=true&answerVisualsV1=true&sourceFiguresV1=true` |

Snake-case aliases are accepted (`answer_visuals_v1`, `source_figures_v1`).

Send the same flags on every request type. History is projected per request, so
a reload without the flags shows the fallback text, not the component.

Only advertise a flag once the matching components pass the fixtures in
section 8. The server only lets the model create a visual or figure for a turn
whose client advertised the flag, and only while the server's rollout switch is
on.

## 2. What each client receives

Stored messages never change. Each response is projected for the flags of the
request that fetched it.

| Client sends | Answer visual | Source figure |
|---|---|---|
| `inlineRichResponseV1` + `answerVisualsV1` + `sourceFiguresV1` | Marker in text, item in `rich_items` | Marker in text, item with labels and caption |
| `inlineRichResponseV1` only | Its `fallback_markdown` replaces the marker; the item is removed from `rich_items` | Its `fallback_markdown` is inserted before the marker; the item keeps its id with a plain legacy image payload |
| no `inlineRichResponseV1` | `fallback_markdown` at the marker position | `fallback_markdown` at the marker position; the images arrive as `file` parts |

The fallback is a separate Markdown paragraph. When the model put a marker in the
middle of a sentence, the text before and after it becomes separate paragraphs.

## 3. When the data arrives during a stream

| Moment | What you get | What to do |
|---|---|---|
| Model writes the marker | `<!--rich:visual:...-->` or `<!--rich:figure:...-->` inside `text-delta` | Reserve a neutral pending block at the marker |
| During the stream | Nothing else for these items. They are never sent as `data-rich-items`, because they become public only if the final answer places them. | Keep the pending block |
| Completion | `data-assistant-message` with `data.message.metadata.rich_items` | Replace your registry with this list and render every marker from it. Drop anything not in it. |
| Completion, client without `answerVisualsV1` | The stream never shows a visual marker in `text-delta`. If one was held back, a `data-answer-reconcile` frame arrives with the final text, fallback included. | Show `data.content` in place of the draft (the existing reconcile rule) |

Use the same marker parser, registry and components for live streams and
history. A reload must look the same as the finished stream.

## 4. `answer_visual`

Ids start with `visual:`. `display_policy` is always `inline_only`: render a
visual only where its marker is, and never append an unreferenced one.

### Shared shape

```ts
type Citations = number[]; // 1-based indexes into payload.evidence

type Evidence = {
  title?: string;
  url: string;   // always http(s)
  label: string; // short display text, e.g. "bench.example"
};

type AnswerVisualCommon = {
  title: string;              // up to 160 characters
  caption?: string;           // up to 500 characters
  caption_citations: Citations;
  evidence: Evidence[];
  fallback_markdown: string;  // same content and citations as Markdown
};

type AnswerVisualItem = {
  id: string;                 // "visual:<hash>"
  type: "answer_visual";
  display_policy: "inline_only";
  source?: string;            // "answer_visual_tool"
  title?: string;
  provenance?: Record<string, unknown>;
  payload: StepsPayload | BarsPayload | ComparisonTablePayload;
};
```

Citation rule for every variant: render each number `n` in a `citations` array
as a link to `evidence[n - 1].url`, with `evidence[n - 1].label` as the visible
text.

An answer places at most six visuals.

### Variant `steps`

```ts
type StepsPayload = AnswerVisualCommon & {
  variant: "steps";
  steps: Array<{
    title: string;     // up to 160 characters
    note?: string;     // up to 500 characters
    citations: Citations;
  }>;                  // 1 to 12
};
```

- Number the steps in order.
- Never draw a check mark, a success colour or a "done" state. A step is an
  instruction to the reader, not something the app did.

Example (from `edge-google-steps.json`):

```json
{
  "id": "visual:e23b6d9e1e12af505eaf1b48",
  "type": "answer_visual",
  "display_policy": "inline_only",
  "source": "answer_visual_tool",
  "title": "Make Google the default search engine in Edge",
  "payload": {
    "variant": "steps",
    "title": "Make Google the default search engine in Edge",
    "steps": [
      { "title": "Open Settings from the ... menu", "citations": [] },
      { "title": "Choose Privacy, search, and services", "citations": [] },
      { "title": "Open Address bar and search", "citations": [] },
      {
        "title": "Pick Google under Search engine used in the address bar",
        "note": "Google appears only after you have searched with it once.",
        "citations": [1]
      }
    ],
    "caption_citations": [],
    "evidence": [
      {
        "title": "Microsoft Edge help",
        "url": "https://support.example/edge-search",
        "label": "support.example"
      }
    ],
    "fallback_markdown": "**Make Google the default search engine in Edge**\n\n1. Open Settings from the ... menu\n2. ..."
  }
}
```

### Variant `bars`

```ts
type BarsPayload = AnswerVisualCommon & {
  variant: "bars";
  unit: string;                           // one unit per visual, e.g. "FPS"
  basis: "measured" | "calculated" | "illustrative";
  assumptions?: string;                   // present for calculated and illustrative
  scale: { min: 0; max: number };         // shared linear scale, max > 0
  bars: Array<{
    label: string;
    value: number;                        // finite, 0 <= value <= scale.max
    citations: Citations;
  }>;                                     // 1 to 12
};
```

- Bar width is `value / scale.max` of the full track. The axis always starts at
  zero, so 100 and 101 must look almost the same length. Never rescale to
  exaggerate a difference.
- Print the number and unit on every bar.
- Show `basis` visibly on every bars visual.
- For `calculated` and `illustrative`, show `assumptions` next to the chart, so a
  teaching example cannot be read as a benchmark.
- Different units (clock, voltage, FPS) arrive as separate visuals. Do not merge
  them onto one axis.

Example (the FPS visual from `overclock-bars.json`):

```json
{
  "variant": "bars",
  "title": "Average FPS (teaching example)",
  "unit": "FPS",
  "basis": "calculated",
  "assumptions": "Assumes FPS scales exactly with core clock (about +1%); real games rarely scale this way.",
  "scale": { "min": 0, "max": 101.0 },
  "bars": [
    { "label": "Stock", "value": 100.0, "citations": [] },
    { "label": "Overclocked", "value": 101.0, "citations": [] }
  ],
  "caption_citations": [],
  "evidence": [],
  "fallback_markdown": "**Average FPS (teaching example)**\n\n| Item | Value (FPS) |\n| --- | --- |\n| Stock | 100 |\n| Overclocked | 101 |\n\nBasis: calculated — Assumes FPS scales exactly with core clock (about +1%); real games rarely scale this way. Scale: 0 to 101 FPS."
}
```

Values may arrive as floats (`100.0`). Display whole numbers without a decimal
point.

### Variant `comparison_table`

```ts
type ComparisonTablePayload = AnswerVisualCommon & {
  variant: "comparison_table";
  columns: string[];         // 1 to 6 data columns; the row label is an extra first column
  rows: Array<{
    label: string;
    cells: Array<{ text: string; citations: Citations }>; // one per column
    assessment?: {
      status: "positive" | "neutral" | "caution" | "negative";
      label: string;         // the badge text, e.g. "Skip"
      reason: string;        // up to 500 characters
      citations: Citations;
    };
  }>;                        // 1 to 12
};
```

- Columns: row label, then `columns`, then an "Assessment" column if any row has
  an assessment. Leave that cell empty for rows without one.
- Badge colour comes from your theme tokens for `status`. The badge always shows
  `label` as text, followed by `reason`, so the meaning never depends on colour.
- Wide tables scroll horizontally inside their own container.

Example row (from `tweak-assessment-table.json`):

```json
{
  "label": "Disable HPET",
  "cells": [{ "text": "Lower input latency", "citations": [] }],
  "assessment": {
    "status": "caution",
    "label": "Skip",
    "reason": "No measurable change in testing; can break timers.",
    "citations": [1]
  }
}
```

## 5. Source figures

A figure is an ordinary `image` (one picture) or `image_group` (two to four)
whose `source` is `"source_figure"`. Ids start with `figure:`. The model built it
from web images it inspected; the server attached each image's protected URL
and source page.

Fields added for clients that send `sourceFiguresV1`:

```ts
type FigureEvidence = { title?: string; url: string; label: string };

type FigureGroupItem = {
  id: string;                      // "figure:<hash>"
  type: "image_group";
  source: "source_figure";
  display_policy: "inline_only";
  alt_text: string;
  title?: string;
  payload: {
    items: Array<{
      url: string;                 // protected "/web-images/<id>"
      mime_type: string;
      source_url?: string;         // the page this image came from
      width?: number;
      height?: number;
      label?: string;              // up to 160 characters, shown with the image
      alt_text?: string;           // up to 500 characters, the img alt attribute
    }>;                            // 2 to 4; 1 only when comparison_unavailable
    caption?: string;              // up to 500 characters, one for the whole figure
    caption_evidence?: FigureEvidence[];
    fallback_markdown?: string;
    comparison_unavailable?: boolean;
  };
};

type FigureImageItem = {
  id: string;
  type: "image";
  source: "source_figure";
  display_policy: "inline_only";
  alt_text: string;
  payload: {
    url: string;
    mime_type: string;
    source_url?: string;
    width?: number;
    height?: number;
    caption?: string;
    label?: string;
    caption_evidence?: FigureEvidence[];
    fallback_markdown?: string;
    comparison_unavailable?: boolean;
  };
};
```

Rendering rules:

- Load each `url` through your existing authenticated image loader, the same
  way as other `/web-images/` items.
- Show each cell's `label` with its image, and a link to its own `source_url`.
- Use the cell's `alt_text` as the `alt` attribute. Do not show it as visible
  caption text.
- Show one `caption` for the whole figure, followed by `caption_evidence` links.
- Keep each image's aspect ratio. Put cells side by side on wide screens and
  stack them on narrow ones. Support the zoom you already have for images.
- When one image fails to load, show an in-place "image unavailable" block for
  that cell only and keep the others.

### Comparison unavailable

If a comparison loses an image after it was composed, the stored figure has
`comparison_unavailable: true` and no `caption`. The comparison's conclusion no
longer applies, so the server removed it. Render the surviving cells with their
labels and sources, plus a visible note such as "Comparison unavailable".

An `image_group` with a single cell is valid only in this state. A client without
`sourceFiguresV1` never receives one; it gets an ordinary `image` instead.

Example (from `partial-figure-failure.json`):

```json
{
  "id": "figure:242d1a004a09969dee7f9860",
  "type": "image_group",
  "source": "source_figure",
  "display_policy": "inline_only",
  "alt_text": "Comparison unavailable: Shadows: High",
  "payload": {
    "comparison_unavailable": true,
    "items": [
      {
        "url": "/web-images/00000000-0000-4000-8000-000000000001",
        "mime_type": "image/jpeg",
        "source_url": "https://screens1.example/graphics",
        "width": 1280,
        "height": 720,
        "label": "Shadows: High",
        "alt_text": "Car with a soft contact shadow under the chassis"
      }
    ],
    "fallback_markdown": "**Figure**\n\n- Shadows: High — Car with a soft contact shadow under the chassis ([screens1.example](https://screens1.example/graphics))\n\n_The comparison is unavailable because one of its images could not be kept._"
  }
}
```

## 6. Copy and export

When the user copies or exports a message, replace each placed visual or figure
marker with that item's `payload.fallback_markdown`, even on a fully capable
client. The copied text then keeps every value and citation. There is no backend
copy endpoint.

## 7. Failure handling

| Situation | Expected behaviour |
|---|---|
| Marker id not in the final `rich_items` | Render nothing for it, or a neutral "unavailable" block, as for other rich items. The prose still reads on its own. |
| Unknown `payload.variant` | Render `fallback_markdown` as Markdown. |
| Unknown future item type | Neutral unsupported block, as the base contract says. Never render a payload as HTML. |
| Image fetch fails inside a figure | That cell shows an unavailable block; the other cells stay. |
| Stream ends with an error or Stop | Discard pending blocks and refetch history, as today. |

Render `fallback_markdown` through your normal Markdown renderer with raw HTML
disabled. Model-written `<!--` sequences are already escaped as `&lt;!--`.

## 8. Fixtures and acceptance

`tests/fixtures/rich_answer_visuals/` on the backend branch has one JSON file per
case. The backend builds them with its real code, and a test fails if the
backend ever disagrees with them.

| File | Covers |
|---|---|
| `edge-google-steps.json` | `steps` |
| `game-edition-markdown-table.json` | A plain Markdown table; no rich item |
| `overclock-bars.json` | Three `bars` visuals with different units; measured and calculated |
| `tweak-assessment-table.json` | `comparison_table` with badges |
| `graphics-screenshot-figure.json` | Two-image labelled figure |
| `mixed-static-and-images.json` | Selected web image + visual + figure in one answer |
| `stale-transient-after-draft-replacement.json` | An item streamed for a draft that the final answer dropped |
| `partial-figure-failure.json` | Figure that lost a cell |

Each answer fixture has:

- `content` and `rich_items`: what is stored.
- `projections.full`, `visuals_only`, `figures_only`, `base_rich`, `non_rich`:
  the `content` and `rich_items` the client receives for each flag combination.

Before advertising a flag, check that:

- [ ] Every fixture renders correctly for the `full` projection, and for the
      projection matching each flag combination you might send.
- [ ] Stream completion and a history reload show the same items, in the same
      order, with the same labels and captions.
- [ ] Bars use one shared zero-based scale; 100 vs 101 looks nearly equal.
- [ ] `basis` and `assumptions` are visible on bars.
- [ ] Badges show their text label; nothing depends on colour alone.
- [ ] Copy/export produces the fallback Markdown with its citation links.
- [ ] Desktop and phone widths, light and dark themes, keyboard focus and
      contrast all work; wide tables scroll inside their container.
- [ ] A figure image that fails to load leaves the other cells and the prose intact.
- [ ] An interrupted stream (error, Stop) leaves no pending block behind.

Please send back:

1. Your frontend commit or revision, and the exact `ai` / `@ai-sdk/react` versions.
2. Screenshots of each fixture at desktop and phone width, in light and dark themes.
3. Any frame or payload your client rejected, with the raw `data:` line.

## 9. Known limitations

- A figure created before a turn paused for tool approval or for its budget does
  not carry its images over to the resumed turn. Until this is fixed, such a
  figure can show broken images after Continue. The server's figure switch stays
  off until this is resolved.
- While a web-answer draft is shown as a preview overlay, a marker split across
  two chunks can briefly appear as raw text. The reconciled final text is
  correct.
- The `data-interrupt` message payload is not projected for these flags yet.
  Ignore any `visual:` or `figure:` markers you see there.
