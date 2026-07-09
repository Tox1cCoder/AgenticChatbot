# Chat Image Attachment Paste Fix Design

## Scope

Fix the Streamlit chat image attachment experience. This covers only images attached to chat messages through paste, drag/drop, or picker-style selection near the chat input. The document upload panel for PDFs, DOCX, and other knowledge-base files is out of scope and must remain unchanged.

## Problem

The current chat UI has two separate image attachment paths:

- pasted images are captured by the custom zero-height Streamlit component in `app/ui/clipboard_image_capture`;
- selected image files are handled by a separate `st.file_uploader` shown behind the `Attach` button in `demo.py`.

Pasting works for the first turn, but after sending a message and clicking New Chat, images cannot be pasted again until the browser reloads Streamlit. The likely root cause is that the browser-side paste listener is installed once on `window.parent.document`, while the active Streamlit component instance changes as the conversation key moves between `pending_new` and a concrete conversation id. The global listener can keep stale component state across reruns and conversation changes.

## Goals

- Pasting one or more images should work on the first turn and after sending a message, clicking New Chat, and returning to a new pending chat.
- Paste, drag/drop, and click-to-select image inputs should feed the same `pending_image_attachments` queue.
- The visual image attachment affordance should match the paste flow instead of exposing a separate Streamlit file uploader block.
- The existing message send contract should remain unchanged: pending images are sent as `message_data["attachments"]`.
- Existing document upload UI and backend upload behavior must not change.

## Non-Goals

- Do not replace the document upload panel.
- Do not change backend message attachment schemas.
- Do not build a full standalone frontend application.
- Do not refactor unrelated chat rendering or streaming code.

## Approach

Update the custom clipboard component to be the single chat image intake component. It should support paste, drag/drop, and file picker selection, and return a payload shaped like the existing paste payload:

```json
{
  "eventId": "attach-...",
  "images": [
    {
      "name": "image.png",
      "mime": "image/png",
      "data": "data:image/png;base64,..."
    }
  ]
}
```

Each mounted component instance must register itself as the current active receiver. The global paste listener may remain installed once, but it must call the latest active `setComponentValue` callback and should update that callback on every component mount. This prevents stale receiver state after New Chat or conversation id changes.

In `demo.py`, remove the chat image `st.file_uploader` path. Keep the `Attach` control, but make it toggle the custom attachment intake area instead. That area should be compact, visually consistent with pasted attachments, and reuse `_handle_pasted_image_payload()` so all image intake paths deduplicate and append to `pending_image_attachments` the same way.

## Data Flow

1. User focuses the chat message textarea and pastes images, or opens the attachment intake and drops/selects image files.
2. The browser component converts selected image files to data URLs and sends `{eventId, images}` to Streamlit.
3. `demo.py` calls `_handle_pasted_image_payload(payload)`.
4. Valid images are normalized to base64-only `pending_image_attachments` entries.
5. `_render_pending_image_attachments()` shows the compact preview grid and removal controls.
6. On Send, `pending_image_attachments` are copied into `message_data["attachments"]`.
7. On successful send or explicit conversation reset, pending attachments are cleared.

## Error Handling

- Non-image clipboard or dropped files are ignored.
- Unreadable files are skipped by the component.
- Invalid payloads are ignored by `_handle_pasted_image_payload()`.
- Duplicate image data is deduplicated by the existing pending queue logic.

## Tests

Add or update tests around `tests/test_demo_image_paste.py`:

- browser component source re-registers the active receiver on each mount and does not permanently bind paste events to the first component instance;
- chat image intake is mounted near the message input and still calls `_handle_pasted_image_payload()`;
- `demo.py` no longer renders the chat image `st.file_uploader`;
- existing pending queue and consumed paste event tests continue to pass.

Run the targeted image paste tests first, then run any nearby Streamlit UI tests that cover message sending or attachment state.
