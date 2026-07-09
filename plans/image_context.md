# Image Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make attached and pasted chat images available as model-ready visual context to every routed agent, and keep recent image turns available through conversation history.

**Architecture:** Keep the existing API shape: chat images enter as `MessageCreate.attachments` and are persisted in `message_metadata["attachments"]`. Add one shared backend image-context adapter that normalizes attachment dictionaries into LangChain multimodal `HumanMessage` parts, then use it from graph current-turn routing, history conversion, RAG's custom prompt path, and summary input. Remove the existing chat-only image parsing helpers instead of leaving duplicate code paths. Add a small Streamlit clipboard component that pushes Ctrl+V images into the existing pending attachment queue with a consumed paste event id, while removing the current four-image UI cap and avoiding any new app-level payload cap.

**Tech Stack:** Python 3.10+, FastAPI, LangGraph, LangChain `HumanMessage` multimodal blocks, Streamlit, plain JavaScript Streamlit component, pytest.

---

## Requirements Recap

- Images must work for all agents, not only `chat_agent`.
- Pasted images must use the same pending attachment queue and preview/removal UI as uploaded images.
- Remove the current four-image limit.
- Do not add an app-level image count or size cap.
- Images should be part of conversation context. Recent turns should carry actual image parts; older summarized turns should at least preserve text references to attached images.
- Pasting images must not create a continuous Streamlit rerun loop.
- The chat text box and Send/Attach controls must remain visible after a paste event is processed.
- Pending image previews should be compactly grouped near the chat box, not spread sparsely across the full page width.

## Current Code Findings

- `demo.py` already creates `pending_image_attachments`, renders thumbnails, and sends `message_data["attachments"]`.
- `demo.py` has `_MAX_IMAGE_ATTACHMENTS = 4`; `_handle_new_image_attachments()` enforces that cap.
- `demo.py` renders pending attachments with `st.columns(min(len(st.session_state.pending_image_attachments), 4))`, which spreads a small number of thumbnails across full-width columns and looks sparse.
- A paste component must not call `st.rerun()` on every rerun with the same component value. The backend handler should return `True` only when a new, unconsumed paste event adds attachments; otherwise `render_chat_view()` must continue to render the message form.
- `app/schemas/message.py::MessageCreate.attachments` already accepts `list[dict[str, str]]`.
- `app/factories/message_factory.py::create_from_schema_with_role()` persists attachments in `message_metadata["attachments"]`.
- `app/services/message_service.py::_build_user_message_workflow_request()` forwards attachments to `WorkflowExecutionRequest`.
- `app/ai/graph.py::_build_initial_state_from_request()` stores request attachments in graph state.
- `app/ai/graph.py::_chat_node()` is the only graph node that turns current-turn attachments into multimodal content.
- `app/ai/graph.py` routes `search_agent`, `image_generator_agent`, `canvas_agent`, custom agents, planning mode, and generic isolated worker agents with text-only current-turn messages.
- `app/ai/graph.py::_run_agent_in_isolated_context()` builds a RAG worker `AgentMessage` without parent attachments, so delegated RAG workers would still miss current-turn user images unless explicitly fixed.
- `app/ai/graph.py` has chat-specific helpers (`_normalize_attachment_image_url()` and `_build_chat_turn_messages_with_attachments()`) that should be replaced by the shared image-context module, not kept beside it.
- `app/ai/agents/rag_agent.py` has a separate agentic prompt builder. The graph already forwards main RAG-node current-turn attachments into `AgentMessage.attachments`, but `_process_message_agentic()` ignores them and only supports document images through `metadata["agentic_images"]`.
- `app/ai/agents/chat_agent.py::_generate_with_vision()` manually builds LangChain `image_url` content for its direct OpenAI vision path; after `app/ai/image_context.py` exists, that construction should reuse the shared helper.
- `app/ai/history.py::db_message_to_agent_message()` drops user message attachments when loading prompt history.
- `app/ai/memory.py::ConversationMemory._db_to_agent_message()` drops attachments in the legacy fallback path.
- `app/ai/agents/base_agent.py::_convert_history_to_langchain_messages()` converts every user history item to plain text, even if an `AgentMessage` has attachments.
- `app/api/ai_sdk.py::_extract_user_attachments()` already extracts attachments from AI SDK message parts, so backend fixes benefit both Streamlit and AI SDK clients.

## Design

Use a shared image-context module instead of agent-specific image parsing. The module owns four things:

- Normalize attachment payloads from base64, data URLs, remote URLs, and common MIME key variants.
- Build LangChain-compatible content parts: `{"type": "text"}` and `{"type": "image_url"}`.
- Detect whether a content list contains image parts.
- Produce text-only image memory lines for summarization.

The graph should call this module for the current turn before invoking any BaseAgent-backed node, including planning mode and isolated worker agents. RAG should call the same module inside its custom prompt construction, and graph-created RAG worker messages should pass parent attachments through `AgentMessage.attachments`. BaseAgent should call it when converting persisted user history. The UI should keep image data in the same pending attachment shape the backend already accepts, while using paste event ids to avoid replaying stale component values. The paste mount must be non-blocking: when no new paste is consumed, execution continues to the existing chat form.

## Cleanup And Duplication Rules

- Do not add a second attachment parser or image URL normalizer outside `app/ai/image_context.py`.
- Delete `MultiAgentWorkflow._normalize_attachment_image_url()` and replace `MultiAgentWorkflow._build_chat_turn_messages_with_attachments()` with the shared `_build_turn_messages_with_attachments()` helper.
- Keep only one helper for marking image-bearing responses (`_mark_response_has_images()`), and call it from graph nodes instead of duplicating metadata mutation blocks.
- Do not leave compatibility wrappers around removed graph image helpers unless an existing test or caller still imports them; if a wrapper is required, it must delegate directly to the new shared helper and have a removal comment.
- After implementation, run a source scan for `_normalize_attachment_image_url`, `_build_chat_turn_messages_with_attachments`, duplicate `{"type": "image_url"}` construction, and `_MAX_IMAGE_ATTACHMENTS`.

## Non-Goals

- No new database tables.
- No new upload endpoint for chat images.
- No image vector index.
- No visual captioning service in this pass.
- No replay of every image ever attached to a conversation forever; existing history budgets still bound prompt history.

## File Structure

- Create `app/ai/image_context.py`
  - Shared attachment normalization and multimodal content helpers.

- Modify `app/ai/history.py`
  - Preserve `message_metadata["attachments"]` on user `AgentMessage` history rows.

- Modify `app/ai/memory.py`
  - Preserve attachments in the legacy memory fallback path.

- Modify `app/ai/agents/base_agent.py`
  - Convert historical user messages with attachments into multimodal `HumanMessage` content.

- Modify `app/ai/agents/chat_agent.py`
  - Reuse shared image-context helpers in the direct `ChatAgent.process_message()` vision path and remove manual OpenAI `image_url` block construction.

- Modify `app/ai/token_instrumentation.py`
  - Add small per-attachment token overhead for history trimming estimates.

- Modify `app/ai/graph.py`
  - Apply current-turn attachments to all BaseAgent graph nodes, planning mode, generic isolated workers, and graph-created RAG worker messages.
  - Remove chat-only image normalization/conversion helpers after replacing them with the shared module.

- Modify `app/ai/agents/rag_agent.py`
  - Include user attachments in RAG's agentic multimodal prompt.

- Modify `app/ai/conversation_summarizer.py`
  - Include text references to image attachments in summary input.

- Modify `demo.py`
  - Remove `_MAX_IMAGE_ATTACHMENTS`.
  - Add pasted-image payload handling.
  - Mount the clipboard capture component near the chat input.

- Create `app/ui/clipboard_image_capture.py`
  - Python wrapper around the Streamlit component.

- Create `app/ui/clipboard_image_capture/index.html`
  - Browser-side paste listener that returns image data URLs to Streamlit.

- Add tests:
  - `tests/test_image_context.py`
  - `tests/test_message_history_image_context.py`
  - `tests/test_base_agent_image_history.py`
  - `tests/test_graph_image_context.py`
  - `tests/test_rag_agent_image_attachments.py`
  - `tests/test_demo_image_paste.py`

---

## Implementation Tasks

### Task 1: Add Shared Image Context Utilities

**Files:**
- Create: `app/ai/image_context.py`
- Create: `tests/test_image_context.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_image_context.py`:

```python
import base64

from app.ai.image_context import (
    attachment_memory_lines,
    build_multimodal_content,
    has_image_parts,
    image_url_part,
    normalize_image_attachment,
)


PNG_B64 = base64.b64encode(b"fake-png").decode("ascii")


def test_normalize_base64_attachment_to_data_url():
    normalized = normalize_image_attachment(
        {"name": "screen.png", "mime": "image/png", "data": PNG_B64}
    )

    assert normalized == {
        "name": "screen.png",
        "mime": "image/png",
        "url": f"data:image/png;base64,{PNG_B64}",
    }


def test_normalize_data_url_keeps_single_data_prefix():
    url = f"data:image/webp;base64,{PNG_B64}"

    normalized = normalize_image_attachment({"name": "clip.webp", "data": url})

    assert normalized["url"] == url
    assert normalized["mime"] == "image/webp"


def test_normalize_remote_url_and_rejects_blob_url():
    normalized = normalize_image_attachment(
        {"name": "remote.png", "mime": "image/png", "url": "https://example.test/a.png"}
    )

    assert normalized == {
        "name": "remote.png",
        "mime": "image/png",
        "url": "https://example.test/a.png",
    }
    assert normalize_image_attachment({"name": "clip.png", "url": "blob:http://app/123"}) is None


def test_normalize_skips_local_path_and_non_image_mime():
    assert normalize_image_attachment({"path": "C:\\fakepath\\x.png"}) is None
    assert normalize_image_attachment({"path": "C:/fakepath/x.png"}) is None
    assert normalize_image_attachment({"mime": "text/plain", "data": PNG_B64}) is None


def test_image_url_part_uses_single_langchain_shape():
    assert image_url_part("data:image/png;base64,abc") == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc"},
    }


def test_build_multimodal_content_adds_text_before_images():
    content = build_multimodal_content(
        "describe this",
        [{"name": "screen.png", "mime": "image/png", "data": PNG_B64}],
    )

    assert content == [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
    ]
    assert has_image_parts(content) is True


def test_attachment_memory_lines_do_not_include_raw_image_data():
    lines = attachment_memory_lines(
        [
            {"name": "screen.png", "mime": "image/png", "data": PNG_B64},
            {"name": "notes.txt", "mime": "text/plain", "data": PNG_B64},
        ]
    )

    assert lines == ["[Attached image: screen.png, image/png]"]
    assert PNG_B64 not in "\n".join(lines)
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m pytest tests/test_image_context.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `app.ai.image_context`.

- [ ] **Step 3: Implement the helper module**

Create `app/ai/image_context.py`:

```python
from __future__ import annotations

import mimetypes
import re
from typing import Any


_LOCAL_PATH_PREFIXES = ("/", "./", "../")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mime_from_data_url(data_url: str) -> str | None:
    if not data_url.startswith("data:"):
        return None
    header = data_url.split(",", 1)[0]
    media_type = header[5:].split(";", 1)[0].strip()
    return media_type or None


def _looks_like_local_path(value: str) -> bool:
    return bool(_WINDOWS_DRIVE_RE.match(value)) or value.startswith(_LOCAL_PATH_PREFIXES)


def _candidate_payloads(attachment: dict[str, Any]) -> list[Any]:
    return [
        attachment.get("data"),
        attachment.get("url"),
        attachment.get("base64"),
        attachment.get("path"),
        attachment.get("image"),
        attachment.get("source"),
    ]


def normalize_image_attachment(attachment: Any) -> dict[str, str] | None:
    if not isinstance(attachment, dict):
        return None

    name = _clean_str(attachment.get("name") or attachment.get("filename")) or "image"
    mime = _clean_str(
        attachment.get("mime")
        or attachment.get("mimeType")
        or attachment.get("mediaType")
        or attachment.get("contentType")
    )
    if not mime:
        mime = mimetypes.guess_type(name)[0] or "image/jpeg"

    raw_value: str | None = None
    for candidate in _candidate_payloads(attachment):
        if isinstance(candidate, dict):
            candidate = (
                candidate.get("url")
                or candidate.get("data")
                or candidate.get("base64")
                or candidate.get("path")
            )
        raw_value = _clean_str(candidate)
        if raw_value:
            break

    if not raw_value or _looks_like_local_path(raw_value):
        return None

    if raw_value.startswith("data:"):
        inferred = _mime_from_data_url(raw_value)
        mime = inferred or mime
        if not mime.startswith("image/"):
            return None
        return {"name": name, "mime": mime, "url": raw_value}

    if not mime.startswith("image/"):
        return None

    if raw_value.startswith("blob:"):
        # Browser blob URLs are scoped to the page process and are not fetchable
        # by the backend or model provider. The UI must send data URLs instead.
        return None

    if raw_value.startswith(("http://", "https://")):
        return {"name": name, "mime": mime, "url": raw_value}

    return {"name": name, "mime": mime, "url": f"data:{mime};base64,{raw_value}"}


def image_url_part(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def build_multimodal_content(text: Any, attachments: list[Any] | None) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    text_value = str(text or "").strip()
    if text_value:
        parts.append({"type": "text", "text": text_value})

    for attachment in attachments or []:
        normalized = normalize_image_attachment(attachment)
        if normalized is None:
            continue
        parts.append(image_url_part(normalized["url"]))

    return parts


def has_image_parts(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)


def attachment_memory_lines(attachments: list[Any] | None) -> list[str]:
    lines: list[str] = []
    for attachment in attachments or []:
        normalized = normalize_image_attachment(attachment)
        if normalized is None:
            continue
        lines.append(f"[Attached image: {normalized['name']}, {normalized['mime']}]")
    return lines
```

- [ ] **Step 4: Run the tests again**

Run:

```bash
python -m pytest tests/test_image_context.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/ai/image_context.py tests/test_image_context.py
git commit -m "feat: add shared image context utilities"
```

### Task 2: Preserve Attachments In Prompt History

**Files:**
- Modify: `app/ai/history.py`
- Modify: `app/ai/memory.py`
- Create: `tests/test_message_history_image_context.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_message_history_image_context.py`:

```python
from types import SimpleNamespace

from app.ai.history import db_message_to_agent_message
from app.ai.memory import ConversationMemory
from app.ai.schemas import MessageRole
from app.models.enums import MessageRole as DBMessageRole


ATTACHMENTS = [{"name": "screen.png", "mime": "image/png", "data": "abc123"}]


def _db_message(sender=DBMessageRole.user.value):
    return SimpleNamespace(
        id="msg-1",
        sender=sender,
        content="look at this",
        message_metadata={"attachments": ATTACHMENTS},
        created_at=None,
        deleted_at=None,
    )


def test_history_provider_preserves_user_attachments():
    msg = db_message_to_agent_message(_db_message())

    assert msg.role == MessageRole.USER
    assert msg.attachments == ATTACHMENTS


def test_history_provider_keeps_assistant_history_text_only():
    msg = db_message_to_agent_message(_db_message(sender=DBMessageRole.assistant.value))

    assert msg.role == MessageRole.ASSISTANT
    assert msg.attachments is None


def test_legacy_memory_preserves_user_attachments():
    memory = ConversationMemory.__new__(ConversationMemory)

    msg = memory._db_to_agent_message(_db_message())

    assert msg.role == MessageRole.USER
    assert msg.attachments == ATTACHMENTS
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m pytest tests/test_message_history_image_context.py -v
```

Expected: FAIL because user history attachments are dropped.

- [ ] **Step 3: Update `app/ai/history.py`**

Inside `db_message_to_agent_message()`, add attachment extraction before the sender branch:

```python
    raw_metadata = message.message_metadata if isinstance(message.message_metadata, dict) else {}
    raw_attachments = raw_metadata.get("attachments")
    attachments = raw_attachments if isinstance(raw_attachments, list) else None
```

Then pass attachments only for user messages:

```python
    if sender == DBMessageRole.user.value:
        return AgentMessage(
            role=MessageRole.USER,
            content=message.content or "",
            metadata=metadata,
            attachments=attachments,
        )
```

- [ ] **Step 4: Update `app/ai/memory.py`**

Inside `_db_to_agent_message()`, compute attachments after `role`:

```python
            metadata_json = (
                db_message.message_metadata if isinstance(db_message.message_metadata, dict) else {}
            )
            attachments = None
            if role == MessageRole.USER and isinstance(metadata_json.get("attachments"), list):
                attachments = metadata_json["attachments"]
```

Then pass `attachments=attachments` into the returned `AgentMessage`.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_message_history_image_context.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/ai/history.py app/ai/memory.py tests/test_message_history_image_context.py
git commit -m "feat: preserve chat image attachments in history"
```

### Task 3: Convert Historical Images In BaseAgent Prompts

**Files:**
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/token_instrumentation.py`
- Create: `tests/test_base_agent_image_history.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_base_agent_image_history.py`:

```python
from app.ai.agents.chat_agent import ChatAgent
from app.ai.schemas import AgentMessage, MessageRole
from app.ai.token_instrumentation import estimate_agent_message_tokens


def test_base_agent_converts_user_history_images_to_multimodal_content():
    agent = ChatAgent()
    history = [
        AgentMessage(
            role=MessageRole.USER,
            content="previous image",
            attachments=[{"name": "a.png", "mime": "image/png", "data": "abc"}],
        )
    ]

    converted = agent._convert_history_to_langchain_messages(history)

    assert converted[0].content == [
        {"type": "text", "text": "previous image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]


def test_token_estimator_counts_attachment_overhead():
    with_image = AgentMessage(
        role=MessageRole.USER,
        content="short",
        attachments=[{"name": "a.png", "mime": "image/png", "data": "abc"}],
    )
    without_image = AgentMessage(role=MessageRole.USER, content="short")

    assert estimate_agent_message_tokens(with_image) > estimate_agent_message_tokens(without_image)
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m pytest tests/test_base_agent_image_history.py -v
```

Expected: FAIL because BaseAgent history conversion is text-only.

- [ ] **Step 3: Update BaseAgent history conversion**

In `app/ai/agents/base_agent.py`, import:

```python
from ..image_context import build_multimodal_content, has_image_parts
```

In `_convert_history_to_langchain_messages()`, replace the user branch with:

```python
                if role == "user":
                    attachments = getattr(msg, "attachments", None)
                    multimodal_content = build_multimodal_content(content, attachments)
                    if has_image_parts(multimodal_content):
                        langchain_history.append(HumanMessage(content=multimodal_content))
                    else:
                        langchain_history.append(HumanMessage(content=content))
```

- [ ] **Step 4: Update token estimates**

In `app/ai/token_instrumentation.py::estimate_agent_message_tokens()`, replace the return with:

```python
    tokens = estimate_tokens(content) + 4
    attachments = getattr(message, "attachments", None)
    if isinstance(attachments, list):
        tokens += 32 * len(attachments)
    return tokens
```

This is an estimate for trimming and telemetry; it is not an app-level cap.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_base_agent_image_history.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/ai/agents/base_agent.py app/ai/token_instrumentation.py tests/test_base_agent_image_history.py
git commit -m "feat: include historical images in agent prompts"
```

### Task 4: Apply Current-Turn Images To All Graph Agents

**Files:**
- Modify: `app/ai/graph.py`
- Create: `tests/test_graph_image_context.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_graph_image_context.py`:

```python
import inspect

from langchain_core.messages import HumanMessage

from app.ai.graph import MultiAgentWorkflow


ATTACHMENTS = [{"name": "screen.png", "mime": "image/png", "data": "abc"}]


def test_workflow_applies_current_turn_images_to_last_human_message():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    result, has_images = workflow._build_turn_messages_with_attachments(
        [HumanMessage(content="inspect this")],
        ATTACHMENTS,
    )

    assert has_images is True
    assert result[0].content == [
        {"type": "text", "text": "inspect this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]


def test_workflow_leaves_turn_plain_without_valid_images():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    result, has_images = workflow._build_turn_messages_with_attachments(
        [HumanMessage(content="hello")],
        [{"mime": "text/plain", "data": "abc"}],
    )

    assert has_images is False
    assert result[0].content == "hello"


def test_workflow_marks_image_response_metadata_once():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    class Response:
        metadata = None

    response = Response()

    workflow._mark_response_has_images(response, True)

    assert response.metadata == {"has_images": True}


def test_planning_node_applies_current_turn_attachments():
    source = inspect.getsource(MultiAgentWorkflow._planning_node)

    assert "_apply_current_turn_attachments(" in source


def test_isolated_worker_path_applies_parent_attachments():
    source = inspect.getsource(MultiAgentWorkflow._run_agent_in_isolated_context)

    assert "_apply_current_turn_attachments(parent_state, [worker_message])" in source
    assert "attachments=self._get_state_attachments(parent_state)" in source
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m pytest tests/test_graph_image_context.py -v
```

Expected: FAIL because `_build_turn_messages_with_attachments` does not exist.

- [ ] **Step 3: Add shared graph helper**

In `app/ai/graph.py`, import:

```python
from .image_context import build_multimodal_content, has_image_parts
```

Delete `_normalize_attachment_image_url()` entirely. Replace `_build_chat_turn_messages_with_attachments()` with these shared graph helpers:

```python
    def _build_turn_messages_with_attachments(
        self,
        current_turn_messages: list[Any],
        attachments: list[Any],
    ) -> tuple[list[Any], bool]:
        if not current_turn_messages:
            current_turn_messages = [HumanMessage(content="")]

        messages_copy = list(current_turn_messages)
        last_human_idx = self._find_last_human_message_index(messages_copy)
        if last_human_idx is None:
            last_human_idx = len(messages_copy)
            messages_copy.append(HumanMessage(content=""))

        original_message = messages_copy[last_human_idx]
        original_content = original_message.content
        user_text = coerce_response_text(original_content)
        multimodal_content = build_multimodal_content(user_text, attachments)

        if not has_image_parts(multimodal_content):
            return messages_copy, False

        kwargs: dict[str, Any] = {"content": multimodal_content}
        message_id = getattr(original_message, "id", None)
        if message_id:
            kwargs["id"] = message_id
        messages_copy[last_human_idx] = HumanMessage(**kwargs)
        return messages_copy, True

    @staticmethod
    def _mark_response_has_images(response: AgentResponse, has_images: bool) -> None:
        if not has_images:
            return
        response.metadata = dict(response.metadata or {})
        response.metadata["has_images"] = True

    def _apply_current_turn_attachments(
        self,
        state: GraphState,
        current_turn_messages: list[Any],
    ) -> tuple[list[Any], bool]:
        attachments = self._get_state_attachments(state)
        if not attachments:
            return current_turn_messages, False
        return self._build_turn_messages_with_attachments(current_turn_messages, attachments)
```

- [ ] **Step 4: Use helper in each BaseAgent graph node**

In `_chat_node()`, `_custom_agent_node()`, `_search_node()`, `_image_generator_node()`, `_canvas_node()`, and `_planning_node()`, add this immediately after `current_turn_messages` is computed:

```python
        current_turn_messages, has_images = self._apply_current_turn_attachments(
            state,
            current_turn_messages,
        )
```

After each node receives `response`, call the shared metadata helper:

```python
        self._mark_response_has_images(response, has_images)
```

Remove the old chat-only attachment conversion call in `_chat_node()`.

In `_run_agent_in_isolated_context()`, replace the single worker message initialization:

```python
        worker_message = HumanMessage(content=task_prompt)
```

with:

```python
        worker_message = HumanMessage(content=task_prompt)
        worker_messages, worker_has_images = self._apply_current_turn_attachments(
            parent_state,
            [worker_message],
        )
```

Then remove the later duplicate initialization:

```python
        worker_messages: list[Any] = [worker_message]
```

In the RAG worker `AgentMessage`, pass parent attachments through to `RAGAgent._process_message_agentic()`:

```python
                agent_msg = AgentMessage(
                    role=MessageRole.USER,
                    content=task_prompt,
                    metadata={
                        "persona": persona,
                        "history": [],
                        "original_query": task_prompt,
                        "tool_context": list(tool_context),
                        "agentic_images": list(rag_context.get("agentic_images") or []),
                        "model_request": model_request,
                        "user_id": user_id,
                        "device_id": device_id,
                        "history_summary": worker_history_summary,
                        "run_config": run_config,
                    },
                    attachments=self._get_state_attachments(parent_state),
                )
```

Before every successful generic worker return, mark the response if images were applied:

```python
                    self._mark_response_has_images(response, worker_has_images)
                    return response
```

- [ ] **Step 5: Run focused graph tests**

Run:

```bash
python -m pytest tests/test_graph_image_context.py -v
```

Expected: PASS.

- [ ] **Step 6: Run adjacent graph tests**

Run:

```bash
python -m pytest tests/test_custom_agents_graph.py tests/test_graph_handoff_streaming.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/ai/graph.py tests/test_graph_image_context.py
git commit -m "feat: apply current image context to all agents"
```

### Task 5: Add User Images To RAG Agentic Prompting

**Files:**
- Modify: `app/ai/agents/rag_agent.py`
- Create: `tests/test_rag_agent_image_attachments.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_rag_agent_image_attachments.py`:

```python
import asyncio
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.ai.agents.rag_agent import RAGAgent
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def test_rag_agent_includes_user_attachments_in_multimodal_query():
    agent = object.__new__(RAGAgent)
    agent.tools = []
    agent._build_skills_suffix = lambda **_kwargs: ""
    agent._get_tools_for_binding = lambda **_kwargs: []

    async def init_tools():
        agent.tools = []

    agent._init_tools = init_tools

    agent._resolve_runtime_model_config = lambda *_args, **_kwargs: SimpleNamespace(
        provider="openai",
        model="gpt-4o",
        capabilities={"supports_vision": True},
        fallback_config=None,
        warnings=[],
    )

    captured = {}

    async def fake_invoke(**kwargs):
        captured["messages"] = kwargs["messages"]
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="ok"),
            metadata={},
        )

    agent._invoke_agentic_rag_model = fake_invoke

    asyncio.run(
        agent._process_message_agentic(
            AgentMessage(
                role=MessageRole.USER,
                content="compare image to docs",
                attachments=[{"name": "screen.png", "mime": "image/png", "data": "abc"}],
                metadata={"original_query": "compare image to docs"},
            ),
            conversation_id="conv-1",
        ),
    )

    human_messages = [m for m in captured["messages"] if isinstance(m, HumanMessage)]
    assert any(
        isinstance(m.content, list)
        and {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        in m.content
        for m in human_messages
    )
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
python -m pytest tests/test_rag_agent_image_attachments.py -v
```

Expected: FAIL because `_process_message_agentic()` ignores `AgentMessage.attachments`.

- [ ] **Step 3: Update RAG prompt construction**

In `app/ai/agents/rag_agent.py`, import:

```python
from ..image_context import build_multimodal_content, has_image_parts, image_url_part
```

Inside `_process_message_agentic()`, read attachments:

```python
        user_attachments = message.attachments or []
```

Replace the current `if agentic_images:` prompt branch with:

```python
        human_content = build_multimodal_content("\n".join(context_parts), user_attachments)

        for img in agentic_images:
            img_data = img.get("data")
            mime_type = img.get("mime_type", "image/jpeg")
            caption = img.get("caption", "")
            page = img.get("page_number", "?")

            if img_data:
                human_content.append(
                    image_url_part(f"data:{mime_type};base64,{img_data}")
                )
                if caption:
                    human_content.append(
                        {
                            "type": "text",
                            "text": f"[Image from page {page}: {caption}]",
                        }
                    )

        has_prompt_images = has_image_parts(human_content)
        if has_prompt_images:
            messages.append(HumanMessage(content=human_content))
        else:
            messages.append(HumanMessage(content="\n".join(context_parts)))
```

Use `has_prompt_images` in the vision fallback check and `_invoke_agentic_rag_model()` call:

```python
        if has_prompt_images and not runtime_config.capabilities.get("supports_vision", False):
```

```python
                has_images=has_prompt_images,
                agentic_images_count=len(agentic_images) if agentic_images else 0,
```

- [ ] **Step 4: Run RAG tests**

Run:

```bash
python -m pytest tests/test_rag_agent_image_attachments.py tests/test_rag_agent.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/ai/agents/rag_agent.py tests/test_rag_agent_image_attachments.py
git commit -m "feat: pass user images into rag prompts"
```

### Task 6: Preserve Image References In Long-Term Summary Input

**Files:**
- Modify: `app/ai/conversation_summarizer.py`
- Modify: `tests/test_conversation_summarizer.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_conversation_summarizer.py`:

```python
from app.ai.conversation_summarizer import _agent_message_to_langchain
from app.ai.schemas import AgentMessage, MessageRole


def test_summarizer_mentions_user_image_attachments_without_raw_base64():
    msg = AgentMessage(
        role=MessageRole.USER,
        content="remember this screenshot",
        attachments=[{"name": "screen.png", "mime": "image/png", "data": "abc123"}],
    )

    converted = _agent_message_to_langchain(msg)

    assert "remember this screenshot" in converted.content
    assert "[Attached image: screen.png, image/png]" in converted.content
    assert "abc123" not in converted.content
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python -m pytest tests/test_conversation_summarizer.py::test_summarizer_mentions_user_image_attachments_without_raw_base64 -v
```

Expected: FAIL because summary input omits image references.

- [ ] **Step 3: Update summarizer conversion**

In `app/ai/conversation_summarizer.py`, import:

```python
from app.ai.image_context import attachment_memory_lines
```

Replace `_agent_message_to_langchain()` with:

```python
def _agent_message_to_langchain(message: AgentMessage):
    content = message.content or ""
    if message.role == MessageRole.USER:
        image_lines = attachment_memory_lines(message.attachments)
        if image_lines:
            content = "\n".join([content, *image_lines]).strip()
        return HumanMessage(content=content)
    return AIMessage(content=content)
```

- [ ] **Step 4: Run summarizer test**

Run:

```bash
python -m pytest tests/test_conversation_summarizer.py::test_summarizer_mentions_user_image_attachments_without_raw_base64 -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/ai/conversation_summarizer.py tests/test_conversation_summarizer.py
git commit -m "feat: preserve image references in summaries"
```

### Task 7: Remove Streamlit Image Count Limit And Add Paste Payload Handler

**Files:**
- Modify: `demo.py`
- Create: `tests/test_demo_image_paste.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_demo_image_paste.py`:

```python
from pathlib import Path

from tests.test_demo_plan_widget import _import_demo_with_ui_stubs


class Upload:
    def __init__(self, name: str, data: bytes, mime: str = "image/png"):
        self.name = name
        self.type = mime
        self._data = data

    def read(self):
        return self._data

    def seek(self, _pos):
        return None


def test_handle_new_image_attachments_has_no_four_image_limit(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub.session_state.pending_image_attachments = []

    uploads = [Upload(f"{idx}.png", f"img-{idx}".encode("ascii")) for idx in range(6)]

    demo._handle_new_image_attachments(uploads)

    assert len(streamlit_stub.session_state.pending_image_attachments) == 6


def test_handle_pasted_image_payload_reuses_pending_attachment_queue(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub.session_state.pending_image_attachments = []

    demo._handle_pasted_image_payload(
        [
            {
                "name": "clipboard-1.png",
                "mime": "image/png",
                "data": "data:image/png;base64,YWJj",
            }
        ]
    )

    pending = streamlit_stub.session_state.pending_image_attachments
    assert len(pending) == 1
    assert pending[0]["name"] == "clipboard-1.png"
    assert pending[0]["mime"] == "image/png"
    assert pending[0]["data"] == "YWJj"


def test_handle_pasted_image_payload_does_not_replay_consumed_event(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub.session_state.pending_image_attachments = []

    payload = {
        "eventId": "paste-1",
        "images": [
            {
                "name": "clipboard-1.png",
                "mime": "image/png",
                "data": "data:image/png;base64,YWJj",
            }
        ],
    }

    assert demo._handle_pasted_image_payload(payload) is True
    streamlit_stub.session_state.pending_image_attachments = []

    assert demo._handle_pasted_image_payload(payload) is False
    assert streamlit_stub.session_state.pending_image_attachments == []


def test_pending_image_preview_uses_compact_grid_helper():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    assert "_PENDING_IMAGE_PREVIEW_COLUMNS = 8" in source
    assert "def _render_pending_image_attachments(" in source
    assert "_render_pending_image_attachments()" in source
    assert "st.columns(min(len(st.session_state.pending_image_attachments), 4))" not in source
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
python -m pytest tests/test_demo_image_paste.py -v
```

Expected: FAIL because the four-image limit remains and paste handler is missing.

- [ ] **Step 3: Remove the image count cap**

In `demo.py`:

- Delete `_MAX_IMAGE_ATTACHMENTS = 4`.
- Add compact preview constants near the remaining module-level UI constants. These are layout controls, not attachment caps:

```python
_PENDING_IMAGE_PREVIEW_COLUMNS = 8
_PENDING_IMAGE_PREVIEW_WIDTH = 72
```

- In `_handle_new_image_attachments()`, remove `remaining` logic and replace the ending with:

```python
    if not new_items:
        return

    pending.extend(new_items)
    st.session_state.pending_image_attachments = pending
```

- Update upload help text from `help=f"Up to {_MAX_IMAGE_ATTACHMENTS} images"` to:

```python
                help="Attach images",
```

- [ ] **Step 4: Add pasted payload handler**

Add near `_handle_new_image_attachments()`:

```python
def _attachment_from_pasted_payload(item: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None

    raw_data = str(item.get("data") or "").strip()
    if not raw_data:
        return None

    mime = str(item.get("mime") or item.get("type") or "image/png").strip() or "image/png"
    if raw_data.startswith("data:"):
        header, _, payload = raw_data.partition(",")
        raw_data = payload or ""
        if ";" in header:
            inferred = header[5:].split(";", 1)[0].strip()
            if inferred:
                mime = inferred

    if not raw_data:
        return None

    return {
        "token": str(uuid.uuid4()),
        "name": str(item.get("name") or "clipboard-image.png"),
        "mime": mime,
        "data": raw_data,
    }


def _paste_event_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    event_id = str(payload.get("eventId") or payload.get("event_id") or "").strip()
    return event_id or None


def _paste_payload_images(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("images"), list):
        return payload["images"]
    return []


def _handle_pasted_image_payload(payload: Any) -> bool:
    images = _paste_payload_images(payload)
    if not images:
        return False

    event_id = _paste_event_id(payload)
    consumed_events = st.session_state.setdefault("consumed_image_paste_events", set())
    if event_id and event_id in consumed_events:
        return False

    pending = st.session_state.get("pending_image_attachments", [])
    existing_data = {item["data"] for item in pending if isinstance(item, dict) and "data" in item}
    new_items: list[dict[str, str]] = []

    for raw_item in images:
        attachment = _attachment_from_pasted_payload(raw_item)
        if not attachment:
            continue
        if attachment["data"] in existing_data:
            continue
        new_items.append(attachment)
        existing_data.add(attachment["data"])

    if event_id:
        consumed_events.add(event_id)
        st.session_state.consumed_image_paste_events = consumed_events

    if new_items:
        pending.extend(new_items)
        st.session_state.pending_image_attachments = pending
        return True

    return False
```

- [ ] **Step 5: Replace sparse pending attachment preview**

Add this helper near `_handle_pasted_image_payload()`:

```python
def _render_pending_image_attachments() -> None:
    pending = st.session_state.get("pending_image_attachments", [])
    if not pending:
        return

    st.caption(f"{len(pending)} attachment(s) ready")
    for row_start in range(0, len(pending), _PENDING_IMAGE_PREVIEW_COLUMNS):
        row = pending[row_start : row_start + _PENDING_IMAGE_PREVIEW_COLUMNS]
        cols = st.columns(_PENDING_IMAGE_PREVIEW_COLUMNS)
        for idx, att in enumerate(row):
            if not isinstance(att, dict):
                continue
            with cols[idx]:
                try:
                    image_bytes = base64.b64decode(att["data"])
                except Exception:
                    continue
                st.image(
                    image_bytes,
                    caption=att.get("name") or "image",
                    width=_PENDING_IMAGE_PREVIEW_WIDTH,
                )
                if st.button(
                    "",
                    icon=":material/close:",
                    key=f"remove_{att.get('token')}",
                    help="Remove attachment",
                ):
                    st.session_state.pending_image_attachments = [
                        item
                        for item in pending
                        if isinstance(item, dict) and item.get("token") != att.get("token")
                    ]
                    st.rerun()
```

In `render_chat_view()`, replace the entire existing pending attachment preview block:

```python
        if st.session_state.pending_image_attachments:
            st.caption(f"{len(st.session_state.pending_image_attachments)} attachment(s) ready")
            cols = st.columns(min(len(st.session_state.pending_image_attachments), 4))
            for idx, att in enumerate(st.session_state.pending_image_attachments):
                with cols[idx % len(cols)]:
                    image_bytes = base64.b64decode(att["data"])
                    st.image(image_bytes, caption=att["name"], width=80)
                    if st.button(
                        "Remove",
                        icon=":material/close:",
                        key=f"remove_{att['token']}",
                        help="Remove attachment",
                    ):
                        st.session_state.pending_image_attachments = [
                            item
                            for item in st.session_state.pending_image_attachments
                            if item["token"] != att["token"]
                        ]
                        st.rerun()
```

with:

```python
        _render_pending_image_attachments()
```

- [ ] **Step 6: Run tests**

Run:

```bash
python -m pytest tests/test_demo_image_paste.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add demo.py tests/test_demo_image_paste.py
git commit -m "feat: remove chat image attachment count limit"
```

### Task 8: Add Ctrl+V Clipboard Image Capture Component

**Files:**
- Create: `app/ui/clipboard_image_capture.py`
- Create: `app/ui/clipboard_image_capture/index.html`
- Modify: `demo.py`
- Modify: `tests/test_demo_image_paste.py`
- Modify: `tests/test_demo_plan_widget.py`

- [ ] **Step 1: Add component source tests**

Append to `tests/test_demo_image_paste.py`:

```python
def test_demo_mounts_clipboard_capture_near_chat_input():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    assert "capture_pasted_images(" in source
    assert "_handle_pasted_image_payload(" in source


def test_paste_mount_does_not_block_message_form_rendering():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    paste_pos = source.index("pasted_payload = capture_pasted_images")
    form_pos = source.index('with st.form("message_form"')
    paste_block = source[paste_pos:form_pos]

    assert paste_pos < form_pos
    assert "if _handle_pasted_image_payload(pasted_payload):" in paste_block
    assert "st.rerun()" in paste_block
    assert "return" not in paste_block


def test_clipboard_component_filters_to_focused_message_textarea():
    html = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "ui"
        / "clipboard_image_capture"
        / "index.html"
    ).read_text(encoding="utf-8")

    assert 'placeholder === "Type your message..."' in html
    assert "clipboardData" in html
    assert "eventId" in html
    assert "images" in html
    assert "__chatImagePasteSetComponentValue" in html
    assert "setFrameHeight(0)" in html
    assert "streamlit:setComponentValue" in html
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
python -m pytest tests/test_demo_image_paste.py::test_demo_mounts_clipboard_capture_near_chat_input tests/test_demo_image_paste.py::test_clipboard_component_filters_to_focused_message_textarea -v
```

Expected: FAIL because the component is missing.

- [ ] **Step 3: Update the Streamlit test stub**

In `tests/test_demo_plan_widget.py`, extend `_import_demo_with_ui_stubs()` so modules that call `streamlit.components.v1.declare_component()` can import under the existing stub:

```python
    components_v1_module.html = lambda *args, **kwargs: None
    components_v1_module.declare_component = (
        lambda *args, **kwargs: (lambda **_component_kwargs: _component_kwargs.get("default"))
    )
```

- [ ] **Step 4: Create Python wrapper**

Create `app/ui/clipboard_image_capture.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_COMPONENT_DIR = Path(__file__).with_name("clipboard_image_capture")
_capture_component = components.declare_component(
    "clipboard_image_capture",
    path=str(_COMPONENT_DIR),
)


def capture_pasted_images(*, key: str) -> dict[str, Any] | list[dict[str, Any]]:
    value = _capture_component(key=key, default={})
    return value if isinstance(value, (dict, list)) else {}
```

- [ ] **Step 5: Create JavaScript component**

Create `app/ui/clipboard_image_capture/index.html`:

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <script>
      function sendStreamlitMessage(type, data) {
        window.parent.postMessage(
          {
            isStreamlitMessage: true,
            type: type,
            ...data,
          },
          "*"
        );
      }

      function setComponentValue(value) {
        sendStreamlitMessage("streamlit:setComponentValue", { value: value });
      }

      function setFrameHeight(height) {
        sendStreamlitMessage("streamlit:setFrameHeight", { height: height });
      }

      function componentReady() {
        sendStreamlitMessage("streamlit:componentReady", { apiVersion: 1 });
        setFrameHeight(0);
      }

      function activeChatTextarea(doc) {
        const active = doc.activeElement;
        if (!active || active.tagName !== "TEXTAREA") return null;
        const placeholder = active.getAttribute("placeholder") || "";
        const label = active.getAttribute("aria-label") || "";
        if (placeholder === "Type your message..." || label === "Message") {
          return active;
        }
        return null;
      }

      async function fileToPayload(file, index) {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => {
            resolve({
              name: file.name || `clipboard-image-${Date.now()}-${index}.png`,
              mime: file.type || "image/png",
              data: String(reader.result || ""),
            });
          };
          reader.onerror = () => reject(reader.error);
          reader.readAsDataURL(file);
        });
      }

      function install() {
        let topDoc;
        try {
          topDoc = window.parent.document;
        } catch (_) {
          return;
        }
        if (!topDoc) return;
        topDoc.__chatImagePasteSetComponentValue = setComponentValue;
        if (topDoc.__chatImagePasteInstalled) return;
        topDoc.__chatImagePasteInstalled = true;

        topDoc.addEventListener(
          "paste",
          async function (event) {
            if (!activeChatTextarea(topDoc)) return;
            const clipboard = event.clipboardData;
            if (!clipboard || !clipboard.items) return;

            const files = [];
            for (const item of clipboard.items) {
              if (item.kind === "file" && item.type && item.type.startsWith("image/")) {
                const file = item.getAsFile();
                if (file) files.push(file);
              }
            }

            if (!files.length) return;
            event.preventDefault();

            const payload = [];
            for (let index = 0; index < files.length; index += 1) {
              payload.push(await fileToPayload(files[index], index + 1));
            }
            const eventId = `paste-${Date.now()}-${Math.random().toString(36).slice(2)}`;
            topDoc.__chatImagePasteSetComponentValue({
              eventId: eventId,
              images: payload,
            });
          },
          true
        );
      }

      componentReady();
      install();
    </script>
  </head>
  <body></body>
</html>
```

- [ ] **Step 6: Mount component in `demo.py`**

Add import:

```python
from app.ui.clipboard_image_capture import capture_pasted_images
```

Inside `render_chat_view()`, after the uploader block and before the form:

```python
        pasted_payload = capture_pasted_images(key=f"chat_image_paste_{conversation_id}")
        if _handle_pasted_image_payload(pasted_payload):
            st.rerun()
```

Do not add `return` after this block. On reruns where `_handle_pasted_image_payload()` returns `False`, execution must continue into `with st.form("message_form", clear_on_submit=True):` so the chat text box and Send/Attach controls remain visible.

- [ ] **Step 7: Run component tests**

Run:

```bash
python -m pytest tests/test_demo_image_paste.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add app/ui/clipboard_image_capture.py app/ui/clipboard_image_capture/index.html demo.py tests/test_demo_image_paste.py tests/test_demo_plan_widget.py
git commit -m "feat: support pasted chat image attachments"
```

### Task 9: Add API Attachment Flow Regressions

**Files:**
- Modify: `tests/test_workflow_request_conversion.py`
- Modify: `tests/test_ai_sdk_v6_stream_contract.py`

- [ ] **Step 1: Add workflow conversion regression**

Append to `tests/test_workflow_request_conversion.py`:

```python
from app.ai.schemas import WorkflowExecutionRequest as AIWorkflowExecutionRequest
from app.schemas.workflow import WorkflowExecutionRequest as ServiceWorkflowExecutionRequest


def test_workflow_request_conversion_preserves_attachments():
    attachments = [{"name": "a.png", "mime": "image/png", "data": "abc"}]
    request = ServiceWorkflowExecutionRequest(message="see image", attachments=attachments)

    ai_request = AIWorkflowExecutionRequest.model_validate(request.model_dump(mode="python"))

    assert ai_request.attachments == attachments
```

- [ ] **Step 2: Add AI SDK extraction regression**

Append to `tests/test_ai_sdk_v6_stream_contract.py`:

```python
from app.api.ai_sdk import _extract_user_attachments


def test_ai_sdk_extracts_file_part_data_url_attachment():
    payload = [
        {
            "role": "user",
            "parts": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "file",
                    "name": "screen.png",
                    "mediaType": "image/png",
                    "url": "data:image/png;base64,abc",
                },
            ],
        }
    ]

    assert _extract_user_attachments(payload) == [
        {"name": "screen.png", "mime": "image/png", "data": "data:image/png;base64,abc"}
    ]
```

- [ ] **Step 3: Run regression tests**

Run:

```bash
python -m pytest tests/test_workflow_request_conversion.py tests/test_ai_sdk_v6_stream_contract.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_workflow_request_conversion.py tests/test_ai_sdk_v6_stream_contract.py
git commit -m "test: cover image attachment request flow"
```

### Task 10: Remove Legacy Image-Context Duplication

**Files:**
- Modify: `app/ai/graph.py`
- Modify: `app/ai/agents/chat_agent.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `demo.py`

- [ ] **Step 1: Scan for old graph image helpers**

Run:

```bash
rg -n "_normalize_attachment_image_url|_build_chat_turn_messages_with_attachments" app/ai/graph.py
```

Expected: no matches.

- [ ] **Step 2: Replace duplicate ChatAgent direct vision construction**

In `app/ai/agents/chat_agent.py`, import:

```python
from ..image_context import build_multimodal_content, normalize_image_attachment
```

In `_generate_with_vision()`, replace the OpenAI branch's manual `content` construction:

```python
                    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                    for attachment in attachments:
                        raw_data = str(attachment.get("data") or "").strip()
                        mime_type = str(attachment.get("mime") or "image/jpeg").strip()
                        if raw_data.startswith("data:"):
                            image_url = raw_data
                        else:
                            image_url = f"data:{mime_type};base64,{raw_data}"
                        if raw_data:
                            content.append({"type": "image_url", "image_url": {"url": image_url}})
```

with:

```python
                    content = build_multimodal_content(prompt, attachments)
```

In the Gemini branch, normalize attachments before decoding so data URLs and MIME inference use the same backend rules:

```python
                parts = [types.Part(text=prompt)]
                for attachment in attachments:
                    try:
                        normalized = normalize_image_attachment(attachment)
                        if normalized is None:
                            continue
                        image_url = normalized["url"]
                        if not image_url.startswith("data:"):
                            continue
                        _header, _separator, raw_data = image_url.partition(",")
                        if not raw_data:
                            continue

                        image_data = base64.b64decode(raw_data)
                        parts.append(
                            types.Part.from_bytes(
                                data=image_data,
                                mime_type=normalized["mime"],
                            )
                        )
                    except Exception as img_err:
                        logger.error("Failed to process image attachment: %s", img_err)
```

- [ ] **Step 3: Scan for duplicate backend image URL part construction**

Run:

```bash
rg -n '"type": "image_url"' app/ai/graph.py app/ai/agents app/services app/api
```

Expected: no matches.

Run:

```bash
rg -n "image_url_part\\(" app/ai/graph.py app/ai/agents
```

Expected: `app/ai/agents/rag_agent.py` may call `image_url_part()` for document images. Graph and ChatAgent should not manually construct LangChain image parts.

If `app/ai/agents/rag_agent.py` still manually builds `{"type": "image_url"}`, replace it with:

```python
image_url_part(f"data:{mime_type};base64,{img_data}")
```

- [ ] **Step 4: Scan for removed Streamlit count cap**

Run:

```bash
rg -n "_MAX_IMAGE_ATTACHMENTS|Up to \\{_MAX_IMAGE_ATTACHMENTS\\}|Some images ignored|remaining = _MAX_IMAGE_ATTACHMENTS" demo.py
```

Expected: no matches.

- [ ] **Step 5: Scan for stale paste replay risks**

Run:

```bash
rg -n "consumed_image_paste_events|eventId|__chatImagePasteSetComponentValue" demo.py app/ui/clipboard_image_capture/index.html
```

Expected: matches for all three identifiers. `demo.py` should contain `consumed_image_paste_events`; the component HTML should contain `eventId` and `__chatImagePasteSetComponentValue`.

- [ ] **Step 6: Remove unused imports**

Run:

```bash
python -m ruff check app/ai/graph.py app/ai/agents/chat_agent.py app/ai/agents/rag_agent.py demo.py --select F401,F841
```

Expected: no unused imports or unused local variables.

- [ ] **Step 7: Commit**

```bash
git add app/ai/graph.py app/ai/agents/chat_agent.py app/ai/agents/rag_agent.py demo.py
git commit -m "chore: remove redundant image context code"
```

### Task 11: Verification And Manual QA

**Files:**
- No planned code changes.

- [ ] **Step 1: Run focused image-context suite**

Run:

```bash
python -m pytest tests/test_image_context.py tests/test_message_history_image_context.py tests/test_base_agent_image_history.py tests/test_graph_image_context.py tests/test_rag_agent_image_attachments.py tests/test_demo_image_paste.py -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent agent and streaming tests**

Run:

```bash
python -m pytest tests/test_custom_agents_graph.py tests/test_graph_handoff_streaming.py tests/test_message_history_pipeline.py tests/test_chat_agent_image_search_binding.py tests/test_ai_sdk_v6_stream_contract.py tests/test_ai_sdk_context_window.py -q
```

Expected: PASS.

- [ ] **Step 3: Run lint on touched files**

Run:

```bash
python -m ruff check app/ai/image_context.py app/ai/history.py app/ai/memory.py app/ai/agents/base_agent.py app/ai/agents/chat_agent.py app/ai/agents/rag_agent.py app/ai/graph.py app/ai/conversation_summarizer.py demo.py tests/test_image_context.py tests/test_message_history_image_context.py tests/test_base_agent_image_history.py tests/test_graph_image_context.py tests/test_rag_agent_image_attachments.py tests/test_demo_image_paste.py tests/test_demo_plan_widget.py
```

Expected: no lint errors.

- [ ] **Step 4: Manual QA**

Run the Streamlit demo and verify:

1. Upload one PNG, ask what is in it, and confirm the answer refers to the image.
2. Paste an image with Ctrl+V while the message box is focused and confirm it appears in pending attachments without continuous page refreshing.
3. After the paste rerun settles, confirm the chat text box plus Send and Attach controls are still visible.
4. Upload or paste at least five images and confirm all appear in a compact pending-preview grid near the chat box.
5. Send an image-bearing request that routes to a non-chat agent and confirm the agent does not report missing image context.
6. Send an image-bearing request while planning mode is active and confirm the planning response or delegated worker does not report missing image context.
7. Paste an image, send it, wait for the rerun, and confirm the same pasted image is not re-added to the pending attachment queue.
8. Ask a follow-up about a recently attached image and confirm the model can still inspect it while that user message remains in prompt history.

---

## Risks

- Removing app-level caps does not remove external limits. Browsers, HTTP clients, databases, providers, and context windows can still reject large payloads.
- Historical image replay increases prompt size. Existing history trimming remains the guardrail.
- Long-term summaries in this plan retain image references, not full visual captions. A later captioning/indexing feature would be needed for durable visual recall after raw images fall out of prompt history.
- The paste component uses `window.parent.document`, matching the existing lightbox pattern in `demo.py`. If Streamlit changes iframe isolation, the component may need explicit frontend focus plumbing.
- Browser `blob:` URLs are intentionally rejected in backend normalization because they are page-local. Clipboard images must arrive as data URLs.

## Self-Review

- Spec coverage: all-agent current-turn context is covered by Tasks 1, 4, and 5, including planning and isolated worker paths. History and memory are covered by Tasks 2, 3, and 6. Paste support, no paste refresh loop, chat box visibility after paste, compact pending image layout, and removal of the four-image limit are covered by Tasks 7 and 8. API preservation is covered by Task 9. Redundant/legacy cleanup is covered by Task 10.
- Placeholder scan: no unfinished marker or deferred work language is intentionally included.
- Type consistency: all backend helpers use the existing `AgentMessage.attachments` and `MessageCreate.attachments` shape with `name`, `mime`, and `data`.
- Scope check: this is one cohesive change set and does not introduce a separate visual asset store.

---

## Implementation Progress Log

> Execution environment: Windows, branch `Thai-Postgre-FastAPI`. Tests run with `.venv/Scripts/python.exe -m pytest` (Python 3.13.7, pytest 9.0.3). `rg` scans replaced with the editor Grep tool. Per-task commits follow the plan.

### Design Decisions

**Post-implementation hardening (production-ready pass, commit `0f5d6a0`):**
- **Token estimate corrected (real fix, not cosmetic):** the plan's `32`-token-per-image placeholder made `trim_history_to_budget` treat an image as ~free, so image-bearing history could evade the trim guardrail and overflow the real context window — contradicting the plan's own "history trimming remains the guardrail" claim. Replaced with a documented `IMAGE_ATTACHMENT_TOKEN_ESTIMATE = 1200` constant (a conservative middle across Anthropic ~(w·h)/750≤~1600 and OpenAI high-detail ~765+). Trimming-only estimate; still not an app-level cap. Sole consumer is `trim_history_to_budget`; the plan's test (`with_image > without_image`) still passes; history/trim suites green.
- **conversation_summarizer.py** made fully lint-clean by wrapping 3 pre-existing over-length logger strings via implicit concatenation (logged text unchanged).
- **chat_agent.py** Gemini branch: dropped unused `partition` throwaways.
- **Deliberate scope decision — demo.py baseline left alone:** 46 pre-existing `E501` + 1 pre-existing `E402` (a mid-file `stream_markdown` re-export) are unrelated to image context and predate this branch. Reformatting scattered lines across a 9600-line UI file would balloon the diff and risk regressions in code this feature never touched; my additions to demo.py are lint-clean. Left as separately-tracked tech debt rather than mixed into this feature.
- **Full suite after hardening:** 1396 passed; only the 2 pre-existing environmental failures remain (`test_live_server_integration` doc-upload; `test_brave_image_search_config` — a real Brave key in the env vs the "empty default" assertion). Neither touches image-context code.

**Task 1 (image_context.py):** Implemented verbatim from the plan; no deviations. 7/7 tests pass. `normalize_image_attachment` correctly passes through data URLs, infers MIME from filename/data-url header, rejects `blob:` URLs and local paths, and rejects non-image MIME types. Committed as `ad19dba`.

**Task 2 (history.py, memory.py):** Implemented as planned. Verified `AgentMessage.attachments` already exists in `schemas.py`, `MessageRole` values are USER/ASSISTANT, and `DBMessageRole` is an `IntEnum` (user=1, assistant=2) matching both `history.py`'s value comparison and `memory.py`'s `{1: USER, 2: ASSISTANT}` map. Assistant history stays text-only. 3/3 tests pass. Committed as `1ce9403`.

**Task 3 (base_agent.py, token_instrumentation.py):** Implemented as planned. `image_context` import placed alphabetically after `hand_off_tool`. History conversion only upgrades a user message to multimodal when `has_image_parts` is true, otherwise keeps the plain string (avoids wrapping text-only history in list form unnecessarily). Token estimator adds a per-attachment overhead for trimming — an estimate, not a cap (initially 32; later revised to a realistic `IMAGE_ATTACHMENT_TOKEN_ESTIMATE = 1200` — see the post-implementation hardening note above). 2/2 tests pass. Committed as `9a969bf`.

**Task 4 (graph.py):** Implemented. Deleted `_normalize_attachment_image_url`, replaced `_build_chat_turn_messages_with_attachments` with `_build_turn_messages_with_attachments` + added `_mark_response_has_images` and `_apply_current_turn_attachments`. Wired the apply+mark pair into `_chat_node`, `_custom_agent_node`, `_search_node`, `_image_generator_node`, `_canvas_node`, `_planning_node`, and the isolated-worker RAG `AgentMessage`.

**Design decision (plan vs. test conflict, isolated worker):** The plan's snippet writes `self._apply_current_turn_attachments(parent_state, [worker_message])` across multiple lines, but `tests/test_graph_image_context.py::test_isolated_worker_path_applies_parent_attachments` asserts that exact call as a single-line substring. A single-line assignment `worker_messages, worker_has_images = self._apply_current_turn_attachments(...)` would exceed the project's 100-char ruff limit. Resolved in favor of the executable test: call on one line into a `worker_turn` temp, then unpack `worker_messages, worker_has_images = worker_turn`. Satisfies the substring assertion and stays ≤100 chars. Import placed after `hitl_config` (isort order). `_rag_node` already forwarded attachments and was left unchanged. Focused suite 5/5, adjacent `test_custom_agents_graph`/`test_graph_handoff_streaming` 29/29, ruff clean. Committed as `bba667a`.

**Task 5 (rag_agent.py):** Implemented. `_process_message_agentic` now reads `message.attachments`, seeds `human_content` via `build_multimodal_content(context_text, user_attachments)`, then appends document (`agentic_images`) parts via `image_url_part`. Vision-fallback check and `has_images` telemetry now key off `has_prompt_images` (covers both user + document images), not just `agentic_images`. Import placed after `context_overflow` (isort). 35/35 RAG tests pass (1 new + 34 existing), ruff clean. Committed as `4005930`.

**Task 6 (conversation_summarizer.py):** Implemented. `_agent_message_to_langchain` appends `attachment_memory_lines(...)` to USER content (text refs only, never raw base64); non-USER roles become `AIMessage`. Minor test deviation: merged `_agent_message_to_langchain` into the existing top-level import rather than duplicating `AgentMessage`/`MessageRole` imports mid-file (keeps ruff clean). 2/2 tests pass. **Pre-existing (not mine):** `conversation_summarizer.py` has 3 baseline `E501` warnings on unrelated logger f-strings (lines 102/146/171) — confirmed present at HEAD via `git stash`; left untouched as out-of-scope. My change is ruff-clean. Committed as `37e5877`.

**Task 7 (demo.py):** Implemented. Removed `_MAX_IMAGE_ATTACHMENTS` (0 refs remain) and its cap logic in `_handle_new_image_attachments`. Added `_PENDING_IMAGE_PREVIEW_COLUMNS=8`/`_WIDTH=72` layout constants, paste helpers (`_attachment_from_pasted_payload`, `_paste_event_id`, `_paste_payload_images`, `_handle_pasted_image_payload` with `consumed_image_paste_events` dedup), and `_render_pending_image_attachments` compact grid. Replaced the sparse 4-column preview block and uploader help text. `uuid`/`base64`/`Any` already imported. 4/4 tests pass, ruff F401/F811/F841 clean. Committed as `d8ff69b`.

**Task 8 (clipboard component):** Implemented. Created `app/ui/clipboard_image_capture.py` wrapper + `app/ui/clipboard_image_capture/index.html` paste listener, extended the test stub with `declare_component`, and mounted `capture_pasted_images()` in `render_chat_view()`.

**Design decision (mount placement):** The plan says "mount after the uploader and before the form", but the region between them contains a stop-rerun early `return`, and `test_paste_mount_does_not_block_message_form_rendering` asserts `"return" not in` the slice from the mount to the form. Placed the mount immediately before `with st.form("message_form", ...)` (after the stop-rerun return) so the assertion holds and the stop-rerun path is not disturbed. No `return` after the mount, so a `False` result falls through to the form (text box + Send/Attach stay visible). 8/8 paste tests + 13/13 plan-widget tests pass. F401/F811/F841 clean; demo.py E501 count unchanged at 46 (pre-existing baseline, 0 added by Tasks 7–8). Committed as `40ae958`.

**Task 9 (regression tests):** Added. Verified both `WorkflowExecutionRequest` schemas already carry `attachments` and `_extract_user_attachments` already parses AI SDK file parts (data-URL → `data` key). These are guards for existing behavior, so no red phase. Minor test deviation: placed the new imports at the top of each file (not appended mid-file with the test body) to avoid ruff `E402`. 14/14 tests pass, ruff clean. Committed as `de40acf`.

**Task 10 (dedup cleanup):** Verified via scans: no `_normalize_attachment_image_url`/`_build_chat_turn_messages_with_attachments` remain in graph.py; the only `{"type": "image_url"}` construction is in `image_context.py` (the `image_url_part` helper); `image_url_part(` is called only by `image_context.py` internally and `rag_agent.py` (document images). Replaced ChatAgent `_generate_with_vision` manual OpenAI block with `build_multimodal_content(prompt, attachments)` and the Gemini block with `normalize_image_attachment`-based decoding (unifies data-URL/MIME handling). `_MAX_IMAGE_ATTACHMENTS`/"Some images ignored" gone from demo.py; `consumed_image_paste_events`/`eventId`/`__chatImagePasteSetComponentValue` present where expected. Only chat_agent.py needed edits (graph/rag/demo already clean from their tasks). F401/F841 clean; 6/6 chat-agent+history tests pass. Committed as `8c14304`.

**Task 11 (verification & QA):**
- Step 1 focused image-context suite: 25/25 pass.
- Step 2 adjacent agent/streaming suite: 60/60 pass.
- Step 3 lint scan on all touched files: fixed 5 `I001` (double-blank-line after imports, from the plan's snippets) in the new/test files (commit `69b543c`). Remaining 49 `E501` + 1 `E402` are **pre-existing baseline** on unrelated lines in `demo.py` (46 E501 + 1 E402 mid-file `stream_markdown` import) and `conversation_summarizer.py` (3 E501) — all confirmed present at/before HEAD. My additions are lint-clean.
- Full suite: **1396 passed, 2 failed**. Both failures are pre-existing environmental and untouched by this work: `test_live_server_integration` (documented flaky doc-upload) and `test_brave_image_search_config::test_brave_image_search_defaults` (a real Brave API key is present in the env, so the "defaults to empty" assertion fails). `git diff` vs base confirms no brave/settings/live-server file was modified.
- **Regression found & fixed during full-suite run:** demo.py now imports the clipboard component at module load, which calls `declare_component`. Four other demo-importing test helpers (`test_demo_custom_agents`, `test_demo_meaningful_widgets`, `test_demo_stop_generation`, `test_demo_sidecar_auth`) stub `streamlit.components.v1` without it, so demo import broke (20 failures, test-order-dependent via module caching). Added `declare_component` to all four stubs (commit `3d395de`); all 53 demo tests now pass.
- Step 4 manual browser QA (8 items: Ctrl+V paste, no-rerun-loop, controls visible, compact grid, non-chat/planning agents see images, no stale replay, follow-up image recall): **requires an interactive Streamlit session with a human** — cannot be driven in this non-interactive environment. Programmatic substitute done: all modified backend modules import cleanly, the component wrapper imports and `capture_pasted_images` is callable, demo.py compiles, and the component HTML asset is present and asserted by tests. **The 8 interactive checks remain for the user to run.**

### Task Status

- [x] Task 1: Add Shared Image Context Utilities
- [x] Task 2: Preserve Attachments In Prompt History
- [x] Task 3: Convert Historical Images In BaseAgent Prompts
- [x] Task 4: Apply Current-Turn Images To All Graph Agents
- [x] Task 5: Add User Images To RAG Agentic Prompting
- [x] Task 6: Preserve Image References In Long-Term Summary Input
- [x] Task 7: Remove Streamlit Image Count Limit And Add Paste Payload Handler
- [x] Task 8: Add Ctrl+V Clipboard Image Capture Component
- [x] Task 9: Add API Attachment Flow Regressions
- [x] Task 10: Remove Legacy Image-Context Duplication
- [x] Task 11: Verification And Manual QA (automated verification complete; 8 interactive browser QA items pending user)
- [ ] Task 11: Verification And Manual QA
