import base64
import mimetypes
import uuid
from copy import deepcopy

import streamlit as st
import requests
import html
import re
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional
from upload_support import render_upload_section, render_document_list
from datetime import datetime, timedelta
from dateutil import parser
import markdown as _markdown

API_BASE_URL = "http://localhost:8000"

_MAX_PERSONA_LENGTH = 2000
_MAX_IMAGE_ATTACHMENTS = 4

PERSONA_TEMPLATES: Dict[str, str] = {
    "Friendly Tutor": (
        "You are a patient programming tutor. Explain topics with simple analogies, "
        "show step-by-step examples, and confirm the learner's understanding before moving on."
    ),
    "Domain Expert": (
        "You are a senior data analyst who answers with evidence. Highlight key metrics, "
        "call out assumptions, and recommend the next investigative steps."
    ),
    "Motivational Coach": (
        "You are a supportive productivity coach. Celebrate wins, emphasize progress, "
        "and end each reply with one clear, actionable suggestion."
    ),
}


_SELF_CLOSING_TAGS = {"br", "hr"}
_ALLOWED_TAGS = {
    "p",
    "ul",
    "ol",
    "li",
    "strong",
    "em",
    "code",
    "pre",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
}.union(_SELF_CLOSING_TAGS)
APP_STYLE = """
<style>
    .main-container { max-width: 1200px; margin: 0 auto; }
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        gap: 16px;
        height: calc(100vh - 220px);
        min-height: 460px;
        position: relative;
    }
    .chat-messages-container {
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        background: #ffffff;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .chat-messages-scroll {
        padding: 20px;
        flex: 1 1 auto;
        overflow-y: auto;
    }
    .chat-messages-scroll::-webkit-scrollbar {
        width: 8px;
    }
    .chat-messages-scroll::-webkit-scrollbar-thumb {
        background: #cbd5f5;
        border-radius: 4px;
    }
    .chat-input-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        padding: 25px;
        margin-top: 4px;
        position: sticky;
        bottom: 0;
        z-index: 5;
        flex-shrink: 0;
    }
    .chat-input-container form {
        margin: 0;
    }
    .user-message {
        background: #ffffff;
        color: #1f2937;
        border: 1px solid #93c5fd;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px auto;
        max-width: 72%;
        word-wrap: break-word;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.16);
    }
    .bot-message {
        background: #ffffff;
        color: #1f2937;
        border: 1px solid #86efac;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px auto 8px 0;
        max-width: 60%;
        white-space: normal;
        line-height: 1.5;
        word-wrap: break-word;
        overflow-wrap: anywhere;
        box-shadow: 0 8px 20px rgba(34, 197, 94, 0.16);
    }
    .message-attachments {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 6px 0 0;
    }
    .message-attachments .attachment-thumb {
        width: 72px;
        height: 72px;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
        background: #f8fafc;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .message-attachments .attachment-thumb:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.2);
    }
    .message-attachments .attachment-thumb-link {
        display: inline-flex;
        border-radius: 12px;
        text-decoration: none;
    }
    .message-attachments .attachment-thumb-link:focus-visible {
        outline: 2px solid #38bdf8;
        outline-offset: 2px;
    }
    .pending-attachments .attachment-thumb img,
    .message-attachments .attachment-thumb img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    .user-message p,
    .bot-message p {
        margin: 0 0 0.75rem 0;
    }
    .user-message p:last-child,
    .bot-message p:last-child {
        margin-bottom: 0;
    }
    .user-message ul,
    .user-message ol,
    .bot-message ul,
    .bot-message ol {
        margin: 0.5rem 0 0.5rem 1.25rem;
        padding-left: 1.25rem;
    }
    .user-message code,
    .bot-message code {
        background: #f9fafb;
        padding: 0.15rem 0.35rem;
        border-radius: 4px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 0.85rem;
    }
    .user-message pre,
    .bot-message pre {
        background: #f9fafb;
        padding: 0.75rem;
        border-radius: 8px;
        overflow-x: auto;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
    .sidebar-conversation {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    .sidebar-conversation:hover { background: #e9ecef; transform: translateX(5px); }
    .sidebar-conversation.active {
        background: linear-gradient(135deg, #28a745, #1e7e34);
        color: white;
        border-color: #1e7e34;
    }
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 40px 20px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .message-timestamp { font-size: 0.8em; color: #64748b; margin-top: 5px; white-space: normal; display: block; max-width: 100%; overflow-wrap: anywhere; }
    div:empty { display: none !important; }
</style>
"""

SESSION_STATE_DEFAULTS: Dict[str, Callable[[], Any] | Any] = {
    "current_user_id": lambda: None,
    "current_conversation_id": lambda: None,
    "messages": list,
    "conversations_list": list,
    "show_conversation_manager": lambda: False,
    "show_instructions": lambda: False,
    "auth_token": lambda: None,
    "conversation_messages_meta": lambda: None,
    "conversation_messages_page": lambda: 0,
    "has_more_messages": lambda: True,
    "pending_persona_prompt": str,
    "persona_editor_origin": lambda: None,
    "persona_editor_value": str,
    "persona_feedback": lambda: None,
    "persona_editor_pending_value": str,
    "persona_editor_pending": lambda: False,
    "pending_image_attachments": list,
    "message_image_thumbnails": dict,
    "show_attachment_uploader": lambda: False,
    "image_viewer_open": lambda: False,
    "image_viewer_payload": lambda: None,
    "API_BASE_URL": lambda: API_BASE_URL,
}


def initialize_session_state() -> None:
    """Ensure all expected session-state keys exist with sensible defaults."""
    for key, factory in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = factory() if callable(factory) else factory

    if "show_login" not in st.session_state:
        st.session_state.show_login = not bool(st.session_state.get("auth_token"))


class _SafeHTMLRenderer(HTMLParser):
    """Allow-list HTML sanitizer for rendered message content."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.result: List[str] = []
        self._tag_stack: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag not in _ALLOWED_TAGS:
            return

        if tag in _SELF_CLOSING_TAGS:
            self.result.append(f"<{tag}>")
            return

        self.result.append(f"<{tag}>")
        self._tag_stack.append(tag)

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if not self._tag_stack:
            return

        if self._tag_stack and self._tag_stack[-1] == tag:
            self.result.append(f"</{tag}>")
            self._tag_stack.pop()

    def handle_startendtag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in _SELF_CLOSING_TAGS:
            self.result.append(f"<{tag}>")
        elif tag in _ALLOWED_TAGS:
            self.result.append(f"<{tag}></{tag}>")

    def _inside_block(self, tag: str) -> bool:
        return tag in self._tag_stack

    def handle_data(self, data: str):
        if not data:
            return

        escaped = html.escape(data, quote=False)
        self.result.append(escaped)

    def handle_entityref(self, name: str):
        self.result.append(f"&{name};")

    def handle_charref(self, name: str):
        self.result.append(f"&#{name};")


def _sanitize_rendered_html(html_fragment: str) -> str:
    parser = _SafeHTMLRenderer()
    parser.feed(html_fragment)
    parser.close()
    return "".join(parser.result)


def _attachment_from_upload(uploaded_file) -> Optional[Dict[str, str]]:
    try:
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
    except Exception:
        return None

    if not raw_bytes:
        return None

    data_b64 = base64.b64encode(raw_bytes).decode("utf-8")
    mime = (
        uploaded_file.type
        or mimetypes.guess_type(uploaded_file.name or "")[0]
        or "application/octet-stream"
    )

    return {
        "token": str(uuid.uuid4()),
        "name": uploaded_file.name or "image",
        "mime": mime,
        "data": data_b64,
    }


def _handle_new_image_attachments(uploaded_files: List) -> None:
    if not uploaded_files:
        return

    pending = st.session_state.get("pending_image_attachments", [])
    existing_data = {item["data"] for item in pending}

    new_items: List[Dict[str, str]] = []
    for file_obj in uploaded_files:
        attachment = _attachment_from_upload(file_obj)
        if not attachment:
            continue

        if attachment["data"] in existing_data or any(
            item["data"] == attachment["data"] for item in new_items
        ):
            continue

        new_items.append(attachment)
        existing_data.add(attachment["data"])

    if not new_items:
        return

    remaining = _MAX_IMAGE_ATTACHMENTS - len(pending)
    if remaining <= 0:
        st.warning(f"Maximum of {_MAX_IMAGE_ATTACHMENTS} images per message reached.")
        return

    if len(new_items) > remaining:
        st.info("Some images were ignored because the limit was reached.")

    pending.extend(new_items[:remaining])
    st.session_state.pending_image_attachments = pending


def render_pending_attachment_preview(allow_remove: bool = True):
    attachments = st.session_state.get("pending_image_attachments", [])
    if not attachments:
        return

    st.caption(
        f"Attachments ready to send ({len(attachments)}/{_MAX_IMAGE_ATTACHMENTS}):"
    )
    columns = st.columns(min(len(attachments), 4))
    remove_token: Optional[str] = None

    for idx, attachment in enumerate(attachments):
        column = columns[idx % len(columns)]
        with column:
            image_bytes = base64.b64decode(attachment["data"])
            st.image(image_bytes, caption=attachment["name"], width=96, clamp=True)
            if allow_remove:
                if st.button(
                    "Remove",
                    key=f"remove_pending_{attachment['token']}",
                    use_container_width=True,
                ):
                    remove_token = attachment["token"]

    if remove_token:
        st.session_state.pending_image_attachments = [
            item for item in attachments if item["token"] != remove_token
        ]
        st.rerun()


def _guess_extension(mime: Optional[str]) -> str:
    if not mime:
        return "png"
    base_mime = mime.split(";")[0]
    ext = mimetypes.guess_extension(base_mime)
    if ext:
        return ext.lstrip(".")
    if base_mime.endswith("jpeg"):
        return "jpg"
    return "png"

def _format_image_only_message(attachments: List[Dict[str, str]]) -> str:
    """Generate fallback message content for image-only submissions."""
    if not attachments:
        return "[Image attachments]"

    names = [
        att.get("name")
        for att in attachments
        if isinstance(att, dict) and att.get("name")
    ]

    if not names:
        count = len(attachments)
        return "[Image attachment]" if count == 1 else f"[Image attachments: {count} files]"

    if len(names) == 1:
        return f"[Image attachment: {names[0]}]"

    displayed = ", ".join(names[:3])
    if len(names) > 3:
        displayed += ", ..."

    return f"[Image attachments: {displayed}]"


def render_message_attachments(message_id: str, align: str = "left"):
    attachments = st.session_state.get("message_image_thumbnails", {}).get(message_id)
    if not attachments:
        return

    prepared: List[Dict[str, Any]] = []
    thumbnail_fragments: List[str] = []

    for idx, att in enumerate(attachments, start=1):
        if not isinstance(att, dict):
            continue

        mime = att.get("mime", "image/png")
        data_str = att.get("data")
        url_str = att.get("url")

        data_str = data_str.strip() if isinstance(data_str, str) else ""
        url_str = url_str.strip() if isinstance(url_str, str) else ""

        has_data = bool(data_str)
        has_url = bool(url_str)

        if not has_data and not has_url:
            continue

        name = att.get("name") or f"Attachment {idx}"
        escaped_name = html.escape(name, quote=True)
        display_src = f"data:{mime};base64,{data_str}" if has_data else url_str
        escaped_src = html.escape(display_src, quote=True)
        download_name = att.get("download_name") or name
        if "." not in download_name:
            download_name = f"{download_name}.{_guess_extension(mime)}"

        rel_value = "noopener"
        if has_url and not has_data:
            rel_value = "noopener noreferrer"
        rel_attr = f' rel="{rel_value}"' if rel_value else ""

        download_attr = (
            f' download="{html.escape(download_name, quote=True)}"' if has_data else ""
        )

        thumbnail_fragments.append(
            f'<a class="attachment-thumb-link" href="{escaped_src}" target="_blank"{rel_attr} '
            f'aria-label="Open {escaped_name}" title="{escaped_name}"{download_attr}>'
            f'<div class="attachment-thumb">'
            f'<img src="{escaped_src}" alt="{escaped_name}" loading="lazy" />'
            f"</div></a>"
        )

        prepared.append(
            {
                "token": att.get("token") or f"{message_id}_{idx}",
                "name": name,
                "mime": mime,
                "has_data": has_data,
                "data": data_str if has_data else None,
                "url": url_str if has_url else None,
                "display_src": display_src,
                "download_name": download_name,
            }
        )

    if not thumbnail_fragments:
        return

    justify = "flex-end" if align == "right" else "flex-start"
    margin_css = "margin:6px 45px 0 0;" if align == "right" else "margin:6px 0 0 45px;"
    st.markdown(
        f'<div class="message-attachments-wrapper" '
        f'style="display:flex; justify-content:{justify}; {margin_css}">'
        f'<div class="message-attachments">{"".join(thumbnail_fragments)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    viewer_items = [
        {
            "name": item["name"],
            "mime": item["mime"],
            "has_data": item["has_data"],
            "data": item["data"],
            "url": item["url"],
            "display_src": item["display_src"],
            "download_name": item["download_name"],
        }
        for item in prepared
    ]


def render_agent_images(message_metadata: dict, message_id: str):
    """Render images from agent responses"""
    if not message_metadata:
        return

    images = message_metadata.get("images", [])
    if not images:
        return

    prepared: List[Dict[str, Any]] = []
    image_fragments: List[str] = []

    for idx, img in enumerate(images, start=1):
        if isinstance(img, dict):
            if "url" in img:
                url = img.get("url", "")
                description = img.get("description", "")
                if url:
                    escaped_url = html.escape(url, quote=True)
                    text = description or f"Image {idx}"
                    escaped_text = html.escape(text, quote=True)
                    image_fragments.append(
                        f'<a class="attachment-thumb-link" href="{escaped_url}" target="_blank" '
                        f'rel="noopener noreferrer" aria-label="Open {escaped_text}" '
                        f'title="{escaped_text}">'
                        f'<div class="attachment-thumb">'
                        f'<img src="{escaped_url}" alt="{escaped_text}" loading="lazy" />'
                        f"</div></a>"
                    )
                    prepared.append(
                        {
                            "token": img.get("token") or f"{message_id}_url_{idx}",
                            "name": text,
                            "mime": img.get("mime", "image/png"),
                            "has_data": False,
                            "data": None,
                            "url": url,
                            "display_src": url,
                            "download_name": (
                                text
                                if "." in text
                                else f"{text}.{_guess_extension(img.get('mime'))}"
                            ),
                        }
                    )
            elif "data" in img:
                data = img.get("data", "")
                mime = img.get("mime", "image/png")
                name = img.get("name", "Generated image")
                if data:
                    escaped_name = html.escape(name, quote=True)
                    data_url = f"data:{mime};base64,{data}"
                    image_fragments.append(
                        f'<a class="attachment-thumb-link" href="{data_url}" target="_blank" rel="noopener" '
                        f'aria-label="Open {escaped_name}" title="{escaped_name}" download="{escaped_name}">'
                        f'<div class="attachment-thumb">'
                        f'<img src="{data_url}" alt="{escaped_name}" loading="lazy" />'
                        f"</div></a>"
                    )
                    prepared.append(
                        {
                            "token": img.get("token") or f"{message_id}_data_{idx}",
                            "name": name,
                            "mime": mime,
                            "has_data": True,
                            "data": data,
                            "url": None,
                            "display_src": data_url,
                            "download_name": (
                                name
                                if "." in name
                                else f"{name}.{_guess_extension(mime)}"
                            ),
                        }
                    )

    if image_fragments:
        st.markdown(
            f'<div class="message-attachments-wrapper" '
            f'style="display:flex; justify-content:flex-start; margin:6px 0 0 45px;">'
            f'<div class="message-attachments">{"".join(image_fragments)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        viewer_items = [
            {
                "name": item["name"],
                "mime": item["mime"],
                "has_data": item["has_data"],
                "data": item["data"],
                "url": item["url"],
                "display_src": item["display_src"],
                "download_name": item["download_name"],
            }
            for item in prepared
        ]


def normalize_persona_input(raw: str) -> str:
    """Normalize persona text similarly to backend sanitization."""
    if not isinstance(raw, str):
        return ""

    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r" +", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.strip()

    if len(normalized) > _MAX_PERSONA_LENGTH:
        normalized = normalized[:_MAX_PERSONA_LENGTH]

    return normalized


def persona_preview(text: Optional[str], limit: int = 160) -> str:
    """Return a compact preview of persona text for UI surfaces."""
    if not text:
        return ""

    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned

    return cleaned[:limit].rstrip() + "..."


def sanitize_message_content(content: Any) -> str:
    """Render limited markdown to HTML while preventing unsafe tags."""
    if not isinstance(content, str):
        return ""

    normalized = (
        html.unescape(content).replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    if not normalized:
        return ""

    # Replace inline images with accessible text fallback
    normalized = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)", r"\1 (\2)", normalized, flags=re.MULTILINE
    )

    rendered = _markdown.markdown(
        normalized,
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )

    safe_html = _sanitize_rendered_html(rendered)
    safe_html = re.sub(r"(?:<br>\s*){3,}", "<br><br>", safe_html)
    safe_html = safe_html.replace("<p></p>", "")
    safe_html = safe_html.replace("<p><br></p>", "<br>")
    safe_html = re.sub(r"\s*(</?p>)\s*", r"\1", safe_html)
    safe_html = safe_html.strip()

    return safe_html


st.set_page_config(
    page_title="ChatBot", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(APP_STYLE, unsafe_allow_html=True)

initialize_session_state()


def reset_conversation_state() -> None:
    st.session_state.messages = []
    st.session_state.conversation_messages_meta = None
    st.session_state.conversation_messages_page = 0
    st.session_state.has_more_messages = True
    st.session_state.pending_persona_prompt = ""
    st.session_state.persona_editor_origin = None
    st.session_state.persona_editor_value = ""
    st.session_state.persona_editor_pending_value = ""
    st.session_state.persona_editor_pending = False
    st.session_state.pending_image_attachments = []
    st.session_state.show_attachment_uploader = False
    st.session_state.image_viewer_open = False
    st.session_state.image_viewer_payload = None
    st.session_state.message_image_thumbnails = {}


def refresh_conversations_list(
    *, fallback_conversation: Optional[Dict[str, Any]] = None
) -> None:
    """Reload conversations list from the API, optionally seeding with a fallback."""
    get_conversations.clear()
    refreshed = get_conversations(include_messages=False)
    if refreshed and refreshed.get("data"):
        st.session_state.conversations_list = refreshed["data"]["items"]
        return

    if fallback_conversation:
        st.session_state.conversations_list = [
            fallback_conversation,
            *(
                conv
                for conv in st.session_state.conversations_list
                if conv.get("id") != fallback_conversation.get("id")
            ),
        ]


def make_api_request(method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    if st.session_state.get("auth_token"):
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

    try:
        response = getattr(requests, method.lower())(url, json=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
    except requests.exceptions.HTTPError as http_error:
        st.error(f"HTTP error {http_error.response.status_code}: {http_error}")
        return {}
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. Make sure the FastAPI server is running on localhost:8000."
        )
        return {}
    except ValueError:
        st.error("Received an unexpected response from the API.")
        return {}
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return {}

    if not response_data.get("success"):
        error_code = response_data.get("code", "unknown_error")
        error_message = response_data.get("message", "An unknown error occurred.")

        if error_code == "unauthenticated":
            st.error(
                f"Authentication required. Please log in. (Error: {error_message})"
            )
            st.session_state.auth_token = None
            st.session_state.current_user_id = None
            st.session_state.show_login = True
        else:
            st.error(f"API Error ({error_code}): {error_message}")
        return {}

    return response_data


@st.cache_data(show_spinner=False)
def get_user(user_id: str) -> Dict[str, Any]:
    response = make_api_request("GET", f"/users/{user_id}")
    return response.get("data", {})


@st.cache_data(show_spinner=False)
def get_conversations(
    page: int = 1,
    limit: int = 20,
    include_messages: bool = False,
    latest_messages: int = 3,
) -> Dict[str, Any]:
    """Get paginated conversations with optional message inclusion"""
    endpoint = f"/conversations/?page={page}&limit={limit}"
    if include_messages:
        endpoint += f"&include=messages&latestMessages={latest_messages}"
    response = make_api_request("GET", endpoint)
    return response


@st.cache_data(show_spinner=False)
def get_messages(
    conversation_id: str,
    page: int = 1,
    limit: int = 10,
    order_by: str = "createdAt",
    order_direction: str = "desc",
) -> Dict[str, Any]:
    """Get paginated conversation messages"""
    endpoint = (
        f"/conversations/{conversation_id}/messages"
        f"?page={page}&limit={limit}&orderBy={order_by}&orderDirection={order_direction}&include=feedback"
    )
    response = make_api_request("GET", endpoint)
    return response  # Returns full response with meta and items


@st.cache_data(show_spinner=False)
def get_user_messages_paginated(
    page: int = 1,
    limit: int = 10,
    order_by: str = "createdAt",
    order_direction: str = "asc",
) -> Dict[str, Any]:
    """Get paginated user messages"""
    response = make_api_request(
        "GET",
        f"/messages/?page={page}&limit={limit}&orderBy={order_by}&orderDirection={order_direction}",
    )
    return response  # Returns full response with meta and items


def get_feedback(message_id: str) -> Optional[Dict[str, Any]]:
    """Get feedback for a specific message (1-1 relationship)"""
    response = make_api_request("GET", f"/messages/{message_id}/feedbacks")
    if not response:
        return None

    data = response.get("data")

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item

    return None


def render_login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("# ChatBot")
        st.markdown("### Welcome back!")

        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

        with tab1:
            with st.form("login_form"):
                st.markdown("#### Sign in to your account")
                email = st.text_input("Email", placeholder="Enter your email")
                password = st.text_input(
                    "Password", type="password", placeholder="Enter your password"
                )

                if st.form_submit_button("Sign In", use_container_width=True):
                    auth_response = make_api_request(
                        "POST", "/auth/login", {"email": email, "password": password}
                    )
                    if auth_response and "data" in auth_response:
                        st.session_state.auth_token = auth_response["data"][
                            "accessToken"
                        ]
                        st.session_state.current_user_id = auth_response["data"][
                            "userId"
                        ]
                        st.session_state.show_login = False
                        st.success("Signed in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

        with tab2:
            with st.form("signup_form"):
                st.markdown("#### Create a new account")
                username = st.text_input("Username", placeholder="Choose a username")
                email = st.text_input("Email", placeholder="Enter your email")
                password = st.text_input(
                    "Password", type="password", placeholder="Create a password"
                )
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Confirm your password",
                )

                if st.form_submit_button("Create Account", use_container_width=True):
                    if not username or not email or not password:
                        st.error("Please fill out all fields")
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        user_data = {
                            "username": username,
                            "email": email,
                            "password": password,
                        }
                        result = make_api_request("POST", "/auth/signup", user_data)
                        if result:
                            auth_response = make_api_request(
                                "POST",
                                "/auth/login",
                                {"email": email, "password": password},
                            )
                            if auth_response and "data" in auth_response:
                                st.session_state.auth_token = auth_response["data"][
                                    "accessToken"
                                ]
                                st.session_state.current_user_id = auth_response[
                                    "data"
                                ]["userId"]
                                st.session_state.show_login = False
                                st.success(
                                    "Account created and signed in successfully!"
                                )
                                st.rerun()
                            else:
                                st.success("Account created! Please sign in.")
                                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def render_conversation_sidebar():
    with st.sidebar:
        st.markdown("### Conversations")

        if st.button("New Chat", use_container_width=True):
            st.session_state.current_conversation_id = "pending_new"
            reset_conversation_state()
            st.rerun()

        if st.button("Manage Conversations", use_container_width=True):
            st.session_state.show_conversation_manager = True
            st.rerun()

        if st.button("Instructions", use_container_width=True):
            st.session_state.show_instructions = True
            st.rerun()

        st.divider()

        if (
            not st.session_state.conversations_list
            and st.session_state.current_user_id
            and st.session_state.auth_token
        ):
            conversations_response = get_conversations(include_messages=False)
            if conversations_response and conversations_response.get("data"):
                st.session_state.conversations_list = conversations_response["data"][
                    "items"
                ]

        if st.session_state.conversations_list:
            sorted_conversations = sorted(
                st.session_state.conversations_list,
                key=lambda x: parser.parse(x.get("createdAt")),
                reverse=True,
            )

            for conv in sorted_conversations:
                is_active = conv["id"] == st.session_state.current_conversation_id

                # Show only conversation title in sidebar
                display_title = conv["title"]

                if st.button(
                    display_title,
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    if conv["id"] != st.session_state.current_conversation_id:
                        st.session_state.current_conversation_id = conv["id"]
                        reset_conversation_state()
                        st.rerun()

        # Document upload and list section for selected conversation
        render_upload_section()
        render_document_list()

        st.divider()

        if st.session_state.current_user_id:
            user = get_user(st.session_state.current_user_id)
            if user:
                st.markdown(f"**User: {user['username']}**")
                if st.button("Sign Out", use_container_width=True):
                    st.session_state.current_user_id = None
                    st.session_state.current_conversation_id = None
                    reset_conversation_state()
                    st.session_state.conversations_list = []
                    st.session_state.auth_token = None
                    st.session_state.show_login = True
                    st.session_state.pending_image_attachments = []
                    st.session_state.message_image_thumbnails = {}
                    st.rerun()


def render_instructions_modal():
    """Render instructions modal with persona editing support."""
    if not st.session_state.show_instructions:
        return

    conversation_id = st.session_state.get("current_conversation_id")
    is_new_conversation = conversation_id == "pending_new"
    has_conversation = conversation_id not in (None, "pending_new")

    if st.session_state.persona_editor_pending:
        st.session_state.persona_editor_value = (
            st.session_state.persona_editor_pending_value or ""
        )
        st.session_state.persona_editor_pending = False

    current_conv: Optional[Dict[str, Any]] = None
    if has_conversation:
        current_conv = next(
            (
                conv
                for conv in st.session_state.conversations_list
                if conv.get("id") == conversation_id
            ),
            None,
        )
        if current_conv is None:
            response = make_api_request("GET", f"/conversations/{conversation_id}")
            if response and response.get("data"):
                current_conv = response["data"]

    if st.session_state.persona_editor_origin != conversation_id:
        if has_conversation and current_conv:
            initial_value = current_conv.get("personaPrompt") or ""
        elif is_new_conversation:
            initial_value = st.session_state.get("pending_persona_prompt", "")
        else:
            initial_value = ""
        st.session_state.persona_editor_origin = conversation_id
        st.session_state.persona_editor_value = initial_value or ""

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("### Conversation Instructions")

        if conversation_id is None:
            st.info(
                "Select a conversation or click **New Chat** to configure instructions."
            )
            if st.button("Close", use_container_width=True):
                st.session_state.show_instructions = False
                st.session_state.persona_editor_origin = None
                st.rerun()
            return

        if has_conversation:
            title = current_conv.get("title") if current_conv else "Conversation"
            st.caption(
                f"Assistant responses in **{title}** will follow these instructions."
            )
            if current_conv is None:
                st.warning(
                    "Conversation details are unavailable. The instructions shown may be out of date."
                )
        else:
            st.caption(
                "These instructions will be applied when you send the first message in this chat."
            )

        st.write(
            "Describe how the assistant should behave. This field is optional and limited to "
            f"{_MAX_PERSONA_LENGTH} characters."
        )

        if PERSONA_TEMPLATES:
            with st.expander("Need inspiration?", expanded=False):
                template_cols = st.columns(len(PERSONA_TEMPLATES))
                for idx, (label, template) in enumerate(PERSONA_TEMPLATES.items()):
                    if template_cols[idx].button(label, key=f"persona_template_{idx}"):
                        st.session_state.persona_editor_value = template[
                            :_MAX_PERSONA_LENGTH
                        ]

        st.text_area(
            "Custom persona",
            key="persona_editor_value",
            height=200,
            placeholder="Describe how the AI should behave (optional)",
        )

        current_value = st.session_state.get("persona_editor_value", "")
        char_count = len(current_value)
        exceeds_limit = char_count > _MAX_PERSONA_LENGTH
        st.caption(f"{char_count}/{_MAX_PERSONA_LENGTH} characters")
        if exceeds_limit:
            st.error(
                "Persona exceeds the 2000 character limit. Please shorten it before saving."
            )

        action_cols = st.columns([2, 2, 1])

        with action_cols[0]:
            if is_new_conversation:
                if st.button(
                    "Apply to New Chat",
                    use_container_width=True,
                    disabled=exceeds_limit,
                ):
                    sanitized = normalize_persona_input(current_value)
                    st.session_state.pending_persona_prompt = sanitized
                    st.session_state.persona_editor_pending_value = sanitized
                    st.session_state.persona_editor_pending = True
                    st.session_state.persona_feedback = (
                        "Persona saved for the next new conversation."
                        if sanitized
                        else "Persona cleared for the next new conversation."
                    )
                    st.session_state.show_instructions = False
                    st.session_state.persona_editor_origin = None
                    st.rerun()
            else:
                if st.button(
                    "Save Persona",
                    use_container_width=True,
                    disabled=exceeds_limit,
                ):
                    sanitized = normalize_persona_input(current_value)
                    payload = {"personaPrompt": sanitized or None}
                    response = make_api_request(
                        "PATCH", f"/conversations/{conversation_id}", payload
                    )
                    if response and response.get("data"):
                        updated = response["data"]
                        st.session_state.persona_editor_pending_value = (
                            updated.get("personaPrompt") or ""
                        )
                        st.session_state.persona_editor_pending = True
                        st.session_state.persona_feedback = (
                            "Persona updated for this conversation."
                        )
                        refresh_conversations_list()
                        st.session_state.show_instructions = False
                        st.session_state.persona_editor_origin = None
                        st.rerun()
                    else:
                        st.error("Unable to update persona. Please try again.")

        with action_cols[1]:
            if is_new_conversation:
                if st.button("Clear", use_container_width=True):
                    st.session_state.persona_editor_pending_value = ""
                    st.session_state.persona_editor_pending = True
                    st.session_state.pending_persona_prompt = ""
                    st.rerun()
            else:
                if st.button("Clear Persona", use_container_width=True):
                    response = make_api_request(
                        "PATCH",
                        f"/conversations/{conversation_id}",
                        {"personaPrompt": None},
                    )
                    if response and response.get("data"):
                        refresh_conversations_list()
                        st.session_state.persona_editor_pending_value = ""
                        st.session_state.persona_editor_pending = True
                        st.session_state.persona_feedback = (
                            "Persona removed for this conversation."
                        )
                        st.session_state.show_instructions = False
                        st.session_state.persona_editor_origin = None
                        st.rerun()
                    else:
                        st.error("Unable to clear persona. Please try again.")

        with action_cols[2]:
            if st.button("Close", use_container_width=True):
                st.session_state.show_instructions = False
                st.session_state.persona_editor_origin = None
                st.rerun()


def render_conversation_manager():
    if st.session_state.show_conversation_manager:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("### Conversation Manager")

            if st.session_state.persona_feedback:
                st.success(st.session_state.persona_feedback)
                st.session_state.persona_feedback = None

            # Load conversations with messages for the manager
            if st.session_state.current_user_id:
                with st.spinner("Loading conversations with messages..."):
                    manager_conversations_response = get_conversations(
                        include_messages=True, latest_messages=3
                    )
                    if (
                        manager_conversations_response
                        and manager_conversations_response.get("data")
                    ):
                        manager_conversations = manager_conversations_response["data"][
                            "items"
                        ]
                    else:
                        manager_conversations = []
            else:
                manager_conversations = []

            search_term = st.text_input(
                "Search conversations:", placeholder="Type to search..."
            )

            if manager_conversations:
                filtered_convs = manager_conversations

                if search_term:
                    filtered_convs = []
                    for conv in manager_conversations:
                        # Check if conversation title contains search term
                        if search_term.lower() in conv.get("title", "").lower():
                            if conv not in filtered_convs:
                                filtered_convs.append(conv)
                            continue

                        # Check messages that are already included in the conversation
                        messages = conv.get("messages", [])
                        if messages:
                            # Use pre-loaded messages to avoid API call
                            for msg in messages:
                                if (
                                    search_term.lower()
                                    in msg.get("content", "").lower()
                                ):
                                    if conv not in filtered_convs:
                                        filtered_convs.append(conv)
                                    break

                if filtered_convs:
                    for conv in filtered_convs:
                        with st.expander(
                            f"Conversation - {conv['title']}", expanded=False
                        ):
                            if conv.get("messages"):
                                for msg in conv["messages"]:
                                    sender_value = msg.get("sender")
                                    sender_icon = (
                                        "[User]"
                                        if sender_value in (1, "user", "USER", "User")
                                        else "[Assistant]"
                                    )
                                    sender_name = (
                                        "user"
                                        if sender_value in (1, "user", "USER", "User")
                                        else "assistant"
                                    )
                                    preview = msg.get("content", "")
                                    st.markdown(
                                        f"{sender_icon} **{sender_name}:** {preview[:100]}..."
                                    )
                            else:
                                st.markdown("No messages available")

                            col_open, col_delete = st.columns(2)
                            with col_open:
                                if st.button("Select", key=f"open_{conv['id']}"):
                                    st.session_state.current_conversation_id = conv[
                                        "id"
                                    ]
                                    reset_conversation_state()
                                    st.session_state.show_conversation_manager = False
                                    st.rerun()

                            with col_delete:
                                if st.button(
                                    "Delete",
                                    key=f"delete_{conv['id']}",
                                    type="secondary",
                                ):
                                    result = make_api_request(
                                        "DELETE", f"/conversations/{conv['id']}"
                                    )
                                    if result:
                                        st.session_state.conversations_list = []
                                        if (
                                            st.session_state.current_conversation_id
                                            == conv["id"]
                                        ):
                                            st.session_state.current_conversation_id = (
                                                None
                                            )
                                            reset_conversation_state()
                                        get_messages.clear()
                                        refresh_conversations_list()
                                        st.success(f"Deleted '{conv['title']}'")
                                        st.rerun()
                else:
                    st.info("No conversations found matching your search.")
            else:
                st.info("No conversations available.")

            if st.button("Close Manager", use_container_width=True):
                st.session_state.show_conversation_manager = False
                st.rerun()


def render_chat_interface():
    conversation_id = st.session_state.get("current_conversation_id")
    user_id = st.session_state.get("current_user_id")

    if not st.session_state.conversations_list and user_id:
        # Load conversations without messages (messages only needed in conversation manager)
        conversations_response = get_conversations(include_messages=False)
        if conversations_response and conversations_response.get("data"):
            st.session_state.conversations_list = conversations_response["data"][
                "items"
            ]

    if st.session_state.persona_feedback:
        st.success(st.session_state.persona_feedback)
        st.session_state.persona_feedback = None

    def load_messages_page(page: int, *, show_spinner: bool = False) -> None:
        conv_id = st.session_state.get("current_conversation_id")
        if not conv_id or conv_id == "pending_new":
            return

        fetch_page = lambda: get_messages(
            conv_id,
            page=page,
            limit=10,
            order_direction="desc",
        )

        if show_spinner:
            with st.spinner("Loading messages..."):
                response = fetch_page()
        else:
            response = fetch_page()

        if response and response.get("data"):
            data = response["data"]
            items = data.get("items", [])
            meta = data.get("meta", {})

            attachments_state = st.session_state.setdefault(
                "message_image_thumbnails", {}
            )

            existing_messages = {msg["id"]: msg for msg in st.session_state.messages}
            for item in items:
                msg_id = item.get("id")
                if msg_id:
                    metadata = item.get("messageMetadata") or {}
                    attachments = metadata.get("attachments") or []
                    normalized_attachments: List[Dict[str, str]] = []

                    for att in attachments:
                        if not isinstance(att, dict):
                            continue
                        data_b64 = att.get("data")
                        if isinstance(data_b64, str):
                            data_b64 = data_b64.strip()
                        else:
                            data_b64 = None

                        url_value = att.get("url")
                        if isinstance(url_value, str):
                            url_value = url_value.strip()
                        else:
                            url_value = None

                        if not data_b64 and not url_value:
                            continue

                        normalized_attachments.append(
                            {
                                "token": att.get("token", str(uuid.uuid4())),
                                "name": att.get("name")
                                or f"Attachment {len(normalized_attachments) + 1}",
                                "mime": att.get("mime", "image/png"),
                                "data": data_b64,
                                "url": url_value,
                            }
                        )

                    key = str(msg_id)
                    if normalized_attachments:
                        attachments_state[key] = normalized_attachments
                    else:
                        attachments_state.pop(key, None)

                    existing_messages[msg_id] = item

            def sort_key(message: Dict[str, Any]):
                timestamp = message.get("createdAt")
                if not timestamp:
                    return datetime.min
                try:
                    return parser.isoparse(timestamp)
                except Exception:
                    return datetime.min

            sorted_messages = sorted(existing_messages.values(), key=sort_key)

            st.session_state.messages = sorted_messages
            st.session_state.conversation_messages_meta = meta

            # API returns camelCase keys (currentPage, lastPage)
            current_page = meta.get("currentPage", meta.get("current_page", page))
            last_page = meta.get("lastPage", meta.get("last_page", current_page))
            st.session_state.conversation_messages_page = current_page
            st.session_state.has_more_messages = current_page < last_page
        else:
            st.session_state.conversation_messages_page = max(
                st.session_state.conversation_messages_page, page
            )
            st.session_state.has_more_messages = False

    if conversation_id and conversation_id != "pending_new":
        if st.session_state.conversation_messages_page == 0:
            load_messages_page(1, show_spinner=True)

    current_conv = next(
        (c for c in st.session_state.conversations_list if c["id"] == conversation_id),
        None,
    )

    if current_conv:
        st.markdown(f"# {current_conv['title']}")
        active_persona = current_conv.get("personaPrompt")
        if active_persona:
            st.caption(f"Instructions active: {persona_preview(active_persona)}")
    elif conversation_id == "pending_new":
        st.markdown("# New Chat - Start typing to begin!")
        queued_persona = st.session_state.get("pending_persona_prompt", "")
        if queued_persona:
            st.caption(f"Instructions queued: {persona_preview(queued_persona)}")
    else:
        st.markdown("# Welcome! Select a conversation to view messages")

    if conversation_id and conversation_id not in (None, "pending_new"):
        if st.session_state.has_more_messages:
            if st.button("Load older messages", key="load_more_messages"):
                next_page = st.session_state.conversation_messages_page + 1
                load_messages_page(next_page, show_spinner=True)
        elif st.session_state.conversation_messages_page > 0:
            st.caption("All caught up - showing the entire thread.")

    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-messages-container"><div class="chat-messages-scroll">',
        unsafe_allow_html=True,
    )

    def format_time(iso_string: str) -> str:
        try:
            dt = parser.isoparse(iso_string)
            now = datetime.now(dt.tzinfo)
            if dt.date() == now.date():
                return dt.strftime("%H:%M")
            if now - timedelta(days=7) < dt <= now:
                return dt.strftime("%a %H:%M")
            return dt.strftime("%b %d, %Y %H:%M")
        except Exception:
            return iso_string

    messages_to_display = (
        st.session_state.messages
        if conversation_id and conversation_id != "pending_new"
        else []
    )

    if not messages_to_display and conversation_id not in (None, "pending_new"):
        st.markdown(
            "<div style='text-align:center; color:#94a3b8;'>No messages yet - send the first one!</div>",
            unsafe_allow_html=True,
        )

    for msg in messages_to_display:
        sender_value = msg.get("sender")
        is_user_message = sender_value in (1, "user", "USER", "User")

        if is_user_message:
            # User message with no feedback option
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px 0; align-items: flex-start; gap: 10px;">
                    <div class="user-message">
                        {sanitize_message_content(msg.get("content"))}
                        <div class="message-timestamp">You - {format_time(msg.get("createdAt", "now"))}</div>
                    </div>
                    <div style="border: 2px solid #007bff; color: #007bff; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">U</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_message_attachments(str(msg.get("id", "")), align="right")
        else:
            # Assistant message with feedback option
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0; align-items: flex-start; gap: 10px;">
                    <div style="border: 2px solid #28a745; color: #28a745; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px; flex-shrink: 0;">AI</div>
                    <div style="display: flex; flex-direction: column;">
                        <div class="bot-message" style="margin: 0;">
                            {sanitize_message_content(msg.get("content"))}
                            <div class="message-timestamp">Assistant - {format_time(msg.get("createdAt", "now"))}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Render agent-sent images (from Tavily or Image Generator)
            render_agent_images(
                msg.get("messageMetadata", {}),
                str(msg.get("id", "")),
            )

            # Add feedback section below the message
            with st.container():
                st.markdown(
                    '<div style="margin-left: 45px;">', unsafe_allow_html=True
                )  # Align with message content

                with st.popover("Feedback", help="Give feedback"):
                    st.markdown("### Provide Feedback")

                    with st.form(f"feedback_form_{msg['id']}"):
                        rating = st.selectbox("Rating", [1, 2, 3, 4, 5], index=4)
                        comment = st.text_area("Comment (optional)", height=100)

                        if st.form_submit_button(
                            "Submit Feedback", use_container_width=True
                        ):
                            feedback_data = {
                                "messageId": msg["id"],
                                "rating": rating,
                                "comment": comment,
                            }
                            response = make_api_request(
                                "POST",
                                f"/messages/{msg['id']}/feedbacks",
                                feedback_data,
                            )
                            if response:
                                st.success("Feedback submitted!")
                                get_messages.clear()
                                st.session_state.conversation_messages_page = 0
                                st.rerun()

                # Show existing feedback directly beneath the bot message
                feedback = msg.get("feedback")
                if isinstance(feedback, dict) and feedback:
                    rating = feedback.get("rating")
                    comment_text = feedback.get("comment")

                    if rating is not None:
                        st.markdown(f"Rating: {rating}/5")

                    if comment_text:
                        preview = comment_text[:100]
                        suffix = "..." if len(comment_text) > 100 else ""
                        st.markdown(f'Comment: *"{preview}{suffix}"*')

                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    if conversation_id:
        st.markdown(
            '<div class="chat-input-container">',
            unsafe_allow_html=True,
        )

        if st.session_state.pending_image_attachments:
            render_pending_attachment_preview()

        file_uploader_key = (
            f"chat_image_uploader_{conversation_id}" if conversation_id else None
        )

        message_input_key = (
            f"message_input_{conversation_id}" if conversation_id else "message_input"
        )
        message_clear_flag_key = f"{message_input_key}_clear_flag"

        if st.session_state.pop(message_clear_flag_key, False):
            st.session_state[message_input_key] = ""

        with st.form("message_form", clear_on_submit=False):

            message_col, button_col = st.columns([5, 1])
            with message_col:
                if st.session_state.show_attachment_uploader and file_uploader_key:
                    uploaded_files = st.file_uploader(
                        "Attach images",
                        type=["png", "jpg", "jpeg", "gif", "webp"],
                        accept_multiple_files=True,
                        key=file_uploader_key,
                        help=f"You can attach up to {_MAX_IMAGE_ATTACHMENTS} images.",
                    )
                    if uploaded_files:
                        _handle_new_image_attachments(uploaded_files)
                message_content = st.text_area(
                    "Message",
                    placeholder="Type your message here...",
                    height=80,
                    label_visibility="collapsed",
                    key=message_input_key,
                )
            with button_col:
                send_button = st.form_submit_button("Send", use_container_width=True)
                attach_button = st.form_submit_button(
                    "Attach Images", use_container_width=True, type="secondary"
                )

            if attach_button:
                st.session_state.show_attachment_uploader = not st.session_state.get(
                    "show_attachment_uploader", False
                )
                st.rerun()

            if send_button:
                pending_attachments = list(
                    st.session_state.get("pending_image_attachments", [])
                )
                stripped_message = message_content.strip()

                if not stripped_message and not pending_attachments:
                    st.warning("Please enter a message or attach images before sending.")
                else:
                    message_to_send = stripped_message or _format_image_only_message(
                        pending_attachments
                    )

                    if conversation_id == "pending_new":
                        saved_attachments = list(pending_attachments)
                        conversation_title_source = stripped_message or message_to_send
                        conversation_data = {
                            "title": (
                                conversation_title_source[:50] + "..."
                                if len(conversation_title_source) > 50
                                else conversation_title_source
                            )
                        }
                        pending_persona = st.session_state.get(
                            "pending_persona_prompt", ""
                        )
                        persona_payload = normalize_persona_input(pending_persona)
                        if persona_payload:
                            conversation_data["personaPrompt"] = persona_payload
                        conv_response = make_api_request(
                            "POST", "/conversations/", conversation_data
                        )
                        if conv_response and conv_response.get("data"):
                            new_conversation = conv_response["data"]
                            st.session_state.current_conversation_id = new_conversation[
                                "id"
                            ]
                            refresh_conversations_list(
                                fallback_conversation=new_conversation
                            )
                            reset_conversation_state()
                            st.session_state.pending_image_attachments = (
                                saved_attachments
                            )
                            conversation_id = st.session_state.current_conversation_id
                            pending_attachments = list(saved_attachments)
                        else:
                            st.error("Failed to create conversation")
                            return

                    message_data = {
                        "content": message_to_send,
                        "conversationId": st.session_state.current_conversation_id,
                    }

                    # Include attachments if present
                    if pending_attachments:
                        message_data["attachments"] = pending_attachments

                    with st.spinner("Thinking..."):
                        response = make_api_request("POST", "/messages/", message_data)

                    if response and response.get("data"):
                        st.session_state.pending_image_attachments = []
                        st.session_state[message_clear_flag_key] = True
                        get_messages.clear()
                        refresh_conversations_list()
                        reset_conversation_state()
                        st.session_state.show_attachment_uploader = False
                        load_messages_page(1)
                        st.rerun()
                    else:
                        st.error("Failed to send message")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    if (
        st.session_state.show_login
        or not st.session_state.current_user_id
        or not st.session_state.auth_token
    ):
        render_login_page()
        return

    render_conversation_sidebar()

    if st.session_state.show_instructions:
        render_instructions_modal()
        return

    if st.session_state.show_conversation_manager:
        render_conversation_manager()
        return

    render_chat_interface()


if __name__ == "__main__":
    main()
