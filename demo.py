import base64
import mimetypes
import uuid
import json

import streamlit as st  # type: ignore
import requests
import html
import re
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple
from upload_support import delete_document, get_uploaded_documents, upload_document
from datetime import datetime, timedelta
from dateutil import parser
import markdown as _markdown  # type: ignore

API_BASE_URL = "http://localhost:8000"

_MAX_PERSONA_LENGTH = 8000
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

COLORS = {
    "primary": "#3b82f6",
    "primary_dark": "#1d4ed8",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "neutral": "#64748b",
    "bg_light": "#f8fafc",
    "bg_card": "#ffffff",
    "border": "#e2e8f0",
}

CONVERSATION_MANAGER_DIALOG_KEY = "conversation_manager_dialog"

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
    "a",
    "span",
    "div",
}.union(_SELF_CLOSING_TAGS)

APP_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .message-bubble {
        border-radius: 16px;
        padding: 14px 18px;
        margin: 12px 0;
        max-width: 75%;
        width: fit-content;
        min-width: 120px;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        animation: slideIn 0.2s ease-out;
    }
    
    .message-bubble-short {
        max-width: 40%;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-bubble {
        background: #ffffff;
        color: #1f2937;
        border: 1px solid #e2e8f0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    .message-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    .message-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 13px;
    }
    
    .user-avatar {
        background: white;
        color: #3b82f6;
        border: 2px solid #3b82f6;
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .message-time {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-left: auto;
    }
    
    /* Message content wrapper - contains the markdown rendered content */
    .message-content-wrapper {
        line-height: 1.6;
    }
    
    .message-content-wrapper > div[data-testid="stMarkdownContainer"] {
        margin: 0;
        padding: 0;
    }
    
    .message-content-wrapper p {
        margin: 0.5em 0;
        line-height: 1.6;
    }
    
    .message-content-wrapper p:first-child {
        margin-top: 0;
    }
    
    .message-content-wrapper p:last-child {
        margin-bottom: 0;
    }
    
    /* Code blocks */
    .message-bubble code {
        background: rgba(0,0,0,0.05);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    
    .user-bubble code {
        background: rgba(255,255,255,0.2);
    }
    
    /* Paragraphs and Lists */
    .message-bubble p {
        margin: 0.5em 0;
        line-height: 1.6;
    }
    
    .message-bubble p:first-child {
        margin-top: 0;
    }
    
    .message-bubble p:last-child {
        margin-bottom: 0;
    }
    
    .message-bubble ul,
    .message-bubble ol {
        margin: 0.75em 0;
        padding-left: 1.5em;
        line-height: 1.6;
    }
    
    .message-bubble ul:first-child,
    .message-bubble ol:first-child {
        margin-top: 0;
    }
    
    .message-bubble ul:last-child,
    .message-bubble ol:last-child {
        margin-bottom: 0;
    }
    
    .message-bubble li {
        margin: 0.35em 0;
        line-height: 1.6;
    }
    
    .message-bubble li > p {
        margin: 0.25em 0;
    }
    
    .message-bubble pre {
        background: rgba(0,0,0,0.05);
        padding: 12px;
        border-radius: 6px;
        overflow-x: auto;
        margin: 0.75em 0;
        line-height: 1.5;
    }
    
    .user-bubble pre {
        background: rgba(255,255,255,0.15);
    }
    
    .message-bubble pre code {
        background: transparent;
        padding: 0;
    }
    
    .message-bubble blockquote {
        border-left: 3px solid #cbd5e1;
        padding-left: 1em;
        margin: 0.75em 0;
        color: #64748b;
    }
    
    .user-bubble blockquote {
        border-left-color: rgba(255,255,255,0.5);
        color: rgba(255,255,255,0.9);
    }
    
    /* Links */
    .message-bubble a {
        color: #2563eb;
        text-decoration: underline;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    
    .message-bubble a:hover {
        color: #1d4ed8;
        text-decoration: underline;
    }
    
    .user-bubble a {
        color: #e0f2fe;
        text-decoration: underline;
    }
    
    .user-bubble a:hover {
        color: #ffffff;
    }
    
    /* Attachments */
    .message-attachments-wrapper {
        display: flex;
        max-width: 75%;
        margin: 4px 0 12px;
    }
    
    .message-attachments-wrapper.align-right {
        margin-left: auto;
        justify-content: flex-end;
    }
    
    .message-attachments-wrapper.align-left {
        margin-right: auto;
        justify-content: flex-start;
    }
    
    .message-attachments {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 0;
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
    
    .message-attachments .attachment-thumb img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    
    .attachment-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
        gap: 8px;
        margin-top: 12px;
    }
    
    .attachment-item {
        position: relative;
        border-radius: 8px;
        overflow: hidden;
        aspect-ratio: 1;
        cursor: pointer;
        transition: transform 0.2s;
        border: 2px solid rgba(0,0,0,0.1);
    }
    
    .attachment-item:hover {
        transform: scale(1.05);
    }
    
    .attachment-item img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    /* Input area */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 12px !important;
        font-size: 0.95rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    }
    
    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        border-radius: 8px;
        background: #f8fafc;
        font-weight: 500;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-processing {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-ready {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-failed {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Citation styling */
    .citation-document {
        margin-bottom: 12px;
        padding: 12px;
        border-radius: 8px;
        background-color: rgba(59, 130, 246, 0.1);
        border: 2px solid rgba(59, 130, 246, 0.3);
    }
    
    .citation-chunk {
        margin-left: 20px;
        margin-bottom: 6px;
        padding: 6px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    
    .citation-score-high {
        color: #22c55e;
    }
    
    .citation-score-medium {
        color: #f59e0b;
    }
    
    .citation-score-low {
        color: #ef4444;
    }
    
    /* Clickable citation button styles */
    .stButton button[data-testid*="cite_"] {
        padding: 4px 8px;
        font-size: 0.85em;
        min-height: 32px;
    }
    
    .stButton button[data-testid*="cite_"]:hover {
        transform: scale(1.05);
        transition: transform 0.2s ease-in-out;
    }
    
    /* Image thumbnail styles in citations */
    .citation-image-thumb {
        width: 100%;
        border-radius: 4px;
        cursor: pointer;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    
    .citation-image-thumb:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Thinking/Reasoning UI Styles - Gemini/ChatGPT inspired */
    .thinking-container {
        border-left: 3px solid #8b5cf6;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(139, 92, 246, 0.04) 100%);
        border-radius: 0 8px 8px 0;
        font-size: 0.9em;
        color: #6b7280;
    }
    
    .thinking-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: #8b5cf6;
        margin-bottom: 8px;
    }
    
    .thinking-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .thinking-dots {
        display: inline-flex;
        gap: 4px;
    }
    
    .thinking-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #8b5cf6;
        animation: thinking-pulse 1.4s infinite ease-in-out;
    }
    
    .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
    .thinking-dot:nth-child(3) { animation-delay: 0s; }
    
    @keyframes thinking-pulse {
        0%, 80%, 100% {
            transform: scale(0.6);
            opacity: 0.4;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .thinking-content {
        line-height: 1.6;
        white-space: pre-wrap;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Collapsed thinking expander styles */
    .thinking-expander-header {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #8b5cf6;
        font-weight: 500;
        cursor: pointer;
    }
    
    .thinking-expander-header:hover {
        color: #7c3aed;
    }
    
    /* Image Lightbox Modal */
    .image-lightbox-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.85);
        z-index: 9999;
        justify-content: center;
        align-items: center;
        cursor: zoom-out;
    }
    
    .image-lightbox-overlay.active {
        display: flex;
    }
    
    .image-lightbox-content {
        max-width: 90%;
        max-height: 90%;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .image-lightbox-close {
        position: absolute;
        top: 20px;
        right: 30px;
        color: white;
        font-size: 32px;
        font-weight: bold;
        cursor: pointer;
        z-index: 10000;
    }
    
    .image-lightbox-close:hover {
        color: #f87171;
    }
    
    /* Image thumbnail hover effects */
    .img-thumb {
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        cursor: pointer;
    }
    
    .img-thumb:hover {
        transform: scale(1.08);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        filter: brightness(1.05);
    }
</style>
"""

# JavaScript for image lightbox with event delegation for Streamlit compatibility
IMAGE_LIGHTBOX_JS = """
<div id="imageLightbox" class="image-lightbox-overlay">
    <span class="image-lightbox-close">&times;</span>
    <img id="lightboxImage" class="image-lightbox-content" src="" alt="Full size image">
</div>
<script>
(function() {
    // Ensure we only initialize once
    if (window._lightboxInitialized) return;
    window._lightboxInitialized = true;
    
    function openImageLightbox(src) {
        var lightbox = document.getElementById('imageLightbox');
        var img = document.getElementById('lightboxImage');
        if (lightbox && img) {
            img.src = src;
            lightbox.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    }
    
    function closeImageLightbox() {
        var lightbox = document.getElementById('imageLightbox');
        if (lightbox) {
            lightbox.classList.remove('active');
            document.body.style.overflow = 'auto';
        }
    }
    
    // Make functions globally available
    window.openImageLightbox = openImageLightbox;
    window.closeImageLightbox = closeImageLightbox;
    
    // Event delegation for image thumbnails - handles dynamically added elements
    document.addEventListener('click', function(e) {
        var target = e.target;
        
        // Check if clicked on an img-thumb image
        if (target.classList.contains('img-thumb')) {
            e.preventDefault();
            e.stopPropagation();
            openImageLightbox(target.src);
            return;
        }
        
        // Check if clicked on lightbox overlay or close button
        var lightbox = document.getElementById('imageLightbox');
        if (lightbox && lightbox.classList.contains('active')) {
            if (target.classList.contains('image-lightbox-overlay') || 
                target.classList.contains('image-lightbox-close')) {
                closeImageLightbox();
            }
        }
    }, true);
    
    // Escape key to close
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closeImageLightbox();
    });
</script>
"""


def safe_api_call(
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    error_message: str = "API request failed",
    success_message: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Wrapper for API calls with consistent error handling and toast notifications.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        endpoint: API endpoint path
        data: Optional request payload
        error_message: Message to show on error
        success_message: Optional message to show on success

    Returns:
        Response data dict or None on error
    """
    try:
        response_data = make_api_request(method, endpoint, data)
        if response_data and response_data.get("success"):
            if success_message:
                st.toast(success_message, icon="✅")
            return response_data
        else:
            st.toast(error_message, icon="❌")
            return None
    except Exception as e:
        st.toast(f"{error_message}: {str(e)}", icon="❌")
        return None


def format_conversation_title(title: str, max_length: int = 40) -> str:
    """
    Format conversation title with truncation and ellipsis.

    Args:
        title: Original title
        max_length: Maximum length before truncation

    Returns:
        Formatted title
    """
    if not title or title == "New Conversation":
        return "New Conversation"
    if len(title) > max_length:
        return title[: max_length - 3] + "..."
    return title


def render_status_badge(status: str) -> str:
    """
    Render a status badge with consistent styling.

    Args:
        status: Status string (processing, ready, failed, etc.)

    Returns:
        HTML for the status badge
    """
    status_lower = status.lower()
    badge_class = f"status-{status_lower}"

    status_icons = {
        "processing": "⏳",
        "ready": "✅",
        "failed": "❌",
        "pending": "⏸️",
        "active": "🟢",
        "inactive": "⚪",
    }

    icon = status_icons.get(status_lower, "")
    display_text = status.replace("_", " ").title()

    return f'<span class="status-badge {badge_class}">{icon} {display_text}</span>'


def format_timestamp(timestamp: str) -> str:
    """
    Format timestamp for display.

    Args:
        timestamp: ISO format timestamp string

    Returns:
        Formatted timestamp string
    """
    try:
        dt = parser.isoparse(timestamp)
        now = datetime.now(dt.tzinfo)
        diff = now - dt

        if diff < timedelta(minutes=1):
            return "Just now"
        elif diff < timedelta(hours=1):
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m ago"
        elif diff < timedelta(days=1):
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"
        elif diff < timedelta(days=7):
            days = diff.days
            return f"{days}d ago"
        else:
            return dt.strftime("%b %d, %Y")
    except Exception:
        return timestamp


def get_agent_display_name(agent: str) -> str:
    """
    Get display-friendly name for an agent.

    Args:
        agent: Agent identifier

    Returns:
        Display name
    """
    agent_names = {
        "chat_agent": "Chat Agent",
        "rag_agent": "RAG Agent",
        "search_agent": "Search Agent",
        "image_generator_agent": "Image Generator",
        "router": "Router",
    }
    return agent_names.get(agent, agent.replace("_", " ").title())


def render_conversation_button(
    conversation: Dict[str, Any],
    is_active: bool,
    on_click_callback: Optional[Callable] = None,
) -> None:
    """
    Render a conversation button with consistent styling.

    Args:
        conversation: Conversation data dict
        is_active: Whether this conversation is currently active
        on_click_callback: Optional callback when button is clicked
    """
    title = format_conversation_title(conversation.get("title", "New Conversation"))
    button_type = "primary" if is_active else "secondary"

    if st.button(
        title,
        key=f"conv_{conversation['id']}",
        use_container_width=True,
        type=button_type,
    ):
        if conversation["id"] != st.session_state.current_conversation_id:
            st.session_state.current_conversation_id = conversation["id"]
            st.session_state.active_view = "chat"
            close_conversation_manager()
            reset_conversation_state()
            if on_click_callback:
                on_click_callback()
            st.rerun()


def group_conversations_by_date(
    conversations: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group conversations by date (Today, Yesterday, Last 7 days, Last 30 days, Older).

    Args:
        conversations: List of conversation dicts

    Returns:
        Dict mapping group names to lists of conversations
    """
    now = datetime.now()
    today = now.date()
    yesterday = today - timedelta(days=1)

    groups: Dict[str, List[Dict[str, Any]]] = {
        "Today": [],
        "Yesterday": [],
        "Last 7 days": [],
        "Last 30 days": [],
        "Older": [],
    }

    for conv in conversations:
        try:
            created_at = parser.parse(conv.get("createdAt", ""))
            conv_date = created_at.date()

            if conv_date == today:
                groups["Today"].append(conv)
            elif conv_date == yesterday:
                groups["Yesterday"].append(conv)
            elif (today - conv_date).days <= 7:
                groups["Last 7 days"].append(conv)
            elif (today - conv_date).days <= 30:
                groups["Last 30 days"].append(conv)
            else:
                groups["Older"].append(conv)
        except Exception:
            groups["Older"].append(conv)

    # Return only non-empty groups
    return {name: convs for name, convs in groups.items() if convs}


# ==================== SESSION STATE DEFAULTS ====================

SESSION_STATE_DEFAULTS: Dict[str, Callable[[], Any] | Any] = {
    "current_user_id": lambda: None,
    "current_conversation_id": lambda: None,
    "messages": list,
    "conversations_list": list,
    "conversations_loaded": lambda: False,
    "conversations_last_fetch_params": lambda: None,
    "show_conversation_manager": lambda: False,
    "conversation_manager_visible": lambda: False,
    CONVERSATION_MANAGER_DIALOG_KEY: lambda: False,
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
    "message_chunks": dict,
    "show_attachment_uploader": lambda: False,
    "image_viewer_open": lambda: False,
    "image_viewer_payload": lambda: None,
    "API_BASE_URL": lambda: API_BASE_URL,
    "active_view": lambda: "chat",
    "mcp_tools_list": lambda: None,
    "mcp_servers_list": lambda: None,
    "selected_tool": lambda: None,
    "tool_execution_result": lambda: None,
    "selected_chunk_info": lambda: None,
    "chunk_preview_dialog_key": lambda: False,
    # Planning mode state
    "planning_status": lambda: None,
    "task_plans_list": list,
    "planning_generate_input": str,
    "planning_manual_input": str,
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

        # Handle anchor tags with href attribute
        if tag == "a":
            href = None
            for attr_name, attr_value in attrs:
                if attr_name.lower() == "href":
                    href = attr_value
                    break

            if href:
                # Sanitize href to prevent javascript: and data: URLs
                href_lower = href.lower().strip()
                if href_lower.startswith(("http://", "https://", "/")):
                    escaped_href = html.escape(href, quote=True)
                    self.result.append(
                        f'<a href="{escaped_href}" target="_blank" rel="noopener noreferrer">'
                    )
                    self._tag_stack.append(tag)
                else:
                    # Skip unsafe URLs but keep the text content
                    return
            else:
                # No href, skip the tag but keep the text content
                return
        # Handle span and div tags with class attribute (for math rendering)
        elif tag in ("span", "div"):
            class_attr = None
            for attr_name, attr_value in attrs:
                if attr_name.lower() == "class":
                    class_attr = attr_value
                    break

            # Allow math-related classes
            if class_attr and (
                "katex" in class_attr.lower() or "math" in class_attr.lower()
            ):
                escaped_class = html.escape(class_attr, quote=True)
                self.result.append(f'<{tag} class="{escaped_class}">')
                self._tag_stack.append(tag)
            else:
                self.result.append(f"<{tag}>")
                self._tag_stack.append(tag)
        else:
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

    def handle_data(self, data: str):
        if not data:
            return
        escaped = html.escape(data, quote=False)
        self.result.append(escaped)

    def handle_entityref(self, name: str):
        self.result.append(f"&{name};")

    def handle_charref(self, name: str):
        self.result.append(f"&#{name};")

    def handle_comment(self, data: str):
        # Preserve comments that contain math placeholders
        if data.startswith("MATH_"):
            self.result.append(f"<!--{data}-->")


def _sanitize_rendered_html(html_fragment: str) -> str:
    parser = _SafeHTMLRenderer()
    parser.feed(html_fragment)
    parser.close()
    return "".join(parser.result)


@st.cache_data(show_spinner=False, max_entries=500)
def sanitize_message_content(content: str) -> str:
    """Render limited markdown to HTML while preventing unsafe tags."""
    if not content or not isinstance(content, str):
        return ""

    normalized = (
        html.unescape(content).replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    if not normalized:
        return ""

    # Preserve LaTeX delimiters by temporarily replacing them with unique markers
    math_expressions = {}
    math_counter = 0

    # Preserve display math ($$...$$)
    def replace_display_math(match):
        nonlocal math_counter
        # Use a unique marker that won't be in normal text
        placeholder = f"ӍӐҬҤ_DISPLAY_{math_counter}_ӍӐҬҤ"
        # Store the math content with markers for later rendering
        math_expressions[placeholder] = {
            "type": "display",
            "content": match.group(1).strip(),  # Extract content between $$
        }
        math_counter += 1
        return placeholder

    normalized = re.sub(
        r"\$\$(.+?)\$\$", replace_display_math, normalized, flags=re.DOTALL
    )

    # Preserve inline math ($...$)
    def replace_inline_math(match):
        nonlocal math_counter
        # Use a unique marker that won't be in normal text
        placeholder = f"ӍӐҬҤ_INLINE_{math_counter}_ӍӐҬҤ"
        # Store the math content with markers for later rendering
        math_expressions[placeholder] = {
            "type": "inline",
            "content": match.group(1).strip(),  # Extract content between $
        }
        math_counter += 1
        return placeholder

    normalized = re.sub(r"\$([^$\n|]+?)\$", replace_inline_math, normalized)

    # Remove image markdown (convert to text links)
    normalized = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)", r"\1 (\2)", normalized, flags=re.MULTILINE
    )

    # Ensure proper list formatting for sane_lists extension
    # Add blank lines before lists and between list type transitions
    lines = normalized.split("\n")
    processed_lines = []
    prev_line_type = None  # 'blank', 'unordered', 'ordered', 'text'

    unordered_pattern = re.compile(r"^\s*[-*+]\s+")
    ordered_pattern = re.compile(r"^\s*\d+\.\s+")

    for line in lines:
        current_line_type = None

        if not line.strip():
            current_line_type = "blank"
        elif unordered_pattern.match(line):
            current_line_type = "unordered"
        elif ordered_pattern.match(line):
            current_line_type = "ordered"
        else:
            current_line_type = "text"

        # Insert blank line when transitioning to a list from non-list content
        # or when list type changes
        if current_line_type in ("unordered", "ordered"):
            if prev_line_type not in (None, "blank", current_line_type):
                processed_lines.append("")

        processed_lines.append(line)
        prev_line_type = current_line_type

    normalized = "\n".join(processed_lines)

    # Render markdown to HTML
    rendered = _markdown.markdown(
        normalized,
        extensions=["extra", "sane_lists", "nl2br"],
        output_format="html5",
    )

    # Sanitize HTML
    safe_html = _sanitize_rendered_html(rendered)

    # Restore LaTeX with proper KaTeX delimiters
    # KaTeX will render content between \( \) for inline and \[ \] for display
    for placeholder, math_info in math_expressions.items():
        latex_content = math_info["content"]
        if math_info["type"] == "display":
            # Display math: use \[ \] delimiters
            latex_html = f"\\[{latex_content}\\]"
        else:  # inline
            # Inline math: use \( \) delimiters
            latex_html = f"\\({latex_content}\\)"

        # Replace both the original placeholder and any HTML-escaped version
        safe_html = safe_html.replace(placeholder, latex_html)
        escaped_placeholder = html.escape(placeholder)
        if escaped_placeholder != placeholder:
            safe_html = safe_html.replace(escaped_placeholder, latex_html)

    safe_html = safe_html.replace("<p></p>", "")
    safe_html = safe_html.replace("<p><br /></p>", "")
    safe_html = safe_html.replace("<p><br></p>", "")

    safe_html = re.sub(r"(<br\s*/?>[\s\n]*){3,}", "<br><br>", safe_html)

    safe_html = safe_html.strip()

    return safe_html


def format_time(iso_string: str) -> str:
    """Format timestamp intelligently"""
    try:
        dt = parser.isoparse(iso_string)
        now = datetime.now(dt.tzinfo)

        if dt.date() == now.date():
            return dt.strftime("%I:%M %p")
        elif now - timedelta(days=1) < dt <= now:
            return "Yesterday " + dt.strftime("%I:%M %p")
        elif now - timedelta(days=7) < dt <= now:
            return dt.strftime("%a %I:%M %p")
        else:
            return dt.strftime("%b %d, %Y")
    except Exception:
        return iso_string


st.set_page_config(
    page_title="ChatBot", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(APP_STYLE, unsafe_allow_html=True)
st.markdown(IMAGE_LIGHTBOX_JS, unsafe_allow_html=True)
initialize_session_state()

# Clear any old cached functions on first run
if "cache_cleared_v2" not in st.session_state:
    st.cache_data.clear()
    st.session_state.cache_cleared_v2 = True


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
    st.session_state.conversations_loaded = False


def open_conversation_manager() -> None:
    """Open the conversation manager dialog on the next render."""
    st.session_state.conversation_manager_visible = True
    st.session_state.show_conversation_manager = True
    st.session_state[CONVERSATION_MANAGER_DIALOG_KEY] = True


def close_conversation_manager() -> None:
    """Close the conversation manager dialog and prevent reopening on rerun."""
    st.session_state.conversation_manager_visible = False
    st.session_state.show_conversation_manager = False
    st.session_state[CONVERSATION_MANAGER_DIALOG_KEY] = False


def refresh_conversations_list(
    *, fallback_conversation: Optional[Dict[str, Any]] = None
) -> None:
    """Reload conversations list from the API, optionally seeding with a fallback."""
    st.session_state.conversations_loaded = False
    refreshed = get_conversations(include_messages=False, fetch_all_pages=True)
    if refreshed and refreshed.get("data"):
        st.session_state.conversations_list = refreshed["data"]["items"]
        st.session_state.conversations_loaded = True
        st.session_state.conversations_last_fetch_params = {
            "include_messages": False,
            "fetch_all_pages": True,
        }
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
        st.toast(f"HTTP error {http_error.response.status_code}", icon="❌")
        return {}
    except requests.exceptions.ConnectionError:
        st.toast("Cannot connect to API", icon="❌")
        return {}
    except ValueError:
        st.toast("Unexpected response from API", icon="❌")
        return {}
    except Exception as exc:
        st.toast(f"Error: {exc}", icon="❌")
        return {}

    if not response_data.get("success"):
        error_code = response_data.get("code", "unknown_error")
        error_message = response_data.get("message", "An unknown error occurred.")

        if error_code == "unauthenticated":
            st.session_state.auth_token = None
            st.session_state.current_user_id = None
            st.session_state.show_login = True
            st.toast("Please log in", icon="🔒")
        else:
            st.toast(f"{error_message}", icon="❌")
        return {}

    return response_data


def make_streaming_request(endpoint: str, data: Optional[Dict] = None):
    """
    Make a streaming API request using Server-Sent Events (SSE).
    Yields parsed JSON events from the stream.
    """
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    if st.session_state.get("auth_token"):
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

    stream_completed = False
    try:
        # Long-running MCP tools can block the stream for several minutes, so use a generous read timeout
        response = requests.post(
            url,
            json=data,
            headers=headers,
            stream=True,
            timeout=(30, 900),
        )
        response.raise_for_status()

        # Parse SSE stream
        for line in response.iter_lines(decode_unicode=True):
            if line:
                # SSE format: "data: {json}"
                if line.startswith("data: "):
                    event_data = line[6:]  # Remove "data: " prefix
                    try:
                        event = json.loads(event_data)
                        event_type = event.get("type")
                        yield event
                        if event_type in ["complete", "error", "interrupt"]:
                            stream_completed = True
                    except json.JSONDecodeError:
                        continue

    except requests.exceptions.HTTPError as http_error:
        st.toast(f"HTTP error {http_error.response.status_code}", icon="❌")
        yield {"type": "error", "error": f"HTTP {http_error.response.status_code}"}
    except requests.exceptions.ConnectionError as conn_error:
        if not stream_completed:
            st.toast("Cannot connect to API", icon="❌")
            yield {"type": "error", "error": "Connection error"}
    except requests.exceptions.Timeout:
        st.toast("Request timed out", icon="⏱️")
        yield {"type": "error", "error": "Timeout"}
    except Exception as exc:
        st.toast(f"Error: {exc}", icon="❌")
        yield {"type": "error", "error": str(exc)}


def get_user(user_id: str) -> Dict[str, Any]:
    response = make_api_request("GET", f"/users/{user_id}")
    return response.get("data", {})


def get_conversations(
    page: int = 1,
    limit: int = 100,
    include_messages: bool = False,
    latest_messages: int = 3,
    fetch_all_pages: bool = False,
) -> Dict[str, Any]:
    """Retrieve conversations with controlled pagination."""
    current_page = page
    aggregated_items: List[Dict[str, Any]] = []
    aggregated_meta: Dict[str, Any] = {}
    last_response: Optional[Dict[str, Any]] = None

    while True:
        endpoint = f"/conversations/?page={current_page}&limit={limit}"
        if include_messages:
            endpoint += f"&include=messages&latestMessages={latest_messages}"

        response = make_api_request("GET", endpoint)
        if not response:
            return {}

        if not response.get("success", False):
            return response

        data = response.get("data") or {}
        items = data.get("items") or []

        if not items:
            break

        aggregated_items.extend(items)

        meta = data.get("meta") or {}
        aggregated_meta = meta
        last_response = response

        if not fetch_all_pages:
            break

        current = meta.get("currentPage", current_page)
        last = meta.get("lastPage", current_page)

        total = meta.get("total", 0)
        if current >= last or (total > 0 and len(aggregated_items) >= total):
            break

        current_page += 1

    if not last_response:
        return {}

    if fetch_all_pages:
        normalized_meta = {
            "total": len(aggregated_items),
            "perPage": len(aggregated_items),
            "currentPage": 1,
            "lastPage": 1,
        }
    else:
        normalized_meta = {
            "total": aggregated_meta.get("total", len(aggregated_items)),
            "perPage": aggregated_meta.get("perPage", limit),
            "currentPage": aggregated_meta.get("currentPage", page),
            "lastPage": aggregated_meta.get("lastPage", 1),
        }

    for key, value in aggregated_meta.items():
        if key not in normalized_meta:
            normalized_meta[key] = value

    result: Dict[str, Any] = {
        "success": last_response.get("success", True),
        "message": last_response.get("message", ""),
        "data": {
            "items": aggregated_items,
            "meta": normalized_meta,
        },
    }

    for optional_key in ("code", "errors"):
        if optional_key in last_response:
            result[optional_key] = last_response[optional_key]

    return result


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
    return response


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
        st.toast(f"{_MAX_IMAGE_ATTACHMENTS} images", icon="⚠️")
        return

    if len(new_items) > remaining:
        st.toast("Some images ignored", icon="⚠️")

    pending.extend(new_items[:remaining])
    st.session_state.pending_image_attachments = pending


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
        return (
            "[Image attachment]"
            if count == 1
            else f"[Image attachments: {count} files]"
        )

    if len(names) == 1:
        return f"[Image attachment: {names[0]}]"

    displayed = ", ".join(names[:3])
    if len(names) > 3:
        displayed += ", ..."

    return f"[Image attachments: {displayed}]"


def get_mcp_servers() -> Optional[Dict[str, Any]]:
    """Fetch list of MCP servers"""
    response = make_api_request("GET", "/mcp/servers")
    return response.get("data") if response else None


def get_mcp_tools(server_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch MCP tools, optionally filtered by server"""
    endpoint = "/mcp/tools"
    if server_name:
        endpoint += f"?serverName={server_name}"
    response = make_api_request("GET", endpoint)
    return response.get("data") if response else None


def get_tool_details(tool_name: str) -> Optional[Dict[str, Any]]:
    """Fetch detailed information about a specific tool"""
    response = make_api_request("GET", f"/mcp/tools/{tool_name}")
    return response.get("data") if response else None


def execute_mcp_tool(
    tool_name: str, arguments: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Execute an MCP tool with provided arguments"""
    response = make_api_request(
        "POST", f"/mcp/tools/{tool_name}/execute", {"arguments": arguments}
    )
    return response.get("data") if response else None


def render_json_output(
    data: Any, label: str = "JSON Output", expanded: Optional[bool] = None
) -> None:
    """Render JSON data with syntax highlighting in an expandable section.

    Args:
        data: Dictionary or list to render as JSON
        label: Label for the expander
        expanded: Whether to expand by default (auto-determined if None)
    """
    try:
        json_string = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False)
    except (TypeError, ValueError):
        # Fallback for non-serializable data
        json_string = str(data)

    # Auto-determine expanded state based on content size
    if expanded is None:
        expanded = len(json_string) < 500

    # Add size caption
    if isinstance(data, dict):
        size_caption = f"{len(data)} keys"
    elif isinstance(data, list):
        size_caption = f"{len(data)} items"
    else:
        size_caption = f"{len(json_string)} characters"

    with st.expander(f"{label} ({size_caption})", expanded=expanded):
        st.code(json_string, language="json", line_numbers=False)


def render_tool_result_payload(payload: Any, use_expander: bool = False) -> None:
    if payload is None:
        st.write("No data returned.")
        return

    if isinstance(payload, (dict, list)):
        if use_expander:
            render_json_output(payload, label="Result Data", expanded=True)
        else:
            # Render directly without expander
            json_string = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            st.code(json_string, language="json", line_numbers=False)
        return

    if isinstance(payload, (bytes, bytearray)):
        text = payload.decode("utf-8", errors="replace")
    else:
        text = str(payload)

    stripped = text.strip()
    if not stripped:
        st.write("Result was empty.")
        return

    try:
        parsed = json.loads(stripped)
    except (TypeError, json.JSONDecodeError):
        language = "json" if stripped[:1] in ("{", "[") else "text"
        st.code(text, language=language)
    else:
        if isinstance(parsed, (dict, list)):
            if use_expander:
                render_json_output(parsed, label="Result Data", expanded=True)
            else:
                # Render directly without expander
                json_string = json.dumps(
                    parsed, indent=2, ensure_ascii=False, default=str
                )
                st.code(json_string, language="json", line_numbers=False)
        else:
            st.code(json.dumps(parsed, ensure_ascii=False), language="json")


def add_mcp_server(server_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Add a new MCP server"""
    response = make_api_request("POST", "/mcp/servers", server_config)
    return response.get("data") if response else None


def remove_mcp_server(server_name: str) -> Optional[Dict[str, Any]]:
    """Remove an MCP server"""
    response = make_api_request("DELETE", f"/mcp/servers/{server_name}")
    return response.get("data") if response else None


def toggle_mcp_server(server_name: str, enabled: bool) -> Optional[Dict[str, Any]]:
    """Enable or disable an MCP server"""
    response = make_api_request(
        "PATCH", f"/mcp/servers/{server_name}/toggle?enabled={enabled}"
    )
    return response.get("data") if response else None


def add_mcp_server_from_url(url_config: Dict[str, Any]) -> bool:
    """Add a new MCP server from URL"""
    response = make_api_request("POST", "/mcp/servers/from-url", url_config)
    return response.get("success", False) if response else False


def render_login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# ChatBot")
        st.markdown("### Welcome! Please sign in to continue")

        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

        with tab1:
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("📧 Email", placeholder="your@email.com")
                password = st.text_input(
                    "🔒 Password", type="password", placeholder="Enter password"
                )

                if st.form_submit_button(
                    "Sign In", use_container_width=True, type="primary"
                ):
                    with st.spinner("Signing in..."):
                        auth_response = make_api_request(
                            "POST",
                            "/auth/login",
                            {"email": email, "password": password},
                        )
                        if auth_response and "data" in auth_response:
                            st.session_state.auth_token = auth_response["data"][
                                "accessToken"
                            ]
                            st.session_state.current_user_id = auth_response["data"][
                                "userId"
                            ]
                            st.session_state.show_login = False
                            st.toast("Welcome back!", icon="✅")
                            st.rerun()
                        else:
                            st.error("Invalid credentials")

        with tab2:
            with st.form("signup_form", clear_on_submit=False):
                username = st.text_input("👤 Username", placeholder="Choose a username")
                email = st.text_input("📧 Email", placeholder="your@email.com")
                password = st.text_input(
                    "🔒 Password", type="password", placeholder="Create password"
                )
                confirm_password = st.text_input(
                    "🔒 Confirm", type="password", placeholder="Confirm password"
                )

                if st.form_submit_button(
                    "Create Account", use_container_width=True, type="primary"
                ):
                    if not username or not email or not password:
                        st.error("Please fill all fields")
                    elif password != confirm_password:
                        st.error("Passwords don't match")
                    else:
                        with st.spinner("Creating account..."):
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
                                    st.toast("✅ Account created!", icon="✅")
                                    st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown("# Multi-agent ChatBot")

        # New chat button
        if st.button("New Chat", use_container_width=True, type="primary"):
            st.session_state.current_conversation_id = "pending_new"
            st.session_state.active_view = "chat"
            close_conversation_manager()
            reset_conversation_state()
            st.rerun()

        # Manage conversations button
        if st.button("Manage Conversations", use_container_width=True):
            if st.session_state.get("conversation_manager_visible"):
                close_conversation_manager()
            else:
                open_conversation_manager()
            st.rerun()

        st.divider()

        # Load conversations if needed
        if (
            not st.session_state.conversations_loaded
            and st.session_state.current_user_id
            and st.session_state.auth_token
        ):
            conversations_response = get_conversations(
                include_messages=False, fetch_all_pages=True
            )
            if conversations_response and conversations_response.get("data"):
                st.session_state.conversations_list = conversations_response["data"][
                    "items"
                ]
                st.session_state.conversations_loaded = True
                st.session_state.conversations_last_fetch_params = {
                    "include_messages": False,
                    "fetch_all_pages": True,
                }

        # Grouped conversations
        if st.session_state.conversations_list:
            st.markdown("### 💬 Conversations")

            sorted_conversations = sorted(
                st.session_state.conversations_list,
                key=lambda x: parser.parse(x.get("createdAt")),
                reverse=True,
            )

            grouped = group_conversations_by_date(sorted_conversations)

            for group_name, convs in grouped.items():
                with st.expander(
                    f"📅 {group_name} ({len(convs)})", expanded=(group_name == "Today")
                ):
                    for conv in convs:
                        is_active = (
                            conv["id"] == st.session_state.current_conversation_id
                        )
                        render_conversation_button(conv, is_active)

        st.divider()

        # User section
        if st.session_state.current_user_id:
            user = get_user(st.session_state.current_user_id)
            if user:
                st.markdown(f"**{user['username']}**")
                if st.button("Sign Out", use_container_width=True):
                    st.session_state.current_user_id = None
                    st.session_state.current_conversation_id = None
                    close_conversation_manager()
                    reset_conversation_state()
                    st.session_state.conversations_list = []
                    st.session_state.conversations_loaded = False
                    st.session_state.conversations_last_fetch_params = None
                    st.session_state.auth_token = None
                    st.session_state.show_login = True
                    st.session_state.pending_image_attachments = []
                    st.session_state.message_image_thumbnails = {}
                    st.toast("👋 Goodbye!", icon="👋")
                    st.rerun()


def _guess_extension(mime: Optional[str]) -> str:
    """Guess file extension from MIME type"""
    if not mime:
        return "png"
    base_mime = mime.split(";")[0]
    ext = mimetypes.guess_extension(base_mime)
    if ext:
        return ext.lstrip(".")
    if base_mime.endswith("jpeg"):
        return "jpg"
    return "png"


def _build_attachment_thumbnail(
    href: str, name: str, *, download: Optional[str] = None
) -> str:
    """Create HTML anchor for a single attachment thumbnail."""
    escaped_href = html.escape(str(href), quote=True)
    escaped_name = html.escape(str(name), quote=True)
    download_attr = (
        f' download="{html.escape(str(download), quote=True)}"' if download else ""
    )
    return (
        f'<a class="attachment-thumb-link" href="{escaped_href}" target="_blank" '
        f'rel="noopener noreferrer" aria-label="Open {escaped_name}" '
        f'title="{escaped_name}"{download_attr}>'
        f'<div class="attachment-thumb">'
        f'<img src="{escaped_href}" alt="{escaped_name}" loading="lazy" />'
        f"</div></a>"
    )


def render_attachment_gallery(attachments: List[Dict[str, str]], *, align: str) -> None:
    """Render a compact row of clickable image thumbnails.

    - URL images (from search agent): open in new tab
    - Base64 images (generated/stored): open in lightbox modal
    """
    if not attachments:
        return

    # Filter valid attachments (must have data or url)
    valid_attachments = []
    for idx, attachment in enumerate(attachments, start=1):
        name = attachment.get("name") or f"Image {idx}"
        data_b64 = attachment.get("data")
        url_value = attachment.get("url")

        if isinstance(data_b64, str) and data_b64:
            valid_attachments.append(
                {
                    "name": name,
                    "data": data_b64,
                    "mime": attachment.get("mime", "image/png"),
                    "type": "base64",
                }
            )
        elif isinstance(url_value, str) and url_value:
            valid_attachments.append({"name": name, "url": url_value, "type": "url"})

    if not valid_attachments:
        return

    # Build compact HTML gallery with clickable thumbnails
    # Using class="img-thumb" for event delegation (no inline onclick needed)
    gallery_html_parts = []
    for att in valid_attachments:
        # Truncate name for display (max 20 chars)
        display_name = att["name"]
        if len(display_name) > 20:
            display_name = display_name[:17] + "..."

        escaped_name = html.escape(display_name, quote=True)
        escaped_full_name = html.escape(att["name"], quote=True)

        if att["type"] == "base64":
            # Base64 images: use lightbox modal via event delegation
            src = f"data:{att.get('mime', 'image/png')};base64,{att['data']}"
            escaped_src = html.escape(src, quote=True)
            gallery_html_parts.append(
                f'<div style="display: inline-block; margin: 4px; text-align: center; vertical-align: top;" '
                f'title="{escaped_full_name}">'
                f'<img src="{escaped_src}" alt="{escaped_full_name}" class="img-thumb" '
                f'style="width: 72px; height: 72px; object-fit: cover; border-radius: 8px; '
                f'border: 1px solid #e2e8f0; cursor: pointer;" />'
                f'<div style="font-size: 10px; color: #64748b; max-width: 72px; overflow: hidden; '
                f'text-overflow: ellipsis; white-space: nowrap; margin-top: 2px;">{escaped_name}</div>'
                f"</div>"
            )
        else:
            # URL images: also use lightbox for consistent experience
            escaped_url = html.escape(att["url"], quote=True)
            gallery_html_parts.append(
                f'<div style="display: inline-block; margin: 4px; text-align: center; vertical-align: top;" '
                f'title="{escaped_full_name}">'
                f'<img src="{escaped_url}" alt="{escaped_full_name}" class="img-thumb" '
                f'style="width: 72px; height: 72px; object-fit: cover; border-radius: 8px; '
                f'border: 1px solid #e2e8f0; cursor: pointer;" />'
                f'<div style="font-size: 10px; color: #64748b; max-width: 72px; overflow: hidden; '
                f'text-overflow: ellipsis; white-space: nowrap; margin-top: 2px;">{escaped_name}</div>'
                f"</div>"
            )

    wrapper_align = "flex-end" if align == "right" else "flex-start"
    gallery_html = (
        f'<div style="display: flex; flex-wrap: wrap; gap: 4px; justify-content: {wrapper_align}; '
        f'margin: 8px 0;">' + "".join(gallery_html_parts) + "</div>"
    )

    st.markdown(gallery_html, unsafe_allow_html=True)


def render_agent_images(message_metadata: dict):
    """Render images from agent responses as thumbnails (Tavily, Image Generator)"""
    if not message_metadata:
        return

    images = message_metadata.get("images") or []
    if not images:
        return

    gallery_items: List[Dict[str, str]] = []

    for idx, image in enumerate(images, start=1):
        if not isinstance(image, dict):
            continue

        url_value = image.get("url")
        data_value = image.get("data")

        if isinstance(url_value, str) and url_value:
            gallery_items.append(
                {
                    "url": url_value,
                    "name": image.get("description") or f"Image {idx}",
                }
            )
        elif isinstance(data_value, str) and data_value:
            gallery_items.append(
                {
                    "data": data_value,
                    "mime": image.get("mime", "image/png"),
                    "name": image.get("name") or f"Generated image {idx}",
                }
            )

    if gallery_items:
        render_attachment_gallery(gallery_items, align="left")


def render_tool_artifacts(tool_artifacts: List[Dict[str, Any]]):
    """
    Render tool execution artifacts as collapsible sections.
    Shows tool name, status, arguments, and results.
    """
    if not tool_artifacts:
        return

    st.markdown("#### 🔧 Tool Executions", unsafe_allow_html=True)

    for idx, artifact in enumerate(tool_artifacts, start=1):
        tool_name = artifact.get("tool", "unknown_tool")
        has_error = artifact.get("error") is not None

        # Status badge styling
        if has_error:
            status_badge = "🔴 Error"
            badge_color = COLORS["error"]
        else:
            status_badge = "🟢 Success"
            badge_color = COLORS["success"]

        # Create expander for each tool
        with st.expander(f"**[{idx}] {tool_name}** - {status_badge}", expanded=False):
            # Show execution status
            st.markdown(
                f'<div style="background-color: {badge_color}15; padding: 8px; border-radius: 4px; margin-bottom: 8px;">'
                f'<strong style="color: {badge_color};">Status:</strong> {status_badge}'
                f"</div>",
                unsafe_allow_html=True,
            )

            # Show arguments if present
            args = artifact.get("args", {})
            if args and isinstance(args, dict):
                st.markdown("**Arguments:**")
                # Display arguments in a nice format
                for key, value in args.items():
                    st.code(f"{key}: {value}", language="text")

            # Show output/result if present
            output = artifact.get("output")
            if output is not None:
                st.markdown("**Output:**")
                if isinstance(output, (dict, list)):
                    render_json_output(
                        output, label=f"{tool_name} Output", expanded=False
                    )
                else:
                    st.code(str(output), language="text")

            # Show error if present
            if has_error:
                error_msg = artifact.get("error", "Unknown error")
                st.markdown("**Error:**")
                st.error(error_msg)

                # Show recovery hint if available
                hint = artifact.get("hint")
                if hint:
                    st.markdown("**Recovery Hint:**")
                    st.info(hint)

            # Show execution time if available
            execution_time = artifact.get("execution_time")
            if execution_time:
                st.caption(f"⏱️ Execution time: {execution_time:.2f}s")


def render_citations(message_metadata: Dict[str, Any], msg_id: Optional[str] = None):
    """
    Render citations from document metadata in a user-friendly format.
    Shows documents cited with chunk and page information.
    """
    if not message_metadata:
        return

    # Check for new grouped structure
    documents_cited = message_metadata.get("documents_cited", [])

    if not documents_cited:
        legacy_citations = message_metadata.get("citations", [])
        if legacy_citations:
            with st.expander(
                f"Sources ({len(legacy_citations)} references)", expanded=False
            ):
                for idx, citation in enumerate(legacy_citations, start=1):
                    source = citation.get("source", "unknown")
                    score = citation.get("score", 0.0)

                    # Determine relevance color
                    if score >= 0.3:
                        relevance_color = COLORS["success"]
                    elif score >= 0.2:
                        relevance_color = COLORS["warning"]
                    else:
                        relevance_color = COLORS["error"]

                    st.markdown(
                        f'<div class="citation-chunk" style="margin-bottom: 8px; padding: 8px; border-left: 3px solid {relevance_color}; background-color: {relevance_color}15;">'
                        f"<strong>[{idx}]</strong> {source}<br>"
                        f'<span style="color: {relevance_color}; font-size: 0.9em;">Relevance: ({score:.1%})</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        return

    # Render new grouped structure
    with st.expander(f"Sources ({len(documents_cited)} documents)", expanded=False):
        # Get chunk data from session state
        chunks_data = st.session_state.get("message_chunks", {}).get(msg_id, {})
        chunks_map = chunks_data.get("chunks_map", {})

        for doc_entry in documents_cited:
            doc_num = doc_entry.get("document_number", "?")
            doc_id = doc_entry.get("document_id")
            source = doc_entry.get("source", "unknown")
            total_chunks = doc_entry.get("total_chunks", 0)
            avg_score = doc_entry.get("avg_score", 0.0)
            chunks = doc_entry.get("chunks", [])

            # Determine overall document relevance color
            if avg_score >= 0.3:
                doc_relevance_color = COLORS["success"]
            elif avg_score >= 0.2:
                doc_relevance_color = COLORS["warning"]
            else:
                doc_relevance_color = COLORS["error"]

            # Document header - make it clickable if only one chunk
            if total_chunks == 1 and chunks and msg_id:
                chunk = chunks[0]
                chunk_idx = chunk.get("chunk_index", 0)
                chunk_data = chunks_map.get(doc_num, {}).get(chunk_idx, {})

                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    st.markdown(
                        f'<div class="citation-document" style="margin-bottom: 12px; padding: 12px; border: 2px solid {doc_relevance_color}; border-radius: 8px; background-color: {doc_relevance_color}10;">'
                        f'<strong style="font-size: 1.1em;">[Document {doc_num}] {source}</strong><br>'
                        f'<span style="color: {doc_relevance_color}; font-size: 0.9em;">Overall Relevance: ({avg_score:.1%})</span> | '
                        f'<span style="font-size: 0.9em;">{total_chunks} chunk(s)</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button(
                        "View",
                        key=f"cite_doc_{msg_id}_{doc_num}",
                        help="View chunk content",
                        use_container_width=True,
                    ):
                        st.session_state["selected_chunk_info"] = {
                            "source": source,
                            "document_number": doc_num,
                            "chunk_index": chunk_idx,
                            "content": chunk_data.get("content", ""),
                            "score": chunk_data.get("score", chunk.get("score", 0.0)),
                            "page_number": chunk_data.get("page_number"),
                            "character_count": chunk_data.get("character_count", 0),
                        }
                        st.session_state["chunk_preview_dialog_key"] = True
                        st.rerun()
            else:
                st.markdown(
                    f'<div class="citation-document" style="margin-bottom: 12px; padding: 12px; border: 2px solid {doc_relevance_color}; border-radius: 8px; background-color: {doc_relevance_color}10;">'
                    f'<strong style="font-size: 1.1em;">[Document {doc_num}] {source}</strong><br>'
                    f'<span style="color: {doc_relevance_color}; font-size: 0.9em;">Overall Relevance: ({avg_score:.1%})</span> | '
                    f'<span style="font-size: 0.9em;">{total_chunks} chunk(s)</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Show individual chunks if more than one
            if total_chunks > 1:
                st.markdown("**Chunks:**")
                for chunk in chunks:
                    chunk_idx = chunk.get("chunk_index", "?")
                    chunk_score = chunk.get("score", 0.0)

                    # Chunk relevance color
                    if chunk_score >= 0.3:
                        chunk_color = COLORS["success"]
                    elif chunk_score >= 0.2:
                        chunk_color = COLORS["warning"]
                    else:
                        chunk_color = COLORS["error"]

                    # Make individual chunks clickable
                    if msg_id:
                        chunk_data = chunks_map.get(doc_num, {}).get(chunk_idx, {})
                        col1, col2 = st.columns([0.85, 0.15])
                        with col1:
                            st.markdown(
                                f'<div class="citation-chunk" style="margin-left: 20px; margin-bottom: 6px; padding: 6px; border-left: 2px solid {chunk_color}; background-color: {chunk_color}08;">'
                                f'<span style="font-size: 0.9em;">Chunk {chunk_idx} '
                                f'<span style="color: {chunk_color};">({chunk_score:.1%})</span></span>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        with col2:
                            if st.button(
                                "Details",
                                key=f"cite_chunk_{msg_id}_{doc_num}_{chunk_idx}",
                                help="View chunk content",
                                use_container_width=True,
                            ):
                                st.session_state["selected_chunk_info"] = {
                                    "source": source,
                                    "document_number": doc_num,
                                    "chunk_index": chunk_idx,
                                    "content": chunk_data.get("content", ""),
                                    "score": chunk_data.get("score", chunk_score),
                                    "page_number": chunk_data.get("page_number"),
                                    "character_count": chunk_data.get(
                                        "character_count", 0
                                    ),
                                }
                                st.session_state["chunk_preview_dialog_key"] = True
                                st.rerun()
                    else:
                        st.markdown(
                            f'<div class="citation-chunk" style="margin-left: 20px; margin-bottom: 6px; padding: 6px; border-left: 2px solid {chunk_color}; background-color: {chunk_color}08;">'
                            f'<span style="font-size: 0.9em;">Chunk {chunk_idx} '
                            f'<span style="color: {chunk_color};">({chunk_score:.1%})</span></span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Show images for this document
            images = message_metadata.get("images", [])
            if images:
                # Filter images that belong to this document
                doc_images = []
                for img in images:
                    # Match by document ID or source filename
                    img_name = img.get("name", "")
                    if doc_id and doc_id in img_name:
                        doc_images.append(img)
                    elif source and source in img_name:
                        doc_images.append(img)

                if doc_images:
                    st.markdown("**Images:**")
                    # Build compact clickable gallery for document images
                    gallery_parts = []
                    for img in doc_images:
                        data_b64 = img.get("data", "")
                        mime = img.get("mime", "image/png")
                        caption = img.get("caption") or img.get("name", "Image")
                        page_num = img.get("page_number")

                        if data_b64:
                            caption_text = caption
                            if page_num:
                                caption_text = f"{caption} (p. {page_num})"

                            # Truncate caption for display
                            display_caption = caption_text
                            if len(display_caption) > 25:
                                display_caption = display_caption[:22] + "..."

                            src = f"data:{mime};base64,{data_b64}"
                            escaped_src = html.escape(src, quote=True)
                            escaped_caption = html.escape(display_caption, quote=True)
                            escaped_full = html.escape(caption_text, quote=True)

                            # Use lightbox for base64 images via event delegation
                            gallery_parts.append(
                                f'<div style="display: inline-block; margin: 4px; text-align: center; vertical-align: top;" '
                                f'title="{escaped_full}">'
                                f'<img src="{escaped_src}" alt="{escaped_full}" class="img-thumb" '
                                f'style="width: 80px; height: 80px; object-fit: cover; border-radius: 8px; '
                                f'border: 1px solid #e2e8f0; cursor: pointer;" />'
                                f'<div style="font-size: 10px; color: #64748b; max-width: 80px; overflow: hidden; '
                                f'text-overflow: ellipsis; white-space: nowrap; margin-top: 2px;">{escaped_caption}</div>'
                                f"</div>"
                            )

                    if gallery_parts:
                        st.markdown(
                            f'<div style="display: flex; flex-wrap: wrap; gap: 4px; margin: 4px 0;">'
                            + "".join(gallery_parts)
                            + "</div>",
                            unsafe_allow_html=True,
                        )


def render_thinking_summary(message_metadata: Dict[str, Any]):
    """Render thinking summary from message metadata in a styled collapsible section."""
    thinking_summary = message_metadata.get("thinking_summary")
    if thinking_summary:
        with st.expander("Thought Process", expanded=False):
            # Use custom styled container for thinking content
            st.markdown(
                f"""<div class="thinking-container">
                    <div class="thinking-content">{html.escape(thinking_summary)}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def render_message_bubble(msg: Dict[str, Any], is_user: bool):
    content_text = msg.get("content", "")
    timestamp = format_time(msg.get("createdAt", ""))

    # Use Streamlit's native chat_message which supports LaTeX
    avatar = "user" if is_user else "assistant"

    with st.chat_message(avatar):
        # Show thinking summary first for assistant messages
        if not is_user:
            render_thinking_summary(msg.get("messageMetadata", {}))

        st.markdown(content_text)  # Native markdown with LaTeX support
        st.caption(timestamp)

    # Show attachments if user message
    if is_user:
        attachments = st.session_state.get("message_image_thumbnails", {}).get(
            str(msg.get("id", ""))
        )
        if attachments:
            render_attachment_gallery(attachments, align="right")

    # Show agent-sent images for assistant messages
    if not is_user:
        render_agent_images(msg.get("messageMetadata", {}))

    # Show tool artifacts for assistant messages
    if not is_user:
        message_metadata = msg.get("messageMetadata", {})
        tool_artifacts = message_metadata.get("tool_artifacts")
        if tool_artifacts:
            render_tool_artifacts(tool_artifacts)

    # Show citations for assistant messages
    if not is_user:
        render_citations(msg.get("messageMetadata", {}), str(msg.get("id", "")))

    # Show feedback for assistant messages
    if not is_user:
        render_message_feedback_inline(msg)


def render_message_feedback_inline(msg: Dict[str, Any]):
    """Inline feedback for assistant messages using popover"""
    feedback = msg.get("feedback")

    if isinstance(feedback, dict) and feedback:
        # Show existing feedback
        col1, col2 = st.columns([4, 1])
        with col1:
            rating = feedback.get("rating", 0)
            stars = "⭐" * rating
            st.caption(f"{stars} {rating}/5")
            comment = feedback.get("comment")
            if comment:
                st.caption(f"💬 {comment[:60]}...")
        with col2:
            with st.popover("✏️", help="Edit feedback"):
                st.markdown("**Edit Feedback**")
                with st.form(f"edit_feedback_{msg['id']}", clear_on_submit=True):
                    rating = st.select_slider(
                        "Rating",
                        options=[1, 2, 3, 4, 5],
                        value=feedback.get("rating", 5),
                    )
                    comment = st.text_area(
                        "Comment (optional)",
                        value=feedback.get("comment", ""),
                        height=68,
                    )

                    if st.form_submit_button(
                        "Update", use_container_width=True, type="primary"
                    ):
                        feedback_data = {
                            "messageId": msg["id"],
                            "rating": rating,
                            "comment": comment,
                        }
                        response = make_api_request(
                            "POST", f"/messages/{msg['id']}/feedbacks", feedback_data
                        )
                        if response:
                            st.session_state.conversation_messages_page = 0
                            st.toast("Feedback updated!", icon="✅")
                            st.rerun()
    else:
        # Show add feedback popover
        with st.popover("💬 Feedback", help="Give feedback"):
            st.markdown("**Provide Feedback**")
            with st.form(f"add_feedback_{msg['id']}", clear_on_submit=True):
                rating = st.select_slider("Rating", options=[1, 2, 3, 4, 5], value=5)
                comment = st.text_area("Comment (optional)", height=68)

                if st.form_submit_button(
                    "Submit", use_container_width=True, type="primary"
                ):
                    feedback_data = {
                        "messageId": msg["id"],
                        "rating": rating,
                        "comment": comment,
                    }
                    response = make_api_request(
                        "POST", f"/messages/{msg['id']}/feedbacks", feedback_data
                    )
                    if response:
                        st.session_state.conversation_messages_page = 0
                        st.toast("Feedback submitted!", icon="✅")
                        st.rerun()


def render_tool_parameter_form(
    args_schema: Dict[str, Any], key_prefix: str = ""
) -> Tuple[Dict[str, Any], List[str]]:
    """Render dynamic form fields based on a tool's JSON Schema."""
    parameters: Dict[str, Any] = {}
    parsing_errors: List[str] = []

    if not args_schema or "properties" not in args_schema:
        st.info("This tool doesn't require any parameters.")
        return parameters, parsing_errors

    properties = args_schema.get("properties", {})
    required = args_schema.get("required", [])

    st.markdown("#### Parameters")

    def _stringify_default(value: Any) -> str:
        if value in (None, "", [], {}):
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(value)

    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "string")
        param_desc = param_info.get("description", "")
        is_required = param_name in required

        label = f"{param_name}{'*' if is_required else ''}"
        help_text = param_desc if param_desc else None
        base_key = f"{key_prefix}_{param_name}" if key_prefix else param_name

        if param_type == "boolean":
            default_val = bool(param_info.get("default", False))
            parameters[param_name] = st.checkbox(
                label,
                value=default_val,
                help=help_text,
                key=f"{base_key}_bool",
            )
        elif param_type == "integer":
            default_val = int(param_info.get("default", 0) or 0)
            value = st.number_input(
                label,
                value=default_val,
                step=1,
                format="%d",
                help=help_text,
                key=f"{base_key}_int",
            )
            parameters[param_name] = int(value)
        elif param_type == "number":
            default_val = float(param_info.get("default", 0.0) or 0.0)
            parameters[param_name] = st.number_input(
                label,
                value=default_val,
                step=0.1,
                format="%.6f",
                help=help_text,
                key=f"{base_key}_number",
            )
        elif param_type == "string":
            if "enum" in param_info:
                enum_values = param_info["enum"]
                parameters[param_name] = st.selectbox(
                    label,
                    options=enum_values,
                    help=help_text,
                    key=f"{base_key}_enum",
                )
            else:
                default_val = _stringify_default(param_info.get("default", ""))
                parameters[param_name] = st.text_input(
                    label,
                    value=default_val,
                    help=help_text,
                    key=f"{base_key}_text",
                )
        elif param_type in {"object", "array"}:
            default_val = param_info.get(
                "default", {} if param_type == "object" else []
            )
            default_text = _stringify_default(default_val)
            placeholder = (
                "Enter JSON object value"
                if param_type == "object"
                else "Enter JSON array value"
            )
            raw_value = st.text_area(
                label,
                value=default_text,
                help=help_text or placeholder,
                placeholder=placeholder,
                key=f"{base_key}_json",
                height=150,
            )
            if raw_value.strip():
                try:
                    parameters[param_name] = json.loads(raw_value)
                except json.JSONDecodeError as exc:
                    parsing_errors.append(
                        f"{param_name}: Invalid JSON ({exc.msg} at column {exc.colno})"
                    )
                    st.warning(f"Invalid JSON provided for {param_name}.")
                    parameters[param_name] = raw_value
            else:
                parameters[param_name] = default_val
        else:
            raw_value = st.text_area(
                label,
                help=help_text or f"Enter {param_type} value (JSON supported)",
                placeholder='Enter value (JSON supported, e.g. 42 or {"key": "value"})',
                key=f"{base_key}_fallback",
                height=120,
            )
            if raw_value.strip():
                try:
                    parameters[param_name] = json.loads(raw_value)
                except json.JSONDecodeError as exc:
                    parsing_errors.append(
                        f"{param_name}: Invalid JSON ({exc.msg} at column {exc.colno})"
                    )
                    st.warning(f"Invalid JSON provided for {param_name}.")
                    parameters[param_name] = raw_value
            else:
                parameters[param_name] = None

    return parameters, parsing_errors


def render_tools_tab():
    """Render the MCP Tools management and testing interface"""
    st.markdown("# 🔧 MCP Tools Management")
    st.markdown(
        "Discover and test Model Context Protocol (MCP) tools available to the chatbot."
    )

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    st.markdown("---")

    # Fetch servers and tools
    servers_data = get_mcp_servers()
    tools_data = get_mcp_tools()

    if not servers_data or not tools_data:
        st.error("Failed to load MCP data. Make sure the API is running.")
        return

    # Server Management Section
    with st.expander("**MCP Servers Management**", expanded=False):
        servers = servers_data.get("servers", [])

        # Add Server Section
        st.markdown("#### Add New Server")

        tab1, tab2, tab3 = st.tabs(["JSON Config", "Form", "URL"])

        with tab1:
            st.markdown("Paste your MCP server configuration in JSON format:")
            json_config = st.text_area(
                "JSON Configuration",
                height=250,
                placeholder="""{
"mcpServers": {
    "my-server": {
    "command": "python",
    "args": ["path/to/server.py"],
    "description": "My custom MCP server"
    }
}
}""",
                label_visibility="collapsed",
            )

            if st.button("Add Server from JSON", use_container_width=True):
                if json_config.strip():
                    try:
                        config = json.loads(json_config)

                        servers_to_add = []

                        # Check if this is a full config file format
                        if "mcpServers" in config or "mcp_servers" in config:
                            # Extract servers from the wrapper
                            servers_dict = config.get("mcpServers") or config.get(
                                "mcp_servers"
                            )
                            for server_name, server_config in servers_dict.items():
                                # Add the name to the config
                                server_config["name"] = server_name
                                # Add default transport if not specified
                                if "transport" not in server_config:
                                    server_config["transport"] = "stdio"
                                servers_to_add.append(server_config)

                        # Check if this is a single server config with name
                        elif "name" in config:
                            if "transport" not in config:
                                config["transport"] = "stdio"
                            servers_to_add.append(config)

                        # Invalid format
                        else:
                            st.error(
                                "❌ Invalid format. Please use one of these formats:"
                            )
                            servers_to_add = []

                        # Add all servers
                        if servers_to_add:
                            success_count = 0
                            failed_servers = []

                            for server_config in servers_to_add:
                                server_name = server_config.get("name", "unknown")

                                with st.spinner(f"Adding server '{server_name}'..."):
                                    # Make API request
                                    response = make_api_request(
                                        "POST", "/mcp/servers", server_config
                                    )

                                    if response and response.get("success"):
                                        success_count += 1
                                        st.success(
                                            f"✅ Server '{server_name}' added successfully!"
                                        )
                                    else:
                                        error_msg = (
                                            response.get("message", "Unknown error")
                                            if response
                                            else "No response from API"
                                        )
                                        failed_servers.append(
                                            f"{server_name}: {error_msg}"
                                        )
                                        st.error(
                                            f"❌ Failed to add '{server_name}': {error_msg}"
                                        )

                            # Show summary
                            if success_count > 0:
                                st.info(
                                    f"✅ Successfully added {success_count} server(s). Refreshing..."
                                )
                                # Small delay to ensure file is written
                                import time

                                time.sleep(0.5)
                                st.rerun()

                            if failed_servers:
                                st.warning(
                                    f"Failed to add {len(failed_servers)} server(s)"
                                )
                                for failure in failed_servers:
                                    st.text(f"  • {failure}")
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a JSON configuration")

        with tab2:
            with st.form("add_server_form"):
                st.markdown("Fill in the server details:")

                server_name_input = st.text_input(
                    "Server Name*", placeholder="my-server"
                )
                transport_input = st.selectbox(
                    "Transport Type*",
                    options=["stdio", "http", "sse", "streamable_http"],
                    index=0,
                )

                if transport_input == "stdio":
                    command_input = st.text_input(
                        "Command*", value="python", placeholder="python"
                    )
                    args_input = st.text_input(
                        "Arguments (comma-separated)*",
                        placeholder="app/ai/mcp_servers/my_server.py",
                    )
                    env_input = st.text_area(
                        "Environment Variables (JSON, optional)",
                        placeholder='{"API_KEY": "value"}',
                        height=100,
                    )
                else:
                    url_input = st.text_input(
                        "URL*", placeholder="http://localhost:8080"
                    )
                    headers_input = st.text_area(
                        "Headers (JSON, optional)",
                        placeholder='{"Authorization": "Bearer token"}',
                        height=100,
                    )

                description_input = st.text_area(
                    "Description (optional)",
                    placeholder="Brief description of the server",
                )
                enabled_input = st.checkbox("Enable server", value=True)

                if st.form_submit_button("Add Server", use_container_width=True):
                    if not server_name_input:
                        st.error("Server name is required")
                    else:
                        try:
                            config = {
                                "name": server_name_input,
                                "transport": transport_input,
                                "enabled": enabled_input,
                            }

                            if description_input:
                                config["description"] = description_input

                            if transport_input == "stdio":
                                if not command_input or not args_input:
                                    st.error(
                                        "Command and arguments are required for stdio transport"
                                    )
                                else:
                                    config["command"] = command_input
                                    config["args"] = [
                                        arg.strip() for arg in args_input.split(",")
                                    ]

                                    if env_input.strip():
                                        try:
                                            config["env"] = json.loads(env_input)
                                        except json.JSONDecodeError:
                                            st.error(
                                                "Invalid JSON in environment variables"
                                            )

                                    with st.spinner("Adding server..."):
                                        result = add_mcp_server(config)
                                        if result:
                                            st.success(
                                                f"✅ Server '{server_name_input}' added!"
                                            )
                                            st.rerun()
                                        else:
                                            st.error("Failed to add server")
                            else:
                                if not url_input:
                                    st.error("URL is required for HTTP transport")
                                else:
                                    config["url"] = url_input

                                    if headers_input.strip():
                                        try:
                                            config["headers"] = json.loads(
                                                headers_input
                                            )
                                        except json.JSONDecodeError:
                                            st.error("Invalid JSON in headers")

                                    with st.spinner("Adding server..."):
                                        result = add_mcp_server(config)
                                        if result:
                                            st.success(
                                                f"✅ Server '{server_name_input}' added!"
                                            )
                                            st.rerun()
                                        else:
                                            st.error("Failed to add server")
                        except Exception as e:
                            st.error(f"Error: {e}")

        with tab3:
            st.markdown("Add a server using a URL. Supports two formats:")
            st.markdown(
                """
**Local NPX Command:**
```
npx @smithery/cli@latest run @ThinkFar/clear-thought-mcp
```

**Remote HTTP/HTTPS URL:**
```
https://server.smithery.ai/reddit/mcp
```

> **Note:** Flags like `--playground`, `--verbose`, and `--debug` are automatically removed to prevent non-JSON output that breaks the MCP protocol.
"""
            )

            url_input = st.text_input(
                "MCP Server URL*",
                placeholder="npx @smithery/cli@latest run @ThinkFar/clear-thought-mcp",
                help="Enter either an npx command or HTTP/HTTPS URL. Verbose flags will be filtered automatically.",
            )

            server_name_url = st.text_input(
                "Custom Server Name (optional)",
                placeholder="Auto-generated from URL if not provided",
            )

            description_url = st.text_area(
                "Description (optional)", placeholder="Brief description of the server"
            )

            enabled_url = st.checkbox("Enable server", value=True, key="url_enabled")

            if st.button("Add Server from URL", use_container_width=True):
                if not url_input.strip():
                    st.error("URL is required")
                else:
                    try:
                        url_config = {
                            "url": url_input.strip(),
                            "enabled": enabled_url,
                        }

                        if server_name_url.strip():
                            url_config["name"] = server_name_url.strip()

                        if description_url.strip():
                            url_config["description"] = description_url.strip()

                        with st.spinner("Adding server from URL..."):
                            result = add_mcp_server_from_url(url_config)
                            if result:
                                st.success("✅ Server added successfully from URL!")
                                # Small delay to ensure file is written
                                import time

                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error("Failed to add server from URL")
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("#### Configured Servers")

        if not servers:
            st.info("No MCP servers configured.")
        else:
            for server in servers:
                server_name = server.get("name", "Unknown")
                enabled = server.get("enabled", False)
                tool_count = server.get("toolCount", 0)
                transport = server.get("transport", "unknown")
                description = server.get("description", "No description")

                status_color = "🟢" if enabled else "🔴"
                status_text = "Enabled" if enabled else "Disabled"

                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(
                        f"""
                    **{status_color} {server_name}** - {status_text}
                    - Transport: `{transport}`
                    - Tools: {tool_count}
                    - {description if description else "No description available"}
                    """
                    )

                with col2:
                    toggle_label = "Disable" if enabled else "Enable"
                    if st.button(toggle_label, key=f"toggle_{server_name}"):
                        with st.spinner(f"{toggle_label}ing server..."):
                            result = toggle_mcp_server(server_name, not enabled)
                            if result:
                                st.rerun()

                with col3:
                    if st.button("Remove", key=f"remove_{server_name}"):
                        with st.spinner("Removing server..."):
                            result = remove_mcp_server(server_name)
                            if result:
                                st.success(f"Removed '{server_name}'")
                                st.rerun()

                st.markdown("---")

    # Tools List Section
    st.markdown("### Available Tools")

    tools = tools_data.get("tools", [])
    total_count = tools_data.get("totalCount", 0)

    if not tools:
        st.info("No tools available. Enable MCP servers to load tools.")
        return

    st.markdown(
        f"**{total_count} tools** available from {tools_data.get('serversCount', 0)} servers"
    )

    # Tool selection
    tool_names = [tool.get("name", "") for tool in tools]

    # Search/filter
    search_query = st.text_input(
        "Search tools", placeholder="Filter by name or description..."
    )

    filtered_tools = tools
    if search_query:
        search_lower = search_query.lower()
        filtered_tools = [
            tool
            for tool in tools
            if search_lower in tool.get("name", "").lower()
            or search_lower in tool.get("description", "").lower()
        ]

    if not filtered_tools:
        st.warning(f"No tools match '{search_query}'")
        return

    # Display tools as selectbox
    selected_tool_name = st.selectbox(
        "Select a tool to test",
        options=[tool.get("name") for tool in filtered_tools],
        format_func=lambda x: f"{x} ({next((t.get('serverName', '') for t in filtered_tools if t.get('name') == x), '')})",
    )

    if not selected_tool_name:
        return

    # Get selected tool details
    selected_tool = next(
        (t for t in filtered_tools if t.get("name") == selected_tool_name), None
    )

    if not selected_tool:
        return

    # Display tool details
    st.markdown("---")
    st.markdown(f"## {selected_tool.get('name')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Server:** `{selected_tool.get('serverName', 'Unknown')}`")
    with col2:
        st.markdown(f"**Type:** Tool")

    st.markdown(
        f"**Description:** {selected_tool.get('description', 'No description available')}"
    )

    # Tool parameter form
    st.markdown("---")
    args_schema = selected_tool.get("argsSchema", {})

    with st.form(key=f"tool_execute_form_{selected_tool_name}"):
        st.markdown("### Execute Tool")

        # Render parameter inputs
        parameters, parameter_errors = render_tool_parameter_form(
            args_schema, key_prefix=selected_tool_name
        )

        # Submit button
        execute_button = st.form_submit_button(
            "▶Execute Tool", use_container_width=True
        )

        if execute_button:
            if parameter_errors:
                for error_msg in parameter_errors:
                    st.error(error_msg)
            else:
                with st.spinner(f"Executing {selected_tool_name}..."):
                    result = execute_mcp_tool(selected_tool_name, parameters)

                    if result:
                        st.session_state.tool_execution_result = result
                    else:
                        st.error("Tool execution failed. Check API logs.")

    # Display execution result
    if (
        hasattr(st.session_state, "tool_execution_result")
        and st.session_state.tool_execution_result
    ):
        result = st.session_state.tool_execution_result

        st.markdown("---")
        st.markdown("### Execution Result")

        success = result.get("success", False)
        execution_time = result.get("executionTime", 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            status_label = "✅ Success" if success else "❌ Failed"
            st.markdown(f"**Status:** {status_label}")
        with col2:
            st.markdown(f"**Time:** {execution_time:.3f}s")
        with col3:
            st.markdown(f"**Tool:** {result.get('toolName', 'Unknown')}")

        if success:
            st.success("Tool executed successfully!")
        else:
            error_msg = result.get("error", "Unknown error")
            st.error(f"Execution failed: {error_msg}")

        payload = result.get("result")
        if payload is not None:
            with st.expander("Result Data", expanded=True):
                render_tool_result_payload(payload)

        # Clear button
        if st.button("Clear Result"):
            st.session_state.tool_execution_result = None
            st.rerun()


def render_interrupt_approval_ui():
    """Render the UI for approving/rejecting/editing tool executions"""
    interrupt_info = st.session_state.get("pending_interrupt", {})

    if not interrupt_info:
        return

    thread_id = interrupt_info.get("thread_id")
    interrupt_id = interrupt_info.get("interrupt_id")
    action_requests = interrupt_info.get("action_requests", [])

    if not action_requests:
        st.warning("No tool actions to approve")
        if st.button("Cancel"):
            st.session_state.pop("pending_interrupt", None)
            st.rerun()
        return

    st.warning("**Tool Execution Requires Approval**", icon="⏸️")
    st.markdown(
        "The AI assistant wants to execute the following tool(s). Please review and approve:"
    )

    # Initialize decisions in session state if not present
    if "pending_decisions" not in st.session_state:
        st.session_state.pending_decisions = {}

    for idx, action_request in enumerate(action_requests):
        tool_name = action_request.get("action", "unknown")
        tool_args = action_request.get("args", {})
        description = action_request.get("description", "")
        tool_call_id = (
            action_request.get("tool_call_id")
            or action_request.get("toolCallId")
            or action_request.get("id")
        )
        task_id = (
            action_request.get("task_id")
            or action_request.get("taskId")
            or tool_call_id
        )

        # Check if this tool already has a decision
        current_decision = st.session_state.pending_decisions.get(task_id)

        st.markdown(f"### Tool {idx + 1}: `{tool_name}`")
        if description:
            st.markdown(f"**Description:** {description}")
        if task_id:
            st.caption(f"Task ID: `{task_id}`")

        # Show decision status if already decided
        if current_decision:
            decision_type = current_decision.get("type", "")
            if decision_type in ("accept", "approve"):
                st.success(f"Approved", icon="✅")
            elif decision_type == "edit":
                st.info(f"Edited and approved", icon="✏️")
            elif decision_type in ("reject", "respond"):
                st.error(f"Rejected", icon="❌")

            # Option to change decision
            if st.button(f"Change decision", key=f"change_{idx}"):
                st.session_state.pending_decisions.pop(task_id, None)
                st.session_state.pop(f"editing_tool_{idx}", None)
                st.rerun()
        else:
            # Display tool arguments
            with st.expander("Tool Arguments", expanded=True):
                st.json(tool_args)

            # Decision options
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    f"Accept",
                    key=f"accept_{idx}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.pending_decisions[task_id] = {
                        "type": "accept",
                        "task_id": task_id,
                        "action": tool_name,
                        "args": None,
                    }
                    st.rerun()

            with col2:
                if st.button(f"Edit Args", key=f"edit_{idx}", use_container_width=True):
                    st.session_state[f"editing_tool_{idx}"] = True
                    st.rerun()

            with col3:
                if st.button(f"Reject", key=f"reject_{idx}", use_container_width=True):
                    st.session_state.pending_decisions[task_id] = {
                        "type": "respond",
                        "task_id": task_id,
                        "action": tool_name,
                        "args": {"message": f"User rejected execution of {tool_name}"},
                    }
                    st.rerun()

            # Show edit form if editing
            if st.session_state.get(f"editing_tool_{idx}"):
                st.markdown("**Edit Arguments:**")
                with st.form(f"edit_form_{idx}"):
                    edited_args_text = st.text_area(
                        "Arguments (JSON format)",
                        value=json.dumps(tool_args, indent=2, ensure_ascii=False),
                        height=200,
                    )

                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.form_submit_button(
                            "Save & Accept", use_container_width=True, type="primary"
                        ):
                            try:
                                edited_args = json.loads(edited_args_text)
                                st.session_state.pending_decisions[task_id] = {
                                    "type": "edit",
                                    "task_id": task_id,
                                    "action": tool_name,
                                    "args": edited_args,
                                }
                                st.session_state.pop(f"editing_tool_{idx}", None)
                                st.rerun()
                            except json.JSONDecodeError:
                                st.error("Invalid JSON format")

                    with col_cancel:
                        if st.form_submit_button("Cancel", use_container_width=True):
                            st.session_state.pop(f"editing_tool_{idx}", None)
                            st.rerun()

        if idx < len(action_requests) - 1:
            st.divider()

    # Check if all tools have decisions
    all_task_ids = set()
    for req in action_requests:
        task_id = (
            req.get("task_id")
            or req.get("taskId")
            or req.get("tool_call_id")
            or req.get("toolCallId")
            or req.get("id")
        )
        if task_id:
            all_task_ids.add(task_id)

    decided_task_ids = set(st.session_state.pending_decisions.keys())
    all_decided = all_task_ids == decided_task_ids and len(all_task_ids) > 0

    st.divider()

    # Show progress
    st.progress(len(decided_task_ids) / max(len(all_task_ids), 1))
    st.caption(f"Decided: {len(decided_task_ids)} / {len(all_task_ids)} tools")

    # Submit button - only enabled when all tools have decisions
    col_submit, col_approve_all, col_cancel = st.columns(3)

    with col_submit:
        if st.button(
            "Submit Decisions",
            use_container_width=True,
            type="primary",
        ):
            _submit_interrupt_decisions(thread_id, interrupt_id, action_requests)

    with col_approve_all:
        if st.button("Accept All", use_container_width=True):
            # Auto-approve all remaining tools
            for req in action_requests:
                task_id = (
                    req.get("task_id")
                    or req.get("taskId")
                    or req.get("tool_call_id")
                    or req.get("toolCallId")
                    or req.get("id")
                )
                if task_id and task_id not in st.session_state.pending_decisions:
                    st.session_state.pending_decisions[task_id] = {
                        "type": "accept",
                        "task_id": task_id,
                        "action": req.get("action"),
                        "args": None,
                    }
            # Submit immediately
            _submit_interrupt_decisions(thread_id, interrupt_id, action_requests)

    with col_cancel:
        if st.button("Cancel All", use_container_width=True):
            # Reject all tools
            for req in action_requests:
                task_id = (
                    req.get("task_id")
                    or req.get("taskId")
                    or req.get("tool_call_id")
                    or req.get("toolCallId")
                    or req.get("id")
                )
                if task_id:
                    st.session_state.pending_decisions[task_id] = {
                        "type": "respond",
                        "task_id": task_id,
                        "action": req.get("action"),
                        "args": {"message": "User cancelled all tool executions"},
                    }
            # Submit immediately
            _submit_interrupt_decisions(thread_id, interrupt_id, action_requests)


def _submit_interrupt_decisions(thread_id, interrupt_id, action_requests):
    """Helper function to submit interrupt decisions to the backend"""
    conversation_id = st.session_state.get("interrupt_conversation_id")
    decisions = list(st.session_state.pending_decisions.values())

    resume_payload = {
        "threadId": thread_id,
        "conversationId": conversation_id,
        "interruptId": interrupt_id,
        "decisions": decisions,
    }

    with st.spinner("Resuming execution..."):
        response = make_api_request(
            "POST", "/messages/resume-interrupt", resume_payload
        )

        if response and response.get("success"):
            response_data = response.get("data") if isinstance(response, dict) else None
            next_interrupt = (
                response_data.get("interrupt")
                if isinstance(response_data, dict)
                else None
            )

            # Clear current decisions
            st.session_state.pop("pending_decisions", None)
            for idx in range(len(action_requests)):
                st.session_state.pop(f"editing_tool_{idx}", None)

            if next_interrupt:
                st.session_state.pending_interrupt = next_interrupt
                st.session_state.interrupt_conversation_id = conversation_id
                st.toast("Additional tool approval required.", icon="⚠️")
                st.rerun()
            else:
                # Clear interrupt state
                st.session_state.pop("pending_interrupt", None)
                st.session_state.pop("interrupt_conversation_id", None)

                # Refresh messages
                st.toast("Tool execution completed!", icon="✅")
                st.session_state.conversation_messages_page = 0
                st.rerun()
        else:
            st.error("Failed to resume execution")


def render_chat_view():
    """Main chat interface"""
    conversation_id = st.session_state.get("current_conversation_id")

    # Load messages
    def load_messages_page(page: int, *, show_spinner: bool = False) -> None:
        conv_id = st.session_state.get("current_conversation_id")
        if not conv_id or conv_id == "pending_new":
            return

        fetch_page = lambda: get_messages(
            conv_id, page=page, limit=10, order_direction="desc"
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
            chunks_state = st.session_state.setdefault("message_chunks", {})
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

                    # Store chunk data for citation modals
                    documents_cited = metadata.get("documents_cited", [])
                    if documents_cited:
                        chunks_map = {}
                        for doc in documents_cited:
                            doc_num = doc.get("document_number")
                            if doc_num is not None:
                                chunks_map[doc_num] = {}
                                for chunk in doc.get("chunks", []):
                                    chunk_idx = chunk.get("chunk_index")
                                    if chunk_idx is not None:
                                        chunks_map[doc_num][chunk_idx] = {
                                            "content": chunk.get("content", ""),
                                            "score": chunk.get("score", 0.0),
                                            "page_number": chunk.get("page_number"),
                                            "character_count": chunk.get(
                                                "character_count", 0
                                            ),
                                        }
                        chunks_state[key] = {
                            "documents_cited": documents_cited,
                            "chunks_map": chunks_map,
                        }
                    else:
                        chunks_state.pop(key, None)

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

    # Show conversation title
    current_conv = next(
        (c for c in st.session_state.conversations_list if c["id"] == conversation_id),
        None,
    )

    if current_conv:
        st.markdown(f"# {current_conv['title']}")
        active_persona = current_conv.get("personaPrompt")
        if active_persona:
            st.info(f"**Instructions active:** {persona_preview(active_persona, 100)}")

        # Show planning mode status in chat view
        planning_mode = current_conv.get("planningModeEnabled", False)
        if planning_mode:
            status = get_planning_status(conversation_id)
            if status:
                progress_pct = status.get("progressPercentage", 0)
                next_task = status.get("nextTask")
                pending = status.get("pendingTasks", 0)

                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if next_task:
                            st.success(
                                f"**Current Task:** {next_task.get('description', 'N/A')} ({progress_pct:.0f}% complete, {pending} remaining)"
                            )
                        else:
                            st.success(
                                f"✅ **All tasks completed!** ({progress_pct:.0f}% complete)"
                            )
                    with col2:
                        if st.button(
                            "📋",
                            key="goto_planning_from_chat",
                            help="View Planning Tab",
                        ):
                            st.session_state.active_view = "planning"
                            st.rerun()
    elif conversation_id == "pending_new":
        st.markdown("# New Chat")
        queued_persona = st.session_state.get("pending_persona_prompt", "")
        if queued_persona:
            st.info(f"**Instructions queued:** {persona_preview(queued_persona, 100)}")
    else:
        st.markdown("# Welcome!")
        st.info(
            "Select a conversation from the sidebar or create a new chat to get started."
        )
        return

    # Load more button
    if conversation_id and conversation_id != "pending_new":
        if st.session_state.has_more_messages:
            if st.button("Load older messages", use_container_width=True):
                next_page = st.session_state.conversation_messages_page + 1
                load_messages_page(next_page, show_spinner=True)

    st.divider()

    # Messages
    messages_to_display = (
        st.session_state.messages
        if conversation_id and conversation_id != "pending_new"
        else []
    )

    if not messages_to_display and conversation_id not in (None, "pending_new"):
        st.info("💬 No messages yet. Start the conversation!")

    for msg in messages_to_display:
        sender_value = msg.get("sender")
        is_user_message = sender_value in (1, "user", "USER", "User")
        render_message_bubble(msg, is_user_message)

    st.divider()

    # Show interrupt approval UI if there's a pending interrupt
    if st.session_state.get("pending_interrupt"):
        render_interrupt_approval_ui()
        return  # Don't show input area while interrupt is pending

    # Input area
    if conversation_id:
        # Show pending attachments
        if st.session_state.pending_image_attachments:
            st.caption(
                f"{len(st.session_state.pending_image_attachments)} attachment(s) ready"
            )
            cols = st.columns(min(len(st.session_state.pending_image_attachments), 4))
            for idx, att in enumerate(st.session_state.pending_image_attachments):
                with cols[idx % len(cols)]:
                    image_bytes = base64.b64decode(att["data"])
                    st.image(image_bytes, caption=att["name"], width=80)
                    if st.button("❌", key=f"remove_{att['token']}"):
                        st.session_state.pending_image_attachments = [
                            item
                            for item in st.session_state.pending_image_attachments
                            if item["token"] != att["token"]
                        ]
                        st.rerun()

        # File uploader
        file_uploader_key = (
            f"chat_image_uploader_{conversation_id}" if conversation_id else None
        )

        if st.session_state.show_attachment_uploader and file_uploader_key:
            uploaded_files = st.file_uploader(
                "Attach images",
                type=["png", "jpg", "jpeg", "gif", "webp"],
                accept_multiple_files=True,
                key=file_uploader_key,
                help=f"Up to {_MAX_IMAGE_ATTACHMENTS} images",
            )
            if uploaded_files:
                _handle_new_image_attachments(uploaded_files)

        # Message form
        with st.form("message_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([6, 1, 1])

            with col1:
                message_content = st.text_area(
                    "Message",
                    placeholder="Type your message...",
                    height=100,
                    label_visibility="collapsed",
                    key=f"msg_input_{conversation_id}",
                )

            with col2:
                send_button = st.form_submit_button(
                    "\nSend", use_container_width=True, type="primary"
                )

            with col3:
                attach_button = st.form_submit_button(
                    "Attach", use_container_width=True
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
                    st.toast("Please enter a message", icon="⚠️")
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

                        with st.status(
                            "Creating conversation...", expanded=True
                        ) as status:
                            conv_response = make_api_request(
                                "POST", "/conversations/", conversation_data
                            )
                            if conv_response and conv_response.get("data"):
                                new_conversation = conv_response["data"]
                                st.session_state.current_conversation_id = (
                                    new_conversation["id"]
                                )
                                refresh_conversations_list(
                                    fallback_conversation=new_conversation
                                )
                                reset_conversation_state()
                                st.session_state.pending_image_attachments = (
                                    saved_attachments
                                )
                                conversation_id = (
                                    st.session_state.current_conversation_id
                                )
                                pending_attachments = list(saved_attachments)
                                status.update(
                                    label="Conversation created!", state="complete"
                                )
                            else:
                                st.toast("Failed to create conversation", icon="❌")
                                return

                    message_data = {
                        "content": message_to_send,
                        "conversationId": st.session_state.current_conversation_id,
                    }

                    if pending_attachments:
                        message_data["attachments"] = pending_attachments

                    # Use streaming endpoint for real-time response
                    with st.status("Sending message...", expanded=True) as status:
                        # Create placeholder for streaming response
                        response_placeholder = st.empty()
                        thinking_placeholder = st.empty()
                        accumulated_content = ""  # Initialize empty for accumulation
                        accumulated_thinking = ""  # Accumulate thinking content
                        final_message = None
                        interrupt_data = None
                        selected_agent = None  # Track which agent is processing

                        # Stream the response
                        for event in make_streaming_request(
                            "/messages/stream", message_data
                        ):
                            event_type = event.get("type")

                            if event_type == "user_message_created":
                                status.update(
                                    label="Generating response...", state="running"
                                )

                            elif event_type == "agent_selected":
                                # Track which agent was selected for processing
                                selected_agent = event.get("agent", "unknown")
                                status.update(
                                    label=f"{selected_agent.replace('_', ' ').title()} processing...",
                                    state="running",
                                )

                            elif event_type == "thinking":
                                # Accumulate and display thinking content with animated indicator
                                content = event.get("content", "")
                                accumulated_thinking += content
                                with thinking_placeholder.container():
                                    # Animated thinking header with dots
                                    st.markdown(
                                        """<div class="thinking-container">
                                            <div class="thinking-header">
                                                <span class="thinking-indicator">
                                                    Thinking
                                                    <span class="thinking-dots">
                                                        <span class="thinking-dot"></span>
                                                        <span class="thinking-dot"></span>
                                                        <span class="thinking-dot"></span>
                                                    </span>
                                                </span>
                                            </div>
                                            <div class="thinking-content">"""
                                        + html.escape(accumulated_thinking)
                                        + """</div>
                                        </div>""",
                                        unsafe_allow_html=True,
                                    )
                                status.update(label="Thinking...", state="running")

                            elif event_type == "token":
                                # Accumulate and display tokens in real-time
                                content = event.get("content", "")
                                accumulated_content += (
                                    content  # Append each token chunk
                                )
                                # Collapse thinking to expander when answer starts
                                if accumulated_thinking and accumulated_content:
                                    with thinking_placeholder.container():
                                        with st.expander(
                                            "Thought Process", expanded=False
                                        ):
                                            st.markdown(
                                                f"""<div class="thinking-container">
                                                    <div class="thinking-content">{html.escape(accumulated_thinking)}</div>
                                                </div>""",
                                                unsafe_allow_html=True,
                                            )
                                # Display with native markdown for LaTeX support
                                response_placeholder.markdown(accumulated_content)

                            elif event_type == "tool":
                                # Show tool execution
                                tool_name = event.get("name", "unknown")
                                tool_status = event.get("status", "running")
                                status_icon = "✓" if tool_status == "success" else "⚠"
                                status.update(
                                    label=f"Tool: {tool_name} {status_icon}",
                                    state="running",
                                )

                            elif event_type == "interrupt":
                                # Workflow paused for human approval
                                thread_id = event.get("thread_id")
                                pending_tool_calls = (
                                    event.get("pending_tool_calls") or []
                                )
                                # Extract the full interrupt response data
                                interrupt_data = event.get("interrupt")

                                status.update(
                                    label="⏸ Workflow paused - Tool approval required",
                                    state="running",
                                )

                                # Store interrupt state in session for the approval UI
                                if interrupt_data:
                                    st.session_state.pending_interrupt = interrupt_data
                                    st.session_state.interrupt_conversation_id = (
                                        conversation_id
                                    )
                                else:
                                    # No interrupt data - this shouldn't happen with proper HITL setup
                                    st.error(
                                        "❌ Interrupt detected but no interrupt data provided. Check HITL configuration."
                                    )

                                # Display info message
                                st.info(
                                    "🔧 The assistant wants to use tools. Please review and approve below."
                                )

                                # Stop processing further events and rerun to show approval UI
                                break

                            elif event_type == "complete":
                                # Store final message and complete
                                final_message = event.get("message")
                                status.update(label="Message sent!", state="complete")

                            elif event_type == "error":
                                # Handle error
                                error_msg = event.get("error", "Unknown error")
                                status.update(
                                    label=f"Error: {error_msg}", state="error"
                                )
                                st.toast(f"Error: {error_msg}", icon="❌")
                                break

                        # Handle interrupt - show approval UI
                        if st.session_state.get("pending_interrupt"):
                            st.rerun()

                        # If successful, update UI
                        if final_message:
                            st.session_state.pending_image_attachments = []
                            refresh_conversations_list()
                            reset_conversation_state()
                            st.session_state.show_attachment_uploader = False
                            load_messages_page(1)
                            st.toast("Message sent!", icon="✅")
                            st.rerun()
                        elif event_type != "error" and not interrupt_data:
                            st.toast("Failed to send message", icon="❌")
    # Note: Legacy tool approval UI removed. Now using modern interrupt-based approval in render_interrupt_approval_ui()


def render_manage_modal():
    """Conversation management modal dialog"""
    should_show = st.session_state.get(CONVERSATION_MANAGER_DIALOG_KEY, False)

    st.session_state.conversation_manager_visible = should_show
    st.session_state.show_conversation_manager = should_show

    if not should_show:
        return

    @st.dialog(
        "Manage Conversations",
        width="large",
    )
    def manage_dialog():
        def _deduplicate_conversations(
            conversations: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            """Return conversations with duplicate IDs removed, preserving order."""
            seen = set()
            deduped: List[Dict[str, Any]] = []
            for conv in conversations:
                conv_id = str(conv.get("id", ""))
                if not conv_id or conv_id in seen:
                    continue
                seen.add(conv_id)
                deduped.append(conv)
            return deduped

        if st.session_state.current_user_id:
            with st.status("Loading conversations...", expanded=False):
                # Get conversations with latest 3 messages for preview
                manager_conversations_response = get_conversations(
                    include_messages=True, latest_messages=3, fetch_all_pages=True
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

        # Search
        search_term = st.text_input(
            "Search conversations", placeholder="Type to search..."
        )

        if manager_conversations:
            manager_conversations = _deduplicate_conversations(manager_conversations)

            if search_term:
                filtered_map: Dict[str, Dict[str, Any]] = {}
                search_lower = search_term.lower()

                for conv in manager_conversations:
                    conv_id = str(conv.get("id", ""))
                    if not conv_id:
                        continue

                    if search_lower in (conv.get("title") or "").lower():
                        filtered_map.setdefault(conv_id, conv)
                        continue

                    messages = conv.get("messages") or []
                    for msg in messages:
                        if search_lower in (msg.get("content") or "").lower():
                            filtered_map.setdefault(conv_id, conv)
                            break

                filtered_convs = list(filtered_map.values())
            else:
                filtered_convs = manager_conversations

            display_conversations = _deduplicate_conversations(filtered_convs)

            if display_conversations:
                st.caption(f"Found {len(display_conversations)} conversation(s)")

                for idx, conv in enumerate(display_conversations):
                    conv_id = conv.get("id")
                    conv_id_str = str(conv_id) if conv_id is not None else ""
                    conv_title = conv.get("title") or "Untitled conversation"
                    with st.expander(conv_title, expanded=False):
                        # Show stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            messages = conv.get("messages") or []
                            message_count = conv.get("messageCount", len(messages))
                            st.metric("Messages", message_count)
                        with col2:
                            created = format_time(conv.get("createdAt", ""))
                            st.metric("Created", created)
                        with col3:
                            persona = conv.get("personaPrompt")
                            st.metric("Persona", "Yes" if persona else "No")

                        # Show latest 3 message previews
                        if conv.get("messages"):
                            st.markdown("**Recent messages:**")
                            for msg in conv["messages"][:3]:
                                sender_value = msg.get("sender")
                                sender_tag = (
                                    "USER"
                                    if sender_value in (1, "user", "USER", "User")
                                    else "ASSISTANT"
                                )
                                preview = msg.get("content", "")[:80]
                                st.caption(f"{sender_tag}: {preview}...")
                        else:
                            st.caption("No messages")

                        # Actions
                        col1, col2 = st.columns(2)
                        with col1:
                            open_button_key = (
                                f"conversation_manager_open_{conv_id_str}_{idx}"
                                if conv_id_str
                                else f"conversation_manager_open_{idx}"
                            )
                            if st.button(
                                "Open",
                                key=open_button_key,
                                use_container_width=True,
                                type="primary",
                            ):
                                if conv_id is not None:
                                    st.session_state.current_conversation_id = conv_id
                                    close_conversation_manager()
                                    reset_conversation_state()
                                    st.rerun()
                                else:
                                    st.toast("Conversation is missing an ID", icon="⚠️")
                        with col2:
                            delete_button_key = (
                                f"conversation_manager_delete_{conv_id_str}_{idx}"
                                if conv_id_str
                                else f"conversation_manager_delete_{idx}"
                            )
                            if st.button(
                                "Delete",
                                key=delete_button_key,
                                use_container_width=True,
                            ):
                                if conv_id is None:
                                    st.toast("Conversation is missing an ID", icon="⚠️")
                                else:
                                    result = make_api_request(
                                        "DELETE", f"/conversations/{conv_id}"
                                    )
                                    if result:
                                        st.session_state.conversations_list = []
                                        if (
                                            st.session_state.current_conversation_id
                                            == conv_id
                                        ):
                                            st.session_state.current_conversation_id = (
                                                None
                                            )
                                            reset_conversation_state()
                                        refresh_conversations_list()
                                        st.toast(
                                            f"Deleted '{conv.get('title', 'Conversation')}'",
                                            icon="✅",
                                        )
                                        st.rerun()
            else:
                st.info("No conversations found matching your search.")
        else:
            st.info("No conversations available.")
        if st.button("Close", use_container_width=True):
            close_conversation_manager()
            st.rerun()

    manage_dialog()
    st.session_state[CONVERSATION_MANAGER_DIALOG_KEY] = False


def render_chunk_preview_modal():
    """Chunk preview modal dialog for viewing retrieved document chunks"""
    should_show = st.session_state.get("chunk_preview_dialog_key", False)

    if not should_show:
        return

    @st.dialog("Document Chunk Preview", width="large")
    def chunk_preview_dialog():
        selected_info = st.session_state.get("selected_chunk_info")

        if not selected_info:
            st.warning("No chunk information available.")
            if st.button("Close", use_container_width=True):
                st.session_state["chunk_preview_dialog_key"] = False
                st.rerun()
            return

        # Extract chunk information
        doc_source = selected_info.get("source", "Unknown")
        doc_num = selected_info.get("document_number", "?")
        chunk_idx = selected_info.get("chunk_index")
        content = selected_info.get("content", "")
        score = selected_info.get("score", 0.0)
        page_num = selected_info.get("page_number")
        char_count = selected_info.get("character_count", len(content))

        # Determine relevance color
        if score >= 0.7:
            relevance_color = COLORS["success"]
        elif score >= 0.5:
            relevance_color = COLORS["warning"]
        else:
            relevance_color = COLORS["error"]

        # Header information
        st.markdown(f"### [Document {doc_num}] {doc_source}")

        # Metadata in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            if chunk_idx is not None:
                st.metric("Chunk Index", f"#{chunk_idx}")
            else:
                st.metric("Chunk Index", "N/A")
        with col2:
            st.markdown(
                f'<div style="padding: 10px; text-align: center;">'
                f'<div style="color: {relevance_color}; font-size: 1.5em; font-weight: bold;">{score:.1%}</div>'
                f'<div style="font-size: 0.9em; color: #888;">Relevance Score</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col3:
            if page_num is not None:
                st.metric("Page Number", page_num)
            else:
                st.metric("Page Number", "N/A")

        st.divider()

        # Content display
        if content:
            st.markdown("**Chunk Content:**")
            st.text_area(
                "Content",
                value=content,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
            st.caption(f"📊 Character count: {char_count}")
        else:
            st.info("Chunk content is not available.")

        st.divider()

        # Close button
        if st.button("Close", use_container_width=True):
            st.session_state["chunk_preview_dialog_key"] = False
            st.session_state["selected_chunk_info"] = None
            st.rerun()

    chunk_preview_dialog()
    st.session_state["chunk_preview_dialog_key"] = False


def render_documents_tab():
    """Documents management workspace (moved from the sidebar)."""
    st.markdown("# 📄 Documents")

    conversation_id = st.session_state.get("current_conversation_id")
    if conversation_id in (None, "pending_new"):
        st.info(
            "Select an existing conversation to upload or manage documents. "
            "Start a new chat and send your first message to unlock uploads."
        )
        return

    current_conv = next(
        (
            conv
            for conv in st.session_state.conversations_list
            if conv.get("id") == conversation_id
        ),
        None,
    )
    if current_conv:
        title = current_conv.get("title") or "Conversation"
        st.caption(f"Managing documents for **{title}**")
    else:
        st.caption("Managing documents for the active conversation.")

    st.markdown(
        "Upload supporting files and monitor their processing status for retrieval."
    )

    upload_col, tips_col = st.columns([1.25, 1])

    with upload_col:
        st.subheader("Upload a Document")
        st.caption(
            "Files attach to this conversation and become searchable once processing completes."
        )
        uploader_key = f"doc_uploader_{conversation_id}"
        uploaded_file = st.file_uploader(
            "Select a file",
            type=["txt", "pdf", "docx", "md"],
            key=uploader_key,
            help="Supported formats: TXT, PDF, DOCX, MD",
        )

        if uploaded_file is not None:
            file_size_kb = uploaded_file.size / 1024
            st.write(f"**Selected:** {uploaded_file.name} ({file_size_kb:.1f} KB)")

            if st.button(
                "Upload & Process",
                key=f"upload_doc_{conversation_id}",
                use_container_width=True,
                type="primary",
            ):
                with st.spinner("Uploading document..."):
                    upload_result = upload_document(uploaded_file)

                if upload_result:
                    st.success("Upload complete. Processing has started.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Upload failed. Please try again.")

    with tips_col:
        st.subheader("Workspace Tips")
        st.markdown(
            "- Keep documents focused on the active conversation.\n"
            "- Replace outdated files to avoid stale context.\n"
            "- Refresh the library to pick up the latest status updates."
        )
        if st.button(
            "Refresh document list",
            key="documents_refresh",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.rerun()

    def _format_timestamp(value: Optional[str]) -> str:
        if not value:
            return "Unknown"
        try:
            dt_value = parser.parse(value)
            return dt_value.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return value[:16]

    st.markdown("---")

    header_col, metric_col = st.columns([3, 1])
    with header_col:
        st.subheader("Conversation Library")

    docs_response = get_uploaded_documents()
    if not docs_response:
        st.warning("Unable to load documents. Try refreshing the list.")
        return

    doc_data = docs_response.get("data") or {}
    documents = doc_data.get("documents") or []
    total_docs = doc_data.get("total", len(documents))

    with metric_col:
        st.metric("Documents", total_docs)

    if not documents:
        st.info("No documents uploaded yet. Use the uploader above to add one.")
        return

    status_map = {
        1: {"label": "Processing", "icon": "⏳", "help": "Indexing in progress"},
        2: {"label": "Ready", "icon": "✅", "help": "Available for retrieval"},
        3: {"label": "Failed", "icon": "⚠️", "help": "Processing failed"},
    }

    for doc in documents:
        status = status_map.get(
            doc.get("status"),
            {"label": "Unknown", "icon": "❔", "help": "Status unavailable"},
        )

        with st.container():
            info_col, status_col, action_col = st.columns([3, 1.2, 1])

            with info_col:
                st.markdown(f"**{doc.get('filename', 'Untitled document')}**")
                st.caption(f"Uploaded {_format_timestamp(doc.get('upload_time'))}")

            with status_col:
                st.markdown(f"{status['icon']} **{status['label']}**")
                st.caption(status["help"])

            with action_col:
                if st.button(
                    "Delete",
                    key=f"delete_doc_{doc.get('id')}",
                    use_container_width=True,
                ):
                    with st.spinner("Removing document..."):
                        if delete_document(doc.get("id")):
                            st.cache_data.clear()
                            st.toast("Document deleted.", icon="🗑️")
                            st.rerun()
                        else:
                            st.error("Delete failed. Please try again.")

        st.divider()


# ==================== PLANNING TAB ====================

TASK_STATUS_MAP = {
    "pending": {"label": "Pending", "icon": "⏳", "color": "#f59e0b"},
    "in_progress": {"label": "In Progress", "icon": "🔄", "color": "#3b82f6"},
    "completed": {"label": "Completed", "icon": "✅", "color": "#10b981"},
    "skipped": {"label": "Skipped", "icon": "⏭️", "color": "#64748b"},
}


def get_task_plans(
    conversation_id: str, include_completed: bool = True
) -> List[Dict[str, Any]]:
    """Fetch task plans for a conversation."""
    endpoint = f"/conversations/{conversation_id}/task-plans?include_completed={str(include_completed).lower()}"
    response = make_api_request("GET", endpoint)
    if response and response.get("success"):
        return response.get("data", [])
    return []


def get_planning_status(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Fetch planning status for a conversation."""
    endpoint = f"/conversations/{conversation_id}/planning-status"
    response = make_api_request("GET", endpoint)
    if response and response.get("success"):
        return response.get("data")
    return None


def create_task_plan_ai(
    conversation_id: str, user_message: str
) -> Optional[List[Dict[str, Any]]]:
    """Create a task plan using AI from user message."""
    endpoint = f"/conversations/{conversation_id}/task-plans"
    response = make_api_request("POST", endpoint, {"userMessage": user_message})
    if response and response.get("success"):
        return response.get("data", [])
    return None


def create_task_plan_manual(
    conversation_id: str, descriptions: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """Create task plans manually from a list of descriptions."""
    endpoint = f"/conversations/{conversation_id}/task-plans/manual"
    response = make_api_request("POST", endpoint, {"taskDescriptions": descriptions})
    if response and response.get("success"):
        return response.get("data", [])
    return None


def update_task_plan(
    task_id: str, update_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Update a task plan."""
    endpoint = f"/task-plans/{task_id}"
    response = make_api_request("PATCH", endpoint, update_data)
    if response and response.get("success"):
        return response.get("data")
    return None


def complete_task_plan(task_id: str) -> Optional[Dict[str, Any]]:
    """Mark a task as completed."""
    endpoint = f"/task-plans/{task_id}/complete"
    response = make_api_request("POST", endpoint)
    if response and response.get("success"):
        return response.get("data")
    return None


def delete_task_plan(task_id: str) -> bool:
    """Delete a task plan."""
    endpoint = f"/task-plans/{task_id}"
    response = make_api_request("DELETE", endpoint)
    return response and response.get("success", False)


def render_planning_tab():
    """Render the Planning tab for managing task plans."""
    st.markdown("# 📋 Planning")

    conversation_id = st.session_state.get("current_conversation_id")
    is_new_conversation = conversation_id == "pending_new"
    has_conversation = conversation_id not in (None, "pending_new")

    if conversation_id is None:
        st.info("Select a conversation or create a new chat to manage plans.")
        return

    if is_new_conversation:
        st.info(
            "Create a conversation first by sending a message, then you can create task plans."
        )
        return

    # Get current conversation info
    current_conv: Optional[Dict[str, Any]] = next(
        (
            conv
            for conv in st.session_state.conversations_list
            if conv.get("id") == conversation_id
        ),
        None,
    )

    if current_conv:
        st.markdown(f"**Conversation:** {current_conv.get('title', 'Untitled')}")

    # Fetch planning status
    status = get_planning_status(conversation_id)

    # Progress Section (if tasks exist)
    if status and status.get("totalTasks", 0) > 0:
        st.markdown("### Progress")

        total = status.get("totalTasks", 0)
        completed = status.get("completedTasks", 0)
        pending = status.get("pendingTasks", 0)
        in_progress = status.get("inProgressTasks", 0)
        skipped = status.get("skippedTasks", 0)
        progress_pct = status.get("progressPercentage", 0)

        # Progress bar
        st.progress(progress_pct / 100)
        st.caption(f"{progress_pct:.0f}% complete ({completed}/{total} tasks)")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏳ Pending", pending)
        with col2:
            st.metric("🔄 In Progress", in_progress)
        with col3:
            st.metric("✅ Completed", completed)
        with col4:
            st.metric("⏭️ Skipped", skipped)

        # Next task info
        next_task = status.get("nextTask")
        if next_task:
            st.markdown("#### Next Task")
            st.info(
                f"**Task {next_task.get('taskOrder', 0) + 1}:** {next_task.get('description', 'No description')}"
            )

    st.divider()

    # Create New Plan Section
    st.markdown("### Create Task Plan")

    tab_ai, tab_manual = st.tabs(["AI Generated", "Manual Entry"])

    # Check if we need to clear inputs (set by successful task creation)
    if st.session_state.get("clear_planning_generate_input"):
        st.session_state.planning_generate_input = ""
        del st.session_state.clear_planning_generate_input
    if st.session_state.get("clear_planning_manual_input"):
        st.session_state.planning_manual_input = ""
        del st.session_state.clear_planning_manual_input

    with tab_ai:
        st.markdown(
            "Describe your goal or project and the AI will create a structured plan for you."
        )

        ai_input = st.text_area(
            "Describe your goal",
            key="planning_generate_input",
            height=100,
            placeholder="Example: Build a REST API with user authentication, CRUD operations for products, and integrate with a payment gateway.",
        )

        if st.button(
            "Generate Plan",
            key="generate_plan_btn",
            type="primary",
            disabled=not ai_input.strip(),
        ):
            with st.spinner("Generating task plan..."):
                tasks = create_task_plan_ai(conversation_id, ai_input.strip())
                if tasks:
                    st.toast(f"Created {len(tasks)} tasks!", icon="✅")
                    st.session_state.clear_planning_generate_input = True
                    st.rerun()
                else:
                    st.toast("Failed to generate plan. Try again.", icon="❌")

    with tab_manual:
        st.markdown("Enter tasks manually, one per line.")

        manual_input = st.text_area(
            "Task list (one per line)",
            key="planning_manual_input",
            height=150,
            placeholder="Set up project structure\nCreate database models\nImplement authentication\nWrite unit tests",
        )

        if st.button(
            "Create Tasks",
            key="create_manual_tasks_btn",
            type="primary",
            disabled=not manual_input.strip(),
        ):
            descriptions = [
                line.strip()
                for line in manual_input.strip().split("\n")
                if line.strip()
            ]
            if descriptions:
                with st.spinner("Creating tasks..."):
                    tasks = create_task_plan_manual(conversation_id, descriptions)
                    if tasks:
                        st.toast(f"Created {len(tasks)} tasks!", icon="✅")
                        st.session_state.clear_planning_manual_input = True
                        st.rerun()
                    else:
                        st.toast("Failed to create tasks. Try again.", icon="❌")
            else:
                st.warning("Please enter at least one task description.")

    st.divider()

    # Task List Section
    st.markdown("### Task List")

    col1, col2 = st.columns([3, 1])
    with col1:
        show_completed = st.checkbox(
            "Show completed tasks", value=True, key="show_completed_tasks"
        )
    with col2:
        if st.button("Refresh", key="refresh_tasks_btn", use_container_width=True):
            st.rerun()

    tasks = get_task_plans(conversation_id, include_completed=show_completed)

    if not tasks:
        st.info("No tasks yet. Create a plan above to get started!")
    else:
        for task in tasks:
            task_id = task.get("id")
            task_order = task.get("taskOrder", 0)
            description = task.get("description", "No description")
            task_status = task.get("status", "pending")
            status_info = TASK_STATUS_MAP.get(task_status, TASK_STATUS_MAP["pending"])
            is_ad_hoc = task.get("taskMetadata", {}).get("ad_hoc", False)
            completed_at = task.get("completedAt")

            # Task card styling
            border_color = status_info["color"]
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border-left: 4px solid {border_color};
                        padding: 12px 16px;
                        margin: 8px 0;
                        background: #f8fafc;
                        border-radius: 0 8px 8px 0;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-weight: 600; color: #1f2937;">
                                    {status_info['icon']} Task {task_order + 1}
                                </span>
                            </div>
                            <span style="color: {border_color}; font-size: 0.85rem; font-weight: 500;">
                                {status_info['label']}
                            </span>
                        </div>
                        <p style="margin: 8px 0 0 0; color: #374151;">{html.escape(description)}</p>
                        {f'<p style="margin: 4px 0 0 0; color: #6b7280; font-size: 0.8rem;">Completed: {completed_at[:16] if completed_at else "N/A"}</p>' if task_status == "completed" else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Action buttons
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

                with col1:
                    if task_status == "pending":  # Pending
                        if st.button(
                            "▶Start", key=f"start_{task_id}", use_container_width=True
                        ):
                            result = update_task_plan(
                                task_id, {"status": "in_progress"}
                            )
                            if result:
                                st.toast("Task started!", icon="🔄")
                                st.rerun()

                with col2:
                    if task_status in (
                        "pending",
                        "in_progress",
                    ):  # Pending or In Progress
                        if st.button(
                            "Complete",
                            key=f"complete_{task_id}",
                            use_container_width=True,
                        ):
                            result = complete_task_plan(task_id)
                            if result:
                                st.toast("Task completed!", icon="✅")
                                st.rerun()

                with col3:
                    if task_status in (
                        "pending",
                        "in_progress",
                    ):  # Pending or In Progress
                        if st.button(
                            "⏭Skip", key=f"skip_{task_id}", use_container_width=True
                        ):
                            result = update_task_plan(task_id, {"status": "skipped"})
                            if result:
                                st.toast("Task skipped!", icon="⏭️")
                                st.rerun()

                with col4:
                    if st.button(
                        "Delete", key=f"delete_{task_id}", use_container_width=True
                    ):
                        if delete_task_plan(task_id):
                            st.toast("Task deleted!", icon="🗑️")
                            st.rerun()
                        else:
                            st.toast("Failed to delete task.", icon="❌")

                st.markdown("---")


def render_settings_view():
    """Settings and instructions view"""
    st.markdown("# ⚙️ Instructions")

    conversation_id = st.session_state.get("current_conversation_id")
    is_new_conversation = conversation_id == "pending_new"
    has_conversation = conversation_id not in (None, "pending_new")

    if conversation_id is None:
        st.info("Select a conversation or create a new chat to configure instructions.")
        return

    # Persona editor
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

    if st.session_state.persona_editor_origin != conversation_id:
        if has_conversation and current_conv:
            initial_value = current_conv.get("personaPrompt") or ""
        elif is_new_conversation:
            initial_value = st.session_state.get("pending_persona_prompt", "")
        else:
            initial_value = ""
        st.session_state.persona_editor_origin = conversation_id
        st.session_state.persona_editor_value = initial_value or ""

    if has_conversation:
        title = current_conv.get("title") if current_conv else "Conversation"
        st.info(f"Editing instructions for: **{title}**")
    else:
        st.info("These instructions will be applied to your new chat")

    # Templates
    with st.expander("Template Library", expanded=False):
        cols = st.columns(len(PERSONA_TEMPLATES))
        for idx, (label, template) in enumerate(PERSONA_TEMPLATES.items()):
            with cols[idx]:
                if st.button(label, key=f"template_{idx}", use_container_width=True):
                    st.session_state.persona_editor_value = template[
                        :_MAX_PERSONA_LENGTH
                    ]
                    st.rerun()

    # Editor
    current_value = st.text_area(
        "Custom Instructions",
        key="persona_editor_value",
        height=200,
        placeholder="Describe how the AI should behave (optional)",
        help=f"Max {_MAX_PERSONA_LENGTH} characters",
    )

    char_count = len(current_value)
    exceeds_limit = char_count > _MAX_PERSONA_LENGTH

    col1, col2 = st.columns([5, 1])
    with col1:
        progress = min(char_count / _MAX_PERSONA_LENGTH, 1.0)
        st.progress(progress)
    with col2:
        st.caption(f"{char_count}/{_MAX_PERSONA_LENGTH}")

    if exceeds_limit:
        st.error("⚠️ Character limit exceeded. Please shorten your instructions.")

    # Actions
    col1, col2 = st.columns(2)

    with col1:
        if is_new_conversation:
            if st.button(
                "Apply to New Chat",
                use_container_width=True,
                disabled=exceeds_limit,
                type="primary",
            ):
                sanitized = normalize_persona_input(current_value)
                st.session_state.pending_persona_prompt = sanitized
                st.session_state.persona_editor_pending_value = sanitized
                st.session_state.persona_editor_pending = True
                st.toast("Persona saved for new chat!", icon="✅")
                st.session_state.active_view = "chat"
                st.rerun()
        else:
            if st.button(
                "Save Persona",
                use_container_width=True,
                disabled=exceeds_limit,
                type="primary",
            ):
                sanitized = normalize_persona_input(current_value)
                payload = {"personaPrompt": sanitized or None}
                response = make_api_request(
                    "PATCH", f"/conversations/{conversation_id}", payload
                )
                if response and response.get("data"):
                    refresh_conversations_list()
                    st.toast("Persona updated!", icon="✅")
                    st.session_state.active_view = "chat"
                    st.rerun()
                else:
                    st.toast("Failed to update persona", icon="❌")

    with col2:
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
                    st.toast("Persona removed!", icon="✅")
                    st.rerun()


def main():
    """Main application entry point"""
    if (
        st.session_state.show_login
        or not st.session_state.current_user_id
        or not st.session_state.auth_token
    ):
        render_login_page()
        return

    render_sidebar()

    # Show manage modal if active
    render_manage_modal()

    # Show chunk preview modal if active
    render_chunk_preview_modal()

    # Tab-based navigation across primary workspaces
    tab_chat, tab_planning, tab_docs, tab_instructions, tab_mcp = st.tabs(
        ["💬 Chat", "📋 Planning", "📄 Documents", "⚙️ Instructions", "🔧 MCP Config"]
    )

    with tab_chat:
        render_chat_view()

    with tab_planning:
        render_planning_tab()

    with tab_docs:
        render_documents_tab()

    with tab_instructions:
        render_settings_view()

    with tab_mcp:
        render_tools_tab()


if __name__ == "__main__":
    main()
