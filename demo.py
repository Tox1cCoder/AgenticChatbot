# ruff: noqa: E501

import base64
import contextlib
import hashlib
import html
import json
import mimetypes
import os
import re
import time
import uuid
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from datetime import time as datetime_time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

import markdown as _markdown  # type: ignore
import requests
import streamlit as st  # type: ignore
import streamlit.components.v1 as _stc
from dateutil import parser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.services.event_streaming.compat import infer_tool_state, normalize_tool_phase
from app.ui.clipboard_image_capture import capture_pasted_images
from app.ui.hitl_decisions import (
    approval_tool_label,
    attach_stream_context,
    build_interrupt_decision,
    interrupt_allowed_decisions,
    interrupt_request_target_ids,
    interrupt_stream_context,
)
from app.ui.hitl_recovery import (
    extract_error_code,
    is_recoverable_resume_conflict,
    reconciliation_action,
    should_suppress_pending_interrupt,
)
from app.ui.rag_artifacts import (
    RAGArtifactView,
    RAGChunkView,
    RAGDocumentListing,
    extract_rag_artifact_views,
)
from app.ui.stream_markdown import normalize_stream_markdown_text
from app.ui.subagent_activity import (
    build_live_subagent_activity_view,
    build_subagent_activity_view,
)
from upload_support import (
    delete_document,
    get_uploaded_documents,
    upload_documents,
)

# demo.py is a UI for the local client_backend sidecar (not the canonical server).
API_BASE_URL = os.environ.get("CHATBOT_API_BASE_URL", "http://127.0.0.1:8100")
WIDGET_WS_BASE_URL = os.environ.get("CHATBOT_WIDGET_WS_BASE_URL", "").rstrip("/")
REQUEST_TIMEOUT = (5, 30)
STREAM_REQUEST_TIMEOUT = (10, 900)

_MAX_PERSONA_LENGTH = 8000
_PENDING_IMAGE_PREVIEW_COLUMNS = 8
_PENDING_IMAGE_PREVIEW_WIDTH = 72
TRACE_PREVIEW_CHAR_LIMIT = 500
_PLACEHOLDER_CONVERSATION_TITLES = {
    "",
    "new conversation",
    "untitled",
    "untitled conversation",
}

PERSONA_TEMPLATES: dict[str, str] = {
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
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,200..700,0..1,-50..200');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .material-symbols-outlined {
        font-variation-settings:
            'FILL' 0,
            'wght' 500,
            'GRAD' 0,
            'opsz' 20;
        font-size: 1em;
        line-height: 1;
        vertical-align: middle;
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

    /* Thinking/Reasoning UI Styles */
    .thinking-container {
        border-left: 3px solid #3b82f6;
        padding: 10px 12px;
        margin: 6px 0 12px 0;
        background: #f8fafc;
        border-radius: 0 8px 8px 0;
        font-size: 0.88em;
        color: #4b5563;
        border-top: 1px solid #dbeafe;
        border-right: 1px solid #dbeafe;
        border-bottom: 1px solid #dbeafe;
    }

    .thinking-content {
        line-height: 1.55;
        white-space: pre-wrap;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #374151;
    }

    .thinking-content-rendered {
        line-height: 1.55;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #374151;
        white-space: normal;
    }

    .thinking-content-rendered p {
        margin: 0.35em 0;
    }

    .thinking-content-rendered p:first-child {
        margin-top: 0;
    }

    .thinking-content-rendered p:last-child {
        margin-bottom: 0;
    }

    .thinking-content strong,
    .thinking-content-rendered strong {
        font-weight: 600;
        color: #1f2937;
    }

    .trace-section-title {
        margin: 0.35rem 0 0.75rem 0;
        color: #1d4ed8;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .trace-text-label {
        margin-bottom: 0.4rem;
        color: #1e3a8a;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .trace-tool-card {
        margin: 0.8rem 0 1rem 0;
        padding: 0.85rem 0.95rem;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: #ffffff;
    }

    .trace-tool-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        flex-wrap: wrap;
    }

    .trace-tool-title {
        color: #0f172a;
        font-size: 0.96rem;
        font-weight: 600;
    }

    .trace-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.28rem 0.65rem;
        border: 1px solid transparent;
        font-size: 0.76rem;
        font-weight: 700;
        line-height: 1;
        white-space: nowrap;
    }

    .trace-status-running {
        color: #1d4ed8;
        background: rgba(59, 130, 246, 0.12);
        border-color: rgba(59, 130, 246, 0.22);
    }

    .trace-status-completed {
        color: #047857;
        background: rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.24);
    }

    .trace-status-error {
        color: #b91c1c;
        background: rgba(239, 68, 68, 0.12);
        border-color: rgba(239, 68, 68, 0.24);
    }

    .trace-status-rejected {
        color: #b45309;
        background: rgba(245, 158, 11, 0.15);
        border-color: rgba(245, 158, 11, 0.28);
    }

    .trace-status-unknown {
        color: #475569;
        background: rgba(148, 163, 184, 0.16);
        border-color: rgba(148, 163, 184, 0.28);
    }

    .subagent-activity-shell {
        margin: 0.7rem 0 0.85rem 0;
        padding: 0.8rem 0.95rem;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        background: #f8fafc;
    }

    .subagent-activity-live {
        border-color: #bfdbfe;
        border-left-color: #2563eb;
        background: #eff6ff;
    }

    .subagent-activity-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        flex-wrap: wrap;
    }

    .subagent-activity-title {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        color: #0f172a;
        font-size: 0.92rem;
        font-weight: 700;
    }

    .subagent-activity-meta {
        color: #475569;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .subagent-worker-row {
        margin: 0.45rem 0;
        padding: 0.68rem 0.78rem;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: #ffffff;
    }

    .subagent-worker-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.65rem;
        flex-wrap: wrap;
        margin-bottom: 0.35rem;
    }

    .subagent-worker-name {
        color: #0f172a;
        font-size: 0.86rem;
        font-weight: 700;
    }

    .subagent-worker-summary {
        color: #334155;
        font-size: 0.85rem;
        line-height: 1.45;
    }

    .subagent-worker-thinking {
        margin: 0.3rem 0;
        padding: 0.35rem 0.6rem;
        border-left: 3px solid #c7d2fe;
        background: #f8fafc;
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.4;
        font-style: italic;
        max-height: 9rem;
        overflow-y: auto;
    }

    .subagent-worker-model {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        margin: 0.3rem 0 0 0;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: ui-monospace, "JetBrains Mono", "SFMono-Regular", monospace;
    }

    .subagent-worker-model-effort {
        color: #4338ca;
        font-weight: 700;
    }

    .subagent-worker-model-override {
        background: #fef3c7;
        color: #92400e;
    }

    .trace-preview-label {
        margin: 0.8rem 0 0.35rem 0;
        color: #64748b;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .trace-preview {
        padding: 0.7rem 0.8rem;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: #f8fafc;
    }

    .trace-preview pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        color: #1e293b;
        font-size: 0.84rem;
        line-height: 1.55;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }

    .trace-note {
        margin-top: 0.7rem;
        color: #475569;
        font-size: 0.84rem;
    }

    /* Collapsed thinking expander styles */
    .thinking-expander-header {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #3b82f6;
        font-weight: 500;
        cursor: pointer;
        transition: color 0.2s ease;
    }

    .thinking-expander-header:hover {
        color: #1d4ed8;
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

    /* Follow-up suggestion buttons */
    .suggestion-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .suggestion-btn {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 0.9rem;
        color: #0369a1;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: left;
        max-width: 280px;
    }

    .suggestion-btn:hover {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-color: #7dd3fc;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(3, 105, 161, 0.15);
    }

    .suggestion-btn:active {
        transform: translateY(0);
    }

    /* Canvas Artifact */
    .canvas-artifact-wrapper {
        margin: 12px 0 4px 0;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        background: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .canvas-artifact-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 14px;
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-bottom: 1px solid #bae6fd;
        font-size: 0.85rem;
        font-weight: 600;
        color: #0369a1;
        gap: 8px;
    }
    .canvas-artifact-header .canvas-lang-badge {
        background: #0369a1;
        color: #fff;
        border-radius: 6px;
        padding: 1px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    /* Context window indicator (assistant provider/model row) */
    .ctx-window-row {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        color: rgba(49, 51, 63, 0.6);
        line-height: 1.4;
    }
    .ctx-window-circle {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
        border: 1px solid rgba(120, 120, 120, 0.28);
        vertical-align: middle;
        background: conic-gradient(
            currentColor var(--ctx-fill, 0%),
            rgba(148, 163, 184, 0.18) 0
        );
    }
    .ctx-window-circle.unknown {
        background-color: transparent;
        background-image: none;
        border-color: rgba(120, 120, 120, 0.6);
        color: rgba(120, 120, 120, 0.6);
    }
    .ctx-window-circle.ok {
        color: #2ecc71;
    }
    .ctx-window-circle.warn {
        color: #f39c12;
    }
    .ctx-window-circle.danger {
        color: #e74c3c;
    }
</style>
"""

# Image lightbox setup. Streamlit's `st.markdown(..., unsafe_allow_html=True)`
# strips <script> tags, so the JS must run inside a components.v1.html iframe
# and reach into `window.parent.document` to install the delegate + overlay on
# the actual page. The CSS used here is already defined in APP_STYLE
# (`.image-lightbox-overlay`, `.image-lightbox-close`, `.image-lightbox-content`).
_IMAGE_LIGHTBOX_SETUP = """
<script>
(function() {
  var topDoc;
  try { topDoc = window.parent.document; } catch (_) { return; }
  if (!topDoc || topDoc.__imgLightboxInstalled) return;
  topDoc.__imgLightboxInstalled = true;

  var overlay = topDoc.createElement('div');
  overlay.id = 'imageLightbox';
  overlay.className = 'image-lightbox-overlay';
  overlay.innerHTML = (
    '<span class="image-lightbox-close">&times;</span>' +
    '<img id="lightboxImage" class="image-lightbox-content" src="" alt="Full size image">'
  );
  topDoc.body.appendChild(overlay);

  function open(src) {
    var img = topDoc.getElementById('lightboxImage');
    if (!img) return;
    img.src = src;
    overlay.classList.add('active');
    topDoc.body.style.overflow = 'hidden';
  }

  function close() {
    overlay.classList.remove('active');
    topDoc.body.style.overflow = '';
  }

  topDoc.addEventListener('click', function(e) {
    var t = e.target;
    if (!t || !t.classList) return;
    if (t.classList.contains('img-thumb')) {
      e.preventDefault();
      e.stopPropagation();
      open(t.src);
      return;
    }
    if (overlay.classList.contains('active') &&
        (t.classList.contains('image-lightbox-overlay') ||
         t.classList.contains('image-lightbox-close'))) {
      close();
    }
  }, true);

  topDoc.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') close();
  });
})();
</script>
"""


def render_image_lightbox() -> None:
    """Install the page-level image lightbox once per session run."""
    _stc.html(_IMAGE_LIGHTBOX_SETUP, height=0)


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
        "planning_agent": "Planning Agent",
        "canvas_agent": "Canvas Agent",
        "router": "Router",
    }
    # Custom agents carry a runtime id (custom_agent:<uuid>). Resolve the display
    # name from the session cache populated by streamed metadata / attachments.
    if isinstance(agent, str) and agent.startswith("custom_agent:"):
        try:
            cache = st.session_state.get("custom_agent_names", {})
        except Exception:
            cache = {}
        return cache.get(agent, "Custom Agent")
    return agent_names.get(agent, agent.replace("_", " ").title())


def get_message_agent_label(message_metadata: dict[str, Any] | None) -> str | None:
    """Resolve a display label for the agent that produced a persisted message.

    Prefers the canonical ``message_metadata["agent"]`` field, falling back to
    the legacy custom-agent compatibility fields for messages persisted before
    the canonical field existed.
    """
    if not isinstance(message_metadata, dict):
        return None
    agent = message_metadata.get("agent")
    if isinstance(agent, dict):
        name = agent.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        agent_id = agent.get("id")
        if isinstance(agent_id, str) and agent_id.strip():
            return get_agent_display_name(agent_id.strip())

    legacy_name = message_metadata.get("custom_agent_name")
    if isinstance(legacy_name, str) and legacy_name.strip():
        return legacy_name.strip()

    legacy_id = message_metadata.get("runtime_agent_id")
    if isinstance(legacy_id, str) and legacy_id.strip():
        return get_agent_display_name(legacy_id.strip())
    return None


# --------------------------------------------------------------------------- #
# Custom agents — API helpers + management UI
# --------------------------------------------------------------------------- #


def _custom_agent_request(
    method: str,
    endpoint: str,
    json_body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Call a custom-agent backend route. Returns (status_code, payload)."""
    headers: dict[str, str] = {}
    auth_token = st.session_state.get("auth_token")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    try:
        response = get_http_session().request(
            method,
            f"{API_BASE_URL}{endpoint}",
            headers=headers,
            json=json_body,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return 0, {"message": str(exc)}
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    return response.status_code, payload if isinstance(payload, dict) else {}


def _custom_agent_device_query() -> str:
    device_id = st.session_state.get("device_id")
    return f"?deviceId={device_id}" if device_id else ""


def list_custom_agents() -> list[dict[str, Any]]:
    status, payload = _custom_agent_request("GET", "/custom-agents")
    if status == 200:
        agents = payload.get("data") or []
        # Refresh the runtime-id -> name cache used by get_agent_display_name.
        cache = st.session_state.setdefault("custom_agent_names", {})
        for agent in agents:
            runtime_id = agent.get("runtimeAgentId") or f"custom_agent:{agent.get('id')}"
            if agent.get("name"):
                cache[runtime_id] = agent["name"]
        return agents
    return []


def get_custom_agent_options() -> dict[str, Any]:
    status, payload = _custom_agent_request(
        "GET", f"/custom-agents/options{_custom_agent_device_query()}"
    )
    return payload.get("data") or {} if status == 200 else {}


def create_custom_agent(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return _custom_agent_request("POST", f"/custom-agents{_custom_agent_device_query()}", body)


def update_custom_agent(agent_id: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return _custom_agent_request(
        "PATCH", f"/custom-agents/{agent_id}{_custom_agent_device_query()}", body
    )


def delete_custom_agent(agent_id: str) -> tuple[int, dict[str, Any]]:
    return _custom_agent_request("DELETE", f"/custom-agents/{agent_id}")


def get_conversation_custom_agents(conversation_id: str) -> list[dict[str, Any]]:
    status, payload = _custom_agent_request(
        "GET", f"/conversations/{conversation_id}/custom-agents"
    )
    return payload.get("data") or [] if status == 200 else []


def set_conversation_custom_agents(
    conversation_id: str, custom_agent_ids: list[str]
) -> tuple[int, dict[str, Any]]:
    return _custom_agent_request(
        "PUT",
        f"/conversations/{conversation_id}/custom-agents",
        {"customAgentIds": custom_agent_ids},
    )


def _provider_model_options(options: dict[str, Any]) -> dict[str, list[str]]:
    providers: dict[str, list[str]] = {}
    for entry in options.get("providers") or []:
        provider_type = entry.get("provider_type")
        if not provider_type:
            continue
        models = [
            str(m.get("id"))
            for m in (entry.get("models") or [])
            if isinstance(m, dict) and m.get("id")
        ]
        providers[provider_type] = models
    return providers


def render_custom_agents_view() -> None:
    """Custom Agents workspace tab — layout consistent with the Models tab."""
    st.markdown("# Custom Agents")
    st.caption(
        "Create per-user agents with their own prompt, model, tools, and skills, then "
        "attach them to a conversation. They participate in routing, hand-off, and "
        "planning as first-class agents."
    )

    if not (st.session_state.get("current_user_id") and st.session_state.get("auth_token")):
        st.info("Sign in to manage custom agents.")
        return

    render_custom_agents_manager()

    st.divider()
    st.subheader("Attach to the current conversation")
    current_conv = st.session_state.get("current_conversation_id")
    if current_conv:
        # ``current_conv`` may be the "pending_new" sentinel for a freshly
        # started chat; the panel buffers the selection until the conversation
        # is created on first send.
        render_conversation_custom_agents_panel(current_conv)
    else:
        st.caption("Open or start a conversation (Chat tab) to attach custom agents to it.")


def render_custom_agents_manager() -> None:
    """List/create/edit/delete custom agents. Controls warn on 409 (in use)."""
    st.subheader("Your agents")
    options = get_custom_agent_options()
    providers = _provider_model_options(options)
    # The API serializes CustomAgentOptions with camelCase aliases; accept
    # snake_case and the older serverTools field for robustness.
    server_default_tools = (
        options.get("serverDefaultTools") or options.get("server_default_tools") or []
    )
    server_tools = options.get("serverTools") or options.get("server_tools") or []
    client_tools = options.get("clientTools") or options.get("client_tools") or []
    client_servers = options.get("clientServers") or options.get("client_servers") or []
    skills = options.get("skills") or []
    selectable_server_tools = [*server_default_tools, *server_tools]
    server_groups = _custom_agent_server_groups(
        selectable_server_tools, client_tools, client_servers
    )
    server_group_labels = {
        group_key: _custom_agent_server_group_label(group_key, server_groups)
        for group_key in server_groups
    }
    tool_labels = {
        _custom_agent_tool_option_key(t): _custom_agent_tool_label(t)
        for t in selectable_server_tools
    }
    tool_labels.update(
        {_custom_agent_tool_option_key(t): _custom_agent_tool_label(t) for t in client_tools}
    )
    skill_labels = {
        _custom_agent_skill_key(s): f"[{s.get('source')}] {s.get('lookup_name')}"
        for s in skills
        if _custom_agent_skill_key(s) is not None
    }

    agents = list_custom_agents()
    if not agents:
        st.caption("No custom agents yet. Create one below.")
    for agent in agents:
        with st.expander(agent.get("name", "Custom Agent"), expanded=False):
            st.caption(f"{agent.get('providerType')} / {agent.get('model')}")
            edit_name = st.text_input(
                "Name", value=agent.get("name", ""), key=f"ca_edit_name_{agent['id']}"
            )
            edit_desc = st.text_input(
                "Description",
                value=agent.get("description") or "",
                key=f"ca_edit_desc_{agent['id']}",
            )
            edit_prompt = st.text_area(
                "System prompt", value=agent.get("prompt", ""), key=f"ca_edit_prompt_{agent['id']}"
            )
            edit_model = st.text_input(
                "Model", value=agent.get("model", ""), key=f"ca_edit_model_{agent['id']}"
            )
            current_tool_refs = agent.get("toolRefs") or agent.get("tool_refs") or []
            current_skill_refs = agent.get("skillRefs") or agent.get("skill_refs") or []
            tool_refs_editable = _custom_agent_tool_refs_available(
                current_tool_refs,
                selectable_server_tools,
                client_tools,
            )
            skill_refs_editable = _custom_agent_skill_refs_available(current_skill_refs, skills)
            edit_server_group_keys_default = _custom_agent_selected_server_group_keys(
                current_tool_refs,
                selectable_server_tools,
                client_tools,
                client_servers,
            )
            edit_server_names = st.multiselect(
                "All tools from MCP servers",
                list(server_group_labels.keys()),
                default=edit_server_group_keys_default,
                format_func=lambda key: server_group_labels.get(key, key),
                disabled=not tool_refs_editable,
                key=f"ca_edit_server_tools_{agent['id']}",
            )
            edit_excluded_tool_keys = _custom_agent_grouped_tool_keys(
                server_groups, edit_server_names
            )
            edit_tool_options = [key for key in tool_labels if key not in edit_excluded_tool_keys]
            _ca_retain_session_options(f"ca_edit_tools_{agent['id']}", edit_tool_options)
            edit_tool_default = [
                key
                for key in _custom_agent_selected_tool_keys(
                    current_tool_refs,
                    selectable_server_tools,
                    client_tools,
                    excluded_group_keys=edit_server_names,
                )
                if key in edit_tool_options
            ]
            edit_tool_ids = st.multiselect(
                "Individual tools",
                edit_tool_options,
                default=edit_tool_default,
                format_func=lambda tid: tool_labels.get(tid, tid),
                disabled=not tool_refs_editable,
                key=f"ca_edit_tools_{agent['id']}",
            )
            if not tool_refs_editable:
                st.caption("Reconnect the original device to edit this agent's tools.")
            edit_skill_keys = st.multiselect(
                "Skills",
                list(skill_labels.keys()),
                default=_custom_agent_selected_skill_keys(
                    current_skill_refs,
                    skills,
                ),
                format_func=lambda key: skill_labels.get(key, str(key)),
                disabled=not skill_refs_editable,
                key=f"ca_edit_skills_{agent['id']}",
            )
            if not skill_refs_editable:
                st.caption("Reconnect the original device to edit this agent's skills.")
            col_save, col_del = st.columns(2)
            with col_save:
                if st.button("Save", key=f"ca_save_{agent['id']}"):
                    body = {
                        "name": edit_name,
                        "description": edit_desc or None,
                        "prompt": edit_prompt,
                        "model": edit_model,
                    }
                    if tool_refs_editable:
                        body["tool_refs"] = _build_tool_refs(
                            edit_tool_ids,
                            selectable_server_tools,
                            client_tools,
                            selected_server_group_keys=edit_server_names,
                            client_servers=client_servers,
                        )
                    if skill_refs_editable:
                        body["skill_refs"] = _build_skill_refs(edit_skill_keys, skills)
                    status, payload = update_custom_agent(
                        agent["id"],
                        body,
                    )
                    if status == 200:
                        st.success("Saved.")
                        st.rerun()
                    elif status == 409:
                        st.warning("Agent is active or paused — cannot edit right now.")
                    else:
                        st.error(_extract_api_error_message(status, payload))
            with col_del:
                if st.button("Delete", key=f"ca_del_{agent['id']}"):
                    status, payload = delete_custom_agent(agent["id"])
                    if status in (200, 204):
                        st.success("Deleted.")
                        st.rerun()
                    elif status == 409:
                        st.warning("Agent is active or paused — cannot delete right now.")
                    else:
                        st.error(_extract_api_error_message(status, payload))

    st.markdown("**Create a custom agent**")
    provider_names = list(providers.keys()) or ["openai", "gemini"]
    # Provider selector lives OUTSIDE the form so changing it reruns the page and
    # refreshes the dependent Model list (st.form defers reruns until submit,
    # which would otherwise leave the model list stale). Matches the Models tab.
    provider_type = st.selectbox("Provider", provider_names, key="ca_new_provider")
    model_choices = providers.get(provider_type, [])
    # The server-group selector lives OUTSIDE the form (like Provider): st.form
    # defers reruns until submit, but the individual-tool list must drop a
    # server's tools the moment that whole server is selected.
    selected_server_names = st.multiselect(
        "All tools from MCP servers",
        list(server_group_labels.keys()),
        format_func=lambda key: server_group_labels.get(key, key),
        key="ca_new_server_tools",
    )
    new_excluded_tool_keys = _custom_agent_grouped_tool_keys(server_groups, selected_server_names)
    new_tool_options = [key for key in tool_labels if key not in new_excluded_tool_keys]
    _ca_retain_session_options("ca_new_tools", new_tool_options)

    with st.form("create_custom_agent_form", clear_on_submit=False):
        name = st.text_input("Name", key="ca_new_name")
        description = st.text_input("Description", key="ca_new_desc")
        prompt = st.text_area("System prompt", key="ca_new_prompt")
        # No persistent key on Model: options change with Provider, and a stale
        # stored selection would raise "value not in options" on switch.
        model = (
            st.selectbox("Model", model_choices)
            if model_choices
            else st.text_input("Model", key="ca_new_model_text")
        )
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1, key="ca_new_temp")
        selected_tool_ids = st.multiselect(
            "Individual tools",
            new_tool_options,
            format_func=lambda tid: tool_labels.get(tid, tid),
            key="ca_new_tools",
        )
        selected_skill_keys = st.multiselect(
            "Skills",
            list(skill_labels.keys()),
            format_func=lambda key: skill_labels.get(key, str(key)),
            key="ca_new_skills",
        )
        submitted = st.form_submit_button("Create")
        if submitted:
            tool_refs = _build_tool_refs(
                selected_tool_ids,
                selectable_server_tools,
                client_tools,
                selected_server_group_keys=selected_server_names,
                client_servers=client_servers,
            )
            skill_refs = _build_skill_refs(selected_skill_keys, skills)
            body = {
                "name": name,
                "description": description or None,
                "prompt": prompt,
                "provider_type": provider_type,
                "model": model,
                "temperature": temperature,
                "tool_refs": tool_refs,
                "skill_refs": skill_refs,
            }
            validation_error = _validate_custom_agent_create_body(body)
            if validation_error:
                st.warning(validation_error)
            else:
                status, payload = create_custom_agent(body)
                if status == 201:
                    st.success("Custom agent created.")
                    st.rerun()
                else:
                    st.error(_extract_api_error_message(status, payload))


def _custom_agent_value(tool: dict[str, Any], snake: str, camel: str | None = None) -> Any:
    return tool.get(snake) if snake in tool else tool.get(camel or snake)


def _custom_agent_tool_option_key(tool: dict[str, Any]) -> str:
    tool_type = str(_custom_agent_value(tool, "type") or "")
    qualified_id = str(_custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId") or "")
    if tool_type == "client":
        device_id = _custom_agent_value(tool, "device_id", "deviceId") or ""
        session_id = _custom_agent_value(tool, "session_id", "sessionId") or ""
        instance_id = _custom_agent_value(tool, "tool_instance_id", "toolInstanceId") or ""
        return f"client::{device_id}::{session_id}::{instance_id}::{qualified_id}"
    return f"server::{tool_type or 'server_mcp'}::{qualified_id}"


def _custom_agent_server_name(tool: dict[str, Any]) -> str:
    server_name = str(_custom_agent_value(tool, "server_name", "serverName") or "").strip()
    if server_name:
        return server_name
    qualified_id = str(
        _custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId") or ""
    ).strip()
    if "::" in qualified_id:
        return qualified_id.split("::", 1)[0].strip()
    return ""


def _custom_agent_server_tool_groups(
    server_tools: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    seen_by_server: dict[str, set[str]] = {}
    for tool in server_tools:
        server_name = _custom_agent_server_name(tool)
        qualified_id = str(
            _custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId") or ""
        ).strip()
        if not server_name or not qualified_id:
            continue
        seen = seen_by_server.setdefault(server_name, set())
        if qualified_id in seen:
            continue
        seen.add(qualified_id)
        groups.setdefault(server_name, []).append(tool)

    return {
        server_name: groups[server_name]
        for server_name in sorted(groups)
        if len(groups[server_name]) > 1
    }


def _custom_agent_server_group_key(
    kind: str, server_name: str, device_id: str | None = None
) -> str:
    """Namespaced key for a pickable MCP-server group.

    Backend and client/sidecar servers share one keyspace in the picker, so the
    key carries the kind (and, for client servers, the owning device) to keep a
    backend server and a same-named sidecar server from colliding.
    """
    if kind == "client":
        return f"client::{device_id or ''}::{server_name}"
    return f"server::{server_name}"


def _custom_agent_client_server_groups(
    client_tools: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group client/sidecar tools by ``(device_id, server_name)``.

    The client equivalent of :func:`_custom_agent_server_tool_groups`: it lets a
    whole sidecar MCP server be attached at once. Deduped by qualified id within a
    group; device id is part of the key because the same server name can be live
    on more than one connected device.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_by_group: dict[tuple[str, str], set[str]] = {}
    for tool in client_tools:
        if str(_custom_agent_value(tool, "type") or "") != "client":
            continue
        server_name = _custom_agent_server_name(tool)
        device_id = str(_custom_agent_value(tool, "device_id", "deviceId") or "").strip()
        qualified_id = str(
            _custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId") or ""
        ).strip()
        if not server_name or not device_id or not qualified_id:
            continue
        group_key = (device_id, server_name)
        seen = seen_by_group.setdefault(group_key, set())
        if qualified_id in seen:
            continue
        seen.add(qualified_id)
        groups.setdefault(group_key, []).append(tool)
    return groups


def _custom_agent_server_groups(
    server_tools: list[dict[str, Any]],
    client_tools: list[dict[str, Any]],
    client_servers: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Pickable MCP-server groups for the "all tools from server" selector.

    Surfaces backend MCP servers (``server::<name>``) and client/sidecar MCP
    servers (``client::<device_id>::<name>``) in one namespaced keyspace. The
    sidecar servers to show come from ``client_servers`` (the backend's advertised
    list) when present, falling back to whatever ``client_tools`` imply; their
    member tools — which carry the full client identity needed to build refs —
    always come from ``client_tools``. Single-tool servers are omitted (pick the
    one tool individually), matching the backend grouping.
    """
    groups: dict[str, dict[str, Any]] = {}

    for server_name, tools in _custom_agent_server_tool_groups(server_tools).items():
        groups[_custom_agent_server_group_key("server", server_name)] = {
            "kind": "server",
            "server_name": server_name,
            "device_id": None,
            "tools": tools,
        }

    tools_by_client_group = _custom_agent_client_server_groups(client_tools)
    if client_servers:
        declared: list[tuple[str, str]] = []
        for entry in client_servers:
            server_name = str(_custom_agent_value(entry, "server_name", "serverName") or "").strip()
            device_id = str(_custom_agent_value(entry, "device_id", "deviceId") or "").strip()
            if server_name and device_id and (device_id, server_name) not in declared:
                declared.append((device_id, server_name))
    else:
        declared = list(tools_by_client_group.keys())

    for device_id, server_name in declared:
        tools = tools_by_client_group.get((device_id, server_name), [])
        if len(tools) <= 1:
            continue
        groups[_custom_agent_server_group_key("client", server_name, device_id)] = {
            "kind": "client",
            "server_name": server_name,
            "device_id": device_id,
            "tools": tools,
        }

    return groups


def _custom_agent_server_group_label(
    group_key: str,
    server_groups: dict[str, dict[str, Any]],
) -> str:
    group = server_groups.get(group_key) or {}
    kind = str(group.get("kind") or "server")
    server_name = str(group.get("server_name") or group_key)
    tool_count = len(group.get("tools") or [])
    return f"[{kind}] {server_name} (all {tool_count} tools)"


def _custom_agent_grouped_tool_keys(
    server_groups: dict[str, dict[str, Any]],
    selected_group_keys: list[str] | set[str] | tuple[str, ...] | None,
) -> set[str]:
    """Individual-tool option keys covered by the selected server groups.

    When a whole MCP server is selected, its tools are dropped from the
    individual-tool picker so a whole-server selection and a redundant individual
    selection of the same tool can never coexist.
    """
    selected = {str(key) for key in selected_group_keys or []}
    covered: set[str] = set()
    for group_key, group in server_groups.items():
        if group_key in selected:
            for tool in group.get("tools") or []:
                covered.add(_custom_agent_tool_option_key(tool))
    return covered


def _ca_retain_session_options(state_key: str, options: list[str]) -> None:
    """Drop persisted multiselect picks that are no longer offered.

    Streamlit raises when a widget's session-state value contains entries absent
    from its current options. Pruning before the widget renders keeps the
    individual-tool list stable when a whole server is selected and its tools
    leave the list.
    """
    current = st.session_state.get(state_key)
    if not isinstance(current, list):
        return
    allowed = set(options)
    filtered = [value for value in current if value in allowed]
    if len(filtered) != len(current):
        st.session_state[state_key] = filtered


def _custom_agent_client_tool_stable_key(tool: dict[str, Any]) -> tuple[str, str] | None:
    if str(_custom_agent_value(tool, "type") or "") != "client":
        return None
    device_id = str(_custom_agent_value(tool, "device_id", "deviceId") or "").strip()
    qualified_id = str(
        _custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId") or ""
    ).strip()
    if not device_id or not qualified_id:
        return None
    return (device_id, qualified_id)


def _custom_agent_tool_label(tool: dict[str, Any]) -> str:
    tool_type = str(_custom_agent_value(tool, "type") or "")
    qualified_id = str(_custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId") or "")
    tool_name = _custom_agent_value(tool, "tool_name", "toolName")
    server_name = _custom_agent_value(tool, "server_name", "serverName")
    if tool_type == "client":
        return f"[client] {qualified_id} ({tool_name or server_name or 'tool'})"
    return f"[server] {qualified_id}"


def _custom_agent_skill_key(skill: dict[str, Any]) -> tuple[str, str] | None:
    source = _custom_agent_value(skill, "source")
    lookup_name = _custom_agent_value(skill, "lookup_name", "lookupName")
    if not source or not lookup_name:
        return None
    return (str(source), str(lookup_name))


def _custom_agent_skill_lookup_values(skill: dict[str, Any]) -> set[str]:
    return {
        str(value).strip()
        for value in (
            _custom_agent_value(skill, "lookup_name", "lookupName"),
            _custom_agent_value(skill, "name"),
        )
        if str(value or "").strip()
    }


def _custom_agent_selected_tool_keys(
    tool_refs: list[dict[str, Any]],
    server_tools: list[dict[str, Any]],
    client_tools: list[dict[str, Any]],
    *,
    excluded_group_keys: list[str] | set[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Individual-tool selection keys for the picker default.

    Tools whose whole server is already selected as a group (``excluded_group_keys``
    — namespaced server/client group keys) are dropped so they are not shown
    pre-checked in both the server picker and the individual-tool picker.
    """
    excluded = {str(key) for key in excluded_group_keys or []}
    server_key_by_qid = {
        str(_custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId")): (
            _custom_agent_tool_option_key(tool)
        )
        for tool in server_tools
        if _custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId")
    }
    client_key_by_stable_key = {
        stable_key: _custom_agent_tool_option_key(tool)
        for tool in client_tools
        if (stable_key := _custom_agent_client_tool_stable_key(tool)) is not None
    }
    selected: list[str] = []

    for ref in tool_refs or []:
        ref_type = str(_custom_agent_value(ref, "type") or "")
        if ref_type == "client":
            device_id = str(_custom_agent_value(ref, "device_id", "deviceId") or "").strip()
            group_key = _custom_agent_server_group_key(
                "client", _custom_agent_server_name(ref), device_id
            )
            if group_key in excluded:
                continue
            stable_key = _custom_agent_client_tool_stable_key(ref)
            key = client_key_by_stable_key.get(stable_key) if stable_key else None
            if key and key not in selected:
                selected.append(key)
            continue

        if _custom_agent_server_group_key("server", _custom_agent_server_name(ref)) in excluded:
            continue
        qid = _custom_agent_value(ref, "qualified_tool_id", "qualifiedToolId")
        key = server_key_by_qid.get(str(qid))
        if key and key not in selected:
            selected.append(key)

    return selected


def _custom_agent_selected_server_group_keys(
    tool_refs: list[dict[str, Any]],
    server_tools: list[dict[str, Any]],
    client_tools: list[dict[str, Any]],
    client_servers: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Server group keys whose every member tool is present in ``tool_refs``.

    Used to pre-select the "all tools from MCP servers" picker when editing an
    agent. Spans both backend and client/sidecar servers.
    """
    groups = _custom_agent_server_groups(server_tools, client_tools, client_servers)

    selected_server_qids: set[str] = set()
    selected_client_keys: set[tuple[str, str]] = set()
    for ref in tool_refs or []:
        if str(_custom_agent_value(ref, "type") or "") == "client":
            stable_key = _custom_agent_client_tool_stable_key(ref)
            if stable_key is not None:
                selected_client_keys.add(stable_key)
            continue
        qualified_id = str(
            _custom_agent_value(ref, "qualified_tool_id", "qualifiedToolId") or ""
        ).strip()
        if qualified_id:
            selected_server_qids.add(qualified_id)

    selected: list[str] = []
    for group_key, group in groups.items():
        tools = group.get("tools") or []
        if group.get("kind") == "client":
            member_keys = {
                stable_key
                for tool in tools
                if (stable_key := _custom_agent_client_tool_stable_key(tool)) is not None
            }
            if member_keys and member_keys <= selected_client_keys:
                selected.append(group_key)
        else:
            member_qids = {
                str(_custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId"))
                for tool in tools
                if _custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId")
            }
            if member_qids and member_qids <= selected_server_qids:
                selected.append(group_key)
    return selected


def _custom_agent_tool_refs_available(
    tool_refs: list[dict[str, Any]],
    server_tools: list[dict[str, Any]],
    client_tools: list[dict[str, Any]],
) -> bool:
    server_qids = {
        str(_custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId"))
        for tool in server_tools
        if _custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId")
    }
    client_stable_keys = {
        stable_key
        for tool in client_tools
        if (stable_key := _custom_agent_client_tool_stable_key(tool)) is not None
    }

    for ref in tool_refs or []:
        ref_type = str(_custom_agent_value(ref, "type") or "")
        if ref_type == "client":
            if _custom_agent_client_tool_stable_key(ref) not in client_stable_keys:
                return False
            continue

        qid = _custom_agent_value(ref, "qualified_tool_id", "qualifiedToolId")
        if str(qid) not in server_qids:
            return False

    return True


def _custom_agent_selected_skill_keys(
    skill_refs: list[dict[str, Any]],
    skills: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    available = {
        key for key in (_custom_agent_skill_key(skill) for skill in skills) if key is not None
    }
    client_key_by_lookup = {
        lookup: key
        for skill in skills
        if (key := _custom_agent_skill_key(skill)) is not None and key[0] == "client"
        for lookup in _custom_agent_skill_lookup_values(skill)
    }
    selected: list[tuple[str, str]] = []
    for ref in skill_refs or []:
        key = _custom_agent_skill_key(ref)
        if key not in available and key and key[0] == "server":
            key = next(
                (
                    client_key_by_lookup[lookup]
                    for lookup in _custom_agent_skill_lookup_values(ref)
                    if lookup in client_key_by_lookup
                ),
                None,
            )
        if key in available and key not in selected:
            selected.append(key)
    return selected


def _custom_agent_skill_refs_available(
    skill_refs: list[dict[str, Any]],
    skills: list[dict[str, Any]],
) -> bool:
    available = {
        key for key in (_custom_agent_skill_key(skill) for skill in skills) if key is not None
    }
    client_key_by_lookup = {
        lookup: key
        for skill in skills
        if (key := _custom_agent_skill_key(skill)) is not None and key[0] == "client"
        for lookup in _custom_agent_skill_lookup_values(skill)
    }
    for ref in skill_refs or []:
        key = _custom_agent_skill_key(ref)
        if key in available:
            continue
        if (
            key
            and key[0] == "server"
            and any(
                lookup in client_key_by_lookup for lookup in _custom_agent_skill_lookup_values(ref)
            )
        ):
            continue
        return False
    return True


def _validate_custom_agent_create_body(body: dict[str, Any]) -> str | None:
    if not str(body.get("name") or "").strip():
        return "Name is required."
    if not str(body.get("prompt") or "").strip():
        return "System prompt is required."
    if not str(body.get("model") or "").strip():
        return "Model is required."
    return None


def _build_tool_refs(
    selected_tool_ids: list[str],
    server_default_tools: list[dict[str, Any]],
    client_tools: list[dict[str, Any]],
    *,
    selected_server_group_keys: list[str] | set[str] | tuple[str, ...] | None = None,
    client_servers: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_key_server = {_custom_agent_tool_option_key(t): t for t in server_default_tools}
    by_key_client = {_custom_agent_tool_option_key(t): t for t in client_tools}
    server_groups = _custom_agent_server_groups(server_default_tools, client_tools, client_servers)
    selected_group_keys = {str(key) for key in selected_server_group_keys or []}
    refs: list[dict[str, Any]] = []
    added_server_qids: set[str] = set()
    added_client_keys: set[tuple[str, str]] = set()

    def append_server_ref(tool: dict[str, Any]) -> None:
        qid = _custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId")
        server_name = _custom_agent_value(tool, "server_name", "serverName")
        tool_name = _custom_agent_value(tool, "tool_name", "toolName")
        if not (qid and server_name and tool_name):
            return
        qid_str = str(qid)
        if qid_str in added_server_qids:
            return
        added_server_qids.add(qid_str)
        refs.append(
            {
                "type": "server_mcp",
                "server_name": server_name,
                "tool_name": tool_name,
                "qualified_tool_id": qid,
            }
        )

    def append_client_ref(tool: dict[str, Any]) -> None:
        stable_key = _custom_agent_client_tool_stable_key(tool)
        if stable_key is None or stable_key in added_client_keys:
            return
        added_client_keys.add(stable_key)
        qid = _custom_agent_value(tool, "qualified_tool_id", "qualifiedToolId")
        refs.append(
            {
                "type": "client",
                "device_id": _custom_agent_value(tool, "device_id", "deviceId"),
                "session_id": _custom_agent_value(tool, "session_id", "sessionId"),
                "catalog_version": str(
                    _custom_agent_value(tool, "catalog_version", "catalogVersion") or ""
                ),
                "tool_instance_id": _custom_agent_value(tool, "tool_instance_id", "toolInstanceId"),
                "server_name": _custom_agent_value(tool, "server_name", "serverName"),
                "qualified_tool_id": qid,
                "tool_name": _custom_agent_value(tool, "tool_name", "toolName"),
            }
        )

    for group_key, group in server_groups.items():
        if group_key not in selected_group_keys:
            continue
        appender = append_client_ref if group.get("kind") == "client" else append_server_ref
        for tool in group.get("tools") or []:
            appender(tool)

    for key in selected_tool_ids:
        if key in by_key_server:
            append_server_ref(by_key_server[key])
        elif key in by_key_client:
            append_client_ref(by_key_client[key])

    return refs


def _build_skill_refs(
    selected_skill_keys: list[tuple[str, str]],
    skills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {key: skill for skill in skills if (key := _custom_agent_skill_key(skill)) is not None}
    return [by_key[key] for key in selected_skill_keys if key in by_key]


def render_conversation_custom_agents_panel(conversation_id: str) -> None:
    """Attach/detach custom agents for the active conversation.

    For a not-yet-created chat (``pending_new``) the selection is buffered in
    ``pending_custom_agent_ids`` and applied to the conversation right after it
    is created on the first message send. This mirrors the pending
    persona-prompt flow so users can configure a new chat before sending.
    """
    if not conversation_id:
        return
    all_agents = list_custom_agents()
    if not all_agents:
        return
    label_by_id = {a["id"]: a.get("name", a["id"]) for a in all_agents}

    is_pending = conversation_id == "pending_new"
    if is_pending:
        default_ids = [
            i for i in st.session_state.get("pending_custom_agent_ids", []) if i in label_by_id
        ]
    else:
        attached_ids = [a.get("id") for a in get_conversation_custom_agents(conversation_id)]
        default_ids = [i for i in attached_ids if i in label_by_id]

    selected = st.multiselect(
        "Attached custom agents",
        list(label_by_id.keys()),
        default=default_ids,
        format_func=lambda i: label_by_id.get(i, i),
        key=f"ca_attach_{conversation_id}",
    )

    if is_pending:
        # Buffer the selection; it is attached when the conversation is created
        # on the first message send (see the send handler).
        st.session_state.pending_custom_agent_ids = list(selected)
        st.caption(
            "These custom agents will be attached to your new chat once you send the first message."
        )
        return

    if st.button("Save attachments", key=f"ca_attach_save_{conversation_id}"):
        status, payload = set_conversation_custom_agents(conversation_id, selected)
        if status == 200:
            st.success("Attachments updated.")
            st.rerun()
        elif status == 409:
            st.warning("A custom agent is active or paused — cannot change attachments now.")
        else:
            st.error(_extract_api_error_message(status, payload))


def render_conversation_button(
    conversation: dict[str, Any],
    is_active: bool,
    on_click_callback: Callable | None = None,
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

    if (
        st.button(
            title,
            key=f"conv_{conversation['id']}",
            width="stretch",
            type=button_type,
        )
        and conversation["id"] != st.session_state.current_conversation_id
    ):
        st.session_state.current_conversation_id = conversation["id"]
        st.session_state.active_view = "chat"
        close_conversation_manager()
        reset_conversation_state()
        if on_click_callback:
            on_click_callback()
        st.rerun()


def group_conversations_by_date(
    conversations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
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

    groups: dict[str, list[dict[str, Any]]] = {
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

SESSION_STATE_DEFAULTS: dict[str, Callable[[], Any] | Any] = {
    "current_user_id": lambda: None,
    "current_user_profile": lambda: None,
    "current_conversation_id": lambda: None,
    "messages": list,
    "conversations_list": list,
    "conversations_loaded": lambda: False,
    "conversations_last_fetch_params": lambda: None,
    CONVERSATION_MANAGER_DIALOG_KEY: lambda: False,
    "show_instructions": lambda: False,
    "auth_token": lambda: None,
    "conversation_messages_meta": lambda: None,
    "conversation_messages_page": lambda: 0,
    "has_more_messages": lambda: True,
    "pending_persona_prompt": str,
    "persona_editor_origin": lambda: None,
    "persona_editor_value": str,
    "pending_image_attachments": list,
    "message_image_thumbnails": dict,
    "message_chunks": dict,
    "show_attachment_uploader": lambda: False,
    "chat_image_uploader_nonce": lambda: 0,
    "API_BASE_URL": lambda: API_BASE_URL,
    "active_view": lambda: "chat",
    "mcp_tools_list": lambda: None,
    "mcp_servers_list": lambda: None,
    "selected_tool": lambda: None,
    "tool_execution_result": lambda: None,
    "selected_chunk_info": lambda: None,
    "chunk_preview_dialog_key": lambda: False,
    "stream_trace_items": list,
    "stream_tool_index": dict,
    "stream_trace_expanded": lambda: False,
    "stream_subagent_activity": lambda: None,
    # Provider/model UI state
    "model_config_options_cache": dict,
    "model_config_options_last_fetch": lambda: None,
    "model_config_options_error": lambda: None,
    "model_config_options_needs_form_sync": lambda: False,
    # Planning mode state
    "planning_generate_input": str,
    "planning_manual_input": str,
    "api_cache_version": lambda: 0,
    "usage_cache_version": lambda: 0,
    "live_widget_mounts": dict,
    # localStorage bridge
    "_ls_op": lambda: None,
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
        self.result: list[str] = []
        self._tag_stack: list[str] = []

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
            if class_attr and ("katex" in class_attr.lower() or "math" in class_attr.lower()):
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

    normalized = html.unescape(content).replace("\r\n", "\n").replace("\r", "\n").strip()
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

    normalized = re.sub(r"\$\$(.+?)\$\$", replace_display_math, normalized, flags=re.DOTALL)

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
    normalized = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1 (\2)", normalized, flags=re.MULTILINE)

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
        if current_line_type in ("unordered", "ordered") and prev_line_type not in (
            None,
            "blank",
            current_line_type,
        ):
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


st.set_page_config(page_title="ChatBot", layout="wide", initial_sidebar_state="expanded")

st.markdown(APP_STYLE, unsafe_allow_html=True)
render_image_lightbox()
initialize_session_state()

# ── localStorage session-persistence bridge ──────────────────────────────────
# Allows the auth token to survive F5 / browser refresh.

# Step 1: flush any pending localStorage write/clear from the previous run.
_ls_op = st.session_state.get("_ls_op")
if _ls_op is not None:
    st.session_state._ls_op = None
    if isinstance(_ls_op, dict):  # save
        _tok = json.dumps(_ls_op.get("token", ""))
        _uid = json.dumps(_ls_op.get("uid", ""))
        _stc.html(
            f"<script>try{{localStorage.setItem('cbtoken',{_tok});"
            f"localStorage.setItem('cbuid',{_uid});}}catch(e){{}}</script>",
            height=0,
        )
    elif _ls_op == "clear":  # logout
        _stc.html(
            "<script>try{localStorage.removeItem('cbtoken');"
            "localStorage.removeItem('cbuid');}catch(e){}</script>",
            height=0,
        )

# Step 2: if not authenticated, try to restore from localStorage.
if not st.session_state.get("auth_token"):
    _qp = st.query_params
    if "__t" in _qp and "__u" in _qp:
        # Bridge already fired and injected params — restore session.
        st.session_state.auth_token = _qp["__t"]
        st.session_state.current_user_id = _qp["__u"]
        st.session_state.show_login = False
        # Remove sensitive params from URL; keep __restore=1 to stop the
        # bridge from firing again on the next rerun.
        del st.query_params["__t"]
        del st.query_params["__u"]
    elif "__restore" not in _qp:
        # Inject the bridge that reads localStorage and redirects once.
        _stc.html(
            """<script>
(function(){
  try{
    var u=new URL(window.parent.location.href);
    if(u.searchParams.has('__restore'))return;
    var t=localStorage.getItem('cbtoken');
    var i=localStorage.getItem('cbuid');
    if(t&&i){
      u.searchParams.set('__restore','1');
      u.searchParams.set('__t',t);
      u.searchParams.set('__u',i);
      window.parent.location.replace(u.toString());
    }
  }catch(e){}
})();
</script>""",
            height=0,
        )
# ── end localStorage bridge ──────────────────────────────────────────────────

# Clear any old cached functions on first run
if "cache_cleared_v2" not in st.session_state:
    st.cache_data.clear()
    st.session_state.cache_cleared_v2 = True


def _clear_inflight_state() -> None:
    """Clear all in-flight streaming state keys."""
    st.session_state.stream_inflight = False
    st.session_state.stream_conversation_id = ""
    st.session_state.stream_user_message_id = ""
    st.session_state.stream_partial_text = ""
    st.session_state.stream_partial_thinking = ""
    st.session_state.stream_selected_agent = None
    st.session_state.stream_trace_items = []
    st.session_state.stream_tool_index = {}
    st.session_state.stream_trace_expanded = False
    st.session_state.stream_subagent_activity = None


def _stoppable_stream_conversation_id() -> str:
    """Return the in-flight stream's conversation id, or "" if nothing to stop.

    The stop must target the conversation that was streaming (recorded in
    ``stream_conversation_id`` at stream start), never the conversation the
    view currently shows: mid-generation the user may click "New Chat" (the
    ``pending_new`` sentinel, not a UUID — the backend rejects it with 422)
    or switch to another conversation entirely.
    """
    if not st.session_state.get("stream_inflight"):
        return ""
    if not st.session_state.get("stream_user_message_id"):
        return ""
    conversation_id = str(st.session_state.get("stream_conversation_id") or "")
    if conversation_id == "pending_new":
        return ""
    return conversation_id


def _handle_stop_rerun(conversation_id: str) -> None:
    """
    Phase 2 of two-phase stop: called on the rerun after the streaming
    connection was dropped (by the user clicking Stop or navigating away).
    Calls ``POST /messages/stop`` to signal the backend, then syncs UI.
    """
    user_message_id = st.session_state.get("stream_user_message_id", "")
    partial_preview = st.session_state.get("stream_partial_text", "")

    # Show partial text preview while the stop call completes
    if partial_preview:
        with st.chat_message("assistant"):
            st.markdown(partial_preview + " *(stopped)*")

    if not user_message_id:
        _clear_inflight_state()
        st.rerun()
        return

    # Call the stop endpoint
    stop_data = {
        "conversationId": conversation_id,
        "userMessageId": user_message_id,
    }
    stop_response = make_api_request("POST", "/messages/stop", data=stop_data)

    # Process the response
    if stop_response and stop_response.get("success"):
        result_data = stop_response.get("data", {})
        stop_status = result_data.get("status", "not_inflight")

        viewing_stopped_conversation = (
            str(st.session_state.get("current_conversation_id") or "") == conversation_id
        )
        if stop_status == "cancelled" and result_data.get("message"):
            # Backend persisted a partial message – append to local state,
            # but only when the stopped conversation is the one on screen.
            if viewing_stopped_conversation:
                st.session_state.messages.append(result_data["message"])
            st.toast("Generation stopped", icon=":material/stop_circle:")
        else:
            # Fallback: reset conversation state so next rerun reloads messages
            st.session_state.conversation_messages_page = 0
            st.session_state.has_more_messages = True
            st.toast("Generation stopped", icon=":material/stop_circle:")
    else:
        # Stop call failed or generation already completed – reset for refresh
        st.session_state.conversation_messages_page = 0
        st.session_state.has_more_messages = True

    # The user message (and its attachments) was already persisted before a
    # stoppable stream id was emitted. Do not leave those images queued for the
    # next message after cancelling or reconciling the stream.
    st.session_state.pending_image_attachments = []
    st.session_state.show_attachment_uploader = False
    _clear_inflight_state()
    st.rerun()


def reset_conversation_state() -> None:
    _clear_interrupt_ui_state()
    st.session_state.messages = []
    st.session_state.conversation_messages_meta = None
    st.session_state.conversation_messages_page = 0
    st.session_state.has_more_messages = True
    st.session_state.pending_persona_prompt = ""
    st.session_state.persona_editor_origin = None
    st.session_state.persona_editor_value = ""
    st.session_state.pending_custom_agent_ids = []
    st.session_state.pending_image_attachments = []
    st.session_state.show_attachment_uploader = False
    st.session_state.message_image_thumbnails = {}
    st.session_state.pop(_hitl_reconciliation_key(), None)
    st.session_state.pop("hitl_reconciliation_notice", None)
    # Clear in-flight streaming state
    _clear_inflight_state()


_MANAGER_PAGE_SIZE = 100


def open_conversation_manager() -> None:
    """Open the conversation manager dialog on the next render."""
    st.session_state[CONVERSATION_MANAGER_DIALOG_KEY] = True
    # Reset lazy-load state so the dialog fetches fresh data on open
    st.session_state.pop("manager_conversations", None)
    st.session_state.pop("manager_conv_page", None)
    st.session_state.pop("manager_conv_has_more", None)
    st.session_state.pop("manager_conv_total", None)


def close_conversation_manager() -> None:
    """Close the conversation manager dialog and prevent reopening on rerun."""
    st.session_state[CONVERSATION_MANAGER_DIALOG_KEY] = False
    # Free memory held by the manager conversation cache
    st.session_state.pop("manager_conversations", None)
    st.session_state.pop("manager_conv_page", None)
    st.session_state.pop("manager_conv_has_more", None)
    st.session_state.pop("manager_conv_total", None)


def find_conversation_in_state(
    conversation_id: str | None,
) -> dict[str, Any] | None:
    """Find a conversation from session state by ID."""
    if not conversation_id:
        return None

    return next(
        (conv for conv in st.session_state.conversations_list if conv.get("id") == conversation_id),
        None,
    )


def upsert_conversation_in_state(conversation: dict[str, Any] | None) -> None:
    """Insert or merge a conversation in session state without refetching all items."""
    if not isinstance(conversation, dict):
        return

    conversation_id = conversation.get("id")
    if not conversation_id:
        return

    merged_items: list[dict[str, Any]] = []
    replaced = False
    for existing in st.session_state.conversations_list:
        if existing.get("id") == conversation_id:
            merged_items.append({**existing, **conversation})
            replaced = True
        else:
            merged_items.append(existing)

    if not replaced:
        merged_items.insert(0, conversation)

    st.session_state.conversations_list = merged_items


def is_placeholder_conversation_title(title: str | None) -> bool:
    """Return True when title is still a default placeholder."""
    normalized = (title or "").strip().lower()
    return normalized in _PLACEHOLDER_CONVERSATION_TITLES


def sync_conversation_title_from_server(conversation_id: str | None) -> None:
    """
    Fetch one conversation and sync title only when the local title is a placeholder.
    Avoids a full conversations list refresh.
    """
    if not conversation_id or conversation_id == "pending_new":
        return

    current_conversation = find_conversation_in_state(conversation_id)
    current_title = current_conversation.get("title") if current_conversation else ""
    if not is_placeholder_conversation_title(current_title):
        return

    response = make_api_request("GET", f"/conversations/{conversation_id}")
    if not response:
        return

    server_conversation = response.get("data")
    if not isinstance(server_conversation, dict):
        return

    server_title = server_conversation.get("title")
    if not server_title:
        return

    upsert_conversation_in_state(server_conversation)


def refresh_conversations_list(*, fallback_conversation: dict[str, Any] | None = None) -> None:
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
        upsert_conversation_in_state(fallback_conversation)


@st.cache_resource(show_spinner=False)
def get_http_session() -> requests.Session:
    """Reuse HTTP connections to reduce API call latency across reruns."""
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=20,
        pool_maxsize=40,
    )
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@st.cache_data(show_spinner=False, ttl=10, max_entries=1000)
def _cached_get_request(endpoint: str, auth_token: str, cache_version: int) -> dict[str, Any]:
    """Cache GET responses briefly to avoid refetching on every rerun."""
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    response = get_http_session().get(
        f"{API_BASE_URL}{endpoint}",
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    payload: Any = {}
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    return {
        "status_code": response.status_code,
        "payload": payload if isinstance(payload, dict) else {},
    }


def _extract_api_error_message(status_code: int | None, payload: Any) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()

        error = payload.get("error")
        if isinstance(error, dict) and error:
            first_key = next(iter(error))
            first_value = error[first_key]
            if isinstance(first_value, list) and first_value:
                return f"{first_key}: {first_value[0]}"
            if isinstance(first_value, str) and first_value.strip():
                return f"{first_key}: {first_value.strip()}"
        if isinstance(error, str) and error.strip():
            return error.strip()

    if status_code:
        return f"HTTP error {status_code}"
    return "Request failed"


def _last_api_error_message(default: str = "Request failed") -> str:
    message = st.session_state.get("_last_api_error_message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return default


def _transition_to_login() -> None:
    """Discard user-scoped UI state before returning to the login screen."""
    _clear_skill_hitl_session_state()
    st.session_state.auth_token = None
    st.session_state.current_user_id = None
    st.session_state.current_user_profile = None
    st.session_state.show_login = True


def make_api_request(
    method: str,
    endpoint: str,
    data: dict | None = None,
    *,
    use_cache: bool = True,
) -> dict:
    method = method.strip().upper()
    auth_token = st.session_state.get("auth_token")
    response_data: dict[str, Any]
    st.session_state["_last_api_error_message"] = None

    try:
        if method == "GET" and data is None and use_cache:
            cached = _cached_get_request(
                endpoint=endpoint,
                auth_token=str(auth_token or ""),
                cache_version=int(st.session_state.get("api_cache_version", 0)),
            )
            status_code = int(cached.get("status_code") or 0)
            response_data = cached.get("payload") or {}
            if status_code >= 400:
                error_message = _extract_api_error_message(status_code, response_data)
                st.session_state["_last_api_error_message"] = error_message
                if status_code == 401:
                    _transition_to_login()
                    st.toast("Please log in", icon=":material/lock:")
                    return {}
                st.toast(error_message, icon=":material/cancel:")
                return {}
        else:
            url = f"{API_BASE_URL}{endpoint}"
            headers = {}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            response = get_http_session().request(
                method,
                url,
                json=data,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            parsed = response.json()
            response_data = parsed if isinstance(parsed, dict) else {}
    except requests.exceptions.HTTPError as http_error:
        status_code = http_error.response.status_code if http_error.response is not None else None
        payload: Any = {}
        if http_error.response is not None:
            try:
                payload = http_error.response.json()
            except ValueError:
                payload = {}
        error_message = _extract_api_error_message(status_code, payload)
        st.session_state["_last_api_error_message"] = error_message
        if status_code == 401:
            _transition_to_login()
            st.toast("Please log in", icon=":material/lock:")
            return {}
        st.toast(error_message, icon=":material/cancel:")
        return {}
    except requests.exceptions.ConnectionError:
        st.toast("Cannot connect to API", icon=":material/cancel:")
        return {}
    except ValueError:
        st.toast("Unexpected response from API", icon=":material/cancel:")
        return {}
    except Exception as exc:
        st.toast(f"Error: {exc}", icon=":material/cancel:")
        return {}

    if not response_data.get("success"):
        error_code = response_data.get("code", "unknown_error")
        error_message = response_data.get("message", "An unknown error occurred.")
        st.session_state["_last_api_error_message"] = error_message

        if error_code == "unauthenticated":
            _transition_to_login()
            st.toast("Please log in", icon=":material/lock:")
        else:
            st.toast(f"{error_message}", icon=":material/cancel:")
        return {}

    if method in {"POST", "PUT", "PATCH", "DELETE"}:
        st.session_state.api_cache_version = int(st.session_state.get("api_cache_version", 0)) + 1

    return response_data


def _discover_system_zone_name() -> str | None:
    """Return an IANA timezone name when the host exposes one."""
    configured = os.environ.get("TZ", "").strip()
    candidates = [configured] if configured else []
    local_tz = datetime.now().astimezone().tzinfo
    local_key = getattr(local_tz, "key", None)
    if isinstance(local_key, str):
        candidates.append(local_key)
    for candidate in candidates:
        try:
            ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError):
            continue
        return candidate
    return None


def get_local_timezone_name() -> str:
    """Select a portable dashboard timezone, falling back to UTC."""
    return _discover_system_zone_name() or "UTC"


def align_usage_boundary(
    value: date | datetime,
    *,
    bucket: str,
    zone: ZoneInfo,
    exclusive_end: bool = False,
) -> datetime:
    """Align a UI boundary to the API's local bucket contract."""
    if bucket not in {"hour", "day"}:
        raise ValueError("bucket must be 'hour' or 'day'")

    value_is_datetime = isinstance(value, datetime)
    if value_is_datetime:
        localized = value.replace(tzinfo=zone) if value.tzinfo is None else value.astimezone(zone)
    else:
        localized = datetime.combine(value, datetime_time.min, tzinfo=zone)

    if bucket == "day":
        boundary_date = localized.date() + (timedelta(days=1) if exclusive_end else timedelta())
        return datetime.combine(boundary_date, datetime_time.min, tzinfo=zone)
    if exclusive_end and not value_is_datetime:
        next_date = localized.date() + timedelta(days=1)
        return datetime.combine(next_date, datetime_time.min, tzinfo=zone)
    return localized.replace(minute=0, second=0, microsecond=0)


def _build_usage_query_boundaries(
    *,
    start_date: date,
    end_date: date,
    bucket: str,
    zone: ZoneInfo,
    start_hour: datetime_time | None = None,
    end_hour: datetime_time | None = None,
) -> tuple[date | datetime, date | datetime]:
    """Convert inclusive date-picker values into API query boundaries."""
    if bucket == "day":
        return start_date, end_date
    if bucket != "hour":
        raise ValueError("bucket must be 'hour' or 'day'")
    if start_hour is None or end_hour is None:
        raise ValueError("hour boundaries require start_hour and end_hour")

    exclusive_end_date = end_date
    if end_hour.replace(tzinfo=None) == datetime_time.min:
        exclusive_end_date += timedelta(days=1)
    return (
        datetime.combine(start_date, start_hour, tzinfo=zone),
        datetime.combine(exclusive_end_date, end_hour, tzinfo=zone),
    )


def _usage_cache_identity() -> str:
    """Build a non-secret cache partition for the active authenticated user."""
    user_id = str(st.session_state.get("current_user_id") or "anonymous")
    auth_token = str(st.session_state.get("auth_token") or "")
    token_fingerprint = hashlib.sha256(auth_token.encode("utf-8")).hexdigest()
    return f"{user_id}:{token_fingerprint}"


@st.cache_data(show_spinner=False, ttl=30, max_entries=500)
def _cached_usage_get_request(
    endpoint: str,
    *,
    auth_identity: str,
    cache_version: int,
) -> dict[str, Any]:
    """Cache authenticated usage reads without sharing entries across users."""
    del auth_identity, cache_version  # Values intentionally participate in the cache key.
    return make_api_request("GET", endpoint, use_cache=False)


def _usage_get(endpoint: str) -> dict[str, Any]:
    return _cached_usage_get_request(
        endpoint,
        auth_identity=_usage_cache_identity(),
        cache_version=int(st.session_state.get("usage_cache_version", 0)),
    )


def _clear_usage_cache_after_completed_turn() -> None:
    """Invalidate analytics only after the server completes an AI turn."""
    st.session_state.usage_cache_version = int(st.session_state.get("usage_cache_version", 0)) + 1
    _cached_usage_get_request.clear()


def _empty_usage_totals() -> dict[str, int]:
    return {
        "inputTokens": 0,
        "outputTokens": 0,
        "totalTokens": 0,
        "reasoningTokens": 0,
        "cachedInputTokens": 0,
        "generatedImages": 0,
        "requestCount": 0,
    }


def _normalized_usage_totals(value: Any) -> dict[str, int]:
    source = value if isinstance(value, dict) else {}
    totals = _empty_usage_totals()
    for key in totals:
        candidate = source.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            totals[key] = candidate
    return totals


def _normalize_usage_dashboard(response: Any) -> dict[str, Any] | None:
    """Extract known dashboard fields while tolerating forward-compatible additions."""
    if not isinstance(response, dict) or response.get("success") is not True:
        return None
    data = response.get("data")
    if not isinstance(data, dict):
        return None
    array_fields = (
        "outcomes",
        "series",
        "byProvider",
        "byModel",
        "byOperation",
        "byAgent",
        "topConversations",
    )
    return {
        "totals": _normalized_usage_totals(data.get("totals")),
        **{
            key: list(data.get(key) or []) if isinstance(data.get(key), list) else []
            for key in array_fields
        },
        "coverage": dict(data.get("coverage") or {})
        if isinstance(data.get("coverage"), dict)
        else {},
        "range": dict(data.get("range") or {}) if isinstance(data.get("range"), dict) else {},
        "generatedAt": data.get("generatedAt"),
    }


def _normalize_conversation_usage(response: Any) -> dict[str, Any] | None:
    if not isinstance(response, dict) or response.get("success") is not True:
        return None
    data = response.get("data")
    if not isinstance(data, dict):
        return None
    return {
        "totals": _normalized_usage_totals(data.get("totals")),
        "byProvider": list(data.get("byProvider") or [])
        if isinstance(data.get("byProvider"), list)
        else [],
        "byModel": list(data.get("byModel") or []) if isinstance(data.get("byModel"), list) else [],
        "coverage": dict(data.get("coverage") or {})
        if isinstance(data.get("coverage"), dict)
        else {},
        "latestContextWindow": data.get("latestContextWindow")
        if isinstance(data.get("latestContextWindow"), dict)
        else None,
        "range": dict(data.get("range") or {}) if isinstance(data.get("range"), dict) else {},
        "generatedAt": data.get("generatedAt"),
    }


def get_usage_dashboard(
    *,
    start: date | datetime,
    end: date | datetime,
    bucket: str,
    timezone_name: str,
    conversation_id: str | None = None,
) -> dict[str, Any] | None:
    zone = ZoneInfo(timezone_name)
    start_at = align_usage_boundary(start, bucket=bucket, zone=zone)
    end_at = align_usage_boundary(end, bucket=bucket, zone=zone, exclusive_end=True)
    params = {
        "from": start_at.isoformat(),
        "to": end_at.isoformat(),
        "bucket": bucket,
        "timezone": timezone_name,
    }
    if conversation_id:
        params["conversationId"] = conversation_id
    response = _usage_get(f"/usage/dashboard?{urlencode(params)}")
    return _normalize_usage_dashboard(response)


def get_conversation_usage(conversation_id: str) -> dict[str, Any] | None:
    """Fetch the retained-range cumulative summary for one owned conversation."""
    return _normalize_conversation_usage(_usage_get(f"/usage/conversations/{conversation_id}"))


def _build_usage_trend_frame(series: Any) -> list[dict[str, Any]]:
    frame: list[dict[str, Any]] = []
    if not isinstance(series, list):
        return frame
    for point in series:
        if not isinstance(point, dict):
            continue
        start = point.get("start")
        totals = _normalized_usage_totals(point.get("totals"))
        frame.extend(
            [
                {"start": start, "tokenType": "Input", "tokens": totals["inputTokens"]},
                {"start": start, "tokenType": "Output", "tokens": totals["outputTokens"]},
            ]
        )
    return frame


def _build_usage_outcome_frame(
    outcomes: Any,
    *,
    total_requests: int,
) -> list[dict[str, Any]]:
    frame: list[dict[str, Any]] = []
    if not isinstance(outcomes, list):
        return frame
    denominator = total_requests if total_requests > 0 else 0
    for item in outcomes:
        if not isinstance(item, dict):
            continue
        requests_count = _normalized_usage_totals(item.get("totals"))["requestCount"]
        frame.append(
            {
                "outcome": str(item.get("key") or "unknown"),
                "requests": requests_count,
                "rate": requests_count / denominator if denominator else 0.0,
            }
        )
    return frame


def _build_usage_breakdown_frame(items: Any) -> list[dict[str, Any]]:
    frame: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return frame
    for item in items:
        if not isinstance(item, dict):
            continue
        totals = _normalized_usage_totals(item.get("totals"))
        frame.append(
            {
                "name": str(item.get("key") or "Unknown"),
                "totalTokens": totals["totalTokens"],
                "requests": totals["requestCount"],
            }
        )
    return frame


def make_streaming_request(endpoint: str, data: dict | None = None):
    """
    Make a streaming API request using Server-Sent Events (SSE).
    Yields parsed JSON events from the stream.
    Heartbeat events from the server are silently consumed (no-op) so the
    connection stays alive and Streamlit gets frequent yield-points.
    """
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    if st.session_state.get("auth_token"):
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

    stream_completed = False
    response = None
    try:
        # Long-running MCP tools can block the stream for several minutes, so use a generous read timeout
        response = get_http_session().post(
            url,
            json=data,
            headers=headers,
            stream=True,
            timeout=STREAM_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        st.session_state.api_cache_version = int(st.session_state.get("api_cache_version", 0)) + 1

        # Parse SSE stream
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                # SSE format: "data: {json}"
                event_data = line[6:]  # Remove "data: " prefix
                try:
                    event = json.loads(event_data)
                    event_type = event.get("type")

                    # Silently consume heartbeat events (keep-alive)
                    if event_type == "heartbeat":
                        continue

                    yield event
                    if event_type in ["complete", "error", "interrupt"]:
                        stream_completed = True
                except json.JSONDecodeError:
                    continue

    except requests.exceptions.HTTPError as http_error:
        status_code = http_error.response.status_code if http_error.response is not None else None
        payload: Any = {}
        if http_error.response is not None:
            with contextlib.suppress(ValueError):
                payload = http_error.response.json()
        error_message = _extract_api_error_message(status_code, payload)
        st.toast(error_message, icon=":material/cancel:")
        yield {
            "type": "error",
            "error": error_message,
            "status_code": status_code,
            "error_code": extract_error_code(payload) if isinstance(payload, dict) else None,
        }
    except requests.exceptions.ConnectionError:
        if not stream_completed:
            st.toast("Cannot connect to API", icon=":material/cancel:")
            yield {"type": "error", "error": "Connection error"}
    except requests.exceptions.Timeout:
        st.toast("Request timed out", icon=":material/schedule:")
        yield {"type": "error", "error": "Timeout"}
    except Exception as exc:
        st.toast(f"Error: {exc}", icon=":material/cancel:")
        yield {"type": "error", "error": str(exc)}
    finally:
        # Ensure the response connection is closed to avoid resource leaks
        if response is not None:
            with contextlib.suppress(Exception):
                response.close()


def get_user(user_id: str) -> dict[str, Any]:
    response = make_api_request("GET", f"/users/{user_id}")
    return response.get("data", {})


def get_conversations(
    page: int = 1,
    limit: int = 100,
    include_messages: bool = False,
    latest_messages: int = 3,
    fetch_all_pages: bool = False,
) -> dict[str, Any]:
    """Retrieve conversations with controlled pagination."""
    current_page = page
    aggregated_items: list[dict[str, Any]] = []
    aggregated_meta: dict[str, Any] = {}
    last_response: dict[str, Any] | None = None

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

        # If we received fewer items than the limit, this is the last page
        if len(items) < limit:
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

    result: dict[str, Any] = {
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
) -> dict[str, Any]:
    """Get paginated conversation messages"""
    endpoint = (
        f"/conversations/{conversation_id}/messages"
        f"?page={page}&limit={limit}&orderBy={order_by}&orderDirection={order_direction}&include=feedback"
    )
    response = make_api_request("GET", endpoint)
    return response


def upsert_provider(
    provider_type: str,
    api_key: str,
    *,
    is_default: bool = False,
    provider_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "provider_type": provider_type,
        "api_key": api_key,
        "is_default": is_default,
        "provider_metadata": provider_metadata or {},
    }
    response = make_api_request("POST", "/providers", payload)
    return response.get("data") if response else None


def delete_provider(provider_type: str) -> bool:
    response = make_api_request("DELETE", f"/providers/{provider_type}")
    return bool(response)


def _cache_bust_query(force_refresh: bool = False) -> str:
    if not force_refresh:
        return ""
    return f"?_ts={uuid.uuid4().hex}"


def fetch_provider_models(
    provider_type: str, *, force_refresh: bool = False
) -> list[dict[str, Any]] | None:
    response = make_api_request(
        "GET",
        f"/providers/{provider_type}/models{_cache_bust_query(force_refresh)}",
    )
    if not response:
        return None
    data = response.get("data", [])
    return data if isinstance(data, list) else []


def get_model_config_options(*, force_refresh: bool = False) -> dict[str, Any]:
    response = make_api_request("GET", f"/model-config/options{_cache_bust_query(force_refresh)}")
    data = response.get("data", {}) if response else {}
    return data if isinstance(data, dict) else {}


def patch_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Upsert one or more agent configs."""
    response = make_api_request("PATCH", "/model-config", payload)
    data = response.get("data", {}) if response else {}
    return data if isinstance(data, dict) else {}


def reset_model_config() -> dict[str, Any]:
    """Reset all persisted agent configs back to defaults."""
    response = make_api_request("POST", "/model-config/reset", {})
    data = response.get("data", {}) if response else {}
    return data if isinstance(data, dict) else {}


def _normalize_provider_type(value: Any) -> str:
    provider_type = str(value or "").strip().lower()
    return provider_type if provider_type else "gemini"


def _provider_display_name(provider_type: str) -> str:
    names = {
        "gemini": "Gemini",
        "openai": "OpenAI",
    }
    return names.get(provider_type, provider_type.replace("_", " ").title())


def _snapshot_provider_list(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    providers = snapshot.get("providers", [])
    return providers if isinstance(providers, list) else []


def _snapshot_provider_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _normalize_provider_type(
            provider.get("providerType") or provider.get("provider_type")
        ): provider
        for provider in _snapshot_provider_list(snapshot)
        if isinstance(provider, dict)
    }


def _snapshot_agent_config(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    config = snapshot.get("agentConfig") or snapshot.get("agent_config") or {}
    return config if isinstance(config, dict) else {}


def _provider_models(provider_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    models = provider_snapshot.get("models", [])
    return models if isinstance(models, list) else []


def _provider_model_ids(provider_snapshot: dict[str, Any]) -> list[str]:
    model_ids: list[str] = []
    for model in _provider_models(provider_snapshot):
        model_id = str(model.get("id") or "").strip()
        if model_id and model_id not in model_ids:
            model_ids.append(model_id)
    return model_ids


def _sync_model_config_form_state(snapshot: dict[str, Any]) -> None:
    provider_map = _snapshot_provider_map(snapshot)
    agent_config = _snapshot_agent_config(snapshot)

    for agent_key in ("chat", "rag", "search", "planning"):
        cfg = agent_config.get(agent_key, {}) if isinstance(agent_config, dict) else {}
        provider = _normalize_provider_type(cfg.get("provider"))
        provider_snapshot = provider_map.get(provider, {})
        catalog_ids = _provider_model_ids(provider_snapshot)
        current_model = str(cfg.get("model") or "").strip()
        current_temperature = cfg.get("temperature", 1.0)
        is_custom_model = bool(cfg.get("isCustomModel") or cfg.get("is_custom_model"))

        selected_model = (
            current_model
            if current_model in catalog_ids
            else (catalog_ids[0] if catalog_ids else current_model)
        )
        custom_model = current_model if is_custom_model else ""

        st.session_state[f"model_cfg_provider_{agent_key}"] = provider
        st.session_state[f"model_cfg_model_select_{agent_key}"] = selected_model
        st.session_state[f"model_cfg_model_custom_{agent_key}"] = custom_model
        st.session_state[f"model_cfg_allow_custom_{agent_key}"] = is_custom_model
        st.session_state[f"model_cfg_temperature_{agent_key}"] = (
            float(current_temperature) if isinstance(current_temperature, (int, float)) else 1.0
        )


def refresh_model_config_options_cache(
    *, force_refresh: bool = False, defer_form_state_sync: bool = False
) -> dict[str, Any]:
    snapshot = get_model_config_options(force_refresh=force_refresh)
    if snapshot:
        st.session_state.model_config_options_cache = snapshot
        st.session_state.model_config_options_last_fetch = datetime.now(timezone.utc).isoformat()
        st.session_state.model_config_options_error = None
        if defer_form_state_sync:
            st.session_state.model_config_options_needs_form_sync = True
        else:
            _sync_model_config_form_state(snapshot)
            st.session_state.model_config_options_needs_form_sync = False
        return snapshot

    st.session_state.model_config_options_error = "Failed to load model configuration options."
    return st.session_state.get("model_config_options_cache") or {}


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


def persona_preview(text: str | None, limit: int = 160) -> str:
    """Return a compact preview of persona text for UI surfaces."""
    if not text:
        return ""
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _attachment_from_upload(uploaded_file) -> dict[str, str] | None:
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


def _handle_new_image_attachments(uploaded_files: list) -> None:
    if not uploaded_files:
        return

    new_items: list[dict[str, str]] = []
    for file_obj in uploaded_files:
        attachment = _attachment_from_upload(file_obj)
        if not attachment:
            continue

        new_items.append(attachment)

    if not new_items:
        return

    pending = st.session_state.get("pending_image_attachments", [])
    pending.extend(new_items)
    st.session_state.pending_image_attachments = pending


def _chat_image_uploader_key(conversation_id: str) -> str:
    try:
        nonce = int(st.session_state.get("chat_image_uploader_nonce", 0) or 0)
    except (TypeError, ValueError):
        nonce = 0
    return f"chat_image_uploader_{conversation_id}_{nonce}"


def _advance_chat_image_uploader_nonce() -> None:
    try:
        nonce = int(st.session_state.get("chat_image_uploader_nonce", 0) or 0)
    except (TypeError, ValueError):
        nonce = 0
    st.session_state.chat_image_uploader_nonce = nonce + 1


def _pending_message_draft_key(conversation_id: str) -> str:
    return f"pending_message_draft_{conversation_id}"


def _preserve_message_draft_for_attachment_toggle(conversation_id: str, message: str) -> None:
    st.session_state[_pending_message_draft_key(conversation_id)] = message


def _consume_preserved_message_draft(conversation_id: str) -> str:
    draft = st.session_state.pop(_pending_message_draft_key(conversation_id), "")
    return draft if isinstance(draft, str) else ""


def _consume_chat_image_uploader(uploader_key: str) -> None:
    uploaded_files = st.session_state.get(uploader_key) or []
    if not uploaded_files:
        return
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    _handle_new_image_attachments(uploaded_files)
    _advance_chat_image_uploader_nonce()


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


def _render_pending_image_attachments() -> None:
    pending = st.session_state.get("pending_image_attachments", [])
    if not pending:
        return

    count = len(pending)
    st.caption(f"{count} attachment{'s' if count != 1 else ''} ready")
    for row_start in range(0, len(pending), _PENDING_IMAGE_PREVIEW_COLUMNS):
        row = pending[row_start : row_start + _PENDING_IMAGE_PREVIEW_COLUMNS]
        cols = st.columns(len(row))
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


def _format_image_only_message(attachments: list[dict[str, str]]) -> str:
    """Generate fallback message content for image-only submissions."""
    if not attachments:
        return "[Image attachments]"

    names = [att.get("name") for att in attachments if isinstance(att, dict) and att.get("name")]

    if not names:
        count = len(attachments)
        return "[Image attachment]" if count == 1 else f"[Image attachments: {count} files]"

    if len(names) == 1:
        return f"[Image attachment: {names[0]}]"

    displayed = ", ".join(names[:3])
    if len(names) > 3:
        displayed += ", ..."

    return f"[Image attachments: {displayed}]"


def get_mcp_servers() -> dict[str, Any] | None:
    """Fetch list of MCP servers"""
    response = make_api_request("GET", "/mcp/servers")
    return response.get("data") if response else None


def get_mcp_tools(server_name: str | None = None) -> dict[str, Any] | None:
    """Fetch MCP tools, optionally filtered by server"""
    endpoint = "/mcp/tools"
    if server_name:
        endpoint += f"?serverName={server_name}"
    response = make_api_request("GET", endpoint)
    return response.get("data") if response else None


def execute_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    server_name: str | None = None,
    qualified_tool_id: str | None = None,
) -> dict[str, Any] | None:
    """Execute an MCP tool with provided arguments"""
    payload: dict[str, Any] = {"arguments": arguments}
    if server_name:
        payload["serverName"] = server_name
    if qualified_tool_id:
        payload["qualifiedToolId"] = qualified_tool_id
    response = make_api_request("POST", f"/mcp/tools/{tool_name}/execute", payload)
    return response.get("data") if response else None


def _mcp_tool_execution_results() -> dict[str, dict[str, Any]]:
    """Return tester results keyed by server-qualified tool identity."""
    results = st.session_state.get("mcp_tool_execution_results")
    if not isinstance(results, dict):
        results = {}
        st.session_state["mcp_tool_execution_results"] = results
    return results


def _remember_mcp_tool_execution_result(qualified_tool_id: str, result: dict[str, Any]) -> None:
    _mcp_tool_execution_results()[qualified_tool_id] = result


def _clear_mcp_tool_execution_result(qualified_tool_id: str) -> None:
    _mcp_tool_execution_results().pop(qualified_tool_id, None)


def get_hitl_settings() -> dict[str, Any] | None:
    """Fetch this device's editable HITL settings through the local sidecar."""
    st.session_state.pop("_hitl_settings_unavailable_reason", None)
    response = make_api_request("GET", "/hitl/settings", use_cache=False)
    data = response.get("data") if response else None
    if not isinstance(data, dict):
        detail = _last_api_error_message(
            "The local sidecar did not return valid HITL settings"
        ).rstrip(". ")
        st.session_state["_hitl_settings_unavailable_reason"] = (
            f"Human approval settings are unavailable: {detail}. "
            "Confirm the local sidecar is connected, then refresh."
        )
        return None
    expected_device_id = str(st.session_state.get("device_id") or "").strip()
    response_device_id = str(data.get("deviceId") or "").strip()
    if expected_device_id and response_device_id != expected_device_id:
        st.session_state["_hitl_settings_unavailable_reason"] = (
            "Human approval settings are unavailable because the response belongs "
            "to another device. Reconnect the local sidecar and refresh."
        )
        return None
    return data


def _hitl_settings_unavailable_message() -> str:
    """Return safe, actionable guidance for an unavailable HITL policy."""
    message = st.session_state.get("_hitl_settings_unavailable_reason")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return (
        "Human approval settings are unavailable. Confirm the local sidecar is "
        "connected, then refresh."
    )


def get_hitl_interrupt_state(interrupt_id: str) -> dict[str, Any] | None:
    """Fetch the canonical lifecycle state without reusing a stale GET cache."""
    response = make_api_request("GET", f"/hitl/interrupts/{interrupt_id}", use_cache=False)
    return response.get("data") if response else None


def _hitl_resume_lock_key(interrupt_id: str | None) -> str:
    return f"hitl_resume_inflight_{interrupt_id or 'unknown'}"


def _hitl_reconciliation_key() -> str:
    return "hitl_reconciling_interrupt_id"


def _clear_interrupt_decision_state() -> None:
    for key in list(st.session_state):
        key_text = str(key)
        if key_text == "pending_decisions" or key_text.startswith(
            ("pending_decisions_", "editing_tool_")
        ):
            del st.session_state[key]


def _clear_interrupt_ui_state(interrupt_id: str | None = None) -> None:
    """Clear paused approval UI while deliberately retaining reconciliation state."""
    _clear_interrupt_decision_state()
    if interrupt_id:
        st.session_state.pop(_hitl_resume_lock_key(interrupt_id), None)
    else:
        for key in list(st.session_state):
            if str(key).startswith("hitl_resume_inflight_"):
                del st.session_state[key]
    st.session_state.pop("pending_interrupt", None)
    st.session_state.pop("interrupt_conversation_id", None)
    st.session_state.conversation_messages_page = 0


def _reconcile_interrupt(interrupt_id: str) -> str:
    """Reconcile one duplicate resume against durable lifecycle state without replaying it."""
    lifecycle_state = get_hitl_interrupt_state(interrupt_id)
    status = lifecycle_state.get("status") if lifecycle_state else None
    action = reconciliation_action(status)
    marker_key = _hitl_reconciliation_key()

    if action == "restore_form":
        st.session_state.pop(_hitl_resume_lock_key(interrupt_id), None)
        if st.session_state.get(marker_key) == interrupt_id:
            st.session_state.pop(marker_key, None)
    elif action == "show_processing":
        _clear_interrupt_decision_state()
        st.session_state[marker_key] = interrupt_id
    elif action == "refresh_history":
        _clear_interrupt_ui_state(interrupt_id)
        st.session_state[marker_key] = interrupt_id
        st.session_state["hitl_reconciliation_notice"] = (
            "Approval completed elsewhere; conversation refreshed."
        )
    else:
        # Failed, expired, and unavailable lifecycle reads require a new turn.
        _clear_interrupt_ui_state(interrupt_id)
        st.session_state[marker_key] = interrupt_id
        st.session_state["hitl_reconciliation_notice"] = (
            "This approval can no longer be resumed. Send a new message."
        )

    return action


def set_hitl_setting(
    tool_origin: str,
    scope_type: str,
    scope_value: str,
    require_approval: bool,
) -> dict[str, Any] | None:
    """Upsert one device-scoped client HITL approval rule."""
    payload = {
        "items": [
            {
                "scopeType": scope_type,
                "scopeValue": scope_value,
                "toolOrigin": tool_origin,
                "requireApproval": require_approval,
            }
        ]
    }
    response = make_api_request("POST", "/hitl/settings", payload)
    return response.get("data") if response else None


def clear_hitl_setting(
    tool_origin: str, scope_type: str, scope_value: str
) -> dict[str, Any] | None:
    """Delete one device-scoped client rule (revert to inherit/default)."""
    response = make_api_request(
        "DELETE",
        f"/hitl/settings?scope_type={scope_type}&scope_value={scope_value}"
        f"&tool_origin={tool_origin}",
    )
    return response.get("data") if response else None


def _persist_skill_hitl_mode(
    widget_key: str,
    skill_qualified_id: str,
    current_mode: str,
) -> None:
    """Persist one deliberate radio change and restore server state on failure."""
    chosen = str(st.session_state.get(widget_key) or current_mode)
    if chosen == current_mode:
        return
    if chosen == "Inherit":
        result = clear_hitl_setting("client_skill", "tool", skill_qualified_id)
    else:
        result = set_hitl_setting("client_skill", "tool", skill_qualified_id, chosen == "Require")
    if result is None:
        st.session_state[widget_key] = current_mode
        st.session_state[f"{widget_key}_error"] = _last_api_error_message(
            "Failed to update skill approval"
        )


def _clear_skill_hitl_session_state() -> None:
    """Remove user-scoped skill approval and credential widgets during logout."""
    for key in list(st.session_state):
        if str(key).startswith("hitl_skill_mode_") or str(key).startswith("skill_secret_"):
            del st.session_state[key]


def render_json_output(data: Any, label: str = "JSON Output", expanded: bool | None = None) -> None:
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

    image_content: Any = None
    if isinstance(payload, dict):
        candidate = payload.get("content")
        if isinstance(candidate, (dict, list)):
            image_content = candidate
    elif isinstance(payload, list):
        image_content = payload

    image_blocks = (
        [image_content]
        if isinstance(image_content, dict)
        else image_content
        if isinstance(image_content, list)
        else []
    )
    if any(
        isinstance(block, dict) and str(block.get("type") or "").strip().lower() == "image"
        for block in image_blocks
    ):
        _render_image_tool_result({"content": image_blocks})
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
                json_string = json.dumps(parsed, indent=2, ensure_ascii=False, default=str)
                st.code(json_string, language="json", line_numbers=False)
        else:
            st.code(json.dumps(parsed, ensure_ascii=False), language="json")


def _get_render_structured_content(render: dict[str, Any]) -> Any:
    if not isinstance(render, dict):
        return None
    return render.get("structured_content") or render.get("structuredContent")


def _render_table_tool_result(render: dict[str, Any]) -> bool:
    structured = _get_render_structured_content(render)
    if not isinstance(structured, dict):
        return False

    columns = structured.get("columns")
    rows = structured.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return False

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(row)
        elif isinstance(row, list):
            normalized_rows.append(
                {
                    str(column): row[index] if index < len(row) else None
                    for index, column in enumerate(columns)
                }
            )

    if not normalized_rows:
        st.caption("Table result contained no rows.")
        return True

    st.dataframe(normalized_rows, width="stretch", hide_index=True)
    return True


def _render_chart_tool_result(render: dict[str, Any]) -> bool:
    structured = _get_render_structured_content(render)
    if not isinstance(structured, dict):
        return False

    labels = structured.get("labels")
    datasets = structured.get("datasets")
    if not isinstance(labels, list) or not isinstance(datasets, list):
        return False

    normalized_datasets: list[tuple[str, list[Any]]] = []
    used_names: dict[str, int] = {}
    for dataset_index, dataset in enumerate(datasets, start=1):
        if not isinstance(dataset, dict):
            continue
        base_name = str(dataset.get("label") or f"Series {dataset_index}")
        occurrence = used_names.get(base_name, 0) + 1
        used_names[base_name] = occurrence
        name = base_name if occurrence == 1 else f"{base_name} ({occurrence})"
        data = dataset.get("data")
        normalized_datasets.append((name, data if isinstance(data, list) else []))

    chart_rows: list[dict[str, Any]] = []
    for label_index, label in enumerate(labels):
        row: dict[str, Any] = {"label": label}
        for name, data in normalized_datasets:
            if label_index < len(data):
                row[name] = data[label_index]
        chart_rows.append(row)

    if not chart_rows:
        st.caption("Chart result contained no plottable values.")
        return True

    chart_type = str(structured.get("chart_type") or structured.get("chartType") or "line").lower()
    series_names = list(dict.fromkeys(key for row in chart_rows for key in row if key != "label"))
    chart_data: dict[str, list[Any]] = {
        "label": [row.get("label") for row in chart_rows],
        **{name: [row.get(name) for row in chart_rows] for name in series_names},
    }
    if chart_type == "bar":
        st.bar_chart(chart_data, x="label", y=series_names)
    elif chart_type == "area":
        st.area_chart(chart_data, x="label", y=series_names)
    else:
        st.line_chart(chart_data, x="label", y=series_names)

    with st.expander("Chart data", expanded=False):
        st.dataframe(chart_rows, width="stretch", hide_index=True)
    return True


def _render_resource_tool_result(render: dict[str, Any]) -> bool:
    resources = render.get("resources")
    if not isinstance(resources, list) or not resources:
        return False

    for resource in resources:
        if not isinstance(resource, dict):
            continue
        uri = str(resource.get("uri") or "")
        title = str(resource.get("title") or uri or "Resource")
        mime_type = str(resource.get("mime_type") or resource.get("mimeType") or "")
        if uri.startswith(("http://", "https://")):
            st.link_button(title, uri, width="stretch")
        else:
            st.markdown(f"**{title}**")
            st.code(uri, language="text")
        if mime_type:
            st.caption(mime_type)
    return True


def _render_image_tool_result(render: dict[str, Any]) -> bool:
    """Render MCP image content blocks instead of falling back to raw JSON."""
    from app.ui.rich_response import build_inline_image_html

    raw_content = render.get("content")
    if isinstance(raw_content, dict):
        content = [raw_content]
    elif isinstance(raw_content, list):
        content = raw_content
    else:
        content = []

    rendered = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").strip().lower()
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                st.markdown(text.strip())
                rendered = True
            continue
        if block_type != "image":
            continue

        raw_url = block.get("url") or block.get("src")
        if isinstance(raw_url, dict):
            raw_url = raw_url.get("url")
        url = raw_url.strip() if isinstance(raw_url, str) else ""
        raw_data = block.get("data") or block.get("base64")
        data = raw_data.strip() if isinstance(raw_data, str) else ""
        mime = str(block.get("mimeType") or block.get("mime_type") or "image/png")
        src = url or (
            data if data.startswith("data:") else f"data:{mime};base64,{data}" if data else ""
        )
        if not src:
            continue

        caption = block.get("title") or block.get("alt") or block.get("description")
        st.markdown(
            build_inline_image_html(src, caption=str(caption) if caption else None),
            unsafe_allow_html=True,
        )
        rendered = True

    # Image resources use the same normalized resource list as ordinary links.
    # Keep them available when a provider returns URLs rather than content blocks.
    return _render_resource_tool_result(render) or rendered


def _render_mcp_app_tool_result(render: dict[str, Any]) -> bool:
    template_uri = render.get("template_uri") or render.get("templateUri")
    if not isinstance(template_uri, str) or not template_uri.strip():
        return False

    title = render.get("title") or "MCP app view"
    st.info(
        "This tool returned an MCP app template. The Streamlit demo cannot mount "
        "arbitrary MCP app iframes yet, but the frontend can use this URI to render it."
    )
    st.markdown(f"**{html.escape(str(title))}**", unsafe_allow_html=True)
    st.code(template_uri, language="text")

    structured = _get_render_structured_content(render)
    if structured not in (None, "", [], {}):
        with st.expander("Structured app data", expanded=False):
            render_tool_result_payload(structured, use_expander=False)
    return True


def _render_subagent_dispatch_tool_result(render: dict[str, Any]) -> bool:
    structured = _get_render_structured_content(render)
    if not isinstance(structured, dict):
        return False

    results = structured.get("results")
    if not isinstance(results, list):
        return False

    status = str(structured.get("status") or "unknown").strip().lower()
    rationale = structured.get("rationale")
    completed = sum(
        1
        for item in results
        if isinstance(item, dict) and str(item.get("status") or "").lower() == "completed"
    )
    st.caption(f"Dispatch status: {status} | {completed}/{len(results)} workers completed")
    if isinstance(rationale, str) and rationale.strip():
        st.caption(rationale.strip())

    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue

        worker_id = str(item.get("id") or f"worker-{index}")
        agent_name = item.get("agent_name") or get_agent_display_name(
            str(item.get("agent") or "unknown_agent")
        )
        worker_status = str(item.get("status") or "unknown").strip().lower()
        elapsed_ms = item.get("elapsed_ms")
        elapsed = ""
        if isinstance(elapsed_ms, (int, float)) and elapsed_ms >= 0:
            elapsed = f" | {float(elapsed_ms) / 1000:.2f}s"

        if worker_status == "completed":
            icon = ":material/check_circle:"
        elif worker_status in {"failed", "timeout"}:
            icon = ":material/error:"
        elif worker_status == "requires_approval":
            icon = ":material/pending_actions:"
        else:
            icon = ":material/help:"

        with st.expander(
            f"{icon} {worker_id} - {agent_name} - {worker_status}{elapsed}",
            expanded=worker_status != "completed",
        ):
            related = item.get("related_todo_ids")
            if isinstance(related, list) and related:
                st.caption("Related todos: " + ", ".join(str(value) for value in related))

            summary = item.get("summary")
            if isinstance(summary, str) and summary.strip():
                st.markdown(summary.strip())
            else:
                st.caption("Worker returned no summary.")

            error = item.get("error")
            if isinstance(error, str) and error.strip():
                st.error(error.strip())

            artifacts = item.get("artifacts")
            if isinstance(artifacts, list) and artifacts:
                with st.expander("Worker artifacts", expanded=False):
                    render_tool_result_payload(artifacts, use_expander=False)

    return True


def render_tool_render_payload(render: Any, fallback_output: Any = None) -> bool:
    if not isinstance(render, dict):
        return False

    render_type = str(render.get("type") or "").lower()
    title = render.get("title")
    if isinstance(title, str) and title.strip():
        st.markdown(f"**{html.escape(title.strip())}**", unsafe_allow_html=True)

    if render_type == "error":
        error = render.get("error") or render.get("text") or fallback_output
        st.error(str(error or "Tool execution failed."))
        return True

    if render_type == "subagent_dispatch":
        rendered = _render_subagent_dispatch_tool_result(render)
    elif render_type == "mcp_app":
        rendered = _render_mcp_app_tool_result(render)
    elif render_type == "table":
        rendered = _render_table_tool_result(render)
    elif render_type == "chart":
        rendered = _render_chart_tool_result(render)
    elif render_type == "resource":
        rendered = _render_resource_tool_result(render)
    elif render_type == "image":
        rendered = _render_image_tool_result(render)
    elif render_type == "text":
        text = render.get("text") or fallback_output
        if text not in (None, ""):
            st.markdown(str(text))
            rendered = True
        else:
            rendered = False
    else:
        structured = _get_render_structured_content(render)
        if structured not in (None, "", [], {}):
            render_tool_result_payload(structured, use_expander=False)
            rendered = True
        else:
            rendered = False

    with st.expander("Raw render metadata", expanded=False):
        render_tool_result_payload(render, use_expander=False)

    return rendered


def add_mcp_server(server_config: dict[str, Any]) -> dict[str, Any] | None:
    """Add a new MCP server"""
    response = make_api_request("POST", "/mcp/servers", server_config)
    return response.get("data") if response else None


def remove_mcp_server(server_name: str) -> dict[str, Any] | None:
    """Remove an MCP server"""
    response = make_api_request("DELETE", f"/mcp/servers/{server_name}")
    return response.get("data") if response else None


def toggle_mcp_server(server_name: str, enabled: bool) -> dict[str, Any] | None:
    """Enable or disable an MCP server"""
    response = make_api_request("PATCH", f"/mcp/servers/{server_name}/toggle?enabled={enabled}")
    return response.get("data") if response else None


def add_mcp_server_from_url(url_config: dict[str, Any]) -> bool:
    """Add a new MCP server from URL"""
    response = make_api_request("POST", "/mcp/servers/from-url", url_config)
    return response.get("success", False) if response else False


# ── Skills API helpers ─────────────────────────────────────────


def get_skills_list() -> dict[str, Any] | None:
    """Fetch the local sidecar's skills."""
    response = make_api_request("GET", "/skills")
    return response.get("data") if response else None


def get_skill_detail(name: str) -> dict[str, Any] | None:
    """Fetch full detail for one local sidecar skill."""
    response = make_api_request("GET", f"/skills/{name}")
    return response.get("data") if response else None


def get_skill_secrets(name: str) -> dict[str, Any] | None:
    """Fetch configured credential names for one local skill, never values."""
    response = make_api_request("GET", f"/skills/{name}/secrets")
    return response.get("data") if response else None


def set_skill_secret(
    name: str,
    secret_name: str,
    value: str,
) -> dict[str, Any] | None:
    """Save one local credential binding without exposing its value."""
    response = make_api_request(
        "POST",
        f"/skills/{name}/secrets",
        {"name": secret_name, "value": value},
    )
    return response.get("data") if response else None


def delete_skill_secret(name: str, secret_name: str) -> dict[str, Any] | None:
    """Remove one local credential binding for a skill."""
    response = make_api_request("DELETE", f"/skills/{name}/secrets/{secret_name}")
    return response.get("data") if response else None


def _save_skill_secret(
    skill_name: str,
    secret_name_key: str,
    value_key: str,
) -> None:
    """Persist the credential from widget state, then clear its password field."""
    secret_name = str(st.session_state.get(secret_name_key) or "").strip()
    value = str(st.session_state.get(value_key) or "")
    status_key = f"{value_key}_status"

    if not secret_name or not value:
        st.session_state[status_key] = ("error", "Enter both a credential name and value.")
        return
    if set_skill_secret(skill_name, secret_name, value) is None:
        st.session_state[status_key] = (
            "error",
            _last_api_error_message("Failed to save local credential"),
        )
        return

    st.session_state[value_key] = ""
    st.session_state[status_key] = ("success", f"{secret_name} configured")


def _delete_skill_secret(skill_name: str, secret_name: str, status_key: str) -> None:
    """Delete one named credential binding and retain only a safe status message."""
    if delete_skill_secret(skill_name, secret_name) is None:
        st.session_state[status_key] = (
            "error",
            _last_api_error_message("Failed to remove local credential"),
        )
        return
    st.session_state[status_key] = ("success", f"{secret_name} removed")


def reload_skills() -> dict[str, Any] | None:
    """Rescan the local sidecar's skill roots."""
    response = make_api_request("POST", "/skills/reload")
    return response.get("data") if response else None


def render_login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# ChatBot")
        st.markdown("### Welcome! Please sign in to continue")

        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

        with tab1, st.form("login_form", clear_on_submit=False):
            email = st.text_input(":material/mail: Email", placeholder="your@email.com")
            password = st.text_input(
                ":material/lock: Password",
                type="password",
                placeholder="Enter password",
            )

            if st.form_submit_button("Sign In", width="stretch", type="primary"):
                with st.spinner("Signing in..."):
                    auth_response = make_api_request(
                        "POST",
                        "/auth/login",
                        {"email": email, "password": password},
                    )
                    if auth_response and "data" in auth_response:
                        data = auth_response["data"]
                        session_token = data.get("localSessionToken") or data["accessToken"]
                        st.session_state.auth_token = session_token
                        st.session_state.current_user_id = data["userId"]
                        st.session_state.device_id = data.get("deviceId")
                        st.session_state.current_user_profile = None
                        st.session_state.active_view = "chat"
                        st.session_state.show_login = False
                        st.session_state._ls_op = {
                            "token": session_token,
                            "uid": data["userId"],
                        }
                        st.toast("Welcome back!", icon=":material/check_circle:")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

        with tab2, st.form("signup_form", clear_on_submit=False):
            username = st.text_input(":material/person: Username", placeholder="Choose a username")
            email = st.text_input(":material/mail: Email", placeholder="your@email.com")
            password = st.text_input(
                ":material/lock: Password",
                type="password",
                placeholder="Create password",
            )
            confirm_password = st.text_input(
                ":material/lock: Confirm",
                type="password",
                placeholder="Confirm password",
            )

            if st.form_submit_button("Create Account", width="stretch", type="primary"):
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
                                data = auth_response["data"]
                                session_token = data.get("localSessionToken") or data["accessToken"]
                                st.session_state.auth_token = session_token
                                st.session_state.current_user_id = data["userId"]
                                st.session_state.device_id = data.get("deviceId")
                                st.session_state.current_user_profile = None
                                st.session_state.active_view = "chat"
                                st.session_state.show_login = False
                                st.session_state._ls_op = {
                                    "token": session_token,
                                    "uid": data["userId"],
                                }
                                st.toast(
                                    "Account created!",
                                    icon=":material/check_circle:",
                                )
                                st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown("# Multi-agent ChatBot")

        # New chat button
        if st.button("New Chat", width="stretch", type="primary"):
            st.session_state.current_conversation_id = "pending_new"
            st.session_state.active_view = "chat"
            close_conversation_manager()
            reset_conversation_state()
            st.rerun()

        # Manage conversations button
        if st.button("Manage Conversations", width="stretch"):
            if st.session_state.get(CONVERSATION_MANAGER_DIALOG_KEY):
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
            conversations_response = get_conversations(include_messages=False, fetch_all_pages=True)
            if conversations_response and conversations_response.get("data"):
                st.session_state.conversations_list = conversations_response["data"]["items"]
                st.session_state.conversations_loaded = True
                st.session_state.conversations_last_fetch_params = {
                    "include_messages": False,
                    "fetch_all_pages": True,
                }

        # Grouped conversations
        if st.session_state.conversations_list:
            st.markdown("### :material/chat: Conversations")

            sorted_conversations = sorted(
                st.session_state.conversations_list,
                key=lambda x: parser.parse(x.get("createdAt")),
                reverse=True,
            )

            grouped = group_conversations_by_date(sorted_conversations)

            for group_name, convs in grouped.items():
                with st.expander(
                    f":material/calendar_today: {group_name} ({len(convs)})",
                    expanded=(group_name == "Today"),
                ):
                    for conv in convs:
                        is_active = conv["id"] == st.session_state.current_conversation_id
                        render_conversation_button(conv, is_active)

        st.divider()

        # User section
        if st.session_state.current_user_id:
            user = st.session_state.get("current_user_profile")
            if not isinstance(user, dict) or user.get("id") != st.session_state.current_user_id:
                user = get_user(st.session_state.current_user_id)
                if user:
                    st.session_state.current_user_profile = user
            if user:
                st.markdown(f"**{user['username']}**")
                if st.button("Sign Out", width="stretch"):
                    make_api_request("POST", "/auth/logout")
                    _clear_skill_hitl_session_state()
                    st.session_state.current_user_id = None
                    st.session_state.current_user_profile = None
                    st.session_state.current_conversation_id = None
                    st.session_state.device_id = None
                    close_conversation_manager()
                    reset_conversation_state()
                    st.session_state.conversations_list = []
                    st.session_state.conversations_loaded = False
                    st.session_state.conversations_last_fetch_params = None
                    st.session_state._ls_op = "clear"
                    if "__restore" in st.query_params:
                        del st.query_params["__restore"]
                    st.session_state.auth_token = None
                    st.session_state.show_login = True
                    st.session_state.active_view = "chat"
                    st.session_state.pending_image_attachments = []
                    st.session_state.message_image_thumbnails = {}
                    st.toast("Goodbye!", icon=":material/waving_hand:")
                    st.rerun()


def _normalize_image_for_gallery(image: Any, fallback_name: str) -> dict[str, str] | None:
    """Extract `{src, name}` from an attachment-style dict.

    Accepts dicts shaped like `{"url": ...}` (plain URL or `data:` URI) or
    `{"data": "<base64>", "mime": ...}`. Returns None if neither is usable.
    """
    if not isinstance(image, dict):
        return None

    name = image.get("name") or image.get("description") or image.get("caption") or fallback_name

    url_value = image.get("url")
    if isinstance(url_value, str) and url_value.strip():
        return {"src": url_value.strip(), "name": name}

    data_b64 = image.get("data")
    if isinstance(data_b64, str) and data_b64.strip():
        payload = data_b64.strip()
        if payload.startswith("data:"):
            return {"src": payload, "name": name}
        mime = image.get("mime", "image/png")
        return {"src": f"data:{mime};base64,{payload}", "name": name}

    return None


def _render_thumbnail_gallery(
    items: list[dict[str, str]],
    *,
    thumb_width: int,
    thumb_height: int,
    caption_max_chars: int = 40,
    align: str = "left",
    card_style: bool = False,
    natural: bool = False,
    indent_px: int = 0,
) -> None:
    """Render a flex gallery of clickable `img-thumb` thumbnails.

    Each item must be `{"src": str, "name": str}`. Clicks are handled by
    the page-level lightbox delegate installed by `render_image_lightbox()`.

    `card_style=True` produces the larger framed cards used for agent
    images; `card_style=False` produces the compact inline thumbnails used
    for user attachments. `natural=True` (cards only) shows the full image at
    its real aspect ratio, capped at `thumb_width`, instead of cropping it to a
    fixed box — used for generated images that are the message's actual content.
    `indent_px` insets the whole gallery from the alignment-side edge.
    """
    if not items:
        return

    parts: list[str] = []
    for item in items:
        src = item.get("src", "")
        if not src:
            continue
        name = item.get("name", "") or ""
        display_name = (
            name if len(name) <= caption_max_chars else name[: caption_max_chars - 3] + "..."
        )
        escaped_src = html.escape(src, quote=True)
        escaped_full_name = html.escape(name, quote=True)
        escaped_display_name = html.escape(display_name, quote=True)

        if card_style:
            cap_div = (
                f'<div style="font-size:.75em; color:#64748b; padding:5px 8px; '
                f"line-height:1.3; overflow:hidden; text-overflow:ellipsis; "
                f'white-space:nowrap; max-width:{thumb_width}px;" '
                f'title="{escaped_full_name}">{escaped_display_name}</div>'
                if name
                else ""
            )
            if natural:
                container_style = (
                    f"width:fit-content; max-width:{thumb_width}px; flex:0 1 auto; "
                    "border-radius:10px; overflow:hidden; background:#f8fafc; "
                    "border:1px solid #e2e8f0;"
                )
                img_style = (
                    f"width:auto; height:auto; max-width:min({thumb_width}px, 100%); "
                    "display:block; cursor:zoom-in; border-radius:10px 10px 0 0;"
                )
            else:
                container_style = (
                    f"width:{thumb_width}px; flex-shrink:0; text-align:center; "
                    "border-radius:8px; overflow:hidden; background:#f8fafc; "
                    "border:1px solid #e2e8f0;"
                )
                img_style = (
                    f"width:{thumb_width}px; height:{thumb_height}px; object-fit:cover; "
                    "display:block; cursor:zoom-in; border-radius:8px 8px 0 0;"
                )
            parts.append(
                f'<div style="{container_style}">'
                f'<img src="{escaped_src}" alt="{escaped_display_name}" '
                f'class="img-thumb" loading="lazy" '
                f'title="Click to view full size" style="{img_style}" '
                f"onerror=\"this.parentElement.style.display='none'\" />"
                f"{cap_div}</div>"
            )
        else:
            parts.append(
                f'<div style="display:inline-block; margin:4px; text-align:center; '
                f'vertical-align:top;" title="{escaped_full_name}">'
                f'<img src="{escaped_src}" alt="{escaped_full_name}" '
                f'class="img-thumb" loading="lazy" '
                f'style="width:{thumb_width}px; height:{thumb_height}px; '
                f"object-fit:cover; border-radius:8px; "
                f'border:1px solid #e2e8f0; cursor:pointer;" />'
                f'<div style="font-size:10px; color:#64748b; max-width:{thumb_width}px; '
                f"overflow:hidden; text-overflow:ellipsis; white-space:nowrap; "
                f'margin-top:2px;">{escaped_display_name}</div></div>'
            )

    if not parts:
        return

    wrapper_align = "flex-end" if align == "right" else "flex-start"
    gap = "8px" if card_style else "4px"
    pad = ""
    if indent_px:
        pad = f"padding-{'right' if align == 'right' else 'left'}:{indent_px}px; "
    gallery_html = (
        f'<div style="display:flex; flex-wrap:wrap; gap:{gap}; '
        f"justify-content:{wrapper_align}; align-items:flex-start; "
        f'{pad}margin:8px 0;">' + "".join(parts) + "</div>"
    )
    st.markdown(gallery_html, unsafe_allow_html=True)


def render_attachment_gallery(attachments: list[dict[str, str]], *, align: str) -> None:
    """User-side compact 72×72 inline image thumbnails."""
    if not attachments:
        return
    items: list[dict[str, str]] = []
    for idx, att in enumerate(attachments, start=1):
        normalized = _normalize_image_for_gallery(att, f"Image {idx}")
        if normalized:
            items.append(normalized)
    _render_thumbnail_gallery(
        items,
        thumb_width=72,
        thumb_height=72,
        caption_max_chars=20,
        align=align,
        card_style=False,
        indent_px=12,
    )


def render_canvas_artifact(message_metadata: dict):
    """Render a canvas_artifact (HTML/SVG/React) from agent response metadata."""
    if not message_metadata:
        return

    artifact = message_metadata.get("canvas_artifact")
    if not isinstance(artifact, dict):
        return

    content = artifact.get("content", "").strip()
    if not content:
        return

    language = artifact.get("language") or "html"
    title = artifact.get("title") or "Canvas"

    # Base64-encode the ORIGINAL (unpatched) content so the Open button can
    # recreate it as a Blob URL — avoids the data: URI browser block.
    orig_b64 = base64.b64encode(content.encode()).decode()
    esc_title = html.escape(title)
    esc_lang = html.escape(language.upper())

    # Toolbar injected INTO the iframe content.
    # Uses a Blob URL instead of data: URI so window.open() is not blocked.
    injected_toolbar = f"""<div id="__canvas_tb__" style="
        position:fixed;top:0;left:0;right:0;z-index:2147483647;
        background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
        border-bottom:1px solid #bae6fd;
        padding:5px 12px;
        display:flex;align-items:center;justify-content:space-between;
        font-family:-apple-system,BlinkMacSystemFont,sans-serif;
        font-size:13px;color:#0369a1;
        box-sizing:border-box;height:36px;
    ">
      <span style="font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{esc_title}</span>
      <span style="display:flex;align-items:center;gap:8px;flex-shrink:0">
        <span style="background:#0369a1;color:#fff;border-radius:4px;padding:0 7px;
                     font-size:10px;text-transform:uppercase;letter-spacing:.04em">{esc_lang}</span>
        <button id="__canvas_open__" style="
            border:1px solid #bae6fd;background:#fff;color:#0369a1;
            border-radius:5px;padding:2px 10px;cursor:pointer;font-size:12px;
            font-family:inherit;
        ">&#x2197; Open</button>
      </span>
    </div>
    <div style="height:36px"></div>
    <script>
    (function(){{
      var b64="{orig_b64}";
      function dec(s){{
        var bin=atob(s),arr=new Uint8Array(bin.length);
        for(var i=0;i<bin.length;i++)arr[i]=bin.charCodeAt(i);
        return new TextDecoder().decode(arr);
      }}
      document.getElementById('__canvas_open__').addEventListener('click',function(){{
        try{{
          var src=dec(b64);
          var blob=new Blob([src],{{type:'text/html'}});
          var url=URL.createObjectURL(blob);
          window.open(url,'_blank');
        }}catch(e){{alert('Could not open: '+e.message);}}
      }});
    }})();
    </script>"""

    # Patch content:
    #  1. Inject <base target="_blank"> in <head> — all links/form actions open
    #     in a new tab instead of navigating the iframe and triggering a
    #     Streamlit page reload.
    #  2. Inject the toolbar at the start of <body>.
    patched = content
    lower = patched.lower()
    base_tag = '<base target="_blank">'

    head_open = lower.find("<head>")
    if head_open != -1:
        ins = head_open + len("<head>")
        patched = patched[:ins] + "\n" + base_tag + "\n" + patched[ins:]
        lower = patched.lower()

    body_open = lower.find("<body")
    if body_open != -1:
        body_tag_end = lower.find(">", body_open)
        if body_tag_end != -1:
            ins = body_tag_end + 1
            patched = patched[:ins] + "\n" + injected_toolbar + "\n" + patched[ins:]
    elif head_open == -1:
        # Bare fragment — no html/head/body structure
        patched = base_tag + "\n" + injected_toolbar + "\n" + patched

    # ── Live iframe ───────────────────────────────────────────────────────────
    # Honor a per-artifact `preferred_height` when the backend supplies one
    # (e.g. small SVGs vs. full dashboards). Clamp to a sane range so a stray
    # value can't blow out the page layout.
    preferred_height = artifact.get("preferred_height")
    if isinstance(preferred_height, (int, float)) and preferred_height > 0:
        frame_height = max(200, min(1200, int(preferred_height)))
    else:
        frame_height = 520
    _stc.html(patched, height=frame_height, scrolling=True)

    # ── Collapsible source code ───────────────────────────────────────────────
    with st.expander(f"Source Code ({language})", expanded=False):
        st.code(content, language=language)


def _widget_component_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _build_live_widget_component_html(widget: dict[str, Any], auth_token: str | None) -> str:
    widget_id = str(widget.get("widget_id") or "")
    config = {
        "widget": {
            "widget_id": widget_id,
            "widget_type": str(widget.get("widget_type") or "html"),
            "title": widget.get("title"),
            "status": str(widget.get("status") or "active"),
            "version": widget.get("version", 1),
            "connection_endpoint": str(
                widget.get("connection_endpoint") or f"/widgets/{widget_id}/connection"
            ),
        },
        "apiBaseUrl": API_BASE_URL.rstrip("/"),
        "widgetWsBaseUrl": WIDGET_WS_BASE_URL,
        "authToken": auth_token or "",
    }

    template = """
    <style>
      :root{color-scheme:light}
      *{box-sizing:border-box}
      html,body{margin:0;height:100%;overflow:hidden}
      body{padding:8px;background:radial-gradient(circle at top left,rgba(14,165,233,.12),transparent 34%),radial-gradient(circle at bottom right,rgba(37,99,235,.1),transparent 28%),linear-gradient(180deg,#f8fbff 0%,#eef4ff 100%);font-family:"Segoe UI","Inter",-apple-system,BlinkMacSystemFont,sans-serif;color:#0f172a}
      .lw-card{height:100%;display:flex;flex-direction:column;border:1px solid rgba(148,163,184,.24);border-radius:22px;background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(248,250,252,.96));box-shadow:0 24px 56px rgba(15,23,42,.12);overflow:hidden}
      .lw-head{display:flex;justify-content:space-between;gap:16px;padding:18px 20px 14px;border-bottom:1px solid #e2e8f0;background:radial-gradient(circle at top right,rgba(14,165,233,.18),transparent 40%),linear-gradient(135deg,rgba(239,246,255,.96),rgba(255,255,255,.98))}
      .lw-kicker{font-size:11px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#0284c7;margin-bottom:6px}
      .lw-title{font-size:20px;font-weight:800;color:#0f172a;line-height:1.15}
      .lw-sub{margin-top:7px;font-size:12px;color:#64748b}
      .lw-badge{display:inline-flex;align-items:center;gap:6px;padding:7px 11px;border-radius:999px;font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;white-space:nowrap;align-self:flex-start}
      .lw-badge[data-state="active"]{background:rgba(16,185,129,.14);color:#047857}
      .lw-badge[data-state="closed"]{background:rgba(148,163,184,.18);color:#475569}
      .lw-meta{display:flex;flex-wrap:wrap;gap:10px;padding:14px 20px 0}
      .lw-pill{display:inline-flex;align-items:center;gap:6px;padding:7px 11px;border-radius:999px;font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8}
      .lw-pill[data-conn="connected"]{background:#ecfdf5;border-color:#a7f3d0;color:#047857}
      .lw-pill[data-conn="connecting"],.lw-pill[data-conn="reconnecting"]{background:#fff7ed;border-color:#fed7aa;color:#c2410c}
      .lw-pill[data-conn="error"],.lw-pill[data-conn="disconnected"],.lw-pill[data-conn="auth"]{background:#fef2f2;border-color:#fecaca;color:#b91c1c}
      .lw-error{margin:14px 20px 0;padding:12px 14px;border-radius:14px;background:#fff1f2;border:1px solid #fecdd3;color:#9f1239;font-size:13px}
      .lw-body{flex:1;min-height:0;overflow:auto;padding:18px 20px 22px;background:linear-gradient(180deg,rgba(255,255,255,.78),rgba(241,245,249,.86));scrollbar-gutter:stable}
      .lw-body::-webkit-scrollbar{width:10px}
      .lw-body::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:999px}
      .lw-empty{display:flex;align-items:center;justify-content:center;min-height:220px;border:1px dashed #bfdbfe;border-radius:18px;color:#475569;background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(239,246,255,.72));padding:22px;text-align:center}
      .lw-html-shell{display:grid;gap:12px}
      .lw-html-caption{font-size:12px;color:#475569;line-height:1.55}
      .lw-html-frame{width:100%;border:none;border-radius:18px;background:#fff;box-shadow:0 18px 30px rgba(15,23,42,.08)}
      @media (max-width:760px){.lw-head{flex-direction:column;align-items:stretch}}
    </style>
    <div class="lw-card">
      <div class="lw-head">
        <div>
          <div class="lw-kicker">Live Widget</div>
          <div class="lw-title" id="lw-title"></div>
          <div class="lw-sub" id="lw-sub"></div>
        </div>
        <div class="lw-badge" id="lw-status" data-state="active"></div>
      </div>
      <div class="lw-meta">
        <div class="lw-pill" id="lw-conn" data-conn="connecting"></div>
        <div class="lw-pill" id="lw-version"></div>
      </div>
      <div class="lw-error" id="lw-error" hidden></div>
      <div class="lw-body" id="lw-body"><div class="lw-empty">Connecting widget…</div></div>
    </div>
    <script>
    (() => {
      const cfg = __CFG__;
      const el = {
        title: document.getElementById("lw-title"),
        sub: document.getElementById("lw-sub"),
        status: document.getElementById("lw-status"),
        conn: document.getElementById("lw-conn"),
        version: document.getElementById("lw-version"),
        error: document.getElementById("lw-error"),
        body: document.getElementById("lw-body"),
      };
      const state = {
        data: null,
        status: String(cfg.widget.status || "active"),
        version: Number(cfg.widget.version || 0),
        conn: "connecting",
      };
      let ws = null;
      let reconnectTimer = null;
      let reconnectCount = 0;
      let destroyed = false;
      const MIN_HEIGHT = 260;
      const MAX_HEIGHT = 960;
      function isObj(v){return !!v && typeof v === "object" && !Array.isArray(v);}
      function esc(v){return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));}
      function numeric(v){const n = Number(v); return Number.isFinite(n) ? n : null;}
      function resize(){document.body.style.margin="0";}
      function showError(msg){el.error.hidden=!msg;el.error.textContent=msg || "";resize();}
      function meta(){
        const widgetType = String(cfg.widget.widget_type || "html");
        el.title.textContent = cfg.widget.title || "Live Widget";
        el.sub.textContent = `${widgetType} · ${cfg.widget.widget_id || "pending"}`;
        el.status.textContent = state.status;
        el.status.dataset.state = state.status;
        el.version.textContent = `Version ${state.version || 0}`;
        const connLabel = {connected:"Connected",connecting:"Connecting",reconnecting:"Reconnecting",disconnected:"Disconnected",error:"Connection Error",auth:"Sign In Required"}[state.conn] || "Connecting";
        el.conn.textContent = connLabel;
        el.conn.dataset.conn = state.conn;
      }
      function htmlState(data){
        if(!isObj(data)) return {html: "", caption: "", height: 620};
        const html = String(data.html || "");
        const caption = String(data.caption || "");
        const explicitHeight = numeric(data.height);
        const height = Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, explicitHeight ?? 620));
        return {html, caption, height};
      }
      function renderHtmlWidget(data){
        const view = htmlState(data);
        if(!view.html.trim()){
          return '<div class="lw-empty">No HTML content available yet.</div>';
        }
        return `<div class="lw-html-shell">${view.caption ? `<div class="lw-html-caption">${esc(view.caption)}</div>` : ""}<iframe class="lw-html-frame" sandbox="allow-scripts allow-forms allow-modals allow-downloads" referrerpolicy="no-referrer" style="height:${view.height}px" srcdoc="${esc(view.html)}"></iframe></div>`;
      }
      function renderBody(){
        meta();
        if(!state.data){el.body.innerHTML='<div class="lw-empty">Connecting widget…</div>';resize();return;}
        el.body.innerHTML = renderHtmlWidget(state.data);
        resize();
      }
      function wsBase(){
        const source = cfg.widgetWsBaseUrl || cfg.apiBaseUrl;
        const url = new URL(source);
        url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
        url.pathname = "/";
        url.search = "";
        url.hash = "";
        return url.toString();
      }
      function wsUrl(pathOrUrl){
        if(!pathOrUrl) return "";
        if(pathOrUrl.startsWith("ws://") || pathOrUrl.startsWith("wss://")) return pathOrUrl;
        if(pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")){const url=new URL(pathOrUrl);url.protocol=url.protocol === "https:" ? "wss:" : "ws:";return url.toString();}
        return new URL(pathOrUrl, wsBase()).toString();
      }
      function scheduleReconnect(){
        if(destroyed || state.status === "closed" || reconnectTimer) return;
        state.conn = reconnectCount === 0 ? "reconnecting" : "disconnected";
        meta();
        const delay = Math.min(1000 * Math.pow(2, reconnectCount), 15000);
        reconnectCount += 1;
        reconnectTimer = window.setTimeout(() => {reconnectTimer = null; connect();}, delay);
      }
      function openSocket(url){
        if(!url){showError("Widget connection response did not include a WebSocket URL.");scheduleReconnect();return;}
        if(ws){try{ws.close();}catch{}}
        ws = new WebSocket(url);
        ws.addEventListener("open", () => {reconnectCount = 0; state.conn = "connected"; meta(); showError("");});
        ws.addEventListener("message", (event) => {
          let msg = null;
          try{msg = JSON.parse(event.data);}catch{return;}
          if(!isObj(msg)) return;
          if(msg.type === "ping"){if(ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type:"pong"})); return;}
          if(msg.type === "error"){showError(msg.message || "Widget connection error."); return;}
          if(msg.type === "widget_state_sync" || msg.type === "widget_update" || msg.type === "widget_close"){
            if(isObj(msg.state)) state.data = msg.state;
            if(msg.version !== undefined) state.version = Number(msg.version || 0);
            if(msg.status) state.status = String(msg.status);
            if(msg.type === "widget_close") state.status = "closed";
            renderBody();
          }
        });
        ws.addEventListener("close", () => {if(destroyed) return; if(state.status === "closed"){state.conn = "disconnected"; meta(); return;} state.conn = "disconnected"; meta(); scheduleReconnect();});
        ws.addEventListener("error", () => {state.conn = "error"; meta();});
      }
      async function connect(){
        if(!cfg.widget.widget_id){state.conn = "error"; meta(); showError("Widget metadata is missing widget_id."); renderBody(); return;}
        if(!cfg.authToken){state.conn = "auth"; meta(); showError("Sign in again to connect this widget."); renderBody(); return;}
        state.conn = "connecting";
        meta();
        showError("");
        renderBody();
        try{
          const response = await fetch(new URL(cfg.widget.connection_endpoint, `${cfg.apiBaseUrl}/`).toString(), {method:"POST", headers:{Authorization:`Bearer ${cfg.authToken}`}});
          const statusCode = response.status;
          const raw = await response.text();
          let payload = {};
          if(raw){try{payload = JSON.parse(raw);}catch{throw new Error(`Unexpected widget connection response: ${raw.slice(0,160)}`);}}
          if(isObj(payload) && payload.success === false) throw new Error(payload.message || "Widget connection request failed.");
          if(!response.ok){
            const err = new Error((isObj(payload) && (payload.error || payload.message)) || `Widget connection request failed (${response.status})`);
            err.permanent = statusCode >= 400 && statusCode < 500 && statusCode !== 408 && statusCode !== 429;
            throw err;
          }
          const data = isObj(payload.data) ? payload.data : payload;
          openSocket(wsUrl(String(data.ws_url || "")));
        }catch(error){
          state.conn = "error";
          meta();
          showError(error instanceof Error ? error.message : "Unable to connect widget.");
          if(!(error && error.permanent)) scheduleReconnect();
        }
      }
      window.addEventListener("beforeunload", () => {
        destroyed = true;
        window.clearTimeout(reconnectTimer);
        if(ws){try{ws.close();}catch{}}
      });
      meta();
      renderBody();
      connect();
    })();
    </script>
    """
    return template.replace("__CFG__", _widget_component_json(config))


def _live_widget_frame_height(widget: dict[str, Any]) -> int:
    """Outer Streamlit component height for an HTML live widget.

    The inner iframe is sized from ``state.html``/``state.height`` (260..960),
    which arrives over the WebSocket after mount, so the wrapper uses a fixed
    height with internal scroll rather than trying to size to unknown state.
    """
    del widget  # height is uniform for HTML widgets; arg kept for call sites
    return 820


def render_live_widgets(
    message_metadata: dict,
    *,
    message_key: str = "",
    auto_mount: bool = False,
    inline: bool = False,
):
    """Render live widgets from assistant message metadata.

    Historical widgets stay lightweight until opened, which prevents
    conversation-load reruns from mounting every widget iframe at once.
    Inline items omit attachment chrome so they sit cleanly between markdown
    segments in a rich response.
    """
    if not message_metadata:
        return

    widgets = message_metadata.get("live_widgets") or []
    if not widgets:
        return

    auth_token = str(st.session_state.get("auth_token") or "")
    mount_state = st.session_state.live_widget_mounts
    for index, widget in enumerate(widgets):
        if not isinstance(widget, dict):
            continue

        widget_id = str(widget.get("widget_id") or "")
        if not widget_id:
            continue

        widget_type = str(widget.get("widget_type") or "widget")
        title = str(widget.get("title") or widget_type or f"Widget {index + 1}")
        version = widget.get("version", 1)
        mount_key = f"{message_key}:{widget_id}" if message_key else widget_id
        if auto_mount and mount_key not in mount_state:
            mount_state[mount_key] = True
        is_mounted = bool(mount_state.get(mount_key))

        if not inline:
            status = str(widget.get("status") or "active")
            meta_suffix = f"{widget_type} · v{version} · {widget_id[:16]}…"
            if status != "active":
                meta_suffix = f"{meta_suffix} · {status}"
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #dbeafe;
                    border-left: 4px solid #2563eb;
                    border-radius: 14px;
                    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
                    padding: 12px 14px 12px 16px;
                    margin: 8px 0 10px;
                ">
                  <div style="font-size:11px; font-weight:700; color:#2563eb; text-transform:uppercase; letter-spacing:.06em;">
                    Live Widget
                  </div>
                  <div style="margin-top:4px; font-weight:700; color:#0f172a; font-size:16px;">
                    {html.escape(title)}
                  </div>
                  <div style="margin-top:6px; color:#64748b; font-size:12px;">
                    {html.escape(meta_suffix)}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if inline and is_mounted:
            _stc.html(
                _build_live_widget_component_html(widget, auth_token),
                height=_live_widget_frame_height(widget),
                scrolling=False,
            )
            continue

        controls = st.columns([1, 5])
        with controls[0]:
            if not is_mounted:
                if st.button("Open", key=f"open_live_widget_{mount_key}", width="stretch"):
                    mount_state[mount_key] = True
                    st.rerun()
            else:
                if st.button("Hide", key=f"hide_live_widget_{mount_key}", width="stretch"):
                    mount_state[mount_key] = False
                    st.rerun()
        with controls[1]:
            if not is_mounted:
                st.caption(
                    "Widget is not connected yet. Open it on demand to avoid loading every widget in the conversation history."
                )

        if is_mounted:
            _stc.html(
                _build_live_widget_component_html(widget, auth_token),
                height=_live_widget_frame_height(widget),
                scrolling=False,
            )

        if not inline:
            with st.expander(f"Widget details: {title}", expanded=False):
                st.code(
                    json.dumps(
                        {
                            "widget_id": widget_id,
                            "widget_type": widget.get("widget_type"),
                            "status": widget.get("status"),
                            "version": widget.get("version"),
                            "connection_endpoint": widget.get(
                                "connection_endpoint", f"/widgets/{widget_id}/connection"
                            ),
                        },
                        indent=2,
                    ),
                    language="json",
                )


def render_agent_images(message_metadata: dict):
    """Agent-side image rendering.

    Model-generated images (base64, the message's actual content) render at full
    size and natural aspect ratio; remote result thumbnails (e.g. web search)
    keep the compact 150×120 framed-card grid. Both are inset from the edge.
    """
    if not message_metadata:
        return
    images = message_metadata.get("images") or []
    if not images:
        return
    generated: list[dict[str, str]] = []
    thumbs: list[dict[str, str]] = []
    for idx, image in enumerate(images, start=1):
        normalized = _normalize_image_for_gallery(image, f"Image {idx}")
        if not normalized:
            continue
        # Base64-inlined images are model-generated content; remote URLs are
        # search/result thumbnails that read better as a compact grid.
        is_generated = isinstance(image, dict) and bool(image.get("data")) and not image.get("url")
        (generated if is_generated else thumbs).append(normalized)

    if generated:
        _render_thumbnail_gallery(
            generated,
            thumb_width=380,
            thumb_height=0,
            caption_max_chars=60,
            align="left",
            card_style=True,
            natural=True,
            indent_px=12,
        )
    if thumbs:
        _render_thumbnail_gallery(
            thumbs,
            thumb_width=150,
            thumb_height=120,
            caption_max_chars=40,
            align="left",
            card_style=True,
            indent_px=12,
        )


def get_message_metadata(msg: dict[str, Any]) -> dict[str, Any]:
    """Read metadata regardless of snake_case/camelCase payload shape."""
    if not isinstance(msg, dict):
        return {}

    for key in ("messageMetadata", "message_metadata", "metadata"):
        value = msg.get(key)
        if isinstance(value, dict):
            return value

    return {}


def extract_interrupt_message(interrupt_payload: Any) -> str | None:
    """Extract a displayable message directly from an interrupt payload."""
    if not isinstance(interrupt_payload, dict):
        return None

    for key in ("message", "reason"):
        value = interrupt_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = interrupt_payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("message", "reason"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _ensure_stream_trace_state() -> None:
    st.session_state.setdefault("stream_trace_items", [])
    st.session_state.setdefault("stream_tool_index", {})
    st.session_state.setdefault("stream_trace_expanded", False)
    st.session_state.setdefault("stream_subagent_activity", None)


def _reset_stream_trace_state(expanded: bool = True) -> None:
    _ensure_stream_trace_state()
    st.session_state.stream_trace_items = []
    st.session_state.stream_tool_index = {}
    st.session_state.stream_trace_expanded = expanded
    st.session_state.stream_subagent_activity = None


def _rebuild_stream_tool_index(trace_items: list[dict[str, Any]]) -> dict[str, int]:
    tool_index: dict[str, int] = {}
    for index, item in enumerate(trace_items):
        if item.get("kind") != "tool":
            continue
        tool_call_id = item.get("tool_call_id")
        if tool_call_id:
            tool_index[str(tool_call_id)] = index
    return tool_index


def _payload_contains_data_url(payload: Any) -> bool:
    pending = [payload]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            pending.extend(current.values())
            continue
        if isinstance(current, list):
            pending.extend(current)
            continue
        if isinstance(current, (bytes, bytearray)):
            try:
                candidate = current[:128].decode("utf-8", errors="replace").strip()
            except Exception:
                candidate = ""
            if candidate.startswith("data:") and ";base64," in candidate:
                return True
            continue
        if isinstance(current, str):
            stripped = current.strip()
            if stripped.startswith("data:") and ";base64," in stripped:
                return True
    return False


def _truncate_text_preview(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return f"{text[:max_chars].rstrip()}...", True


def _build_json_preview(payload: Any, max_chars: int) -> tuple[str, bool]:
    encoder = json.JSONEncoder(indent=2, ensure_ascii=False, default=str)
    pieces: list[str] = []
    collected = 0

    for fragment in encoder.iterencode(payload):
        remaining = max_chars - collected
        if remaining <= 0:
            return "".join(pieces).rstrip() + "...", True
        if len(fragment) > remaining:
            pieces.append(fragment[:remaining])
            return "".join(pieces).rstrip() + "...", True
        pieces.append(fragment)
        collected += len(fragment)

    return "".join(pieces), False


def _build_trace_payload_display(
    payload: Any,
    max_chars: int = TRACE_PREVIEW_CHAR_LIMIT,
) -> dict[str, Any]:
    if payload is None:
        return {
            "preview": "",
            "language": "text",
            "truncated": False,
            "omitted": False,
        }

    if _payload_contains_data_url(payload):
        return {
            "preview": "Binary/data payload omitted from preview.",
            "language": "text",
            "truncated": False,
            "omitted": True,
        }

    if isinstance(payload, (dict, list)):
        preview, truncated = _build_json_preview(payload, max_chars)
        language = "json"
    elif isinstance(payload, (bytes, bytearray)):
        preview, truncated = _truncate_text_preview(
            payload[: max_chars + 1].decode("utf-8", errors="replace"),
            max_chars,
        )
        language = "text"
    else:
        preview, truncated = _truncate_text_preview(str(payload), max_chars)
        language = "text"

    return {
        "preview": preview,
        "language": language,
        "truncated": truncated,
        "omitted": False,
    }


def _render_trace_preview_block(label: str, payload: Any, full_label: str) -> None:
    display = _build_trace_payload_display(payload)
    preview = display.get("preview")
    if not isinstance(preview, str) or not preview:
        return

    st.markdown(
        f'<div class="trace-preview-label">{html.escape(label)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="trace-preview"><pre>{html.escape(preview)}</pre></div>',
        unsafe_allow_html=True,
    )

    if display.get("omitted"):
        st.caption("Full payload omitted from the UI because it appears to be a binary/data URL.")
        return

    if display.get("truncated"):
        with st.expander(full_label, expanded=False):
            if isinstance(payload, (dict, list)):
                render_tool_result_payload(payload, use_expander=False)
            else:
                full_text = (
                    payload.decode("utf-8", errors="replace")
                    if isinstance(payload, (bytes, bytearray))
                    else str(payload)
                )
                st.code(full_text, language=str(display["language"]))


def _trace_status_meta(state: str | None) -> tuple[str, str, str]:
    normalized = str(state or "unknown").strip().lower()
    if normalized == "queued":
        return "Queued", "hourglass_top", "running"
    if normalized == "running":
        return "Running", "hourglass_top", "running"
    if normalized == "error":
        return "Error", "error", "error"
    if normalized == "rejected":
        return "Rejected", "block", "rejected"
    if normalized == "completed":
        return "Completed", "check_circle", "completed"
    return "Unknown", "help", "unknown"


def _render_trace_text_block(
    content: str | None,
    *,
    label: str | None = None,
    live: bool = False,
) -> None:
    if not isinstance(content, str) or not content.strip():
        return

    if live:
        rendered_content = re.sub(
            r"\*\*(.*?)\*\*",
            r"<strong>\1</strong>",
            html.escape(content),
        )
        body_html = f'<div class="thinking-content">{rendered_content}</div>'
    elif label:
        header_html = f'<div class="trace-text-label">{html.escape(label)}</div>'
        body_html = (
            f'<div class="thinking-content-rendered">{sanitize_message_content(content)}</div>'
        )
        st.markdown(
            f'<div class="thinking-container">{header_html}{body_html}</div>',
            unsafe_allow_html=True,
        )
        return
    else:
        body_html = (
            f'<div class="thinking-content-rendered">{sanitize_message_content(content)}</div>'
        )

    st.markdown(
        f'<div class="thinking-container">{body_html}</div>',
        unsafe_allow_html=True,
    )


def _trace_panel_title(
    *,
    has_thinking: bool,
    has_reasoning: bool,
    has_tools: bool,
) -> str:
    if has_tools:
        return "Execution Trace"
    if has_thinking and has_reasoning:
        return "Thought Process"
    if has_reasoning:
        return "Reasoning Summary"
    return "Thinking"


def _should_show_thinking_section_title(
    *,
    has_tools: bool,
    has_thinking: bool,
    has_reasoning: bool,
) -> bool:
    return has_tools and (has_thinking or has_reasoning)


def _trace_text_label(
    *,
    section: str,
    has_tools: bool,
    has_thinking: bool,
    has_reasoning: bool,
) -> str | None:
    if section == "reasoning":
        return "Reasoning Summary" if has_tools or has_thinking else None
    if section == "thinking":
        return "Thinking Summary" if has_tools and has_reasoning else None
    return None


def _build_tool_trace_items_from_artifacts(
    tool_artifacts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not isinstance(tool_artifacts, list):
        return []

    trace_items: list[dict[str, Any]] = []
    for index, artifact in enumerate(tool_artifacts, start=1):
        if not isinstance(artifact, dict):
            continue

        tool_name = artifact.get("tool", "unknown_tool")
        render = artifact.get("render")
        render_type = render.get("type") if isinstance(render, dict) else None
        if tool_name == "dispatch_subagents" or render_type == "subagent_dispatch":
            continue

        raw_status = str(artifact.get("status") or "").strip().lower()
        if raw_status == "rejected":
            state = "rejected"
        elif raw_status in {"error", "failed"} or artifact.get("error") not in (None, ""):
            state = "error"
        elif raw_status in {"success", "completed"}:
            state = "completed"
        elif raw_status == "running":
            state = "running"
        else:
            state = "unknown"

        execution_time = artifact.get("execution_time")
        duration_ms = (
            int(float(execution_time) * 1000)
            if isinstance(execution_time, (int, float)) and execution_time >= 0
            else None
        )

        trace_items.append(
            {
                "kind": "tool",
                "tool_call_id": artifact.get("tool_call_id") or f"artifact_{index}",
                "name": tool_name,
                "phase": "end" if state != "running" else "start",
                "state": state,
                "args": artifact.get("args"),
                "result": artifact.get("output"),
                "render": render,
                "error": artifact.get("error"),
                "hint": artifact.get("hint"),
                "duration_ms": duration_ms,
            }
        )

    return trace_items


def _render_trace_tool_card(tool_item: dict[str, Any], index: int) -> None:
    tool_name = str(tool_item.get("name") or "unknown_tool")
    badge_label, icon_name, badge_class = _trace_status_meta(tool_item.get("state"))

    st.markdown(
        f"""
        <div class="trace-tool-card">
            <div class="trace-tool-header">
                <div class="trace-tool-title">[{index}] {html.escape(tool_name)}</div>
                <span class="trace-status-pill trace-status-{badge_class}">
                    <span class="material-symbols-outlined" aria-hidden="true">{icon_name}</span>
                    {html.escape(badge_label)}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    duration_ms = tool_item.get("duration_ms")
    if isinstance(duration_ms, (int, float)) and duration_ms >= 0:
        st.caption(f"Execution time: {float(duration_ms) / 1000:.2f}s")

    args = tool_item.get("args")
    if args not in (None, {}, []):
        _render_trace_preview_block(
            "Input Preview",
            args,
            f"Full input for {tool_name}",
        )

    result = tool_item.get("result")
    render = tool_item.get("render")
    render_error_shown = False
    if isinstance(render, dict):
        st.markdown(
            '<div class="trace-preview-label">Result Preview</div>',
            unsafe_allow_html=True,
        )
        if render_tool_render_payload(render, fallback_output=result):
            render_error_shown = str(render.get("type") or "").lower() == "error"
        else:
            _render_trace_preview_block(
                "Result Preview",
                result,
                f"Full output for {tool_name}",
            )
    elif result not in (None, ""):
        _render_trace_preview_block(
            "Result Preview",
            result,
            f"Full output for {tool_name}",
        )
    elif str(tool_item.get("state") or "").lower() == "running":
        st.markdown(
            '<div class="trace-note">Waiting for tool output...</div>',
            unsafe_allow_html=True,
        )

    error_message = tool_item.get("error")
    if not render_error_shown and isinstance(error_message, str) and error_message.strip():
        st.error(error_message.strip())

    hint = tool_item.get("hint")
    if isinstance(hint, str) and hint.strip():
        st.info(hint.strip())


def render_trace_panel(
    *,
    thinking_content: str | None = None,
    reasoning_summary: str | None = None,
    tool_items: list[dict[str, Any]] | None = None,
    expanded: bool = False,
    live: bool = False,
) -> None:
    normalized_tools = [item for item in tool_items or [] if isinstance(item, dict)]
    has_thinking = isinstance(thinking_content, str) and bool(thinking_content.strip())
    has_reasoning = isinstance(reasoning_summary, str) and bool(reasoning_summary.strip())
    has_tools = bool(normalized_tools)
    if not any(
        [
            has_thinking,
            has_reasoning,
            has_tools,
        ]
    ):
        return

    with st.expander(
        _trace_panel_title(
            has_thinking=has_thinking,
            has_reasoning=has_reasoning,
            has_tools=has_tools,
        ),
        expanded=expanded,
    ):
        if has_reasoning or has_thinking:
            if _should_show_thinking_section_title(
                has_tools=has_tools,
                has_thinking=has_thinking,
                has_reasoning=has_reasoning,
            ):
                st.markdown(
                    '<div class="trace-section-title">Thinking</div>',
                    unsafe_allow_html=True,
                )

            if has_reasoning:
                _render_trace_text_block(
                    reasoning_summary,
                    label=_trace_text_label(
                        section="reasoning",
                        has_tools=has_tools,
                        has_thinking=has_thinking,
                        has_reasoning=has_reasoning,
                    ),
                )
            if has_thinking:
                _render_trace_text_block(
                    thinking_content,
                    label=None
                    if live
                    else _trace_text_label(
                        section="thinking",
                        has_tools=has_tools,
                        has_thinking=has_thinking,
                        has_reasoning=has_reasoning,
                    ),
                    live=live,
                )

        if normalized_tools:
            st.markdown(
                '<div class="trace-section-title">Tool Activity</div>',
                unsafe_allow_html=True,
            )
            for index, tool_item in enumerate(normalized_tools, start=1):
                _render_trace_tool_card(tool_item, index)


def render_message_trace(message_metadata: dict[str, Any], expanded: bool = False) -> None:
    if not isinstance(message_metadata, dict):
        return

    render_trace_panel(
        thinking_content=message_metadata.get("thinking_summary"),
        reasoning_summary=message_metadata.get("reasoning_summary"),
        tool_items=_build_tool_trace_items_from_artifacts(message_metadata.get("tool_artifacts")),
        expanded=expanded,
        live=False,
    )


def _format_subagent_duration(value: Any) -> str:
    if not isinstance(value, (int, float)) or value < 0:
        return ""
    return f"{float(value) / 1000:.2f}s"


def _format_subagent_model_badge(item: dict[str, Any]) -> str | None:
    """Build a compact HTML badge for the model that answered (or will answer)
    a single worker. Prefers ``resolved_model`` (post-run truth); falls back to
    ``requested_model`` (set when the supervisor explicitly assigned a model).
    """
    resolved = item.get("resolved_model") if isinstance(item, dict) else None
    requested = item.get("requested_model") if isinstance(item, dict) else None
    source = resolved if isinstance(resolved, dict) else requested
    if not isinstance(source, dict):
        return None

    provider = str(source.get("provider") or "").strip()
    model_name = str(source.get("model") or "").strip()
    if not provider and not model_name:
        return None

    label_parts: list[str] = []
    if model_name:
        label_parts.append(html.escape(model_name))
    if provider and provider != model_name:
        label_parts.append(f"&middot; {html.escape(provider)}")

    effort = str(source.get("reasoning_effort") or "").strip()
    if effort:
        label_parts.append(
            f'<span class="subagent-worker-model-effort">{html.escape(effort)}</span>'
        )

    # Highlight when the supervisor explicitly assigned a model (override path)
    # vs. inherited the default. ``requested_model`` is only set when the
    # supervisor passed ``model_override`` for that task.
    extra_class = " subagent-worker-model-override" if isinstance(requested, dict) else ""

    title_parts = [f"Model: {model_name or provider}"]
    if isinstance(requested, dict):
        title_parts.append("(explicit override)")
    if effort:
        title_parts.append(f"reasoning: {effort}")
    title = html.escape(" ".join(title_parts))

    return (
        f'<div class="subagent-worker-model{extra_class}" title="{title}">'
        f'<span class="material-symbols-outlined" aria-hidden="true" style="font-size:0.85rem;">'
        f"memory</span>{' '.join(label_parts)}</div>"
    )


def _render_subagent_activity_view(view: dict[str, Any] | None, *, live: bool = False) -> None:
    if not view:
        return

    total = int(view.get("total") or 0)
    completed = int(view.get("completed") or 0)
    blocked = int(view.get("failed") or 0)
    running = int(view.get("running") or 0)
    status = str(view.get("status") or "unknown").strip().lower()
    badge_label, icon_name, badge_class = _trace_status_meta(
        "completed" if status == "completed" else "error" if blocked else "unknown"
    )
    if status == "running":
        badge_label = "Running"
        icon_name = "hourglass_top"
        badge_class = "running"
    elif status == "partial":
        badge_label = "Partial"
        icon_name = "pending"
        badge_class = "running"
    elif status == "failed":
        badge_label = "Blocked"
        icon_name = "error"
        badge_class = "error"
    elif status == "completed":
        badge_label = "Completed"
        icon_name = "check_circle"
        badge_class = "completed"

    meta_parts = [f"{completed}/{total} workers completed"]
    if running:
        meta_parts.append(f"{running} running")
    if blocked:
        meta_parts.append(f"{blocked} need attention")
    shell_class = (
        "subagent-activity-shell subagent-activity-live" if live else "subagent-activity-shell"
    )

    st.markdown(
        f"""
        <div class="{shell_class}">
            <div class="subagent-activity-head">
                <div class="subagent-activity-title">
                    <span class="material-symbols-outlined" aria-hidden="true">hub</span>
                    Subagent Activity
                </div>
                <span class="trace-status-pill trace-status-{badge_class}">
                    <span class="material-symbols-outlined" aria-hidden="true">{icon_name}</span>
                    {html.escape(badge_label)}
                </span>
            </div>
            <div class="subagent-activity-meta">
                {html.escape(" | ".join(meta_parts))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rationales = [value for value in view.get("rationales", []) if isinstance(value, str)]
    for rationale in rationales[:2]:
        st.caption(rationale)

    with st.expander("Worker details", expanded=(live and status == "running") or blocked > 0):
        for index, item in enumerate(view.get("results") or [], start=1):
            if not isinstance(item, dict):
                continue

            worker_status = str(item.get("status") or "unknown").strip().lower()
            worker_badge, worker_icon, worker_class = _trace_status_meta(
                "completed"
                if worker_status == "completed"
                else "error"
                if worker_status in {"failed", "timeout"}
                else "running"
                if worker_status in {"running", "queued", "requires_approval"}
                else "unknown"
            )
            if worker_status == "timeout":
                worker_badge = "Timeout"
            elif worker_status == "requires_approval":
                worker_badge = "Needs Approval"
                worker_icon = "pending_actions"
            elif worker_status == "running":
                worker_badge = "Running"
                worker_icon = "hourglass_top"
            elif worker_status == "queued":
                worker_badge = "Queued"
                worker_icon = "pending"

            worker_id = html.escape(str(item.get("id") or f"worker-{index}"))
            agent_name = html.escape(
                str(
                    item.get("agent_name")
                    or get_agent_display_name(str(item.get("agent") or "unknown_agent"))
                )
            )
            duration = _format_subagent_duration(item.get("elapsed_ms"))
            duration_text = f" | {html.escape(duration)}" if duration else ""
            summary = str(item.get("summary") or "").strip()
            model_badge_html = _format_subagent_model_badge(item) or ""

            summary_html = sanitize_message_content(summary) if summary else "No summary returned."
            thinking = str(item.get("thinking") or "").strip()
            thinking_html = (
                f'<div class="subagent-worker-thinking">{sanitize_message_content(thinking)}</div>'
                if thinking
                else ""
            )
            # Built without newlines/indentation: st.markdown treats indented
            # lines after a blank line (e.g. an empty model badge) as a
            # CommonMark code block and shows the raw HTML.
            st.markdown(
                '<div class="subagent-worker-row">'
                '<div class="subagent-worker-top">'
                '<div class="subagent-worker-name">'
                f"{worker_id} &middot; {agent_name}{duration_text}</div>"
                f'<span class="trace-status-pill trace-status-{worker_class}">'
                f'<span class="material-symbols-outlined" aria-hidden="true">{worker_icon}</span>'
                f"{html.escape(worker_badge)}</span></div>"
                f"{model_badge_html}"
                f"{thinking_html}"
                f'<div class="subagent-worker-summary">{summary_html}</div>'
                "</div>",
                unsafe_allow_html=True,
            )

            related = item.get("related_todo_ids")
            if isinstance(related, list) and related:
                st.caption("Related todos: " + ", ".join(str(value) for value in related))

            error = item.get("error")
            if isinstance(error, str) and error.strip():
                st.error(error.strip())

            artifacts = item.get("artifacts")
            if isinstance(artifacts, list) and artifacts:
                with st.expander(f"Worker artifacts for {worker_id}", expanded=False):
                    render_tool_result_payload(artifacts, use_expander=False)


def render_subagent_activity(message_metadata: dict[str, Any]) -> None:
    _render_subagent_activity_view(build_subagent_activity_view(message_metadata), live=False)


def _render_rag_chunk_card(view: RAGArtifactView, chunk: RAGChunkView) -> None:
    header_bits: list[str] = [f"**[{chunk.rank}] {chunk.source}**"]
    if chunk.score is not None:
        header_bits.append(f"score `{chunk.score:.2%}`")
    if chunk.page_label:
        header_bits.append(chunk.page_label)
    st.markdown(" — ".join(header_bits))

    meta_parts: list[str] = []
    if chunk.document_id:
        meta_parts.append(f"document: `{chunk.document_id}`")
    if chunk.chunk_id:
        meta_parts.append(f"chunk: `{chunk.chunk_id}`")
    if chunk.image_count:
        meta_parts.append(f"images: `{chunk.image_count}`")
    if chunk.table_count is not None:
        meta_parts.append(f"tables: `{chunk.table_count}`")
    elif chunk.has_tables:
        meta_parts.append("tables: `present`")
    if meta_parts:
        st.caption(" | ".join(meta_parts))

    if chunk.image_captions:
        captions = ", ".join(chunk.image_captions[:3])
        more = "…" if len(chunk.image_captions) > 3 else ""
        st.caption(f"image captions: {captions}{more}")

    if chunk.content:
        st.markdown(f"> {chunk.content.strip()}")


def _render_rag_document_listing(listing: RAGDocumentListing) -> None:
    parts: list[str] = [f"**{listing.rank}. {listing.filename or 'unknown'}**"]
    if listing.chunk_count is not None:
        parts.append(f"{listing.chunk_count} chunks")
    if listing.document_id:
        parts.append(f"`{listing.document_id}`")
    st.markdown(" — ".join(parts))


def render_rag_retrieval_artifacts(message_metadata: dict[str, Any]) -> None:
    views = extract_rag_artifact_views(message_metadata)
    if not views:
        return

    label = "Retrieved Evidence"
    total_chunks = sum(len(view.chunks) for view in views)
    total_documents = sum(len(view.documents) for view in views)
    if total_chunks:
        summary_count = total_chunks
        summary_label = "chunk"
    elif total_documents:
        summary_count = total_documents
        summary_label = "document"
    else:
        summary_count = len(views)
        summary_label = "result"
    summary_suffix = "" if summary_count == 1 else "s"

    with st.expander(f"{label} ({summary_count} {summary_label}{summary_suffix})", expanded=False):
        for index, view in enumerate(views, start=1):
            st.markdown(f"**{view.title}**")
            detail_parts: list[str] = []
            if view.query:
                detail_parts.append(f"query: `{view.query}`")
            if view.document_id:
                detail_parts.append(f"document: `{view.document_id}`")
            if view.status:
                detail_parts.append(f"status: `{view.status}`")
            if view.blob_size_bytes:
                detail_parts.append(f"size: `{view.blob_size_bytes} bytes`")
            if detail_parts:
                st.caption(" | ".join(detail_parts))

            if view.chunks:
                for chunk in view.chunks:
                    _render_rag_chunk_card(view, chunk)
            elif view.documents:
                for listing in view.documents:
                    _render_rag_document_listing(listing)
            elif view.preview:
                st.code(view.preview, language="text")

            if (
                view.output
                and view.output != view.preview
                and not view.chunks
                and not view.documents
            ):
                with st.expander(f"Full output for {view.title}", expanded=False):
                    st.code(view.output, language="text")
            elif view.chunks or view.documents:
                with st.expander(f"Raw tool output for {view.title}", expanded=False):
                    st.code(view.output or view.preview, language="text")

            if view.blob_id:
                blob_state_key = f"rag_blob_text:{view.blob_id}"
                cached_full = st.session_state.get(blob_state_key)
                if cached_full:
                    with st.expander(f"Full result for {view.title}", expanded=True):
                        st.code(cached_full, language="text")
                elif st.button(
                    "Load full result",
                    key=f"rag_blob_load:{view.tool_call_id}:{view.blob_id}",
                ):
                    headers = {}
                    if st.session_state.get("auth_token"):
                        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"
                    try:
                        response = get_http_session().get(
                            f"{API_BASE_URL}/tool-results/{view.blob_id}",
                            headers=headers,
                            timeout=REQUEST_TIMEOUT,
                        )
                        response.raise_for_status()
                        st.session_state[blob_state_key] = response.text
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to load full result: {exc}")

            if index < len(views):
                st.markdown("---")


def _upsert_stream_thinking_trace(content: str) -> None:
    _ensure_stream_trace_state()
    trace_items = list(st.session_state.get("stream_trace_items") or [])
    thinking_index = next(
        (index for index, item in enumerate(trace_items) if item.get("kind") == "thinking"),
        None,
    )

    thinking_item = {"kind": "thinking", "content": content}
    if thinking_index is None:
        trace_items.insert(0, thinking_item)
    else:
        trace_items[thinking_index] = thinking_item

    st.session_state.stream_trace_items = trace_items
    st.session_state.stream_tool_index = _rebuild_stream_tool_index(trace_items)


def _upsert_stream_subagent_activity(stream_event: dict[str, Any]) -> bool:
    previous = st.session_state.get("stream_subagent_activity")
    view = build_live_subagent_activity_view(stream_event, previous=previous)
    if view is previous:
        return False

    st.session_state.stream_subagent_activity = view
    return bool(view)


def _has_live_trace_panel_content() -> bool:
    return bool(
        st.session_state.get("stream_trace_items")
        or st.session_state.get("stream_subagent_activity")
    )


def _format_stream_tool_status_label(tool_event: dict[str, Any]) -> str:
    tool_name = str(tool_event.get("name") or "unknown")
    phase = str(tool_event.get("phase") or tool_event.get("status") or "running")
    if tool_name == "dispatch_subagents":
        return "Subagents: dispatching..." if phase == "start" else "Subagents: results received"
    return f"Tool: {tool_name} ({phase})"


def _resolve_stream_tool_trace_id(tool_event: dict[str, Any]) -> str:
    explicit_id = tool_event.get("tool_call_id")
    if explicit_id:
        return str(explicit_id)

    phase = normalize_tool_phase(tool_event.get("phase") or tool_event.get("status"))
    trace_items = st.session_state.get("stream_trace_items") or []

    if phase == "end":
        for item in trace_items:
            if item.get("kind") == "tool" and str(item.get("state")).lower() in {
                "queued",
                "running",
            }:
                fallback_id = item.get("tool_call_id")
                if fallback_id:
                    return str(fallback_id)

    next_index = sum(1 for item in trace_items if item.get("kind") == "tool") + 1
    return f"tool_{next_index}"


def _upsert_stream_tool_trace(tool_event: dict[str, Any]) -> None:
    _ensure_stream_trace_state()
    if _upsert_stream_subagent_activity(tool_event):
        return

    trace_items = list(st.session_state.get("stream_trace_items") or [])
    tool_index = dict(st.session_state.get("stream_tool_index") or {})

    phase = normalize_tool_phase(tool_event.get("phase") or tool_event.get("status")) or "unknown"
    tool_trace_id = _resolve_stream_tool_trace_id(tool_event)
    item_index = tool_index.get(tool_trace_id)
    if item_index is not None and (
        item_index >= len(trace_items)
        or trace_items[item_index].get("kind") != "tool"
        or str(trace_items[item_index].get("tool_call_id")) != tool_trace_id
    ):
        item_index = None
    if item_index is None:
        item_index = next(
            (
                index
                for index, item in enumerate(trace_items)
                if item.get("kind") == "tool" and str(item.get("tool_call_id")) == tool_trace_id
            ),
            None,
        )
    now_ts = datetime.now(timezone.utc).timestamp()

    if item_index is None:
        item = {
            "kind": "tool",
            "tool_call_id": tool_trace_id,
            "name": tool_event.get("name", "unknown"),
            "phase": phase,
            "state": tool_event.get("state") or infer_tool_state(phase=phase),
            "args": None,
            "result": None,
            "started_at": None,
            "ended_at": None,
            "duration_ms": None,
        }
        trace_items.append(item)
        item_index = len(trace_items) - 1
        tool_index[tool_trace_id] = item_index
    else:
        item = dict(trace_items[item_index])

    item["name"] = tool_event.get("name") or item.get("name") or "unknown"
    item["phase"] = phase

    if tool_event.get("args") is not None:
        item["args"] = tool_event.get("args")

    if tool_event.get("result") is not None:
        item["result"] = tool_event.get("result")

    # End events carry the presentation details separately from the model-facing
    # result. Keep them on the live trace item so the same renderer used for
    # persisted artifacts can show charts, tables, images, errors, and hints.
    for field in ("render", "error", "hint"):
        if tool_event.get(field) is not None:
            item[field] = tool_event.get(field)

    if phase == "start" and item.get("started_at") is None:
        item["started_at"] = now_ts

    if phase == "end":
        item["ended_at"] = now_ts
        duration_ms = tool_event.get("duration_ms")
        if isinstance(duration_ms, (int, float)) and duration_ms >= 0:
            item["duration_ms"] = int(duration_ms)
        elif isinstance(item.get("started_at"), (int, float)):
            item["duration_ms"] = int((now_ts - float(item["started_at"])) * 1000)

    item["state"] = tool_event.get("state") or infer_tool_state(
        phase=item.get("phase"),
        result=item.get("result"),
    )

    trace_items[item_index] = item
    st.session_state.stream_trace_items = trace_items
    st.session_state.stream_tool_index = _rebuild_stream_tool_index(trace_items)


def render_live_trace_panel(trace_placeholder: Any) -> None:
    _ensure_stream_trace_state()
    trace_items = [
        item
        for item in (st.session_state.get("stream_trace_items") or [])
        if isinstance(item, dict)
    ]
    thinking_item = next(
        (item for item in trace_items if item.get("kind") == "thinking"),
        None,
    )
    tool_items = [item for item in trace_items if item.get("kind") == "tool"]
    subagent_view = st.session_state.get("stream_subagent_activity")

    if thinking_item is None and not tool_items and not subagent_view:
        trace_placeholder.empty()
        return

    with trace_placeholder.container():
        if isinstance(subagent_view, dict):
            _render_subagent_activity_view(subagent_view, live=True)

        render_trace_panel(
            thinking_content=thinking_item.get("content") if thinking_item else None,
            tool_items=tool_items,
            expanded=bool(st.session_state.get("stream_trace_expanded", True)),
            live=True,
        )


def render_citations(
    message_metadata: dict[str, Any],
    msg_id: str | None = None,
    *,
    include_images: bool = True,
):
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
            with st.expander(f"Sources ({len(legacy_citations)} references)", expanded=False):
                for idx, citation in enumerate(legacy_citations, start=1):
                    source = citation.get("source", "unknown")
                    safe_source = html.escape(str(source))
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
                        f"<strong>[{idx}]</strong> {safe_source}<br>"
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
            safe_doc_num = html.escape(str(doc_num))
            safe_source = html.escape(str(source))
            safe_total_chunks = html.escape(str(total_chunks))

            # Determine overall document relevance color
            if avg_score >= 0.3:
                doc_relevance_color = COLORS["success"]
            elif avg_score >= 0.2:
                doc_relevance_color = COLORS["warning"]
            else:
                doc_relevance_color = COLORS["error"]

            # Document header — same markup whether or not it's clickable;
            # only the surrounding column layout differs.
            doc_header_html = (
                f'<div class="citation-document" style="margin-bottom: 12px; '
                f"padding: 12px; border: 2px solid {doc_relevance_color}; "
                f'border-radius: 8px; background-color: {doc_relevance_color}10;">'
                f'<strong style="font-size: 1.1em;">[Document {safe_doc_num}] {safe_source}</strong><br>'
                f'<span style="color: {doc_relevance_color}; font-size: 0.9em;">'
                f"Overall Relevance: ({avg_score:.1%})</span> | "
                f'<span style="font-size: 0.9em;">{safe_total_chunks} chunk(s)</span></div>'
            )

            if total_chunks == 1 and chunks and msg_id:
                chunk = chunks[0]
                chunk_idx = chunk.get("chunk_index", 0)
                chunk_data = chunks_map.get(doc_num, {}).get(chunk_idx, {})

                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    st.markdown(doc_header_html, unsafe_allow_html=True)
                with col2:
                    if st.button(
                        "View",
                        key=f"cite_doc_{msg_id}_{doc_num}",
                        help="View chunk content",
                        width="stretch",
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
                st.markdown(doc_header_html, unsafe_allow_html=True)

            # Show individual chunks if more than one
            if total_chunks > 1:
                st.markdown("**Chunks:**")
                for chunk in chunks:
                    chunk_idx = chunk.get("chunk_index", "?")
                    safe_chunk_idx = html.escape(str(chunk_idx))
                    chunk_score = chunk.get("score", 0.0)

                    # Chunk relevance color
                    if chunk_score >= 0.3:
                        chunk_color = COLORS["success"]
                    elif chunk_score >= 0.2:
                        chunk_color = COLORS["warning"]
                    else:
                        chunk_color = COLORS["error"]

                    chunk_row_html = (
                        f'<div class="citation-chunk" style="margin-left: 20px; '
                        f"margin-bottom: 6px; padding: 6px; "
                        f"border-left: 2px solid {chunk_color}; "
                        f'background-color: {chunk_color}08;">'
                        f'<span style="font-size: 0.9em;">Chunk {safe_chunk_idx} '
                        f'<span style="color: {chunk_color};">({chunk_score:.1%})</span>'
                        f"</span></div>"
                    )

                    if msg_id:
                        chunk_data = chunks_map.get(doc_num, {}).get(chunk_idx, {})
                        col1, col2 = st.columns([0.85, 0.15])
                        with col1:
                            st.markdown(chunk_row_html, unsafe_allow_html=True)
                        with col2:
                            if st.button(
                                "Details",
                                key=f"cite_chunk_{msg_id}_{doc_num}_{chunk_idx}",
                                help="View chunk content",
                                width="stretch",
                            ):
                                st.session_state["selected_chunk_info"] = {
                                    "source": source,
                                    "document_number": doc_num,
                                    "chunk_index": chunk_idx,
                                    "content": chunk_data.get("content", ""),
                                    "score": chunk_data.get("score", chunk_score),
                                    "page_number": chunk_data.get("page_number"),
                                    "character_count": chunk_data.get("character_count", 0),
                                }
                                st.session_state["chunk_preview_dialog_key"] = True
                                st.rerun()
                    else:
                        st.markdown(chunk_row_html, unsafe_allow_html=True)

            # Show images for this document via the shared thumbnail gallery.
            images = message_metadata.get("images", [])
            if include_images and images:
                doc_image_items: list[dict[str, str]] = []
                for img in images:
                    img_name = img.get("name", "")
                    if not ((doc_id and doc_id in img_name) or (source and source in img_name)):
                        continue
                    normalized = _normalize_image_for_gallery(img, "Image")
                    if not normalized:
                        continue
                    caption = img.get("caption") or normalized["name"]
                    page_num = img.get("page_number")
                    if page_num:
                        caption = f"{caption} (p. {page_num})"
                    normalized["name"] = caption
                    doc_image_items.append(normalized)

                if doc_image_items:
                    st.markdown("**Images:**")
                    _render_thumbnail_gallery(
                        doc_image_items,
                        thumb_width=80,
                        thumb_height=80,
                        caption_max_chars=25,
                        align="left",
                        card_style=False,
                    )


def render_suggestion_buttons(suggestions: list[str], msg_id: str):
    """
    Render follow-up question suggestions as clickable buttons.
    When clicked, the suggestion is stored in session state and used to populate the input.
    """
    if not suggestions:
        return

    # Create columns for horizontal layout
    cols = st.columns(min(len(suggestions), 3))

    for idx, suggestion in enumerate(suggestions[:3]):
        with cols[idx]:
            # Use a unique key based on message ID and suggestion index
            button_key = f"suggestion_{msg_id}_{idx}"
            if st.button(
                f"{suggestion}",
                key=button_key,
                width="stretch",
                help="Click to use this question",
            ):
                # Store in session state so the chat input can pick it up
                st.session_state.pending_suggestion = suggestion
                st.rerun()


def _format_tokens(value: Any) -> str:
    """Format an integer token count as a compact short string (e.g. '12.0k')."""
    if value is None:
        return "?"
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "?"
    if n < 0:
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        formatted = f"{n / 1_000:.1f}k"
        return formatted.replace(".0k", "k")
    return str(n)


def _get_context_window_metadata(message_metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a valid context_window dict from message metadata.

    Returns None if metadata is missing or the context_window field is the wrong shape.
    """
    if not isinstance(message_metadata, dict):
        return None
    cw = message_metadata.get("context_window")
    return cw if isinstance(cw, dict) else None


def _format_context_window_label(context_window: dict[str, Any]) -> str:
    """Build the accessible tooltip for a context-window gauge."""
    return str(_context_window_presentation(context_window).get("tooltip") or "")


def _format_context_tokens(value: Any) -> str:
    if value is None:
        return "?"
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "?"
    if number < 0:
        return "?"
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}k"
    return str(number)


def _format_usage_percentage(ratio: Any) -> str:
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        return "?"
    percentage = max(0.0, float(ratio) * 100)
    if 0 < percentage < 1:
        return f"{percentage:.1f}%"
    return f"{percentage:.0f}%"


def _context_usage_source(context_window: dict[str, Any]) -> tuple[str, str]:
    raw_source = context_window.get("usage_source")
    if not isinstance(raw_source, str) or not raw_source:
        used_source = context_window.get("used_token_source")
        raw_source = (
            "provider_reported"
            if used_source in {"provider_reported_total", "provider_reported_split"}
            else "locally_estimated"
            if used_source == "estimated_total"
            else "unavailable"
        )
    labels = {
        "provider_reported": "Provider reported",
        "mixed_reported_estimated": "Mixed reported + estimated",
        "locally_estimated": "Locally estimated",
        "unavailable": "Unavailable",
    }
    badge = labels.get(raw_source, "Unavailable")
    return badge, badge.lower()


def _valid_ratio(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _context_window_presentation(context_window: dict[str, Any]) -> dict[str, Any]:
    """Normalize context metadata into one safe, ratio-aware UI presentation."""
    if not isinstance(context_window, dict):
        return {
            "tooltip": "Context usage unavailable",
            "raw_ratio": None,
            "visual_ratio": 0.0,
            "source_badge": "Unavailable",
            "state": "unknown",
        }

    source_badge, source_label = _context_usage_source(context_window)
    input_tokens = context_window.get("input_tokens")
    output_tokens = context_window.get("output_tokens")
    total_tokens = context_window.get("total_tokens")
    used_tokens = context_window.get("used_tokens")
    limit_type = str(context_window.get("limit_type") or "unknown")
    raw_ratio = _valid_ratio(context_window.get("usage_ratio"))

    if limit_type == "shared_context" and context_window.get("context_window_tokens"):
        limit = context_window["context_window_tokens"]
        if used_tokens is None:
            used_tokens = total_tokens
        if raw_ratio is None and isinstance(used_tokens, int):
            raw_ratio = used_tokens / int(limit)
        if used_tokens is None:
            tooltip = f"{_format_context_tokens(limit)} token window · {source_label}"
        else:
            tooltip = (
                f"{_format_context_tokens(used_tokens)} / {_format_context_tokens(limit)} "
                f"({_format_usage_percentage(raw_ratio)}) · "
                f"input {_format_context_tokens(input_tokens)} · "
                f"output {_format_context_tokens(output_tokens)} · {source_label}"
            )
    elif limit_type == "separate_io":
        input_limit = context_window.get("max_input_tokens")
        output_limit = context_window.get("max_output_tokens")
        input_ratio = _valid_ratio(context_window.get("input_usage_ratio"))
        output_ratio = _valid_ratio(context_window.get("output_usage_ratio"))
        if input_ratio is None and isinstance(input_tokens, int) and input_limit:
            input_ratio = input_tokens / int(input_limit)
        if output_ratio is None and isinstance(output_tokens, int) and output_limit:
            output_ratio = output_tokens / int(output_limit)
        known_ratios = [ratio for ratio in (input_ratio, output_ratio) if ratio is not None]
        if raw_ratio is None and known_ratios:
            raw_ratio = max(known_ratios)
        tooltip = (
            f"{_format_usage_percentage(raw_ratio)} limiting · "
            f"input {_format_context_tokens(input_tokens)} / "
            f"{_format_context_tokens(input_limit)} ({_format_usage_percentage(input_ratio)}) · "
            f"output {_format_context_tokens(output_tokens)} / "
            f"{_format_context_tokens(output_limit)} ({_format_usage_percentage(output_ratio)}) · "
            f"{source_label}"
        )
    else:
        raw_ratio = None
        total_display = total_tokens if total_tokens is not None else used_tokens
        tooltip = (
            f"{_format_context_tokens(total_display)} total · "
            f"input {_format_context_tokens(input_tokens)} · "
            f"output {_format_context_tokens(output_tokens)} · "
            f"limit unknown · {source_label}"
        )

    display_state = str(context_window.get("display_state") or "").lower()
    if display_state not in {"ok", "warn", "danger", "unknown"}:
        if raw_ratio is None:
            display_state = "unknown"
        elif raw_ratio >= 0.9:
            display_state = "danger"
        elif raw_ratio >= 0.7:
            display_state = "warn"
        else:
            display_state = "ok"
    visual_ratio = min(max(raw_ratio or 0.0, 0.0), 1.0)
    return {
        "tooltip": tooltip,
        "raw_ratio": raw_ratio,
        "visual_ratio": visual_ratio,
        "source_badge": source_badge,
        "state": display_state,
    }


def _render_context_window_indicator(
    provider: str | None,
    model: str | None,
    context_window: dict[str, Any] | None,
) -> None:
    """Render the provider/model line with an inline context-window circle.

    Falls back to st.caption(...) if context_window metadata is absent.
    """
    raw_provider = str(provider).strip() if provider else ""
    raw_model = str(model).strip() if model else ""
    if raw_provider and raw_model:
        plain_label = f"{raw_provider}:{raw_model}"
    else:
        plain_label = raw_model or raw_provider

    if not plain_label:
        return

    if context_window is None:
        st.caption(plain_label)
        return

    presentation = _context_window_presentation(context_window)
    display_state = presentation["state"]
    tooltip = presentation["tooltip"]
    visual_percent = float(presentation["visual_ratio"]) * 100
    raw_ratio = presentation["raw_ratio"]
    safe_label = html.escape(plain_label)
    safe_tooltip = html.escape(str(tooltip), quote=True)
    raw_ratio_attr = "" if raw_ratio is None else str(raw_ratio)

    html_row = (
        '<div class="ctx-window-row">'
        f"<span>{safe_label}</span>"
        f'<span class="ctx-window-circle {display_state}" '
        f'style="--ctx-fill:{visual_percent:.1f}%" data-usage-ratio="{raw_ratio_attr}" '
        f'title="{safe_tooltip}" aria-label="{safe_tooltip}"></span>'
        "</div>"
    )
    st.markdown(html_row, unsafe_allow_html=True)


def render_message_bubble(
    msg: dict[str, Any],
    is_user: bool,
    *,
    auto_mount_live_widgets: bool = False,
):
    content_text = msg.get("content", "")
    timestamp = format_time(msg.get("createdAt", ""))

    # Use Streamlit's native chat_message which supports LaTeX
    avatar = "user" if is_user else "assistant"

    with st.chat_message(avatar):
        message_metadata = get_message_metadata(msg)

        # Show thinking summary first for assistant messages
        if not is_user:
            agent_label = get_message_agent_label(message_metadata)
            if agent_label:
                st.caption(f"Worked by {agent_label}")

            render_message_trace(message_metadata, expanded=False)

            provider = message_metadata.get("provider")
            model = message_metadata.get("model")
            if provider or model:
                context_window = _get_context_window_metadata(message_metadata)
                _render_context_window_indicator(provider, model, context_window)

        # ── Inline rich-response rendering ──────────────────────────────
        # For v1 messages, render markdown and rich items in their authored
        # order rather than emitting all of the body first and rich items
        # below. Legacy messages keep their existing behavior via the view
        # model's single-markdown-segment shape and the post-body helpers.
        view = _build_rich_response_view_for_msg(content_text, message_metadata)
        if view.is_v1:
            _render_rich_segments(
                view.segments,
                message_metadata=message_metadata,
                message_key=str(msg.get("id", "")),
                auto_mount=auto_mount_live_widgets,
            )
        else:
            st.markdown(content_text)  # Native markdown with LaTeX support

        if not is_user:
            render_subagent_activity(message_metadata)
            render_rag_retrieval_artifacts(message_metadata)

        if not is_user:
            fallback = message_metadata.get("provider_fallback")
            if isinstance(fallback, dict):
                from_provider = fallback.get("from")
                to_provider = fallback.get("to")
                reason = fallback.get("reason")
                label = (
                    f"Fallback: {from_provider} -> {to_provider}"
                    if from_provider and to_provider
                    else "Fallback to default provider"
                )
                if isinstance(reason, str) and reason.strip():
                    label = f"{label} ({reason.strip()[:140]})"
                st.caption(label)

        # Keep attachments and generated artifacts in the same chat row as the
        # message they belong to, then show the timestamp after all content.
        if is_user:
            attachments = st.session_state.get("message_image_thumbnails", {}).get(
                str(msg.get("id", ""))
            )
            if attachments:
                render_attachment_gallery(attachments, align="right")
        else:
            if not view.is_v1 and view.use_legacy_image_gallery:
                render_agent_images(message_metadata)
            if view.is_v1:
                _render_append_items(
                    view.append_items,
                    message_metadata=message_metadata,
                    message_key=str(msg.get("id", "")),
                    auto_mount=auto_mount_live_widgets,
                )
            else:
                render_canvas_artifact(message_metadata)
                render_live_widgets(
                    message_metadata,
                    message_key=str(msg.get("id", "")),
                    auto_mount=auto_mount_live_widgets,
                )
            render_citations(
                message_metadata,
                str(msg.get("id", "")),
                include_images=not view.is_v1,
            )
            render_message_feedback_inline(msg)

        st.caption(timestamp)


def _build_rich_response_view_for_msg(content_text: str, message_metadata: dict):
    """Wrap `build_rich_response_view` import to keep top-of-module imports
    minimal and avoid coupling demo.py boot to app modules in tests."""
    from app.ui.rich_response import build_rich_response_view

    return build_rich_response_view(content_text, message_metadata)


def _render_rich_segments(
    segments: list[Any],
    *,
    message_metadata: dict[str, Any],
    message_key: str,
    auto_mount: bool,
) -> None:
    """Render ordered rich-response segments, mounting each widget at most once."""
    mounted_widget_ids: set[str] = set()
    for index, segment in enumerate(segments):
        if segment.kind == "markdown" and segment.text:
            st.markdown(segment.text)
            continue
        if segment.kind == "unavailable":
            st.caption(f":material/error_outline: rich item `{segment.item_id}` is unavailable.")
            continue
        if segment.kind != "rich" or not segment.item:
            continue

        item = segment.item
        item_id = str(item.get("id") or "")
        should_auto_mount = auto_mount
        if item.get("type") == "live_widget":
            should_auto_mount = auto_mount and item_id not in mounted_widget_ids
            mounted_widget_ids.add(item_id)
        _render_inline_rich_item(
            item,
            message_metadata=message_metadata,
            message_key=f"{message_key}:segment:{index}",
            auto_mount=should_auto_mount,
        )


class _StreamingRichResponseRenderer:
    """Incrementally render an active assistant body as ordered rich segments.

    The layout is rebuilt only when marker/item structure changes. Plain text
    deltas update existing markdown slots so an already mounted widget is not
    recreated for every trailing token.
    """

    def __init__(self, placeholder: Any, *, message_key: str) -> None:
        from app.ui.rich_response import RichStreamState

        self.placeholder = placeholder
        self.message_key = message_key
        self.state = RichStreamState(latest=True)
        self._layout_signature: tuple[Any, ...] | None = None
        self._markdown_slots: dict[int, Any] = {}
        self._finalized = False

    def append_text(self, delta: str) -> None:
        self.state.append_text(delta)
        self.render()

    def apply_rich_items_upsert(self, items: list[dict[str, Any]]) -> None:
        self.state.apply_rich_items_upsert(items)
        self._layout_signature = None
        self.render()

    def finalize(self, message: dict[str, Any] | None) -> None:
        if not isinstance(message, dict):
            return
        self._finalized = True
        final_content = message.get("content")
        if isinstance(final_content, str) and final_content != self.state.accumulated_text:
            self.state.accumulated_text = final_content
        metadata = get_message_metadata(message)
        if metadata.get("rich_items_version") == 1:
            self.state.replace_with_finalized(metadata.get("rich_items") or [])
            self._layout_signature = None
        self.render()

    @staticmethod
    def _signature(view: Any) -> tuple[Any, ...]:
        segment_signature = []
        for segment in view.segments:
            if segment.kind == "markdown":
                segment_signature.append(("markdown",))
            elif segment.kind == "rich":
                segment_signature.append(("rich", (segment.item or {}).get("id")))
            else:
                segment_signature.append(("unavailable", segment.item_id))
        append_signature = tuple((item.get("type"), item.get("id")) for item in view.append_items)
        return tuple(segment_signature), append_signature

    def render(self) -> None:
        view = self.state.build_view()
        signature = self._signature(view)
        if signature != self._layout_signature:
            self._rebuild(view)
            self._layout_signature = signature
            return
        for index, segment in enumerate(view.segments):
            if segment.kind == "markdown" and index in self._markdown_slots:
                self._markdown_slots[index].markdown(
                    normalize_stream_markdown_text(segment.text or "")
                )

    def _rebuild(self, view: Any) -> None:
        metadata = {
            "rich_items_version": 1,
            "rich_items": list(self.state.items_by_id.values()),
        }
        self._markdown_slots = {}
        mounted_widget_ids: set[str] = set()
        with self.placeholder.container():
            for index, segment in enumerate(view.segments):
                if segment.kind == "markdown":
                    slot = st.empty()
                    self._markdown_slots[index] = slot
                    slot.markdown(normalize_stream_markdown_text(segment.text or ""))
                    continue
                if segment.kind == "unavailable":
                    if self._finalized:
                        st.caption(
                            f":material/error_outline: rich item `{segment.item_id}` "
                            "is unavailable."
                        )
                    else:
                        _render_pending_rich_placeholder(segment.item_id)
                    continue
                if segment.kind != "rich" or not segment.item:
                    continue
                item = segment.item
                item_id = str(item.get("id") or "")
                should_auto_mount = True
                if item.get("type") == "live_widget":
                    should_auto_mount = item_id not in mounted_widget_ids
                    mounted_widget_ids.add(item_id)
                _render_inline_rich_item(
                    item,
                    message_metadata=metadata,
                    message_key=f"{self.message_key}:segment:{index}",
                    auto_mount=should_auto_mount,
                )

            if self._finalized:
                for index, item in enumerate(view.append_items):
                    _render_inline_rich_item(
                        item,
                        message_metadata=metadata,
                        message_key=f"{self.message_key}:append:{index}",
                        auto_mount=True,
                    )


class _StreamingImagePreviewPanel:
    """Render early-delivery `image_preview` events during an active stream.

    Keeps the latest payload per image index; a partial preview is replaced by
    the next partial/final for the same index. `clear()` runs at `complete` —
    the finalized message owns the authoritative image rendering.
    """

    def __init__(self, placeholder: Any) -> None:
        self.placeholder = placeholder
        self._by_index: dict[int, dict[str, Any]] = {}

    def apply(self, event: dict[str, Any]) -> None:
        data_b64 = event.get("data_b64")
        image_index = event.get("image_index")
        if not data_b64 or not isinstance(image_index, int):
            return
        self._by_index[image_index] = event
        self._render()

    def _render(self) -> None:
        with self.placeholder.container():
            for index in sorted(self._by_index):
                event = self._by_index[index]
                try:
                    raw = base64.b64decode(event.get("data_b64") or "")
                except Exception:
                    continue
                caption = (
                    "Generating image... (preview)"
                    if event.get("status") == "partial"
                    else "Generated image"
                )
                st.image(raw, caption=caption)

    def clear(self) -> None:
        if self._by_index:
            self._by_index = {}
            self.placeholder.empty()


def _render_pending_rich_placeholder(item_id: str | None) -> None:
    """Reserve the authored inline position until a stream upsert resolves it."""
    label = (
        "Preparing interactive widget..."
        if str(item_id or "").startswith("widget:")
        else "Preparing rich content..."
    )
    st.markdown(
        (
            '<div aria-busy="true" style="border:1px dashed #cbd5e1;'
            "border-radius:10px;padding:12px 14px;margin:8px 0;"
            'background:#f8fafc;color:#64748b;font-size:13px;">'
            f"{html.escape(label)}</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_inline_rich_item(
    item: dict[str, Any],
    *,
    message_metadata: dict[str, Any],
    message_key: str,
    auto_mount: bool,
) -> None:
    """Render a single rich-item record at its inline marker position."""
    from app.core.rich_response import GENERIC_IMAGE_ALT_TEXT
    from app.ui.rich_response import build_inline_image_html

    item_type = item.get("type")
    payload = item.get("payload") or {}
    if item_type == "image":
        url = payload.get("url")
        data = payload.get("data")
        mime = payload.get("mime_type") or "image/png"
        alt_text = item.get("alt_text")
        if alt_text == GENERIC_IMAGE_ALT_TEXT:
            alt_text = None
        caption = item.get("title") or alt_text
        src = url if url else (f"data:{mime};base64,{data}" if data else None)
        if src:
            # Render at a capped article width without upscaling (st.image
            # width="stretch" blew small images up to full width and blurred
            # them). The img-thumb class opens the full-resolution source in the
            # page-level lightbox on click.
            st.markdown(
                build_inline_image_html(src, caption=caption),
                unsafe_allow_html=True,
            )
    elif item_type == "live_widget":
        # Reuse existing widget renderer with a single-item metadata shape so
        # it mounts just this widget, not all live widgets at once. The
        # registry stores the human-readable title on the rich-item record
        # (not the payload), so merge it back into the per-widget dict the
        # legacy renderer expects.
        widget_for_render = dict(payload)
        title = item.get("title")
        if isinstance(title, str) and title and not widget_for_render.get("title"):
            widget_for_render["title"] = title
        single = dict(message_metadata)
        single["live_widgets"] = [widget_for_render]
        render_live_widgets(
            single,
            message_key=f"{message_key}::{item.get('id')}",
            auto_mount=auto_mount,
            inline=True,
        )
    elif item_type == "tool_render":
        # Tool renders reuse the existing result renderer; unsupported payloads
        # still remain inspectable as JSON.
        render = payload.get("render") or {}
        try:
            render_tool_render_payload(render)
        except Exception:
            st.json(render)
    elif item_type == "canvas_artifact":
        single = dict(message_metadata)
        single["canvas_artifact"] = payload
        render_canvas_artifact(single)
    elif item_type == "citation":
        source = payload.get("source") or "source"
        page = payload.get("page_number")
        label = f"{source}" + (f" (page {page})" if page else "")
        st.caption(f":material/menu_book: {label}")
    elif item_type == "resource_link":
        title = payload.get("title") or payload.get("url")
        url = payload.get("url")
        if url:
            st.markdown(f"[{title}]({url})")


def _render_append_items(
    items: list[dict[str, Any]],
    *,
    message_metadata: dict[str, Any],
    message_key: str,
    auto_mount: bool,
) -> None:
    """Render unreferenced inline_or_append items below the message body."""
    if not items:
        return
    for item in items:
        _render_inline_rich_item(
            item,
            message_metadata=message_metadata,
            message_key=message_key,
            auto_mount=auto_mount,
        )


def render_message_feedback_inline(msg: dict[str, Any]):
    """Inline feedback for assistant messages using popover"""
    feedback = msg.get("feedback")

    if isinstance(feedback, dict) and feedback:
        # Show existing feedback
        col1, col2 = st.columns([4, 1])
        with col1:
            rating = feedback.get("rating", 0)
            stars = " ".join([":material/star:"] * int(rating)) if rating else ""
            st.caption(f"{stars} {rating}/5".strip())
            comment = feedback.get("comment")
            if comment:
                st.caption(f":material/comment: {comment[:60]}...")
        with col2, st.popover(":material/edit:", help="Edit feedback"):
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

                if st.form_submit_button("Update", width="stretch", type="primary"):
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
                        st.toast("Feedback updated!", icon=":material/check_circle:")
                        st.rerun()
    else:
        # Show add feedback popover
        with st.popover(":material/comment: Feedback", help="Give feedback"):
            st.markdown("**Provide Feedback**")
            with st.form(f"add_feedback_{msg['id']}", clear_on_submit=True):
                rating = st.select_slider("Rating", options=[1, 2, 3, 4, 5], value=5)
                comment = st.text_area("Comment (optional)", height=68)

                if st.form_submit_button("Submit", width="stretch", type="primary"):
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
                        st.toast("Feedback submitted!", icon=":material/check_circle:")
                        st.rerun()


def render_tool_parameter_form(
    args_schema: dict[str, Any], key_prefix: str = ""
) -> tuple[dict[str, Any], list[str]]:
    """Render dynamic form fields based on a tool's JSON Schema."""
    parameters: dict[str, Any] = {}
    parsing_errors: list[str] = []

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
        has_default = "default" in param_info
        schema_default = param_info.get("default")

        label = f"{param_name}{'*' if is_required else ''}"
        help_text = param_desc if param_desc else None
        base_key = f"{key_prefix}_{param_name}" if key_prefix else param_name

        if param_type == "boolean":
            if is_required or has_default:
                default_val = bool(schema_default) if has_default else False
                parameters[param_name] = st.checkbox(
                    label,
                    value=default_val,
                    help=help_text,
                    key=f"{base_key}_bool",
                )
            else:
                value = st.selectbox(
                    label,
                    options=[True, False],
                    index=None,
                    placeholder="Not provided",
                    help=help_text,
                    key=f"{base_key}_bool",
                )
                if value is not None:
                    parameters[param_name] = value
        elif param_type == "integer":
            default_val = int(schema_default or 0) if is_required or has_default else None
            value = st.number_input(
                label,
                value=default_val,
                step=1,
                format="%d",
                help=help_text,
                key=f"{base_key}_int",
            )
            if value is not None:
                parameters[param_name] = int(value)
        elif param_type == "number":
            default_val = float(schema_default or 0.0) if is_required or has_default else None
            value = st.number_input(
                label,
                value=default_val,
                step=0.1,
                format="%.6f",
                help=help_text,
                key=f"{base_key}_number",
            )
            if value is not None:
                parameters[param_name] = value
        elif param_type == "string":
            if "enum" in param_info:
                enum_values = param_info["enum"]
                enum_index = None
                if has_default and schema_default in enum_values:
                    enum_index = enum_values.index(schema_default)
                elif is_required and enum_values:
                    enum_index = 0
                value = st.selectbox(
                    label,
                    options=enum_values,
                    index=enum_index,
                    placeholder="Not provided" if enum_index is None else None,
                    help=help_text,
                    key=f"{base_key}_enum",
                )
                if value is not None:
                    parameters[param_name] = value
            else:
                default_val = _stringify_default(schema_default) if has_default else ""
                value = st.text_input(
                    label,
                    value=default_val,
                    help=help_text,
                    key=f"{base_key}_text",
                )
                if is_required or has_default or value != "":
                    parameters[param_name] = value
        elif param_type in {"object", "array"}:
            default_val = (
                schema_default
                if has_default
                else ({} if param_type == "object" else [])
                if is_required
                else None
            )
            default_text = _stringify_default(default_val)
            placeholder = (
                "Enter JSON object value" if param_type == "object" else "Enter JSON array value"
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
                if is_required or has_default:
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
                if is_required or has_default:
                    parameters[param_name] = schema_default if has_default else None

    return parameters, parsing_errors


def render_tools_tab():
    """Render the MCP Tools management and testing interface"""
    st.markdown("# :material/extension: MCP Tools Management")
    st.markdown("Discover and test Model Context Protocol (MCP) tools available to the chatbot.")

    if st.button("Refresh", icon=":material/refresh:", width="stretch"):
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
        existing_server_names = {
            str(server.get("name") or "").strip()
            for server in servers
            if str(server.get("name") or "").strip()
        }

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

            if st.button("Add Server from JSON", width="stretch"):
                if json_config.strip():
                    try:
                        config = json.loads(json_config)

                        servers_to_add = []

                        # Check if this is a full config file format
                        if "mcpServers" in config or "mcp_servers" in config:
                            # Extract servers from the wrapper
                            servers_dict = config.get("mcpServers") or config.get("mcp_servers")
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
                                "Invalid format.",
                                icon=":material/error:",
                            )
                            servers_to_add = []

                        # Add all servers
                        if servers_to_add:
                            success_count = 0
                            failed_servers = []

                            for server_config in servers_to_add:
                                server_name = str(server_config.get("name") or "unknown").strip()
                                if server_name in existing_server_names:
                                    message = f"Server '{server_name}' already exists"
                                    failed_servers.append(f"{server_name}: {message}")
                                    st.error(message, icon=":material/cancel:")
                                    continue

                                with st.spinner(f"Adding server '{server_name}'..."):
                                    # Make API request
                                    response = make_api_request(
                                        "POST", "/mcp/servers", server_config
                                    )

                                    if response and response.get("success"):
                                        success_count += 1
                                        st.success(
                                            f"Server '{server_name}' added successfully!",
                                            icon=":material/check_circle:",
                                        )
                                    else:
                                        error_msg = _last_api_error_message("Failed to add server")
                                        failed_servers.append(f"{server_name}: {error_msg}")
                                        st.error(
                                            f"Failed to add '{server_name}': {error_msg}",
                                            icon=":material/cancel:",
                                        )

                            # Show summary
                            if success_count > 0:
                                st.info(
                                    f"Successfully added {success_count} server(s). Refreshing...",
                                    icon=":material/refresh:",
                                )
                                # Small delay to ensure file is written
                                time.sleep(0.5)
                                st.rerun()

                            if failed_servers:
                                st.warning(f"Failed to add {len(failed_servers)} server(s)")
                                for failure in failed_servers:
                                    st.text(f"  - {failure}")
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a JSON configuration")

        with tab2, st.form("add_server_form"):
            st.markdown("Fill in the server details:")

            server_name_input = st.text_input("Server Name*", placeholder="my-server")
            transport_input = st.selectbox(
                "Transport Type*",
                options=["stdio", "http", "sse", "streamable_http"],
                index=0,
            )

            if transport_input == "stdio":
                command_input = st.text_input("Command*", value="python", placeholder="python")
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
                url_input = st.text_input("URL*", placeholder="http://localhost:8080")
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

            if st.form_submit_button("Add Server", width="stretch"):
                server_name_value = server_name_input.strip()
                if not server_name_value:
                    st.error("Server name is required")
                elif server_name_value in existing_server_names:
                    st.error(f"Server '{server_name_value}' already exists")
                else:
                    try:
                        config = {
                            "name": server_name_value,
                            "transport": transport_input,
                            "enabled": enabled_input,
                        }

                        if description_input:
                            config["description"] = description_input

                        if transport_input == "stdio":
                            parsed_args = [
                                arg.strip() for arg in args_input.split(",") if arg.strip()
                            ]
                            if not command_input.strip() or not parsed_args:
                                st.error("Command and arguments are required for stdio transport")
                            else:
                                config["command"] = command_input.strip()
                                config["args"] = parsed_args

                                if env_input.strip():
                                    try:
                                        config["env"] = json.loads(env_input)
                                    except json.JSONDecodeError:
                                        st.error("Invalid JSON in environment variables")
                                        config = {}

                                if config:
                                    with st.spinner("Adding server..."):
                                        result = add_mcp_server(config)
                                        if result:
                                            st.success(
                                                f"Server '{server_name_value}' added!",
                                                icon=":material/check_circle:",
                                            )
                                            st.rerun()
                                        else:
                                            st.error(
                                                _last_api_error_message("Failed to add server")
                                            )
                        else:
                            if not url_input.strip():
                                st.error("URL is required for HTTP transport")
                            else:
                                config["url"] = url_input.strip()

                                if headers_input.strip():
                                    try:
                                        config["headers"] = json.loads(headers_input)
                                    except json.JSONDecodeError:
                                        st.error("Invalid JSON in headers")
                                        config = {}

                                if config:
                                    with st.spinner("Adding server..."):
                                        result = add_mcp_server(config)
                                        if result:
                                            st.success(
                                                f"Server '{server_name_value}' added!",
                                                icon=":material/check_circle:",
                                            )
                                            st.rerun()
                                        else:
                                            st.error(
                                                _last_api_error_message("Failed to add server")
                                            )
                    except Exception as e:
                        st.error(f"Error: {e}")

        with tab3:
            st.markdown("Add a server using a URL")

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

            if st.button("Add Server from URL", width="stretch"):
                if not url_input.strip():
                    st.error("URL is required")
                elif server_name_url.strip() and server_name_url.strip() in existing_server_names:
                    st.error(f"Server '{server_name_url.strip()}' already exists")
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
                                st.success(
                                    "Server added successfully from URL!",
                                    icon=":material/check_circle:",
                                )
                                # Small delay to ensure file is written
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(_last_api_error_message("Failed to add server from URL"))
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("#### Configured Servers")

        if not servers:
            st.info("No MCP servers configured.")
        else:
            hitl_settings = get_hitl_settings() or {}
            hitl_master = bool(hitl_settings.get("masterEnabled", True))
            hitl_servers = {
                item["scopeValue"]: item["requireApproval"]
                for item in hitl_settings.get("servers", [])
                if item.get("toolOrigin") == "client_mcp"
            }
            if not hitl_master:
                st.caption(
                    ":material/info: Human-in-the-loop is globally disabled (admin setting); "
                    "approval rules below are inactive until it is enabled."
                )
            for server in servers:
                server_name = server.get("name", "Unknown")
                enabled = server.get("enabled", False)
                tool_count = server.get("toolCount", 0)
                transport = server.get("transport", "unknown")
                description = server.get("description", "No description")

                status_icon = ":material/check_circle:" if enabled else ":material/cancel:"
                status_text = "Enabled" if enabled else "Disabled"

                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.markdown(
                        f"""
                    **{status_icon} {server_name}** - {status_text}
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

                with col4:
                    server_gated = bool(hitl_servers.get(server_name, False))
                    approval_label = "Approval: ON" if server_gated else "Approval: OFF"
                    if st.button(
                        approval_label,
                        key=f"hitl_server_{server_name}",
                        help="Require human approval for all tools from this server",
                    ):
                        with st.spinner("Updating approval rule..."):
                            if (
                                set_hitl_setting(
                                    "client_mcp", "server", server_name, not server_gated
                                )
                                is not None
                            ):
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

    # Search/filter
    search_query = st.text_input("Search tools", placeholder="Filter by name or description...")

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

    # Display tools as selectbox, keyed by qualified id so duplicate tool names
    # across servers each select their own rule (server::tool).
    qualified_tool_options = {
        str(tool.get("qualifiedId") or f"{tool.get('serverName', '')}::{tool.get('name')}"): tool
        for tool in filtered_tools
        if tool.get("name")
    }
    selected_tool_key = st.selectbox(
        "Select a tool to test",
        options=list(qualified_tool_options.keys()),
        format_func=lambda key: (
            f"{qualified_tool_options[key].get('name')} "
            f"({qualified_tool_options[key].get('serverName', '')})"
        ),
    )

    if not selected_tool_key:
        return

    # Get selected tool details
    selected_tool = qualified_tool_options.get(selected_tool_key)

    if not selected_tool:
        return
    selected_tool_name = selected_tool.get("name")

    # Display tool details
    st.markdown("---")
    st.markdown(f"## {selected_tool.get('name')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Server:** `{selected_tool.get('serverName', 'Unknown')}`")
    with col2:
        st.markdown("**Type:** Tool")

    st.markdown(f"**Description:** {selected_tool.get('description', 'No description available')}")

    # Per-tool human-approval mode (tri-state: inherit server / always / never).
    st.markdown("**Human approval**")
    qualified_id = selected_tool_key
    hitl_settings = get_hitl_settings() or {}
    tool_rules = {
        item["scopeValue"]: item["requireApproval"]
        for item in hitl_settings.get("tools", [])
        if item.get("toolOrigin") == "client_mcp"
    }

    if qualified_id in tool_rules:
        current_mode = "Require" if tool_rules[qualified_id] else "Skip"
    else:
        current_mode = "Inherit"

    modes = ["Inherit", "Require", "Skip"]
    chosen = st.radio(
        "Approval mode for this tool",
        modes,
        index=modes.index(current_mode),
        key=f"hitl_tool_mode_{qualified_id}",
        horizontal=True,
        help="Inherit = follow the server rule; Require = always prompt; Skip = never prompt",
    )
    if chosen != current_mode:
        with st.spinner("Updating tool approval..."):
            if chosen == "Inherit":
                result = clear_hitl_setting("client_mcp", "tool", qualified_id)
            else:
                result = set_hitl_setting("client_mcp", "tool", qualified_id, chosen == "Require")
            if result is not None:
                st.rerun()

    # Tool parameter form
    st.markdown("---")
    args_schema = selected_tool.get("argsSchema", {})

    with st.form(key=f"tool_execute_form_{qualified_id}"):
        st.markdown("### Execute Tool")

        # Render parameter inputs
        parameters, parameter_errors = render_tool_parameter_form(
            args_schema, key_prefix=qualified_id
        )

        # Submit button
        execute_button = st.form_submit_button("Execute Tool", width="stretch")

        if execute_button:
            if parameter_errors:
                for error_msg in parameter_errors:
                    st.error(error_msg)
            else:
                with st.spinner(f"Executing {selected_tool_name}..."):
                    result = execute_mcp_tool(
                        selected_tool_name,
                        parameters,
                        server_name=selected_tool.get("serverName"),
                        qualified_tool_id=qualified_id,
                    )

                    if result:
                        _remember_mcp_tool_execution_result(qualified_id, result)
                    else:
                        st.error("Tool execution failed. Check API logs.")

    # Display execution result
    result = _mcp_tool_execution_results().get(qualified_id)
    if result:
        st.markdown("---")
        st.markdown("### Execution Result")

        success = result.get("success", False)
        execution_time = result.get("executionTime", 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            status_label = (
                ":material/check_circle: Success" if success else ":material/cancel: Failed"
            )
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
        if st.button("Clear Result", key=f"clear_tool_result_{qualified_id}"):
            _clear_mcp_tool_execution_result(qualified_id)
            st.rerun()


def render_skills_tab():
    """Render the installed repo skills interface."""
    st.markdown("# :material/psychology: Installed Repo Skills")
    st.markdown(
        "Inspect the server-owned skills checked into this repository. This tab reads the "
        "local `skills/` directory directly and does not call a public server `/skills` API."
    )
    st.caption(
        "Runtime skill selection uses installed repo skills plus device-scoped client skills. "
        "Server repo skills are installed from disk, not managed over HTTP."
    )

    col_refresh, col_reload = st.columns(2)
    with col_refresh:
        if st.button(
            "Refresh",
            icon=":material/refresh:",
            key="skills_refresh",
            width="stretch",
        ):
            st.rerun()
    with col_reload:
        if st.button(
            "Reload from Disk",
            icon=":material/sync:",
            key="skills_reload",
            width="stretch",
        ):
            with st.spinner("Rescanning repo skills folder..."):
                result = reload_skills()
                if result:
                    msg = result.get("message", "Skills reloaded")
                    st.success(msg, icon=":material/check_circle:")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Failed to reload skills. Is the API running?")

    st.markdown("---")

    # Fetch skills
    skills_data = get_skills_list()

    if not skills_data:
        st.error("Failed to load skills. Make sure the API is running.")
        return

    skills = skills_data.get("skills", [])
    total_count = skills_data.get("totalCount", 0)
    enabled_count = skills_data.get("enabledCount", 0)

    # Summary metrics
    col_total, col_enabled, col_disabled = st.columns(3)
    with col_total:
        st.metric("Total Skills", total_count)
    with col_enabled:
        st.metric("Enabled", enabled_count)
    with col_disabled:
        st.metric("Disabled", total_count - enabled_count)

    st.markdown("---")

    if not skills:
        st.info(
            "No skills found. Drop a folder with a `SKILL.md` file into the `skills/` "
            "directory and click **Reload from Disk**.",
            icon=":material/lightbulb:",
        )
        return

    hitl_settings = get_hitl_settings()
    skill_tool_rules: dict[str, bool] = {}
    if hitl_settings is None:
        st.warning(_hitl_settings_unavailable_message())
    else:
        hitl_master = bool(hitl_settings.get("masterEnabled", True))
        skill_tool_rules = {
            item["scopeValue"]: item["requireApproval"]
            for item in hitl_settings.get("tools", [])
            if item.get("toolOrigin") == "client_skill"
        }
        if not hitl_master:
            st.caption(
                ":material/info: Human-in-the-loop is globally disabled (admin setting); "
                "approval rules below are inactive until it is enabled."
            )

    # Render each skill as a card
    skill_hitl_scope = str(st.session_state.get("current_user_id") or "anonymous")
    for skill in skills:
        skill_name = skill.get("name", "Unknown")
        description = skill.get("description", "No description")
        enabled = skill.get("enabled", False)
        folder_path = skill.get("folderPath", "")

        status_icon = ":material/check_circle:" if enabled else ":material/cancel:"
        status_text = "Enabled" if enabled else "Disabled"

        with st.expander(
            f"**{status_icon} {skill_name}** — {status_text}",
            expanded=False,
        ):
            st.markdown(f"**Description:** {description}")
            st.caption(f"Folder: `{folder_path}`")

            runtime_status = str(skill.get("runtimeStatus") or "not_ready")
            setup_status = str(skill.get("setupStatus") or "setup_required")
            command_capable = skill.get("commandCapable", False)
            if runtime_status == "ready":
                st.caption("Command runtime: Ready")
            elif runtime_status == "instruction_only":
                st.caption(
                    "Command runtime: This skill provides instructions only; "
                    "it has no command runtime."
                )
            else:
                st.warning(f"Command runtime is not ready ({setup_status})")

            skill_secret_scope = f"{skill_hitl_scope}_{skill_name}"
            secret_name_key = f"skill_secret_name_{skill_secret_scope}"
            value_key = f"skill_secret_value_{skill_secret_scope}"
            secret_status_key = f"{value_key}_status"
            with st.expander("Local credentials", expanded=False):
                st.caption(
                    "Use the environment-variable name documented by this skill. "
                    "Values stay encrypted on this device."
                )
                st.text_input("Credential name", key=secret_name_key)
                st.text_input("Secret value", type="password", key=value_key)
                st.button(
                    "Save credential",
                    key=f"skill_secret_save_{skill_secret_scope}",
                    on_click=_save_skill_secret,
                    args=(skill_name, secret_name_key, value_key),
                )

                secret_status = st.session_state.pop(secret_status_key, None)
                if secret_status:
                    level, message = secret_status
                    getattr(st, level)(message)

                configured = get_skill_secrets(skill_name) or {"secrets": []}
                for item in configured.get("secrets", []):
                    secret_name = str(item.get("name") or "")
                    if not secret_name:
                        continue
                    st.caption(f"{secret_name} configured")
                    st.button(
                        "Remove",
                        key=f"skill_secret_delete_{skill_secret_scope}_{secret_name}",
                        on_click=_delete_skill_secret,
                        args=(skill_name, secret_name, secret_status_key),
                    )

            if runtime_status == "ready" and hitl_settings is not None and command_capable:
                skill_qualified_id = f"skill::{skill_name}::run_skill_command"
                if skill_qualified_id in skill_tool_rules:
                    current_mode = "Require" if skill_tool_rules[skill_qualified_id] else "Skip"
                else:
                    current_mode = "Inherit"

                st.markdown("**Human approval**")
                modes = ["Inherit", "Require", "Skip"]
                widget_key = f"hitl_skill_mode_{skill_hitl_scope}_{skill_name}"
                st.session_state[widget_key] = current_mode
                st.radio(
                    "Approval mode for this skill command",
                    modes,
                    index=modes.index(current_mode),
                    key=widget_key,
                    horizontal=True,
                    help=(
                        "Inherit = use the safe mutation default; Require = always prompt; "
                        "Skip = preapprove this skill command"
                    ),
                    on_change=_persist_skill_hitl_mode,
                    args=(widget_key, skill_qualified_id, current_mode),
                )
                error_message = st.session_state.pop(f"{widget_key}_error", None)
                if error_message:
                    st.error(error_message)

            col_view, col_spacer = st.columns([1, 1])

            with col_view:
                if st.button(
                    "View Content",
                    key=f"skill_view_{skill_name}",
                    icon=":material/visibility:",
                    width="stretch",
                ):
                    st.session_state[f"skill_detail_{skill_name}"] = True

            # Show full content on demand
            if st.session_state.get(f"skill_detail_{skill_name}", False):
                detail = get_skill_detail(skill_name)
                if detail:
                    content = detail.get("content", "_No content_")
                    st.markdown("---")
                    st.markdown("#### Skill Instructions")
                    st.markdown(content)

                    if st.button(
                        "Hide Content",
                        key=f"skill_hide_{skill_name}",
                        icon=":material/visibility_off:",
                    ):
                        st.session_state[f"skill_detail_{skill_name}"] = False
                        st.rerun()
                else:
                    st.error("Failed to load skill details.")

    # Help section
    st.markdown("---")
    with st.expander(":material/help: How to add a new skill", expanded=False):
        st.markdown(
            """
1. Create a new folder inside `skills/`, e.g. `skills/my-skill/`
2. Add a `SKILL.md` file with YAML front-matter:

```markdown
---
name: my-skill
description: >
  A short description of what this skill does and when to activate it.
---

# My Skill

Your Markdown instructions go here. These will be appended to every
agent's system prompt when the skill is enabled.
```

3. Click **Reload from Disk** above to pick it up.
4. Restart or redeploy the server if you need the running backend process to pick up the new skill.
"""
        )


def render_interrupt_approval_ui():
    """Render the UI for approving/rejecting/editing tool executions"""
    interrupt_info = st.session_state.get("pending_interrupt", {})

    if not interrupt_info:
        return

    interrupt_message = extract_interrupt_message(interrupt_info)

    thread_id = interrupt_info.get("thread_id")
    interrupt_id = interrupt_info.get("interrupt_id")
    action_requests = interrupt_info.get("action_requests", [])

    reconciliation_marker = st.session_state.get(_hitl_reconciliation_key())
    if should_suppress_pending_interrupt(interrupt_id, reconciliation_marker):
        st.info(
            st.session_state.get("hitl_reconciliation_notice")
            or "Approval status is being reconciled."
        )
        check_column, return_column = st.columns(2)
        with check_column:
            if st.button("Check status", key=f"hitl_check_status_{interrupt_id}"):
                _reconcile_interrupt(interrupt_id)
                st.rerun()
        with return_column:
            if st.button("Return to chat", key=f"hitl_return_to_chat_{interrupt_id}"):
                _clear_interrupt_ui_state(interrupt_id)
                st.rerun()
        return

    # Key all pending decisions under the interrupt_id to avoid cross-contamination
    # on reload or when multiple interrupts occur in a single session.
    decisions_key = f"pending_decisions_{interrupt_id}" if interrupt_id else "pending_decisions"
    resume_inflight = bool(st.session_state.get(_hitl_resume_lock_key(interrupt_id)))

    if interrupt_message:
        st.warning(f"**{interrupt_message}**", icon=":material/pause_circle:")

    # Surface the reasoning/partial answer the agent streamed before pausing.
    # On an interrupt turn no assistant message is persisted, so without this the
    # streamed thinking/content is lost on the rerun into this approval view.
    agent_thinking, agent_partial_content = interrupt_stream_context(interrupt_info)
    if agent_partial_content:
        st.markdown(agent_partial_content)
    if agent_thinking:
        with st.expander("Agent reasoning", icon=":material/neurology:"):
            st.markdown(agent_thinking)

    if not action_requests:
        if st.button("Cancel"):
            st.session_state.pop("pending_interrupt", None)
            st.rerun()
        return

    # Initialize decisions in session state if not present for this interrupt
    if decisions_key not in st.session_state:
        st.session_state[decisions_key] = {}

    # Tools arrive one interrupt at a time; offset numbering by tools already
    # resolved this turn so the sequence reads Tool 1, Tool 2, ... not Tool 1 each.
    step_base = int(st.session_state.get("hitl_step_base", 0))

    for idx, action_request in enumerate(action_requests):
        tool_name = (
            action_request.get("action")
            or action_request.get("tool")
            or action_request.get("name")
            or str(idx + 1)
        )
        tool_args = action_request.get("args", {})
        description = action_request.get("description", "")
        task_id, tool_call_id = interrupt_request_target_ids(action_request)

        # Determine which decision types are allowed for this tool
        allowed_decisions = interrupt_allowed_decisions(action_request)

        # Check if this tool already has a decision
        current_decision = st.session_state[decisions_key].get(task_id)

        st.markdown(f"### {approval_tool_label(step_base, idx, tool_name)}")
        if description:
            st.markdown(f"**Description:** {description}")
        if task_id:
            st.caption(f"`{task_id}`")

        # Show decision status if already decided
        if current_decision:
            decision_type = current_decision.get("type", "")
            decision_label = str(decision_type).replace("_", " ").title()
            if decision_type == "approve":
                st.success(decision_label, icon=":material/check_circle:")
            elif decision_type == "edit":
                st.info(decision_label, icon=":material/edit:")
            elif decision_type == "reject":
                st.error(decision_label, icon=":material/cancel:")

            # Option to change decision
            if st.button("Change decision", key=f"change_{idx}", disabled=resume_inflight):
                st.session_state[decisions_key].pop(task_id, None)
                st.session_state.pop(f"editing_tool_{idx}", None)
                st.rerun()
        else:
            # Display tool arguments
            with st.expander("Tool Arguments", expanded=True):
                st.json(tool_args)

            # Decision options — only show buttons for allowed decision types
            col1, col2, col3 = st.columns(3)

            with col1:
                if "approve" in allowed_decisions and st.button(
                    "Approve",
                    key=f"approve_{idx}",
                    width="stretch",
                    type="primary",
                    disabled=resume_inflight,
                ):
                    st.session_state[decisions_key][task_id] = build_interrupt_decision(
                        "approve",
                        action_request,
                        action=tool_name,
                        args=None,
                    )
                    st.rerun()

            with col2:
                if "edit" in allowed_decisions and st.button(
                    "Edit Args",
                    key=f"edit_{idx}",
                    width="stretch",
                    disabled=resume_inflight,
                ):
                    st.session_state[f"editing_tool_{idx}"] = True
                    st.rerun()

            with col3:
                if "reject" in allowed_decisions and st.button(
                    "Reject",
                    key=f"reject_{idx}",
                    width="stretch",
                    disabled=resume_inflight,
                ):
                    st.session_state[decisions_key][task_id] = build_interrupt_decision(
                        "reject",
                        action_request,
                        action=tool_name,
                        args={},
                    )
                    st.rerun()

            # Show edit form if editing
            if st.session_state.get(f"editing_tool_{idx}"):
                st.markdown("**Edit Arguments:**")
                with st.form(f"edit_form_{idx}"):
                    edited_args_text = st.text_area(
                        "Arguments (JSON format)",
                        value=json.dumps(tool_args, indent=2, ensure_ascii=False),
                        height=200,
                        disabled=resume_inflight,
                    )

                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.form_submit_button(
                            "Save & Approve",
                            width="stretch",
                            type="primary",
                            disabled=resume_inflight,
                        ):
                            try:
                                edited_args = json.loads(edited_args_text)
                                st.session_state[decisions_key][task_id] = build_interrupt_decision(
                                    "edit",
                                    action_request,
                                    action=tool_name,
                                    args=edited_args,
                                )
                                st.session_state.pop(f"editing_tool_{idx}", None)
                                st.rerun()
                            except json.JSONDecodeError:
                                st.error("Invalid JSON format")

                    with col_cancel:
                        if st.form_submit_button(
                            "Cancel", width="stretch", disabled=resume_inflight
                        ):
                            st.session_state.pop(f"editing_tool_{idx}", None)
                            st.rerun()

        if idx < len(action_requests) - 1:
            st.divider()

    # Check if all tools have decisions
    all_task_ids = set()
    for req in action_requests:
        task_id, _tool_call_id = interrupt_request_target_ids(req)
        if task_id:
            all_task_ids.add(task_id)

    decided_task_ids = set(st.session_state[decisions_key].keys())
    all_decided = all_task_ids == decided_task_ids and len(all_task_ids) > 0

    st.divider()

    # Show progress
    st.progress(len(decided_task_ids) / max(len(all_task_ids), 1))
    st.caption(f"Decided: {len(decided_task_ids)} / {len(all_task_ids)} tools")

    resume_stream_container = st.container()
    submit_resume = False

    # Submit button — disabled until all tools have a decision
    col_submit, col_approve_all, col_cancel = st.columns(3)

    undecided_requests = []
    for req in action_requests:
        task_id, _tool_call_id = interrupt_request_target_ids(req)
        if task_id not in st.session_state[decisions_key]:
            undecided_requests.append(req)
    can_approve_all = bool(undecided_requests) and all(
        "approve" in interrupt_allowed_decisions(req) for req in undecided_requests
    )
    can_reject_all = bool(undecided_requests) and all(
        "reject" in interrupt_allowed_decisions(req) for req in undecided_requests
    )

    with col_submit:
        if st.button(
            "Submit Decisions",
            width="stretch",
            type="primary",
            disabled=not all_decided or resume_inflight,
        ):
            submit_resume = True

    with col_approve_all:
        if st.button(
            "Approve All",
            width="stretch",
            disabled=resume_inflight or not can_approve_all,
            help=None if can_approve_all else "One or more tools do not allow approval.",
        ):
            # Auto-approve all remaining tools
            for req in action_requests:
                task_id, _tool_call_id = interrupt_request_target_ids(req)
                if task_id and task_id not in st.session_state[decisions_key]:
                    st.session_state[decisions_key][task_id] = build_interrupt_decision(
                        "approve",
                        req,
                        args=None,
                    )
            submit_resume = True

    with col_cancel:
        if st.button(
            "Cancel All",
            width="stretch",
            disabled=resume_inflight or not can_reject_all,
            help=None if can_reject_all else "One or more tools do not allow rejection.",
        ):
            # Reject all tools
            for req in action_requests:
                task_id, _tool_call_id = interrupt_request_target_ids(req)
                if task_id:
                    st.session_state[decisions_key][task_id] = build_interrupt_decision(
                        "reject",
                        req,
                        args={},
                    )
            submit_resume = True

    if submit_resume:
        st.session_state[_hitl_resume_lock_key(interrupt_id)] = True
        with resume_stream_container:
            _submit_interrupt_decisions(thread_id, interrupt_id, action_requests, decisions_key)


def _submit_interrupt_decisions(thread_id, interrupt_id, action_requests, decisions_key=None):
    """Helper function to submit interrupt decisions to the backend"""
    if decisions_key is None:
        decisions_key = f"pending_decisions_{interrupt_id}" if interrupt_id else "pending_decisions"
    conversation_id = st.session_state.get("interrupt_conversation_id")
    decisions = list(st.session_state.get(decisions_key, {}).values())

    resume_payload = {
        "threadId": thread_id,
        "conversationId": conversation_id,
        "interruptId": interrupt_id,
        "decisions": decisions,
        # Match the send path so resumed turns keep the inline rich-response
        # contract — widgets/tool outputs render inline at marker position.
        "inlineRichResponseV1": True,
    }

    with st.status("Resuming execution...", expanded=True) as status:
        next_interrupt = None
        resume_succeeded = False
        resume_error_event: dict[str, Any] | None = None
        trace_placeholder = st.empty()
        response_placeholder = st.empty()
        stream_renderer = _StreamingRichResponseRenderer(
            response_placeholder,
            message_key=f"resume::{conversation_id or thread_id}",
        )
        image_preview_panel = _StreamingImagePreviewPanel(st.empty())
        accumulated_content = ""
        accumulated_thinking = ""

        _reset_stream_trace_state(expanded=True)

        for event in make_streaming_request("/messages/resume-interrupt", resume_payload):
            event_type = event.get("type")

            if event_type == "agent_selected":
                selected_agent = event.get("agent", "unknown")
                display_name = event.get("agent_name") or get_agent_display_name(selected_agent)
                status.update(label=f"{display_name} is processing...", state="running")
                continue

            if event_type == "thinking":
                content = event.get("content", "")
                accumulated_thinking += content
                _upsert_stream_thinking_trace(accumulated_thinking)
                st.session_state.stream_trace_expanded = True
                render_live_trace_panel(trace_placeholder)
                status.update(label="Working...", state="running")
                continue

            if event_type == "tool":
                _upsert_stream_tool_trace(event)
                render_live_trace_panel(trace_placeholder)
                status.update(
                    label=_format_stream_tool_status_label(event),
                    state="running",
                )
                continue

            if event_type == "node_complete":
                if _upsert_stream_subagent_activity(event):
                    render_live_trace_panel(trace_placeholder)
                    status.update(label="Subagents: dispatching...", state="running")
                continue

            if event_type == "subagent":
                if _upsert_stream_subagent_activity(event):
                    render_live_trace_panel(trace_placeholder)
                    status.update(label="Subagents: working...", state="running")
                continue

            if event_type == "token":
                content = event.get("content", "")
                accumulated_content += content
                if _has_live_trace_panel_content():
                    st.session_state.stream_trace_expanded = False
                    render_live_trace_panel(trace_placeholder)
                stream_renderer.append_text(content)
                status.update(label="Resuming response...", state="running")
                continue

            if event_type == "rich_items":
                stream_renderer.apply_rich_items_upsert(event.get("items") or [])
                continue

            if event_type == "image_preview":
                image_preview_panel.apply(event)
                status.update(label="Image ready — finishing response...", state="running")
                continue

            if event_type == "interrupt":
                next_interrupt = event.get("interrupt")
                resume_succeeded = True
                # Carry the reasoning/partial answer streamed during this resume
                # turn onto the follow-up interrupt so it shows in the next prompt.
                attach_stream_context(
                    next_interrupt,
                    thinking=accumulated_thinking,
                    content=accumulated_content,
                )
                interrupt_message = extract_interrupt_message(next_interrupt)
                if interrupt_message:
                    status.update(label=interrupt_message, state="running")
                break

            if event_type == "error":
                resume_error_event = event
                resume_error = event.get("error") or "Failed to resume execution"
                status.update(label=f"Error: {resume_error}", state="error")
                break

            if event_type == "complete":
                image_preview_panel.clear()
                stream_renderer.finalize(event.get("message"))
                status.update(label="Resume completed", state="complete")
                resume_succeeded = True
                break

        _clear_inflight_state()

        if resume_error_event:
            resume_error = str(resume_error_event.get("error") or "Failed to resume execution")
            error_code = extract_error_code(resume_error_event)

            if is_recoverable_resume_conflict(resume_error_event):
                _reconcile_interrupt(interrupt_id)
                st.rerun()
                return

            if (
                error_code
                in {
                    "INTERRUPT_FAILED",
                    "INTERRUPT_EXPIRED",
                    "INTERRUPT_NOT_FOUND",
                }
                or resume_error_event.get("status_code") == 410
            ):
                _clear_interrupt_ui_state(interrupt_id)
                st.session_state[_hitl_reconciliation_key()] = interrupt_id
                st.session_state["hitl_reconciliation_notice"] = resume_error
                st.error(resume_error)
                st.rerun()
                return

            st.session_state.pop(_hitl_resume_lock_key(interrupt_id), None)
            st.error(resume_error)
            return

        # The attachments belong to the original user turn that produced this
        # interrupt. Once resume succeeds, do not leave them queued for the next
        # message (including when execution pauses again on a follow-up tool).
        if resume_succeeded:
            st.session_state.pending_image_attachments = []

        # Clear decisions for this completed interrupt only after a non-error event.
        st.session_state.pop(decisions_key, None)
        for idx in range(len(action_requests)):
            st.session_state.pop(f"editing_tool_{idx}", None)

        st.session_state.pop(_hitl_resume_lock_key(interrupt_id), None)

        if next_interrupt:
            # The just-resolved interrupt's tools are now done; advance the
            # cumulative counter so the follow-up interrupt numbers its tools
            # after these (Tool 1 → Tool 2 → ...) instead of restarting at 1.
            st.session_state.hitl_step_base = int(st.session_state.get("hitl_step_base", 0)) + len(
                action_requests or []
            )
            st.session_state.pending_interrupt = next_interrupt
            st.session_state.interrupt_conversation_id = conversation_id
            next_interrupt_message = extract_interrupt_message(next_interrupt)
            if next_interrupt_message:
                st.toast(next_interrupt_message, icon=":material/warning:")
            st.rerun()

        _clear_interrupt_ui_state(interrupt_id)
        st.rerun()


_PLAN_NEXT_TASK_PREVIEW_CHARS = 50

_PLAN_TASK_STATUS_RENDERING: dict[str, tuple[str, str]] = {
    "completed": (":material/check_circle:", "completed"),
    "in_progress": (":material/refresh:", "in_progress"),
    "skipped": (":material/skip_next:", "skipped"),
    "pending": (":material/pending:", "pending"),
}


def _render_plan_progress_widget(
    conversation_id: str,
    *,
    current_conv: dict[str, Any] | None,
) -> None:
    """Render the inline plan progress widget for the given conversation.

    The widget is driven by the canonical ``planning-status`` payload rather
    than the locally cached ``planningModeEnabled`` flag, so plans that get
    created mid-conversation appear immediately on the next render even when
    the conversations list cache is stale.
    """
    status = get_planning_status(conversation_id)
    if not status or status.get("totalTasks", 0) <= 0:
        return

    if isinstance(current_conv, dict) and not current_conv.get("planningModeEnabled"):
        # Best-effort cache sync so other widgets (sidebar badges, manage modal)
        # see the flag without an extra fetch.
        current_conv["planningModeEnabled"] = True

    progress_pct = status.get("progressPercentage", 0)
    next_task = status.get("nextTask")
    completed = status.get("completedTasks", 0)
    skipped = status.get("skippedTasks", 0)
    total = status.get("totalTasks", 0)

    with st.container():
        col_progress, col_action = st.columns([4, 1])
        with col_progress:
            st.progress(progress_pct / 100.0)
            if next_task:
                next_desc = next_task.get("description", "N/A")
                truncated = (
                    f"{next_desc[:_PLAN_NEXT_TASK_PREVIEW_CHARS]}..."
                    if len(next_desc) > _PLAN_NEXT_TASK_PREVIEW_CHARS
                    else next_desc
                )
                st.success(
                    f":material/checklist: **{completed}/{total}** tasks complete | "
                    f"**Current:** {truncated}"
                )
            else:
                if completed == total:
                    st.success(f":material/check_circle: **All {total} tasks completed!**")
                else:
                    st.success(
                        f":material/check_circle: **All {total} tasks resolved** | "
                        f"{completed} completed, {skipped} skipped"
                    )
        with col_action:
            if st.button(
                ":material/checklist: View All",
                key=f"goto_planning_from_chat_{conversation_id}",
                help="View full task list",
            ):
                st.session_state.active_view = "planning"
                st.rerun()

        tasks = get_task_plans(conversation_id, include_completed=True)
        if not tasks:
            return
        with st.expander(":material/list_alt: Task Progress", expanded=False):
            for task in tasks:
                task_status = str(task.get("status") or "pending")
                task_desc = task.get("description", "No description")
                icon, label = _PLAN_TASK_STATUS_RENDERING.get(
                    task_status, _PLAN_TASK_STATUS_RENDERING["pending"]
                )
                if label == "completed":
                    st.markdown(f"{icon} ~~{task_desc}~~")
                elif label == "in_progress":
                    st.markdown(f"{icon} **{task_desc}** (current)")
                elif label == "skipped":
                    st.markdown(f"{icon} ~~{task_desc}~~ (skipped)")
                else:
                    st.markdown(f"{icon} {task_desc}")


def render_chat_view():
    """Main chat interface"""
    conversation_id = st.session_state.get("current_conversation_id")

    # Load messages
    def load_messages_page(page: int, *, show_spinner: bool = False) -> None:
        conv_id = st.session_state.get("current_conversation_id")
        if not conv_id or conv_id == "pending_new":
            return

        def fetch_page():
            return get_messages(conv_id, page=page, limit=10, order_direction="desc")

        if show_spinner:
            with st.spinner("Loading messages..."):
                response = fetch_page()
        else:
            response = fetch_page()

        if response and response.get("data"):
            data = response["data"]
            items = data.get("items", [])
            meta = data.get("meta", {})

            attachments_state = st.session_state.setdefault("message_image_thumbnails", {})
            chunks_state = st.session_state.setdefault("message_chunks", {})
            existing_messages = {msg["id"]: msg for msg in st.session_state.messages}

            for item in items:
                msg_id = item.get("id")
                if msg_id:
                    metadata = get_message_metadata(item)
                    attachments = metadata.get("attachments") or []
                    normalized_attachments: list[dict[str, str]] = []

                    for att in attachments:
                        if not isinstance(att, dict):
                            continue
                        data_b64 = att.get("data")
                        data_b64 = data_b64.strip() if isinstance(data_b64, str) else None

                        url_value = att.get("url")
                        url_value = url_value.strip() if isinstance(url_value, str) else None

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
                                            "character_count": chunk.get("character_count", 0),
                                        }
                        chunks_state[key] = {
                            "documents_cited": documents_cited,
                            "chunks_map": chunks_map,
                        }
                    else:
                        chunks_state.pop(key, None)

                    existing_messages[msg_id] = item

            def sort_key(message: dict[str, Any]):
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

        # Reload recovery: if no pending_interrupt is in session (e.g. after page
        # refresh) but the latest assistant message shows a paused workflow, restore
        # the interrupt state so the approval UI is presented again.
        if not st.session_state.get("pending_interrupt"):
            _msgs = st.session_state.get("messages", [])
            for _msg in reversed(_msgs):
                if _msg.get("role") == "assistant":
                    _meta = get_message_metadata(_msg)
                    if (
                        _meta.get("paused")
                        and _meta.get("pause_reason") == "tool_approval_required"
                    ):
                        _interrupt_data = _meta.get("interrupt")
                        _thread_id = _meta.get("thread_id")
                        if _interrupt_data and isinstance(_interrupt_data, dict):
                            # thread_id may be stored separately in metadata
                            if _thread_id and not _interrupt_data.get("thread_id"):
                                _interrupt_data = {
                                    **_interrupt_data,
                                    "thread_id": _thread_id,
                                }
                            recovered_interrupt_id = _interrupt_data.get(
                                "interrupt_id"
                            ) or _interrupt_data.get("interruptId")
                            reconciliation_marker = st.session_state.get(_hitl_reconciliation_key())
                            if not should_suppress_pending_interrupt(
                                recovered_interrupt_id, reconciliation_marker
                            ):
                                st.session_state.pending_interrupt = _interrupt_data
                                st.session_state.interrupt_conversation_id = conversation_id
                    break  # only check the most recent assistant message

    if not st.session_state.get("pending_interrupt"):
        reconciliation_notice = st.session_state.pop("hitl_reconciliation_notice", None)
        if reconciliation_notice:
            st.info(reconciliation_notice)

    # Show conversation title
    current_conv = next(
        (c for c in st.session_state.conversations_list if c["id"] == conversation_id),
        None,
    )

    if conversation_id == "pending_new":
        st.markdown("# New Chat")
        queued_persona = st.session_state.get("pending_persona_prompt", "")
        if queued_persona:
            st.info(f"**Instructions queued:** {persona_preview(queued_persona, 100)}")
    elif current_conv:
        st.markdown(f"# {current_conv['title']}")
        active_persona = current_conv.get("personaPrompt")
        if active_persona:
            st.info(f"**Instructions active:** {persona_preview(active_persona, 100)}")
    elif conversation_id:
        # The conversation exists server-side but hasn't landed in our local
        # ``conversations_list`` yet (deep link, manage-modal open, stale cache).
        # Show a minimal header so the plan widget and messages still render —
        # the title syncs on the next list refresh.
        st.markdown("# Conversation")
    else:
        st.markdown("# Welcome!")
        st.info("Select a conversation from the sidebar or create a new chat to get started.")
        return

    # Render the plan progress widget for any persisted conversation (including
    # plans created mid-conversation). It is decoupled from ``current_conv``
    # because the server-side status is authoritative — relying on
    # ``planningModeEnabled`` in the local cache would hide plans created in
    # this turn before the conversation list is refreshed.
    if conversation_id and conversation_id != "pending_new":
        _render_plan_progress_widget(conversation_id, current_conv=current_conv)
        _render_conversation_usage_panel(str(conversation_id))

    # Load more button
    if (
        conversation_id
        and conversation_id != "pending_new"
        and st.session_state.has_more_messages
        and st.button("Load older messages", width="stretch")
    ):
        next_page = st.session_state.conversation_messages_page + 1
        load_messages_page(next_page, show_spinner=True)

    st.divider()

    # Messages
    messages_to_display = (
        st.session_state.messages if conversation_id and conversation_id != "pending_new" else []
    )

    if not messages_to_display and conversation_id not in (None, "pending_new"):
        st.info("No messages yet. Start the conversation!", icon=":material/chat:")

    # Find the last assistant message for showing suggestions (skip hidden HITL markers)
    last_assistant_msg_id = None
    for msg in reversed(messages_to_display):
        sender_value = msg.get("sender")
        if sender_value not in (1, "user", "USER", "User") and not get_message_metadata(msg).get(
            "paused"
        ):
            last_assistant_msg_id = msg.get("id")
            break

    for msg in messages_to_display:
        sender_value = msg.get("sender")
        is_user_message = sender_value in (1, "user", "USER", "User")
        # HITL interrupt markers are hidden from the chat — they exist only for
        # reload recovery and carry no user-visible content.
        if not is_user_message and get_message_metadata(msg).get("paused"):
            continue
        render_message_bubble(
            msg,
            is_user_message,
            auto_mount_live_widgets=(
                not is_user_message and msg.get("id") == last_assistant_msg_id
            ),
        )

        # Show suggestion buttons for the last assistant message only
        if not is_user_message and msg.get("id") == last_assistant_msg_id:
            metadata = get_message_metadata(msg)
            suggestions = msg.get("suggestedQuestions") or metadata.get("suggested_questions")
            if suggestions:
                render_suggestion_buttons(suggestions, str(msg.get("id", "")))

    st.divider()

    # Show interrupt approval UI if there's a pending interrupt
    if st.session_state.get("pending_interrupt"):
        render_interrupt_approval_ui()
        return  # Don't show input area while interrupt is pending

    # Input area
    if conversation_id:
        # Show pending attachments
        _render_pending_image_attachments()

        # File uploader
        file_uploader_key = _chat_image_uploader_key(conversation_id) if conversation_id else None

        if st.session_state.show_attachment_uploader and file_uploader_key:
            st.file_uploader(
                "Attach images",
                type=["png", "jpg", "jpeg", "gif", "webp"],
                accept_multiple_files=True,
                key=file_uploader_key,
                help="Attach images",
                on_change=_consume_chat_image_uploader,
                args=(file_uploader_key,),
            )

        # Message form
        # ── Handle interrupted stream on rerun (Phase 2 of two-phase stop) ──
        stoppable_conversation_id = _stoppable_stream_conversation_id()
        if stoppable_conversation_id:
            _handle_stop_rerun(stoppable_conversation_id)
            return

        # Check for pending suggestion from suggestion buttons
        pending_suggestion = st.session_state.pop("pending_suggestion", "")
        preserved_draft = _consume_preserved_message_draft(conversation_id)

        # ── Message form ──
        # Capture form values first; heavy processing (streaming) happens AFTER
        # the form context exits so we can freely use st.button() etc.
        _form_send = False
        _form_attach = False
        _form_message = ""

        # Mount the clipboard capture component near the chat input. It is
        # non-blocking: when no new paste is consumed, execution falls through
        # to the message form so the text box and Send/Attach controls stay
        # visible. A consumed paste triggers a single rerun, after which the
        # event id is remembered and the stale component value is ignored.
        pasted_payload = capture_pasted_images(key=f"chat_image_paste_{conversation_id}")
        if _handle_pasted_image_payload(pasted_payload):
            st.rerun()

        with st.form("message_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([6, 1, 1])

            with col1:
                _form_message = st.text_area(
                    "Message",
                    value=pending_suggestion or preserved_draft,
                    placeholder="Type your message...",
                    height=100,
                    label_visibility="collapsed",
                    key=f"msg_input_{conversation_id}",
                )

            with col2:
                _form_send = st.form_submit_button("\nSend", width="stretch", type="primary")

            with col3:
                _form_attach = st.form_submit_button("Attach", width="stretch")

        # ── Process form actions OUTSIDE the form context ──
        if _form_attach:
            # Submitting a form with clear_on_submit=True resets its textarea.
            # Carry the current draft across the uploader-toggle rerun.
            _preserve_message_draft_for_attachment_toggle(conversation_id, _form_message)
            st.session_state.show_attachment_uploader = not st.session_state.get(
                "show_attachment_uploader", False
            )
            st.rerun()

        if _form_send:
            pending_attachments = list(st.session_state.get("pending_image_attachments", []))
            stripped_message = _form_message.strip()

            if not stripped_message and not pending_attachments:
                st.toast("Please enter a message", icon=":material/warning:")
            else:
                message_to_send = stripped_message or _format_image_only_message(
                    pending_attachments
                )
                title_sync_conversation_id: str | None = None

                if conversation_id == "pending_new":
                    saved_attachments = list(pending_attachments)

                    with st.status("Creating conversation...", expanded=True) as status:
                        # Use placeholder title - backend will generate and update it in parallel
                        conversation_data = {"title": "New Conversation"}
                        pending_persona = st.session_state.get("pending_persona_prompt", "")
                        persona_payload = normalize_persona_input(pending_persona)
                        if persona_payload:
                            conversation_data["personaPrompt"] = persona_payload

                        status.update(label="Creating conversation...", state="running")
                        conv_response = make_api_request(
                            "POST", "/conversations/", conversation_data
                        )
                        if conv_response and conv_response.get("data"):
                            new_conversation = conv_response["data"]
                            st.session_state.current_conversation_id = new_conversation["id"]
                            upsert_conversation_in_state(new_conversation)
                            st.session_state.conversations_loaded = True
                            # Apply custom agents queued while the chat was
                            # pending_new, BEFORE reset clears the buffer.
                            queued_custom_agent_ids = list(
                                st.session_state.get("pending_custom_agent_ids", [])
                            )
                            if queued_custom_agent_ids:
                                ca_status, _ca_payload = set_conversation_custom_agents(
                                    new_conversation["id"], queued_custom_agent_ids
                                )
                                if ca_status != 200:
                                    st.toast(
                                        "Failed to attach custom agents to the new chat",
                                        icon=":material/warning:",
                                    )
                            reset_conversation_state()
                            st.session_state.pending_image_attachments = saved_attachments
                            conversation_id = st.session_state.current_conversation_id
                            pending_attachments = list(saved_attachments)
                            status.update(label="Conversation created!", state="complete")
                        else:
                            st.toast(
                                "Failed to create conversation",
                                icon=":material/cancel:",
                            )
                            return

                current_conv = find_conversation_in_state(conversation_id)
                if current_conv and is_placeholder_conversation_title(current_conv.get("title")):
                    title_sync_conversation_id = conversation_id

                message_data = {
                    "content": message_to_send,
                    "conversationId": st.session_state.current_conversation_id,
                    # Opt into the inline rich-response v1 contract so the backend
                    # offers the bounded rich-item inventory to the model and the
                    # response-bubble path renders widgets/tool outputs inline at
                    # the author-placed marker position rather than appended.
                    "inlineRichResponseV1": True,
                }

                if pending_attachments:
                    message_data["attachments"] = pending_attachments

                # Placeholder for the "Stop generating" button lives OUTSIDE
                # the form but directly below it, so it appears next to Send /
                # Attach. Created only on the send path so idle reruns don't
                # leave an empty slot under the form.
                stop_button_placeholder = st.empty()

                # Use streaming endpoint for real-time response
                with st.status("Sending message...", expanded=True) as status:
                    # Create placeholder for streaming response
                    trace_placeholder = st.empty()
                    response_placeholder = st.empty()
                    stream_renderer = _StreamingRichResponseRenderer(
                        response_placeholder,
                        message_key=f"stream::{conversation_id}",
                    )
                    image_preview_panel = _StreamingImagePreviewPanel(st.empty())
                    accumulated_content = ""  # Initialize empty for accumulation
                    accumulated_thinking = ""  # Accumulate thinking content
                    final_message = None
                    interrupt_data = None
                    selected_agent = None  # Track which agent is processing
                    received_title_update = False

                    # Mark stream as in-flight BEFORE starting (survives rerun)
                    st.session_state.stream_inflight = True
                    st.session_state.stream_conversation_id = str(conversation_id)
                    st.session_state.stream_partial_text = ""
                    st.session_state.stream_partial_thinking = ""
                    st.session_state.stream_selected_agent = None
                    # New turn → restart the multi-interrupt approval counter so
                    # tool numbering reflects this turn's sequence (Tool 1, 2, ...).
                    st.session_state.hitl_step_base = 0
                    _reset_stream_trace_state(expanded=True)

                    # Render stop button into the placeholder that lives
                    # OUTSIDE the form.  Clicking it triggers a Streamlit
                    # rerun which drops the HTTP connection; on the next
                    # rerun stream_inflight==True triggers _handle_stop_rerun().
                    stop_button_placeholder.button(
                        "Stop generating",
                        key="stop_generating_btn",
                        type="secondary",
                        icon=":material/stop_circle:",
                    )

                    # Stream the response
                    for event in make_streaming_request("/messages/stream", message_data):
                        event_type = event.get("type")

                        if event_type == "user_message_created":
                            # Store user_message_id for stop endpoint
                            user_msg = event.get("message", {})
                            st.session_state.stream_user_message_id = str(user_msg.get("id", ""))
                            status.update(label="Generating response...", state="running")

                        elif event_type == "agent_selected":
                            # Track which agent was selected for processing
                            selected_agent = event.get("agent", "unknown")
                            st.session_state.stream_selected_agent = selected_agent
                            display_name = event.get("agent_name") or get_agent_display_name(
                                selected_agent
                            )
                            status.update(
                                label=f"{display_name} is processing...",
                                state="running",
                            )

                        elif event_type == "thinking":
                            content = event.get("content", "")
                            accumulated_thinking += content
                            st.session_state.stream_partial_thinking = accumulated_thinking
                            _upsert_stream_thinking_trace(accumulated_thinking)
                            st.session_state.stream_trace_expanded = True
                            render_live_trace_panel(trace_placeholder)
                            status.update(label="Working...", state="running")

                        elif event_type == "token":
                            content = event.get("content", "")
                            accumulated_content += content  # Append each token chunk
                            st.session_state.stream_partial_text = accumulated_content
                            if _has_live_trace_panel_content():
                                st.session_state.stream_trace_expanded = False
                                render_live_trace_panel(trace_placeholder)
                            stream_renderer.append_text(content)

                        elif event_type == "tool":
                            _upsert_stream_tool_trace(event)
                            render_live_trace_panel(trace_placeholder)
                            status.update(
                                label=_format_stream_tool_status_label(event),
                                state="running",
                            )

                        elif event_type == "rich_items":
                            stream_renderer.apply_rich_items_upsert(event.get("items") or [])

                        elif event_type == "image_preview":
                            image_preview_panel.apply(event)
                            status.update(
                                label="Image ready — finishing response...",
                                state="running",
                            )

                        elif event_type == "node_complete":
                            if _upsert_stream_subagent_activity(event):
                                render_live_trace_panel(trace_placeholder)
                                status.update(label="Subagents: dispatching...", state="running")

                        elif event_type == "subagent":
                            if _upsert_stream_subagent_activity(event):
                                render_live_trace_panel(trace_placeholder)
                                status.update(label="Subagents: working...", state="running")

                        elif event_type == "interrupt":
                            # Workflow paused for human approval
                            interrupt_data = event.get("interrupt")
                            interrupt_message = extract_interrupt_message(interrupt_data)
                            if interrupt_message:
                                status.update(label=interrupt_message, state="running")

                            # Store interrupt state in session for the approval UI.
                            # Carry the reasoning/partial answer streamed before the
                            # pause so it survives the rerun into the approval view.
                            if interrupt_data:
                                attach_stream_context(
                                    interrupt_data,
                                    thinking=accumulated_thinking,
                                    content=accumulated_content,
                                )
                                st.session_state.pending_interrupt = interrupt_data
                                st.session_state.interrupt_conversation_id = conversation_id
                            else:
                                st.json(event)

                            # Stop processing further events and rerun to show approval UI
                            _clear_inflight_state()
                            break

                        elif event_type == "complete":
                            # Store final message and complete
                            final_message = event.get("message")
                            _clear_usage_cache_after_completed_turn()
                            image_preview_panel.clear()
                            stream_renderer.finalize(final_message)
                            status.update(label="Message sent!", state="complete")

                        elif event_type == "error":
                            # Handle error
                            error_msg = event.get("error", "Unknown error")
                            status.update(label=f"Error: {error_msg}", state="error")
                            st.toast(f"Error: {error_msg}", icon=":material/cancel:")
                            _clear_inflight_state()
                            break

                        elif event_type == "title_updated":
                            # Update conversation title in real-time
                            new_title = event.get("title")
                            if new_title:
                                target_conversation_id = (
                                    event.get("conversation_id") or conversation_id
                                )
                                upsert_conversation_in_state(
                                    {
                                        "id": target_conversation_id,
                                        "title": new_title,
                                    }
                                )
                                received_title_update = True

                    # Clear stop button placeholder after stream ends
                    stop_button_placeholder.empty()

                    # Handle interrupt - show approval UI
                    if st.session_state.get("pending_interrupt"):
                        st.rerun()

                    # Clear inflight state on normal completion
                    _clear_inflight_state()

                    # If successful, update UI
                    if final_message:
                        if title_sync_conversation_id and not received_title_update:
                            sync_conversation_title_from_server(title_sync_conversation_id)
                        st.session_state.pending_image_attachments = []
                        reset_conversation_state()
                        st.session_state.show_attachment_uploader = False
                        load_messages_page(1)
                        st.toast("Message sent!", icon=":material/check_circle:")
                        st.rerun()
                    elif event_type != "error" and not interrupt_data:
                        st.toast("Failed to send message", icon=":material/cancel:")


def render_manage_modal():
    """Conversation management modal dialog"""
    if not st.session_state.get(CONVERSATION_MANAGER_DIALOG_KEY, False):
        return

    @st.dialog(
        "Manage Conversations",
        width="large",
    )
    def manage_dialog():
        def _deduplicate_conversations(
            conversations: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            """Return conversations with duplicate IDs removed, preserving order."""
            seen = set()
            deduped: list[dict[str, Any]] = []
            for conv in conversations:
                conv_id = str(conv.get("id", ""))
                if not conv_id or conv_id in seen:
                    continue
                seen.add(conv_id)
                deduped.append(conv)
            return deduped

        def _load_manager_page(page: int) -> None:
            """Fetch one page of conversations and append to session state."""
            resp = get_conversations(
                page=page,
                limit=_MANAGER_PAGE_SIZE,
                include_messages=True,
                latest_messages=3,
                fetch_all_pages=False,
            )
            if resp and resp.get("data"):
                items = resp["data"]["items"]
                meta = resp["data"].get("meta") or {}
                current_page = meta.get("currentPage", page)
                last_page = meta.get("lastPage", 1)
                total = meta.get("total", 0)
                st.session_state.manager_conversations.extend(items)
                st.session_state.manager_conv_page = current_page
                st.session_state.manager_conv_has_more = current_page < last_page
                st.session_state.manager_conv_total = total
            else:
                st.session_state.manager_conv_has_more = False

        # ------- initial / lazy load -------
        if st.session_state.current_user_id:
            if "manager_conversations" not in st.session_state:
                st.session_state.manager_conversations = []
                st.session_state.manager_conv_page = 0
                st.session_state.manager_conv_has_more = False
                st.session_state.manager_conv_total = 0

            if st.session_state.manager_conv_page == 0:
                with st.status("Loading conversations...", expanded=False):
                    _load_manager_page(1)

            manager_conversations = list(st.session_state.manager_conversations)
        else:
            manager_conversations = []

        # Search
        search_term = st.text_input("Search conversations", placeholder="Type to search...")

        if manager_conversations:
            manager_conversations = _deduplicate_conversations(manager_conversations)

            if search_term:
                filtered_map: dict[str, dict[str, Any]] = {}
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
                total_known = st.session_state.get("manager_conv_total", len(display_conversations))
                loaded_count = len(manager_conversations)
                if total_known > loaded_count:
                    st.caption(
                        f"Showing {len(display_conversations)} of {total_known} conversation(s) "
                    )
                else:
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
                                width="stretch",
                                type="primary",
                            ):
                                if conv_id is not None:
                                    st.session_state.current_conversation_id = conv_id
                                    close_conversation_manager()
                                    reset_conversation_state()
                                    st.rerun()
                                else:
                                    st.toast(
                                        "Conversation is missing an ID",
                                        icon=":material/warning:",
                                    )
                        with col2:
                            delete_button_key = (
                                f"conversation_manager_delete_{conv_id_str}_{idx}"
                                if conv_id_str
                                else f"conversation_manager_delete_{idx}"
                            )
                            if st.button(
                                "Delete",
                                key=delete_button_key,
                                width="stretch",
                            ):
                                if conv_id is None:
                                    st.toast(
                                        "Conversation is missing an ID",
                                        icon=":material/warning:",
                                    )
                                else:
                                    result = make_api_request("DELETE", f"/conversations/{conv_id}")
                                    if result:
                                        st.session_state.conversations_list = []
                                        # Remove from manager cache
                                        st.session_state.manager_conversations = [
                                            c
                                            for c in st.session_state.get(
                                                "manager_conversations", []
                                            )
                                            if c.get("id") != conv_id
                                        ]
                                        if st.session_state.current_conversation_id == conv_id:
                                            st.session_state.current_conversation_id = None
                                            reset_conversation_state()
                                        refresh_conversations_list()
                                        st.toast(
                                            f"Deleted '{conv.get('title', 'Conversation')}'",
                                            icon=":material/check_circle:",
                                        )
                                        st.rerun()

                # ------- Load-more button -------
                has_more = st.session_state.get("manager_conv_has_more", False)
                if has_more and not search_term:
                    remaining = max(
                        0,
                        st.session_state.get("manager_conv_total", 0)
                        - len(st.session_state.get("manager_conversations", [])),
                    )
                    if st.button(
                        f"Load more conversations ({remaining} remaining)",
                        key="manager_load_more",
                        width="stretch",
                    ):
                        next_page = st.session_state.manager_conv_page + 1
                        _load_manager_page(next_page)
            else:
                st.info("No conversations found matching your search.")
        else:
            st.info("No conversations available.")
        if st.button("Close", width="stretch"):
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
            if st.button("Close", width="stretch"):
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
            st.caption(f":material/bar_chart: Character count: {char_count}")
        else:
            st.info("Chunk content is not available.")

        st.divider()

        # Close button
        if st.button("Close", width="stretch"):
            st.session_state["chunk_preview_dialog_key"] = False
            st.session_state["selected_chunk_info"] = None
            st.rerun()

    chunk_preview_dialog()
    st.session_state["chunk_preview_dialog_key"] = False


def render_documents_tab():
    """Documents management workspace (moved from the sidebar)."""
    st.markdown("# :material/description: Documents")

    conversation_id = st.session_state.get("current_conversation_id")
    if conversation_id in (None, "pending_new"):
        st.info(
            "Select an existing conversation to upload or manage documents. "
            "Start a new chat and send your first message to unlock uploads."
        )
        return

    current_conv = next(
        (conv for conv in st.session_state.conversations_list if conv.get("id") == conversation_id),
        None,
    )
    if current_conv:
        title = current_conv.get("title") or "Conversation"
        st.caption(f"Managing documents for **{title}**")
    else:
        st.caption("Managing documents for the active conversation.")

    st.markdown("Upload supporting files and monitor their processing status for retrieval.")

    upload_col, tips_col = st.columns([1.25, 1])

    with upload_col:
        st.subheader("Upload Documents")
        st.caption(
            "Files attach to this conversation and become searchable once processing completes."
        )
        uploader_key = f"doc_uploader_{conversation_id}"
        uploaded_files = st.file_uploader(
            "Select files",
            # Mirror app/api/documents.py::SUPPORTED_UPLOAD_EXTENSIONS.
            type=["txt", "pdf", "docx", "pptx", "xlsx", "html", "md"],
            accept_multiple_files=True,
            key=uploader_key,
            help="Supported formats: TXT, PDF, DOCX, PPTX, XLSX, HTML, MD",
        )

        if uploaded_files:
            st.write(f"**Selected {len(uploaded_files)} file(s):**")
            for uploaded_file in uploaded_files:
                size_kb = uploaded_file.size / 1024
                st.caption(f"- {uploaded_file.name} ({size_kb:.1f} KB)")

            if st.button(
                "Upload & Process",
                key=f"upload_doc_{conversation_id}",
                width="stretch",
                type="primary",
            ):
                with st.spinner("Uploading documents..."):
                    upload_result = upload_documents(uploaded_files)

                if upload_result is not None:
                    data = upload_result.get("data") or {}
                    accepted = int(data.get("accepted_count") or 0)
                    rejected = int(data.get("rejected_count") or 0)

                    if accepted:
                        st.success(f"{accepted} document(s) queued for processing.")
                    if rejected:
                        st.warning(f"{rejected} file(s) were rejected.")
                        for item in data.get("files") or []:
                            if (item.get("status") or "").lower() != "rejected":
                                continue
                            error_code = item.get("error_code") or "REJECTED"
                            label = (
                                "Duplicate filename"
                                if error_code == "DUPLICATE_FILENAME"
                                else error_code
                            )
                            st.caption(
                                f"- {item.get('filename')} — {label}: {item.get('message') or ''}"
                            )

                    if accepted:
                        st.cache_data.clear()
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
            width="stretch",
        ):
            st.cache_data.clear()
            st.rerun()

    def _format_timestamp(value: str | None) -> str:
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
        1: {
            "label": "Processing",
            "icon": ":material/schedule:",
            "help": "Indexing in progress",
        },
        2: {
            "label": "Ready",
            "icon": ":material/check_circle:",
            "help": "Available for retrieval",
        },
        3: {
            "label": "Failed",
            "icon": ":material/warning:",
            "help": "Processing failed",
        },
    }

    for doc in documents:
        status = status_map.get(
            doc.get("status"),
            {
                "label": "Unknown",
                "icon": ":material/help:",
                "help": "Status unavailable",
            },
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
                    width="stretch",
                ):
                    with st.spinner("Removing document..."):
                        if delete_document(doc.get("id")):
                            st.cache_data.clear()
                            st.toast("Document deleted.", icon=":material/delete:")
                            st.rerun()
                        else:
                            st.error("Delete failed. Please try again.")

        st.divider()


# ==================== PLANNING TAB ====================

TASK_STATUS_MAP = {
    "pending": {
        "label": "Pending",
        "icon": '<span class="material-symbols-outlined" aria-hidden="true">schedule</span>',
        "color": "#f59e0b",
    },
    "in_progress": {
        "label": "In Progress",
        "icon": '<span class="material-symbols-outlined" aria-hidden="true">autorenew</span>',
        "color": "#3b82f6",
    },
    "completed": {
        "label": "Completed",
        "icon": '<span class="material-symbols-outlined" aria-hidden="true">check_circle</span>',
        "color": "#10b981",
    },
    "skipped": {
        "label": "Skipped",
        "icon": '<span class="material-symbols-outlined" aria-hidden="true">skip_next</span>',
        "color": "#64748b",
    },
}


def get_task_plans(conversation_id: str, include_completed: bool = True) -> list[dict[str, Any]]:
    """Fetch task plans for a conversation."""
    endpoint = f"/conversations/{conversation_id}/task-plans?include_completed={str(include_completed).lower()}"
    response = make_api_request("GET", endpoint)
    if response and response.get("success"):
        return response.get("data", [])
    return []


def get_planning_status(conversation_id: str) -> dict[str, Any] | None:
    """Fetch planning status for a conversation."""
    endpoint = f"/conversations/{conversation_id}/planning-status"
    response = make_api_request("GET", endpoint)
    if response and response.get("success"):
        return response.get("data")
    return None


def create_task_plan_ai(conversation_id: str, user_message: str) -> list[dict[str, Any]] | None:
    """Create a task plan using AI from user message."""
    endpoint = f"/conversations/{conversation_id}/task-plans"
    response = make_api_request("POST", endpoint, {"userMessage": user_message})
    if response and response.get("success"):
        return response.get("data", [])
    return None


def create_task_plan_manual(
    conversation_id: str, descriptions: list[str]
) -> list[dict[str, Any]] | None:
    """Create task plans manually from a list of descriptions."""
    endpoint = f"/conversations/{conversation_id}/task-plans/manual"
    response = make_api_request("POST", endpoint, {"taskDescriptions": descriptions})
    if response and response.get("success"):
        return response.get("data", [])
    return None


def update_task_plan(task_id: str, update_data: dict[str, Any]) -> dict[str, Any] | None:
    """Update a task plan."""
    endpoint = f"/task-plans/{task_id}"
    response = make_api_request("PATCH", endpoint, update_data)
    if response and response.get("success"):
        return response.get("data")
    return None


def complete_task_plan(task_id: str) -> dict[str, Any] | None:
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
    st.markdown("# :material/checklist: Planning")

    conversation_id = st.session_state.get("current_conversation_id")
    is_new_conversation = conversation_id == "pending_new"

    if conversation_id is None:
        st.info("Select a conversation or create a new chat to manage plans.")
        return

    if is_new_conversation:
        st.info("Create a conversation first by sending a message, then you can create task plans.")
        return

    # Get current conversation info
    current_conv: dict[str, Any] | None = next(
        (conv for conv in st.session_state.conversations_list if conv.get("id") == conversation_id),
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
            st.metric("Pending", pending)
        with col2:
            st.metric("In Progress", in_progress)
        with col3:
            st.metric("Completed", completed)
        with col4:
            st.metric("Skipped", skipped)

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
                    st.toast(f"Created {len(tasks)} tasks!", icon=":material/check_circle:")
                    st.session_state.clear_planning_generate_input = True
                    st.rerun()
                else:
                    st.toast("Failed to generate plan. Try again.", icon=":material/cancel:")

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
                line.strip() for line in manual_input.strip().split("\n") if line.strip()
            ]
            if descriptions:
                with st.spinner("Creating tasks..."):
                    tasks = create_task_plan_manual(conversation_id, descriptions)
                    if tasks:
                        st.toast(
                            f"Created {len(tasks)} tasks!",
                            icon=":material/check_circle:",
                        )
                        st.session_state.clear_planning_manual_input = True
                        st.rerun()
                    else:
                        st.toast(
                            "Failed to create tasks. Try again.",
                            icon=":material/cancel:",
                        )
            else:
                st.warning("Please enter at least one task description.")

    st.divider()

    # Task List Section
    st.markdown("### Task List")

    col1, col2 = st.columns([3, 1])
    with col1:
        show_completed = st.checkbox("Show completed tasks", value=True, key="show_completed_tasks")
    with col2:
        if st.button("Refresh", key="refresh_tasks_btn", width="stretch"):
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
            task.get("taskMetadata", {}).get("ad_hoc", False)
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
                                    {status_info["icon"]} Task {task_order + 1}
                                </span>
                            </div>
                            <span style="color: {border_color}; font-size: 0.85rem; font-weight: 500;">
                                {status_info["label"]}
                            </span>
                        </div>
                        <p style="margin: 8px 0 0 0; color: #374151;">{html.escape(description)}</p>
                        {f'<p style="margin: 4px 0 0 0; color: #6b7280; font-size: 0.8rem;">Completed: {completed_at[:16] if completed_at else "N/A"}</p>' if task_status == "completed" else ""}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Action buttons
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

                with col1:
                    if task_status == "pending" and st.button(  # Pending
                        "Start",
                        icon=":material/play_arrow:",
                        key=f"start_{task_id}",
                        width="stretch",
                    ):
                        result = update_task_plan(task_id, {"status": "in_progress"})
                        if result:
                            st.toast("Task started!", icon=":material/refresh:")
                            st.rerun()

                with col2:
                    if task_status in (
                        "pending",
                        "in_progress",
                    ) and st.button(  # Pending or In Progress
                        "Complete",
                        key=f"complete_{task_id}",
                        width="stretch",
                    ):
                        result = complete_task_plan(task_id)
                        if result:
                            st.toast("Task completed!", icon=":material/check_circle:")
                            st.rerun()

                with col3:
                    if task_status in (
                        "pending",
                        "in_progress",
                    ) and st.button(  # Pending or In Progress
                        "Skip",
                        icon=":material/skip_next:",
                        key=f"skip_{task_id}",
                        width="stretch",
                    ):
                        result = update_task_plan(task_id, {"status": "skipped"})
                        if result:
                            st.toast("Task skipped!", icon=":material/skip_next:")
                            st.rerun()

                with col4:
                    if st.button("Delete", key=f"delete_{task_id}", width="stretch"):
                        if delete_task_plan(task_id):
                            st.toast("Task deleted!", icon=":material/delete:")
                            st.rerun()
                        else:
                            st.toast("Failed to delete task.", icon=":material/cancel:")

                st.markdown("---")


def render_settings_view():
    """Settings and instructions view"""
    st.markdown("# :material/settings: Instructions")

    conversation_id = st.session_state.get("current_conversation_id")
    is_new_conversation = conversation_id == "pending_new"
    has_conversation = conversation_id not in (None, "pending_new")

    if conversation_id is None:
        st.info("Select a conversation or create a new chat to configure instructions.")
        return

    # Persona editor
    current_conv: dict[str, Any] | None = None
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
                if st.button(label, key=f"template_{idx}", width="stretch"):
                    st.session_state.persona_editor_value = template[:_MAX_PERSONA_LENGTH]
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
        st.error(
            "Character limit exceeded. Please shorten your instructions.",
            icon=":material/warning:",
        )

    # Actions
    col1, col2 = st.columns(2)

    with col1:
        if is_new_conversation:
            if st.button(
                "Apply to New Chat",
                width="stretch",
                disabled=exceeds_limit,
                type="primary",
            ):
                sanitized = normalize_persona_input(current_value)
                st.session_state.pending_persona_prompt = sanitized
                st.toast("Persona saved for new chat!", icon=":material/check_circle:")
                st.session_state.active_view = "chat"
                st.rerun()
        else:
            if st.button(
                "Save Persona",
                width="stretch",
                disabled=exceeds_limit,
                type="primary",
            ):
                sanitized = normalize_persona_input(current_value)
                payload = {"personaPrompt": sanitized or None}
                response = make_api_request("PATCH", f"/conversations/{conversation_id}", payload)
                if response and response.get("data"):
                    refresh_conversations_list()
                    st.toast("Persona updated!", icon=":material/check_circle:")
                    st.session_state.active_view = "chat"
                    st.rerun()
                else:
                    st.toast("Failed to update persona", icon=":material/cancel:")

    with col2:
        if is_new_conversation:
            if st.button("Clear", width="stretch"):
                st.session_state.pending_persona_prompt = ""
                st.rerun()
        else:
            if st.button("Clear Persona", width="stretch"):
                response = make_api_request(
                    "PATCH",
                    f"/conversations/{conversation_id}",
                    {"personaPrompt": None},
                )
                if response and response.get("data"):
                    refresh_conversations_list()
                    st.toast("Persona removed!", icon=":material/check_circle:")
                    st.rerun()


def render_models_view() -> None:
    """Model/provider settings rendered from the backend-owned options snapshot."""
    st.markdown("# Models")
    st.caption(
        "Provider status, synced catalogs, and effective agent selections are loaded from one backend snapshot."
    )

    if not st.session_state.get("model_config_options_cache"):
        with st.spinner("Loading model configuration..."):
            refresh_model_config_options_cache()

    snapshot = st.session_state.get("model_config_options_cache") or {}
    if not snapshot:
        error_message = st.session_state.get("model_config_options_error")
        if error_message:
            st.error(error_message)
        else:
            st.info("No model configuration data is available yet.")
        return
    if st.session_state.get("model_config_options_needs_form_sync"):
        _sync_model_config_form_state(snapshot)
        st.session_state.model_config_options_needs_form_sync = False

    provider_map = _snapshot_provider_map(snapshot)
    agent_config = _snapshot_agent_config(snapshot)

    provider_order: list[str] = []
    for provider in _snapshot_provider_list(snapshot):
        provider_type = _normalize_provider_type(
            provider.get("providerType") or provider.get("provider_type")
        )
        if provider_type not in provider_order:
            provider_order.append(provider_type)
    for provider_type in ("gemini", "openai"):
        if provider_type not in provider_order:
            provider_order.append(provider_type)

    controls_col1, controls_col2, controls_col3 = st.columns([1.2, 1.2, 3])
    with controls_col1:
        if st.button("Reload snapshot", width="stretch"):
            with st.spinner("Loading model configuration..."):
                refreshed = refresh_model_config_options_cache(force_refresh=True)
            if refreshed:
                st.toast("Model snapshot reloaded", icon=":material/refresh:")
            st.rerun()

    with controls_col2:
        if st.button("Reset agent defaults", width="stretch"):
            with st.spinner("Resetting..."):
                reset_model_config()
                refreshed = refresh_model_config_options_cache(force_refresh=True)
            if refreshed:
                st.toast("Agent model settings reset", icon=":material/cleaning_services:")
            st.rerun()

    with controls_col3:
        fetched_at = st.session_state.get("model_config_options_last_fetch")
        if fetched_at:
            st.caption(f"Last loaded: {fetched_at}")
        error_message = st.session_state.get("model_config_options_error")
        if error_message:
            st.caption(error_message)
        else:
            st.caption("Mutations refresh this snapshot automatically.")

    st.divider()
    st.subheader("Provider configuration")

    for provider_type in provider_order:
        provider = provider_map.get(provider_type, {})
        provider_name = _provider_display_name(provider_type)
        configured = bool(provider.get("configured"))
        key_source = str(provider.get("keySource") or provider.get("key_source") or "none")
        sync_status = str(provider.get("syncStatus") or provider.get("sync_status") or "unknown")
        last_synced_at = provider.get("lastSyncedAt") or provider.get("last_synced_at")
        sync_error = str(provider.get("syncError") or provider.get("sync_error") or "").strip()
        warnings = provider.get("warnings") or []
        models = _provider_models(provider)

        st.markdown(f"### {provider_name}")
        status_cols = st.columns(4)
        with status_cols[0]:
            st.metric("Configured", "Yes" if configured else "No")
        with status_cols[1]:
            st.metric("Key Source", key_source.upper())
        with status_cols[2]:
            st.metric("Sync Status", sync_status.replace("_", " ").title())
        with status_cols[3]:
            st.metric("Catalog Models", len(models))

        if key_source == "env":
            st.info(
                f"{provider_name} is currently using the server environment fallback. "
                "Save a DB key here if you want a user-specific override."
            )
        if sync_error:
            st.error(sync_error)
        for warning in warnings:
            if isinstance(warning, str) and warning.strip():
                st.warning(warning.strip())
        if last_synced_at:
            st.caption(f"Last synced: {last_synced_at}")

        with st.form(f"provider_form_{provider_type}", clear_on_submit=True):
            placeholder = "sk-..." if provider_type == "openai" else "AIza..."
            api_key = st.text_input(
                f"{provider_name} API key",
                type="password",
                placeholder=placeholder,
            )
            save_submitted = st.form_submit_button(
                f"Save / Update {provider_name} Key",
                width="stretch",
                type="primary",
            )
            if save_submitted:
                api_key = api_key.strip()
                if not api_key:
                    st.toast("Please enter an API key", icon=":material/warning:")
                else:
                    with st.spinner(f"Saving {provider_name} key..."):
                        result = upsert_provider(provider_type, api_key, is_default=False)
                        refreshed = refresh_model_config_options_cache(force_refresh=True)
                    if result and refreshed:
                        st.toast(f"{provider_name} key saved", icon=":material/check_circle:")
                    st.rerun()

        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button(
                f"Sync {provider_name} models",
                key=f"sync_provider_{provider_type}",
                width="stretch",
                disabled=not configured,
            ):
                with st.spinner(f"Syncing {provider_name} models..."):
                    synced_models = fetch_provider_models(provider_type, force_refresh=True)
                    refreshed = (
                        refresh_model_config_options_cache(force_refresh=True)
                        if synced_models is not None
                        else {}
                    )
                if synced_models is not None and refreshed:
                    st.toast(
                        f"Synced {len(synced_models)} {provider_name} models",
                        icon=":material/sync:",
                    )
                    st.rerun()

        with action_cols[1]:
            if st.button(
                f"Delete {provider_name} key",
                key=f"delete_provider_{provider_type}",
                width="stretch",
                disabled=key_source != "db",
            ):
                if delete_provider(provider_type):
                    refresh_model_config_options_cache(force_refresh=True)
                    st.toast(f"{provider_name} key deleted", icon=":material/delete:")
                st.rerun()

        with action_cols[2]:
            if key_source == "db":
                st.caption("Deleting removes the DB key and resets dependent agent configs.")
            elif key_source == "env":
                st.caption("Environment fallback is active; there is no DB key to delete.")
            else:
                st.caption("Save a key to enable sync and per-user model selection.")

        if models:

            def _ctx_window_cell(m: dict[str, Any]) -> str:
                value = m.get("contextWindowTokens") or m.get("context_window_tokens")
                return _format_tokens(value) if value else ""

            def _max_output_cell(m: dict[str, Any]) -> str:
                value = m.get("maxOutputTokens") or m.get("max_output_tokens")
                return _format_tokens(value) if value else ""

            model_rows = [
                {
                    "ID": str(model.get("id") or "").strip(),
                    "Name": str(
                        model.get("displayName") or model.get("display_name") or ""
                    ).strip(),
                    "Recommended": "Yes" if model.get("recommended") else "",
                    "Vision": "Yes"
                    if model.get("supportsVision") or model.get("supports_vision")
                    else "",
                    "Tools": "Yes"
                    if model.get("supportsToolCalling") or model.get("supports_tool_calling")
                    else "",
                    "Streaming": "Yes"
                    if model.get("supportsStreaming") or model.get("supports_streaming")
                    else "",
                    "Reasoning": "Yes"
                    if model.get("supportsReasoning") or model.get("supports_reasoning")
                    else "",
                    "Context Window": _ctx_window_cell(model),
                    "Max Output": _max_output_cell(model),
                }
                for model in models
            ]
            st.dataframe(model_rows, width="stretch", hide_index=True)
        elif configured:
            st.info(f"No synced {provider_name} models yet. Use the sync action above.")
        else:
            st.info(f"Configure {provider_name} above to load its model catalog.")

        st.divider()

    st.subheader("Agent model configuration")
    st.caption("These settings apply automatically to new messages.")

    agents: list[tuple[str, str]] = [
        ("chat", "Chat"),
        ("rag", "RAG"),
        ("search", "Search"),
        ("planning", "Planning"),
    ]

    st.markdown("#### Select providers for each agent")
    provider_cols = st.columns(len(agents))
    for idx, (agent_key, label) in enumerate(agents):
        with provider_cols[idx]:
            current_provider = _normalize_provider_type(
                st.session_state.get(f"model_cfg_provider_{agent_key}")
            )
            if current_provider not in provider_order:
                current_provider = provider_order[0]

            selected_provider = st.selectbox(
                label,
                options=provider_order,
                index=provider_order.index(current_provider),
                key=f"model_cfg_provider_{agent_key}",
                format_func=_provider_display_name,
            )
            provider_snapshot = provider_map.get(selected_provider, {})
            provider_state = (
                "Configured" if provider_snapshot.get("configured") else "Not configured"
            )
            st.caption(provider_state)

    st.divider()
    st.subheader("Configure models and parameters")

    with st.form("agent_model_config_form"):
        for agent_key, label in agents:
            cfg = agent_config.get(agent_key, {}) if isinstance(agent_config, dict) else {}
            selected_provider = _normalize_provider_type(
                st.session_state.get(f"model_cfg_provider_{agent_key}")
            )
            provider_snapshot = provider_map.get(selected_provider, {})
            catalog_ids = _provider_model_ids(provider_snapshot)
            configured = bool(provider_snapshot.get("configured"))
            key_source = str(
                provider_snapshot.get("keySource") or provider_snapshot.get("key_source") or "none"
            )
            sync_status = str(
                provider_snapshot.get("syncStatus")
                or provider_snapshot.get("sync_status")
                or "unknown"
            )
            current_selection = str(
                st.session_state.get(f"model_cfg_model_select_{agent_key}") or ""
            ).strip()
            allow_custom_default = bool(
                st.session_state.get(
                    f"model_cfg_allow_custom_{agent_key}",
                    cfg.get("isCustomModel") or cfg.get("is_custom_model") or False,
                )
            )
            custom_value_default = str(
                st.session_state.get(f"model_cfg_model_custom_{agent_key}") or ""
            ).strip()
            model_options = list(catalog_ids)
            if current_selection and current_selection not in model_options:
                model_options.insert(0, current_selection)
            if not model_options:
                model_options = ["(sync models first)"]

            st.markdown(f"**{label}**")
            st.caption(
                f"Provider: {_provider_display_name(selected_provider)}"
                f" • Key source: {key_source.upper()}"
                f" • Sync: {sync_status.replace('_', ' ').title()}"
            )

            field_cols = st.columns([2.2, 1.4, 1.2])
            with field_cols[0]:
                st.selectbox(
                    "Catalog model",
                    options=model_options,
                    index=0,
                    key=f"model_cfg_model_select_{agent_key}",
                    disabled=not configured or not catalog_ids,
                )
                st.text_input(
                    "Custom model ID",
                    key=f"model_cfg_model_custom_{agent_key}",
                    value=custom_value_default,
                    placeholder="Enter a provider-specific model ID",
                    disabled=not configured,
                    help="Only used when custom override is enabled below.",
                )

            with field_cols[1]:
                st.checkbox(
                    "Allow custom model override",
                    key=f"model_cfg_allow_custom_{agent_key}",
                    value=allow_custom_default,
                    disabled=not configured,
                )

            with field_cols[2]:
                st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=float(
                        st.session_state.get(
                            f"model_cfg_temperature_{agent_key}",
                            cfg.get("temperature", 1.0),
                        )
                    ),
                    step=0.05,
                    key=f"model_cfg_temperature_{agent_key}",
                )

            for warning in cfg.get("warnings") or []:
                if isinstance(warning, str) and warning.strip():
                    st.warning(warning.strip())

        submitted = st.form_submit_button(
            "Save agent model settings",
            type="primary",
            width="stretch",
        )

        if submitted:
            payload: dict[str, Any] = {}
            validation_errors: list[str] = []

            for agent_key, label in agents:
                selected_provider = _normalize_provider_type(
                    st.session_state.get(f"model_cfg_provider_{agent_key}")
                )
                provider_snapshot = provider_map.get(selected_provider, {})
                configured = bool(provider_snapshot.get("configured"))
                selected_model = str(
                    st.session_state.get(f"model_cfg_model_select_{agent_key}") or ""
                ).strip()
                custom_model = str(
                    st.session_state.get(f"model_cfg_model_custom_{agent_key}") or ""
                ).strip()
                allow_custom_model = bool(
                    st.session_state.get(f"model_cfg_allow_custom_{agent_key}")
                )
                temperature = st.session_state.get(f"model_cfg_temperature_{agent_key}", 1.0)
                if not isinstance(temperature, (int, float)):
                    temperature = 1.0

                if not configured:
                    validation_errors.append(
                        f"{label}: configure {_provider_display_name(selected_provider)} before saving."
                    )
                    continue

                model = custom_model if allow_custom_model else selected_model
                if allow_custom_model and not custom_model:
                    validation_errors.append(
                        f"{label}: enter a custom model ID or disable the custom override."
                    )
                    continue
                if not allow_custom_model and (
                    not selected_model or selected_model.startswith("(")
                ):
                    validation_errors.append(
                        f"{label}: sync {_provider_display_name(selected_provider)} models before saving."
                    )
                    continue

                payload[agent_key] = {
                    "provider": selected_provider,
                    "model": model,
                    "temperature": float(temperature),
                    "allow_custom_model": allow_custom_model,
                }

            if validation_errors:
                for error in validation_errors:
                    st.toast(error, icon=":material/warning:")
            else:
                with st.spinner("Saving model settings..."):
                    updated = patch_model_config(payload)
                    refreshed = (
                        refresh_model_config_options_cache(
                            force_refresh=True,
                            defer_form_state_sync=True,
                        )
                        if updated
                        else {}
                    )

                if updated and refreshed:
                    st.toast("Saved model settings", icon=":material/check_circle:")
                    st.rerun()
                st.toast("Failed to save model settings", icon=":material/cancel:")


def _render_usage_chart(frame: list[dict[str, Any]], *, kind: str) -> None:
    """Render a bounded frame with Altair, which ships with Streamlit."""
    if not frame:
        return
    import altair as alt

    data = alt.Data(values=frame)
    if kind == "trend":
        chart = (
            alt.Chart(data)
            .mark_area()
            .encode(
                x=alt.X("start:T", title="Time"),
                y=alt.Y("tokens:Q", stack="zero", title="Tokens"),
                color=alt.Color("tokenType:N", title="Token type"),
                tooltip=["start:T", "tokenType:N", "tokens:Q"],
            )
        )
    elif kind == "outcomes":
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("outcome:N", title="Outcome"),
                y=alt.Y("requests:Q", title="Requests"),
                tooltip=["outcome:N", "requests:Q", alt.Tooltip("rate:Q", format=".1%")],
            )
        )
    else:
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("totalTokens:Q", title="Total tokens"),
                y=alt.Y("name:N", sort="-x", title=None),
                tooltip=["name:N", "totalTokens:Q", "requests:Q"],
            )
        )
    st.altair_chart(chart, width="stretch")


def _usage_breakdown_rows(items: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return rows
    for item in items:
        if not isinstance(item, dict):
            continue
        totals = _normalized_usage_totals(item.get("totals"))
        rows.append(
            {
                "Name": str(item.get("key") or "Unknown"),
                "Total tokens": totals["totalTokens"],
                "Input": totals["inputTokens"],
                "Output": totals["outputTokens"],
                "Requests": totals["requestCount"],
                "Images": totals["generatedImages"],
            }
        )
    return rows


def _render_usage_breakdown(title: str, items: Any) -> None:
    rows = _usage_breakdown_rows(items)
    chart_frame = _build_usage_breakdown_frame(items)
    st.markdown(f"#### {title}")
    if rows:
        _render_usage_chart(chart_frame, kind="breakdown")
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.caption("No usage in this breakdown.")


def _render_context_usage_summary(context_window: dict[str, Any] | None) -> None:
    if not context_window:
        st.caption("Latest context-window usage is unavailable.")
        return
    presentation = _context_window_presentation(context_window)
    provider = str(context_window.get("provider") or "")
    model = str(context_window.get("model") or "")
    _render_context_window_indicator(provider, model, context_window)
    st.caption(f"{presentation['tooltip']} · Source: {presentation['source_badge']}")


def _render_conversation_usage_panel(conversation_id: str) -> None:
    """Render retained cumulative usage without allowing analytics errors to block chat."""
    with st.expander(":material/monitoring: Retained usage", expanded=False):
        if st.session_state.get("stream_inflight"):
            st.caption("Usage refreshes after this turn completes.")
            return
        try:
            usage = get_conversation_usage(conversation_id)
        except Exception:
            usage = None
        if usage is None:
            st.caption("Usage is temporarily unavailable. Chat remains available.")
            if st.button("Retry usage", key=f"retry_usage_{conversation_id}"):
                st.session_state.usage_cache_version = (
                    int(st.session_state.get("usage_cache_version", 0)) + 1
                )
                st.rerun()
            return

        totals = usage["totals"]
        columns = st.columns(4)
        columns[0].metric("Total tokens", _format_tokens(totals["totalTokens"]))
        columns[1].metric("Input", _format_tokens(totals["inputTokens"]))
        columns[2].metric("Output", _format_tokens(totals["outputTokens"]))
        columns[3].metric("Requests", totals["requestCount"])

        model_names = [
            str(item.get("key"))
            for item in usage.get("byModel", [])
            if isinstance(item, dict) and item.get("key")
        ]
        if model_names:
            st.caption(f"Models: {', '.join(model_names)}")
        _render_context_usage_summary(usage.get("latestContextWindow"))
        retained_range = usage.get("range") or {}
        if retained_range.get("from") and retained_range.get("to"):
            st.caption(f"Retained range: {retained_range['from']} to {retained_range['to']}")


def render_usage_view() -> None:
    """Render authenticated, user-scoped usage analytics returned by the API."""
    st.markdown("# Usage")
    st.caption("Token and image usage for your authenticated account.")

    local_zone = get_local_timezone_name()
    timezone_options = sorted(available_timezones())
    if local_zone not in timezone_options:
        timezone_options.insert(0, local_zone)
    zone_index = timezone_options.index(local_zone)

    filter_cols = st.columns([1, 2, 2])
    with filter_cols[0]:
        bucket = st.selectbox("Bucket", ["day", "hour"], key="usage_bucket")
    today = datetime.now(ZoneInfo(local_zone)).date()
    with filter_cols[1]:
        selected_range = st.date_input(
            "Date range",
            value=(today - timedelta(days=29), today),
            key="usage_date_range",
        )
    with filter_cols[2]:
        timezone_name = st.selectbox(
            "Timezone",
            timezone_options,
            index=zone_index,
            key="usage_timezone",
        )

    conversation_map = {
        str(item.get("id")): str(item.get("title") or "Untitled conversation")
        for item in st.session_state.get("conversations_list", [])
        if isinstance(item, dict) and item.get("id")
    }
    conversation_options = [""] + list(conversation_map)
    conversation_id = st.selectbox(
        "Conversation (optional)",
        conversation_options,
        format_func=lambda value: "All conversations" if not value else conversation_map[value],
        key="usage_conversation_filter",
    )

    if not isinstance(selected_range, (tuple, list)) or len(selected_range) != 2:
        st.info("Choose a start and end date.")
        return

    start_hour: datetime_time | None = None
    end_hour: datetime_time | None = None
    selected_zone = ZoneInfo(str(timezone_name))
    if bucket == "hour":
        local_now = datetime.now(selected_zone)
        next_hour = (local_now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        hour_columns = st.columns(2)
        with hour_columns[0]:
            start_hour = st.time_input(
                "From hour",
                value=datetime_time.min,
                step=timedelta(hours=1),
                key="usage_start_hour",
            )
        with hour_columns[1]:
            end_hour = st.time_input(
                "To hour (exclusive)",
                value=next_hour.time(),
                step=timedelta(hours=1),
                key="usage_end_hour",
            )
    query_start, query_end = _build_usage_query_boundaries(
        start_date=selected_range[0],
        end_date=selected_range[1],
        bucket=str(bucket),
        zone=selected_zone,
        start_hour=start_hour,
        end_hour=end_hour,
    )

    with st.spinner("Loading usage..."):
        usage = get_usage_dashboard(
            start=query_start,
            end=query_end,
            bucket=str(bucket),
            timezone_name=str(timezone_name),
            conversation_id=str(conversation_id) or None,
        )
    if usage is None:
        st.warning("Usage is temporarily unavailable.")
        if st.button("Retry dashboard", key="retry_usage_dashboard"):
            st.session_state.usage_cache_version = (
                int(st.session_state.get("usage_cache_version", 0)) + 1
            )
            st.rerun()
        return

    totals = usage["totals"]
    metric_columns = st.columns(5)
    metric_columns[0].metric("Total tokens", _format_tokens(totals["totalTokens"]))
    metric_columns[1].metric("Input", _format_tokens(totals["inputTokens"]))
    metric_columns[2].metric("Output", _format_tokens(totals["outputTokens"]))
    metric_columns[3].metric("Requests", totals["requestCount"])
    metric_columns[4].metric("Generated images", totals["generatedImages"])

    if totals["requestCount"] == 0:
        st.info("No usage was recorded for this range.")

    trend_frame = _build_usage_trend_frame(usage.get("series"))
    outcome_frame = _build_usage_outcome_frame(
        usage.get("outcomes"), total_requests=totals["requestCount"]
    )
    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown("### Input and output trend")
        if trend_frame:
            _render_usage_chart(trend_frame, kind="trend")
        else:
            st.caption("No trend data for this range.")
    with chart_columns[1]:
        st.markdown("### Outcomes")
        if outcome_frame:
            _render_usage_chart(outcome_frame, kind="outcomes")
        else:
            st.caption("No outcome data for this range.")

    breakdown_columns = st.columns(2)
    with breakdown_columns[0]:
        _render_usage_breakdown("Providers", usage.get("byProvider"))
        _render_usage_breakdown("Operations", usage.get("byOperation"))
    with breakdown_columns[1]:
        _render_usage_breakdown("Models", usage.get("byModel"))
        _render_usage_breakdown("Agents", usage.get("byAgent"))

    st.markdown("### Top conversations")
    top_rows: list[dict[str, Any]] = []
    for item in usage.get("topConversations", []):
        if not isinstance(item, dict):
            continue
        row_totals = _normalized_usage_totals(item.get("totals"))
        top_rows.append(
            {
                "Conversation": item.get("title") or "Untitled conversation",
                "Total tokens": row_totals["totalTokens"],
                "Requests": row_totals["requestCount"],
            }
        )
    if top_rows:
        st.dataframe(top_rows, width="stretch", hide_index=True)
    else:
        st.caption("No conversations have usage in this range.")

    coverage = usage.get("coverage") or {}
    known_ratio = coverage.get("knownTotalRatio")
    if isinstance(known_ratio, (int, float)):
        st.caption(
            f"Known token totals: {known_ratio:.1%} "
            f"({coverage.get('requestsWithKnownTotal', 0)} of "
            f"{coverage.get('totalRequests', 0)} requests)."
        )
    if usage.get("generatedAt"):
        st.caption(f"Generated at {usage['generatedAt']}")


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
    (
        tab_chat,
        tab_planning,
        tab_docs,
        tab_instructions,
        tab_models,
        tab_usage,
        tab_custom_agents,
        tab_mcp,
        tab_skills,
    ) = st.tabs(
        [
            ":material/chat: Chat",
            ":material/checklist: Planning",
            ":material/description: Documents",
            ":material/settings: Instructions",
            ":material/smart_toy: Models",
            ":material/monitoring: Usage",
            ":material/robot_2: Custom Agents",
            ":material/extension: MCP Config",
            ":material/psychology: Skills",
        ]
    )

    with tab_chat:
        render_chat_view()

    with tab_planning:
        render_planning_tab()

    with tab_docs:
        render_documents_tab()

    with tab_instructions:
        render_settings_view()

    with tab_models:
        render_models_view()

    with tab_usage:
        render_usage_view()

    with tab_custom_agents:
        render_custom_agents_view()

    with tab_mcp:
        render_tools_tab()

    with tab_skills:
        render_skills_tab()


if __name__ == "__main__":
    main()
