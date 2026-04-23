import base64
import html
import json
import mimetypes
import os
import re
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any

import markdown as _markdown  # type: ignore
import requests
import streamlit as st  # type: ignore
from dateutil import parser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.ai.skills_snapshot import (
    get_repo_skill_detail_for_demo,
    list_repo_skills_for_demo,
    reload_repo_skills_for_demo,
)
from app.services.stream_events import infer_tool_state, normalize_tool_phase
from upload_support import delete_document, get_uploaded_documents, upload_document

API_BASE_URL = os.environ.get("CHATBOT_API_BASE_URL", "http://127.0.0.1:8000")
WIDGET_WS_BASE_URL = os.environ.get("CHATBOT_WIDGET_WS_BASE_URL", "").rstrip("/")
REQUEST_TIMEOUT = (5, 30)
STREAM_REQUEST_TIMEOUT = (10, 900)

_MAX_PERSONA_LENGTH = 8000
_MAX_IMAGE_ATTACHMENTS = 4
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

    /* Status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
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

    /* Thinking/Reasoning UI Styles - Modern blue design */
    .thinking-container {
        border-left: 3px solid #3b82f6;
        padding: 14px 18px;
        margin: 8px 0 16px 0;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.06) 0%, rgba(59, 130, 246, 0.02) 100%);
        border-radius: 0 10px 10px 0;
        font-size: 0.9em;
        color: #4b5563;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .thinking-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: #3b82f6;
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
        background-color: #3b82f6;
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
        line-height: 1.65;
        white-space: pre-wrap;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #374151;
    }

    .thinking-content-rendered {
        line-height: 1.65;
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
        border: 1px solid #dbeafe;
        border-radius: 14px;
        background:
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.08), transparent 42%),
            linear-gradient(180deg, rgba(248, 250, 252, 0.95), rgba(239, 246, 255, 0.9));
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
        border: 1px solid #dbeafe;
        border-radius: 10px;
        background: rgba(239, 246, 255, 0.9);
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
    .canvas-artifact-header a.canvas-newtab {
        color: #0369a1;
        text-decoration: none;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid #bae6fd;
        border-radius: 6px;
        padding: 2px 8px;
        background: #fff;
        cursor: pointer;
        transition: background 0.15s;
    }
    .canvas-artifact-header a.canvas-newlab:hover {
        background: #bae6fd;
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
    data: dict | None = None,
    error_message: str = "API request failed",
    success_message: str | None = None,
) -> dict[str, Any] | None:
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
                st.toast(success_message, icon=":material/check_circle:")
            return response_data
        else:
            st.toast(error_message, icon=":material/cancel:")
            return None
    except Exception as e:
        st.toast(f"{error_message}: {str(e)}", icon=":material/cancel:")
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
        "processing": "schedule",
        "ready": "check_circle",
        "failed": "cancel",
        "pending": "pause_circle",
        "active": "radio_button_checked",
        "inactive": "radio_button_unchecked",
    }

    icon_name = status_icons.get(status_lower)
    icon_html = (
        f'<span class="material-symbols-outlined" aria-hidden="true">{html.escape(icon_name)}</span>'
        if icon_name
        else ""
    )
    display_text = status.replace("_", " ").title()

    return f'<span class="status-badge {badge_class}">{icon_html}{display_text}</span>'


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
    "stream_trace_items": list,
    "stream_tool_index": dict,
    "stream_trace_expanded": lambda: False,
    # Provider/model UI state
    "model_config_options_cache": dict,
    "model_config_options_last_fetch": lambda: None,
    "model_config_options_error": lambda: None,
    "model_config_options_needs_form_sync": lambda: False,
    # Planning mode state
    "planning_status": lambda: None,
    "task_plans_list": list,
    "planning_generate_input": str,
    "planning_manual_input": str,
    "api_cache_version": lambda: 0,
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
st.markdown(IMAGE_LIGHTBOX_JS, unsafe_allow_html=True)
initialize_session_state()

# ── localStorage session-persistence bridge ──────────────────────────────────
# Allows the auth token to survive F5 / browser refresh.
import contextlib  # noqa: E402
import json as _json  # noqa: E402

import streamlit.components.v1 as _stc_ls  # noqa: E402

# Step 1: flush any pending localStorage write/clear from the previous run.
_ls_op = st.session_state.get("_ls_op")
if _ls_op is not None:
    st.session_state._ls_op = None
    if isinstance(_ls_op, dict):  # save
        _tok = _json.dumps(_ls_op.get("token", ""))
        _uid = _json.dumps(_ls_op.get("uid", ""))
        _stc_ls.html(
            f"<script>try{{localStorage.setItem('cbtoken',{_tok});"
            f"localStorage.setItem('cbuid',{_uid});}}catch(e){{}}</script>",
            height=0,
        )
    elif _ls_op == "clear":  # logout
        _stc_ls.html(
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
        _stc_ls.html(
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

        if stop_status == "cancelled" and result_data.get("message"):
            # Backend persisted a partial message – append to local state
            bot_msg = result_data["message"]
            st.session_state.messages.append(bot_msg)
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

    _clear_inflight_state()
    st.rerun()


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
    # Clear in-flight streaming state
    _clear_inflight_state()


_MANAGER_PAGE_SIZE = 100


def open_conversation_manager() -> None:
    """Open the conversation manager dialog on the next render."""
    st.session_state.conversation_manager_visible = True
    st.session_state.show_conversation_manager = True
    st.session_state[CONVERSATION_MANAGER_DIALOG_KEY] = True
    # Reset lazy-load state so the dialog fetches fresh data on open
    st.session_state.pop("manager_conversations", None)
    st.session_state.pop("manager_conv_page", None)
    st.session_state.pop("manager_conv_has_more", None)
    st.session_state.pop("manager_conv_total", None)


def close_conversation_manager() -> None:
    """Close the conversation manager dialog and prevent reopening on rerun."""
    st.session_state.conversation_manager_visible = False
    st.session_state.show_conversation_manager = False
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


def make_api_request(method: str, endpoint: str, data: dict | None = None) -> dict:
    method = method.strip().upper()
    auth_token = st.session_state.get("auth_token")
    response_data: dict[str, Any]
    st.session_state["_last_api_error_message"] = None

    try:
        if method == "GET" and data is None:
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
            st.session_state.auth_token = None
            st.session_state.current_user_id = None
            st.session_state.current_user_profile = None
            st.session_state.show_login = True
            st.toast("Please log in", icon=":material/lock:")
        else:
            st.toast(f"{error_message}", icon=":material/cancel:")
        return {}

    if method in {"POST", "PUT", "PATCH", "DELETE"}:
        st.session_state.api_cache_version = int(st.session_state.get("api_cache_version", 0)) + 1

    return response_data


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
        st.toast(f"HTTP error {http_error.response.status_code}", icon=":material/cancel:")
        yield {"type": "error", "error": f"HTTP {http_error.response.status_code}"}
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

    pending = st.session_state.get("pending_image_attachments", [])
    existing_data = {item["data"] for item in pending}

    new_items: list[dict[str, str]] = []
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
        st.toast(f"{_MAX_IMAGE_ATTACHMENTS} images", icon=":material/warning:")
        return

    if len(new_items) > remaining:
        st.toast("Some images ignored", icon=":material/warning:")

    pending.extend(new_items[:remaining])
    st.session_state.pending_image_attachments = pending


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


def get_tool_details(tool_name: str) -> dict[str, Any] | None:
    """Fetch detailed information about a specific tool"""
    response = make_api_request("GET", f"/mcp/tools/{tool_name}")
    return response.get("data") if response else None


def execute_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
    """Execute an MCP tool with provided arguments"""
    response = make_api_request("POST", f"/mcp/tools/{tool_name}/execute", {"arguments": arguments})
    return response.get("data") if response else None


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

    chart_rows: list[dict[str, Any]] = []
    for label_index, label in enumerate(labels):
        row: dict[str, Any] = {"label": label}
        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            name = str(dataset.get("label") or f"Series {len(row)}")
            data = dataset.get("data")
            if isinstance(data, list) and label_index < len(data):
                row[name] = data[label_index]
        chart_rows.append(row)

    if not chart_rows:
        st.caption("Chart result contained no plottable values.")
        return True

    chart_type = str(structured.get("chart_type") or structured.get("chartType") or "line").lower()
    chart_data = {
        row["label"]: {k: v for k, v in row.items() if k != "label"} for row in chart_rows
    }
    if chart_type == "bar":
        st.bar_chart(chart_data)
    elif chart_type == "area":
        st.area_chart(chart_data)
    else:
        st.line_chart(chart_data)

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

    if render_type == "mcp_app":
        rendered = _render_mcp_app_tool_result(render)
    elif render_type == "table":
        rendered = _render_table_tool_result(render)
    elif render_type == "chart":
        rendered = _render_chart_tool_result(render)
    elif render_type in {"resource", "image"}:
        rendered = _render_resource_tool_result(render)
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
    """Fetch the locally installed server repo skills for the demo."""
    return list_repo_skills_for_demo()


def get_skill_detail(name: str) -> dict[str, Any] | None:
    """Fetch full detail for one installed server repo skill."""
    return get_repo_skill_detail_for_demo(name)


def reload_skills() -> dict[str, Any] | None:
    """Rescan repo skills for this local demo process."""
    return reload_repo_skills_for_demo()


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
                        st.session_state.auth_token = auth_response["data"]["accessToken"]
                        st.session_state.current_user_id = auth_response["data"]["userId"]
                        st.session_state.current_user_profile = None
                        st.session_state.active_view = "chat"
                        st.session_state.show_login = False
                        st.session_state._ls_op = {
                            "token": auth_response["data"]["accessToken"],
                            "uid": auth_response["data"]["userId"],
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
                                st.session_state.auth_token = auth_response["data"]["accessToken"]
                                st.session_state.current_user_id = auth_response["data"]["userId"]
                                st.session_state.current_user_profile = None
                                st.session_state.active_view = "chat"
                                st.session_state.show_login = False
                                st.session_state._ls_op = {
                                    "token": auth_response["data"]["accessToken"],
                                    "uid": auth_response["data"]["userId"],
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
                    st.session_state.current_user_id = None
                    st.session_state.current_user_profile = None
                    st.session_state.current_conversation_id = None
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


def _guess_extension(mime: str | None) -> str:
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


def _build_attachment_thumbnail(href: str, name: str, *, download: str | None = None) -> str:
    """Create HTML anchor for a single attachment thumbnail."""
    escaped_href = html.escape(str(href), quote=True)
    escaped_name = html.escape(str(name), quote=True)
    download_attr = f' download="{html.escape(str(download), quote=True)}"' if download else ""
    return (
        f'<a class="attachment-thumb-link" href="{escaped_href}" target="_blank" '
        f'rel="noopener noreferrer" aria-label="Open {escaped_name}" '
        f'title="{escaped_name}"{download_attr}>'
        f'<div class="attachment-thumb">'
        f'<img src="{escaped_href}" alt="{escaped_name}" loading="lazy" />'
        f"</div></a>"
    )


def render_attachment_gallery(attachments: list[dict[str, str]], *, align: str) -> None:
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
        mime = attachment.get("mime", "image/png")

        if isinstance(data_b64, str):
            data_b64 = data_b64.strip()
            if data_b64.startswith("data:"):
                header, _, payload = data_b64.partition(",")
                if payload:
                    data_b64 = payload.strip()
                    header_mime = header[5:].split(";")[0].strip() if header else ""
                    if "/" in header_mime:
                        mime = header_mime

        if isinstance(url_value, str):
            url_value = url_value.strip()
            if url_value.startswith("data:"):
                header, _, payload = url_value.partition(",")
                if payload:
                    data_b64 = payload.strip()
                    url_value = None
                    header_mime = header[5:].split(";")[0].strip() if header else ""
                    if "/" in header_mime:
                        mime = header_mime

        if isinstance(data_b64, str) and data_b64:
            valid_attachments.append(
                {
                    "name": name,
                    "data": data_b64,
                    "mime": mime,
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

    import streamlit.components.v1 as _stc

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
    _stc.html(patched, height=520, scrolling=True)

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
            "widget_type": str(widget.get("widget_type") or "widget"),
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
      .lw-badge,.lw-pill{display:inline-flex;align-items:center;gap:6px;padding:7px 11px;border-radius:999px;font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase}
      .lw-badge{white-space:nowrap;align-self:flex-start}
      .lw-badge[data-state="active"]{background:rgba(16,185,129,.14);color:#047857}
      .lw-badge[data-state="closed"]{background:rgba(148,163,184,.18);color:#475569}
      .lw-meta{display:flex;flex-wrap:wrap;gap:10px;padding:14px 20px 0}
      .lw-pill{background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8}
      .lw-pill[data-conn="connected"]{background:#ecfdf5;border-color:#a7f3d0;color:#047857}
      .lw-pill[data-conn="connecting"],.lw-pill[data-conn="reconnecting"]{background:#fff7ed;border-color:#fed7aa;color:#c2410c}
      .lw-pill[data-conn="error"],.lw-pill[data-conn="disconnected"],.lw-pill[data-conn="auth"]{background:#fef2f2;border-color:#fecaca;color:#b91c1c}
      .lw-error{margin:14px 20px 0;padding:12px 14px;border-radius:14px;background:#fff1f2;border:1px solid #fecdd3;color:#9f1239;font-size:13px}
      .lw-body{flex:1;min-height:0;overflow:auto;padding:18px 20px 22px;background:linear-gradient(180deg,rgba(255,255,255,.78),rgba(241,245,249,.86));scrollbar-gutter:stable}
      .lw-body::-webkit-scrollbar{width:10px}
      .lw-body::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:999px}
      .lw-empty{display:flex;align-items:center;justify-content:center;min-height:220px;border:1px dashed #bfdbfe;border-radius:18px;color:#475569;background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(239,246,255,.72));padding:22px;text-align:center}
      .lw-table-wrap{overflow:auto;border:1px solid #dbeafe;border-radius:18px;background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(248,250,252,.96))}
      .lw-table{width:100%;border-collapse:separate;border-spacing:0}
      .lw-table th,.lw-table td{padding:13px 15px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top;font-size:13px}
      .lw-table th{position:sticky;top:0;background:rgba(248,250,252,.96);backdrop-filter:blur(10px);color:#0f172a;font-weight:800;z-index:1}
      .lw-table tbody tr:nth-child(even) td{background:rgba(248,250,252,.66)}
      .lw-table tr:last-child td{border-bottom:none}
      .lw-list{display:grid;gap:12px}
      .lw-item{border:1px solid #dbeafe;border-radius:16px;padding:14px 15px;background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(248,250,252,.92));box-shadow:0 8px 24px rgba(59,130,246,.08)}
      .lw-item[data-clickable="true"]{cursor:pointer;transition:border-color .15s ease,transform .15s ease,box-shadow .15s ease}
      .lw-item[data-clickable="true"]:hover{border-color:#60a5fa;transform:translateY(-1px);box-shadow:0 12px 28px rgba(37,99,235,.14)}
      .lw-item[data-selected="true"]{border-color:#3b82f6;background:linear-gradient(180deg,rgba(239,246,255,.96),#fff)}
      .lw-item-title{font-size:14px;font-weight:800;color:#0f172a}
      .lw-item-desc{margin-top:7px;font-size:12px;color:#64748b;line-height:1.55}
      .lw-form{display:grid;gap:16px}
      .lw-field{display:grid;gap:8px}
      .lw-field label{font-size:13px;font-weight:700;color:#0f172a}
      .lw-field input,.lw-field select,.lw-field textarea{width:100%;border:1px solid #cbd5e1;border-radius:14px;padding:11px 12px;background:#fff;color:#0f172a;font:inherit;box-shadow:inset 0 1px 0 rgba(255,255,255,.7)}
      .lw-field textarea{min-height:120px;resize:vertical}
      .lw-field-help,.lw-note{font-size:12px;color:#64748b;line-height:1.5}
      .lw-check{display:flex;align-items:center;gap:10px}
      .lw-chart,.lw-panel,.lw-metric{border:1px solid #dbeafe;border-radius:18px;background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(248,250,252,.95));box-shadow:0 16px 30px rgba(37,99,235,.08)}
      .lw-chart{padding:16px}
      .lw-chart-top{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:14px}
      .lw-chart-type{display:inline-flex;padding:5px 9px;border-radius:999px;background:#eff6ff;color:#1d4ed8;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase}
      .lw-chart-title{font-size:14px;font-weight:800;color:#0f172a}
      .lw-chart-sub{margin-top:6px;font-size:12px;color:#64748b}
      .lw-chart-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:0 0 16px}
      .lw-stat{padding:12px 13px;border-radius:16px;background:linear-gradient(180deg,rgba(239,246,255,.92),rgba(255,255,255,.96));border:1px solid rgba(191,219,254,.9)}
      .lw-stat-label{font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
      .lw-stat-value{margin-top:6px;font-size:20px;font-weight:800;color:#0f172a}
      .lw-legend{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 14px}
      .lw-legend-item{display:inline-flex;align-items:center;gap:7px;color:#334155;font-size:12px;font-weight:600}
      .lw-dot{width:10px;height:10px;border-radius:999px;box-shadow:0 0 0 3px rgba(255,255,255,.7)}
      .lw-chart-surface{padding:12px;border-radius:16px;background:linear-gradient(180deg,rgba(248,250,252,.98),rgba(239,246,255,.86));border:1px solid rgba(219,234,254,.95)}
      .lw-html-shell{display:grid;gap:12px}
      .lw-html-caption{font-size:12px;color:#475569;line-height:1.55}
      .lw-html-frame{width:100%;border:none;border-radius:18px;background:#fff;box-shadow:0 18px 30px rgba(15,23,42,.08)}
      .lw-chart-svg{display:block;width:100%;height:auto}
      .lw-grid-line{stroke:#dbeafe;stroke-dasharray:4 6}
      .lw-axis-line{stroke:#cbd5e1}
      .lw-axis-text{fill:#64748b;font-size:11px;font-weight:600}
      .lw-line-path{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}
      .lw-area-path{opacity:.14}
      .lw-point{stroke:#fff;stroke-width:2}
      .lw-donut{display:grid;grid-template-columns:minmax(180px,240px) minmax(0,1fr);gap:20px;align-items:center}
      .lw-ring{position:relative;aspect-ratio:1;border-radius:999px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.8),0 18px 30px rgba(15,23,42,.08)}
      .lw-ring-hole{position:absolute;inset:22%;display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:999px;background:linear-gradient(180deg,#fff,#eff6ff);border:1px solid rgba(219,234,254,.9);text-align:center}
      .lw-ring-hole strong{font-size:24px;font-weight:800;color:#0f172a}
      .lw-ring-hole span{margin-top:4px;font-size:12px;color:#64748b}
      .lw-donut-legend{display:grid;gap:10px}
      .lw-donut-item{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:10px;padding:10px 12px;border-radius:14px;background:rgba(255,255,255,.78);border:1px solid rgba(226,232,240,.92)}
      .lw-donut-value{font-size:12px;font-weight:700;color:#0f172a}
      .lw-donut-share{font-size:11px;color:#64748b}
      .lw-dash{display:grid;gap:16px}
      .lw-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
      .lw-metric,.lw-panel{padding:15px}
      .lw-metric-title{font-size:12px;color:#64748b;margin-bottom:8px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
      .lw-metric-value{font-size:28px;font-weight:800;color:#0f172a}
      .lw-metric-delta{margin-top:8px;font-size:12px;color:#2563eb}
      .lw-panel-title{font-size:14px;font-weight:800;color:#0f172a;margin-bottom:12px}
      .lw-fallback{display:grid;gap:12px}
      .lw-json{margin:0;padding:14px;border-radius:14px;background:#0f172a;color:#e2e8f0;font-size:12px;line-height:1.55;overflow:auto;white-space:pre-wrap;word-break:break-word}
      .lw-raw{border:1px dashed #cbd5e1;border-radius:14px;background:rgba(248,250,252,.82);padding:12px}
      .lw-raw summary{cursor:pointer;font-size:12px;font-weight:700;color:#475569}
      .lw-muted{color:#94a3b8}
      .lw-controls{display:grid;gap:12px;margin:0 0 14px;padding:14px;border:1px solid #dbeafe;border-radius:18px;background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(239,246,255,.72));box-shadow:0 14px 28px rgba(37,99,235,.08)}
      .lw-controls-head,.lw-table-toolbar,.lw-chart-toolbar{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
      .lw-controls-title{font-size:12px;font-weight:800;color:#1d4ed8;letter-spacing:.08em;text-transform:uppercase}
      .lw-chip-wrap{display:flex;gap:8px;flex-wrap:wrap}
      .lw-chip{display:inline-flex;align-items:center;padding:6px 10px;border-radius:999px;background:#dbeafe;color:#1e3a8a;font-size:11px;font-weight:800;letter-spacing:.04em;text-transform:uppercase}
      .lw-control-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
      .lw-control{display:grid;gap:7px}
      .lw-control label{font-size:12px;font-weight:700;color:#0f172a}
      .lw-control select,.lw-control input{width:100%;border:1px solid #cbd5e1;border-radius:12px;padding:10px 11px;background:#fff;color:#0f172a;font:inherit}
      .lw-segment{display:flex;flex-wrap:wrap;gap:8px}
      .lw-segment button,.lw-legend-item-button,.lw-sort-button{border:1px solid #bfdbfe;background:#eff6ff;color:#1d4ed8;border-radius:999px;padding:8px 11px;font:inherit;font-size:12px;font-weight:700;cursor:pointer;transition:transform .15s ease,box-shadow .15s ease}
      .lw-segment button[data-active="true"]{background:linear-gradient(135deg,#1d4ed8,#2563eb);color:#fff;border-color:transparent;box-shadow:0 10px 24px rgba(37,99,235,.22)}
      .lw-segment button:disabled,.lw-legend-item-button:disabled,.lw-sort-button:disabled{opacity:.55;cursor:not-allowed;transform:none;box-shadow:none}
      .lw-legend-item-button{display:inline-flex;align-items:center;gap:7px;background:#fff;border-color:#dbeafe;color:#1e3a8a}
      .lw-legend-item-button[data-hidden="true"]{opacity:.55;text-decoration:line-through}
      .lw-search{display:flex;align-items:center;gap:8px;min-width:min(100%,280px);padding:10px 12px;border-radius:14px;border:1px solid #dbeafe;background:rgba(255,255,255,.92)}
      .lw-search input{border:none;outline:none;flex:1;min-width:0;background:transparent;padding:0}
      .lw-meta-note{font-size:12px;color:#64748b;font-weight:600}
      .lw-sort-button{width:100%;display:inline-flex;align-items:center;justify-content:space-between;gap:10px;padding:0;border:none;background:transparent;color:inherit;border-radius:0}
      .lw-sort-button[data-active="true"]{color:#1d4ed8}
      @media (max-width:760px){.lw-head{flex-direction:column;align-items:stretch}.lw-donut{grid-template-columns:1fr}.lw-ring{max-width:240px;justify-self:center}}
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
      const palette = ["#2563eb","#0f766e","#ea580c","#7c3aed","#dc2626","#0891b2"];
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
      let patchTimer = null;
      let pendingPatch = null;
      let destroyed = false;
      function isObj(v){return !!v && typeof v === "object" && !Array.isArray(v);}
      function esc(v){return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));}
      function label(v){return (String(v ?? "").replace(/[_-]+/g," ").replace(/\\s+/g," ").trim().replace(/\\b\\w/g, (m) => m.toUpperCase())) || "Widget";}
      function numeric(v){const n = Number(v); return Number.isFinite(n) ? n : null;}
      function formatNumber(value){
        const n = numeric(value);
        if(n === null) return String(value ?? "-");
        const abs = Math.abs(n);
        const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
        return new Intl.NumberFormat(undefined, {maximumFractionDigits: digits}).format(n);
      }
      function truncateText(value, maxLength = 18){
        const text = String(value ?? "");
        return text.length > maxLength ? `${text.slice(0, Math.max(1, maxLength - 1))}…` : text;
      }
      function colorWithAlpha(color, alpha){
        const raw = String(color || "").trim();
        const hex = raw.startsWith("#") ? raw.slice(1) : raw;
        if(/^[0-9a-fA-F]{3}$/.test(hex)){
          const chars = hex.split("");
          const [r, g, b] = chars.map((item) => parseInt(item + item, 16));
          return `rgba(${r},${g},${b},${alpha})`;
        }
        if(/^[0-9a-fA-F]{6}$/.test(hex)){
          const r = parseInt(hex.slice(0, 2), 16);
          const g = parseInt(hex.slice(2, 4), 16);
          const b = parseInt(hex.slice(4, 6), 16);
          return `rgba(${r},${g},${b},${alpha})`;
        }
        return raw || `rgba(37,99,235,${alpha})`;
      }
      function val(v){if(v === null || v === undefined || v === "") return '<span class="lw-muted">-</span>'; if(typeof v === "object") return `<code>${esc(JSON.stringify(v))}</code>`; return esc(typeof v === "number" ? formatNumber(v) : v);}
      function resize(){document.body.style.margin="0";}
      function showError(msg){el.error.hidden=!msg;el.error.textContent=msg || "";resize();}
      function inputValue(target, type){
        if(type === "checkbox") return Boolean(target.checked);
        if(type === "number" || type === "range"){
          const raw = String(target.value ?? "").trim();
          return raw === "" ? null : Number(raw);
        }
        return target.value;
      }
      function meta(){
        const widgetType = label(cfg.widget.widget_type);
        el.title.textContent = cfg.widget.title || widgetType;
        el.sub.textContent = `${widgetType} · ${cfg.widget.widget_id || "pending"}`;
        el.status.textContent = state.status;
        el.status.dataset.state = state.status;
        el.version.textContent = `Version ${state.version || 0}`;
        const connLabel = {connected:"Connected",connecting:"Connecting",reconnecting:"Reconnecting",disconnected:"Disconnected",error:"Connection Error",auth:"Sign In Required"}[state.conn] || "Connecting";
        el.conn.textContent = connLabel;
        el.conn.dataset.conn = state.conn;
      }
      function renderJson(data){return `<pre class="lw-json">${esc(JSON.stringify(data, null, 2))}</pre>`;}
      function renderRawDisclosure(data){return `<details class="lw-raw"><summary>Inspect raw data</summary>${renderJson(data)}</details>`;}
      function normalizeOption(option, index){
        if(isObj(option)){
          const value = option.value ?? option.id ?? option.key ?? option.label ?? `${index}`;
          return {value: String(value), label: String(option.label ?? option.title ?? option.name ?? value)};
        }
        return {value: String(option ?? index), label: String(option ?? index)};
      }
      function deepClone(value){
        if(Array.isArray(value)) return value.map((item) => deepClone(item));
        if(isObj(value)) return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, deepClone(item)]));
        return value;
      }
      function deepMerge(base, override){
        if(!isObj(base) || !isObj(override)) return deepClone(override);
        const merged = deepClone(base);
        Object.entries(override).forEach(([key, value]) => {
          merged[key] = isObj(value) && isObj(merged[key]) ? deepMerge(merged[key], value) : deepClone(value);
        });
        return merged;
      }
      function normalizeControls(data){
        if(!isObj(data) || !Array.isArray(data.controls)) return [];
        return data.controls.filter((item) => isObj(item)).map((control, index) => ({
          key: String(control.key || control.id || control.name || `control_${index}`),
          label: String(control.label || control.title || label(control.key || control.id || control.name || `Control ${index + 1}`)),
          type: String(control.type || "select").toLowerCase(),
          options: Array.isArray(control.options) ? control.options.map(normalizeOption) : [],
          min: numeric(control.min),
          max: numeric(control.max),
          step: numeric(control.step),
          help: String(control.help || control.helpText || control.description || ""),
          value: control.value,
          defaultValue: control.default,
        }));
      }
      function getControlValues(data, controls){
        const explicit = isObj(data.control_values) ? data.control_values : isObj(data.args) ? data.args : {};
        const values = {...explicit};
        controls.forEach((control) => {
          if(values[control.key] !== undefined) return;
          if(control.value !== undefined) values[control.key] = control.value;
          else if(control.defaultValue !== undefined) values[control.key] = control.defaultValue;
          else if(control.options[0]) values[control.key] = control.options[0].value;
          else if(control.type === "checkbox" || control.type === "toggle") values[control.key] = false;
        });
        return values;
      }
      function applyControlValues(data, controlValues){
        const next = isObj(data) ? {...data} : {};
        next.control_values = {...controlValues};
        next.args = {...controlValues};
        if(Array.isArray(next.controls)){
          next.controls = next.controls.map((control, index) => {
            const key = String(control.key || control.id || control.name || `control_${index}`);
            return {...control, value: controlValues[key]};
          });
        }
        return next;
      }
      function parseControlSelector(raw){
        const result = {};
        String(raw || "").replace(/[|,]+/g, "&").split("&").forEach((pair) => {
          const [key, ...rest] = pair.split("=");
          const safeKey = String(key || "").trim();
          if(!safeKey) return;
          result[safeKey] = rest.join("=").trim();
        });
        return result;
      }
      function matchesControlValues(expected, actual){
        return isObj(expected) && Object.entries(expected).every(([key, value]) => String(actual[key]) === String(value));
      }
      function resolveInteractiveData(data){
        if(!isObj(data)) return {payload: data, controls: [], controlValues: {}};
        const controls = normalizeControls(data);
        const controlValues = getControlValues(data, controls);
        let variantState = null;
        if(isObj(data.views)){
          const encoded = Object.keys(controlValues).sort().map((key) => `${key}=${String(controlValues[key])}`).join("&");
          variantState = data.views[encoded] || null;
          if(!variantState && controls.length === 1) variantState = data.views[String(controlValues[controls[0].key])] || null;
          if(!variantState){
            Object.entries(data.views).some(([key, value]) => {
              if(key === "default" || key === "__default") return false;
              if(matchesControlValues(parseControlSelector(key), controlValues)){
                variantState = value;
                return true;
              }
              return false;
            });
          }
          if(!variantState) variantState = data.views.default || data.views.__default || null;
        }
        if(!variantState && Array.isArray(data.variants)){
          data.variants.some((variant) => {
            if(!isObj(variant)) return false;
            const expected = variant.match || variant.when || variant.args;
            if(!matchesControlValues(expected, controlValues)) return false;
            variantState = variant.state || variant.view || variant.data || variant.payload || null;
            return Boolean(variantState);
          });
        }
        const base = deepClone(data);
        delete base.views;
        delete base.variants;
        delete base.default_view;
        return {payload: variantState && isObj(variantState) ? deepMerge(base, variantState) : base, controls, controlValues};
      }
      function renderControls(controls, controlValues, readOnly){
        if(!controls.length) return "";
        const chips = Object.entries(controlValues || {}).map(([key, value]) => `<span class="lw-chip">${esc(label(key))}: ${esc(typeof value === "number" ? formatNumber(value) : value)}</span>`).join("");
        const items = controls.map((control) => {
          const currentValue = controlValues[control.key];
          const disabled = readOnly ? " disabled" : "";
          if(control.type === "segmented" || control.type === "radio" || control.type === "pill"){
            return `<div class="lw-control"><label>${esc(control.label)}</label><div class="lw-segment">${control.options.map((option) => `<button type="button" data-control-option="true" data-control-key="${esc(control.key)}" data-control-value="${esc(option.value)}" data-active="${String(option.value) === String(currentValue)}"${disabled}>${esc(option.label)}</button>`).join("")}</div>${control.help ? `<div class="lw-note">${esc(control.help)}</div>` : ""}</div>`;
          }
          if(control.type === "checkbox" || control.type === "toggle"){
            return `<div class="lw-control"><label>${esc(control.label)}</label><label class="lw-check"><input type="checkbox" data-control-key="${esc(control.key)}" data-control-type="checkbox"${currentValue ? " checked" : ""}${disabled}><span>${esc(control.help || "Enabled")}</span></label></div>`;
          }
          if(control.type === "range" || control.type === "slider"){
            const min = control.min ?? 0;
            const max = control.max ?? 100;
            const step = control.step ?? 1;
            const value = numeric(currentValue) ?? min;
            return `<div class="lw-control"><label>${esc(control.label)} · ${esc(formatNumber(value))}</label><input type="range" min="${esc(min)}" max="${esc(max)}" step="${esc(step)}" value="${esc(value)}" data-control-key="${esc(control.key)}" data-control-type="range"${disabled}>${control.help ? `<div class="lw-note">${esc(control.help)}</div>` : ""}</div>`;
          }
          if(control.options.length){
            return `<div class="lw-control"><label>${esc(control.label)}</label><select data-control-key="${esc(control.key)}" data-control-type="select"${disabled}>${control.options.map((option) => `<option value="${esc(option.value)}"${String(option.value) === String(currentValue) ? " selected" : ""}>${esc(option.label)}</option>`).join("")}</select>${control.help ? `<div class="lw-note">${esc(control.help)}</div>` : ""}</div>`;
          }
          return `<div class="lw-control"><label>${esc(control.label)}</label><input type="${esc(control.type === "number" ? "number" : "text")}" value="${esc(currentValue ?? "")}" data-control-key="${esc(control.key)}" data-control-type="${esc(control.type === "number" ? "number" : "text")}"${disabled}>${control.help ? `<div class="lw-note">${esc(control.help)}</div>` : ""}</div>`;
        }).join("");
        return `<div class="lw-controls"><div class="lw-controls-head"><div class="lw-controls-title">Interactive Controls</div>${chips ? `<div class="lw-chip-wrap">${chips}</div>` : ""}</div><div class="lw-control-grid">${items}</div></div>`;
      }
      function tableUi(data){
        return isObj(data.table_ui) ? data.table_ui : {};
      }
      function chartUi(data){
        return isObj(data.chart_ui) ? data.chart_ui : {};
      }
      function queuePatch(patch, delay = 150){
        if(!patch || state.status === "closed") return;
        const basePatch = isObj(pendingPatch) ? pendingPatch : {};
        pendingPatch = {...basePatch, ...patch};
        window.clearTimeout(patchTimer);
        patchTimer = window.setTimeout(() => {
          const payload = pendingPatch;
          pendingPatch = null;
          sendPatch(payload);
        }, Math.max(0, delay));
      }
      function setTableUi(patch){
        const base = isObj(state.data) ? state.data : {};
        const nextTableUi = {...tableUi(base), ...patch};
        state.data = {...base, table_ui: nextTableUi};
        renderBody();
        queuePatch({table_ui: nextTableUi});
      }
      function setChartUi(patch, delay = 120){
        const base = isObj(state.data) ? state.data : {};
        const nextChartUi = {...chartUi(base), ...patch};
        state.data = {...base, chart_ui: nextChartUi};
        renderBody();
        queuePatch({chart_ui: nextChartUi}, delay);
      }
      function setControlValue(key, value, delay = 150){
        const base = isObj(state.data) ? state.data : {};
        const controls = normalizeControls(base);
        if(!controls.length) return;
        const nextValues = {...getControlValues(base, controls), [key]: value};
        const next = applyControlValues(base, nextValues);
        state.data = next;
        renderBody();
        const patch = {
          control_values: next.control_values || {},
          args: next.args || {},
        };
        if(Array.isArray(next.controls)) patch.controls = next.controls;
        queuePatch(patch, delay);
      }
      function toggleChartSeries(seriesLabel){
        const base = isObj(state.data) ? state.data : {};
        const currentHidden = Array.isArray(chartUi(base).hidden_series) ? chartUi(base).hidden_series.map((item) => String(item)) : [];
        const nextHidden = currentHidden.includes(seriesLabel)
          ? currentHidden.filter((item) => item !== seriesLabel)
          : [...currentHidden, seriesLabel];
        setChartUi({hidden_series: nextHidden});
      }
      function normColumns(data){
        let raw = Array.isArray(data.columns) ? data.columns : Array.isArray(data.cols) ? data.cols : [];
        if(!raw.length && Array.isArray(data.rows) && data.rows[0] && isObj(data.rows[0])) raw = Object.keys(data.rows[0]);
        if(!raw.length && Array.isArray(data.rows) && Array.isArray(data.rows[0])) raw = Array.from({length: data.rows[0].length}, (_, idx) => `column_${idx + 1}`);
        if(!raw.length) raw = ["value"];
        return raw.map((col, i) => typeof col === "string" ? {key: col, label: label(col)} : {key: String(col.key || col.id || col.name || col.label || `column_${i}`), label: String(col.label || col.title || col.name || col.key || `Column ${i + 1}`)});
      }
      function normRows(data, columns){
        const rows = Array.isArray(data.rows) ? data.rows : [];
        return rows.map((row) => {
          if(Array.isArray(row)){const mapped={};columns.forEach((col, idx) => mapped[col.key]=row[idx]);return mapped;}
          if(isObj(row)) return row;
          return {value: row};
        });
      }
      function renderTable(data, readOnly = false){
        const columns = normColumns(data);
        const rows = normRows(data, columns);
        if(!rows.length) return '<div class="lw-empty">No table rows available yet.</div>';
        const ui = tableUi(data);
        const disabled = readOnly ? " disabled" : "";
        const query = String(ui.query || "").trim().toLowerCase();
        let filteredRows = rows.filter((row) => !query || columns.some((col) => String(row[col.key] ?? "").toLowerCase().includes(query)));
        const sortKey = String(ui.sort_by || "");
        const sortDirection = String(ui.sort_dir || "asc");
        if(sortKey){
          filteredRows = [...filteredRows].sort((left, right) => {
            const leftValue = left[sortKey];
            const rightValue = right[sortKey];
            const leftNumeric = numeric(leftValue);
            const rightNumeric = numeric(rightValue);
            let comparison = 0;
            if(leftNumeric !== null && rightNumeric !== null) comparison = leftNumeric - rightNumeric;
            else comparison = String(leftValue ?? "").localeCompare(String(rightValue ?? ""), undefined, {numeric: true, sensitivity: "base"});
            return sortDirection === "desc" ? comparison * -1 : comparison;
          });
        }
        const head = columns.map((col) => {
          const active = sortKey === col.key;
          const symbol = active ? (sortDirection === "desc" ? "↓" : "↑") : "↕";
          return `<th><button type="button" class="lw-sort-button" data-sort-key="${esc(col.key)}" data-active="${active}"${disabled}><span>${esc(col.label)}</span><span>${esc(symbol)}</span></button></th>`;
        }).join("");
        const body = filteredRows.map((row) => `<tr>${columns.map((col) => `<td>${val(row[col.key])}</td>`).join("")}</tr>`).join("");
        return `<div class="lw-table-shell"><div class="lw-table-toolbar"><label class="lw-search"><span>Search</span><input type="text" value="${esc(ui.query || "")}" data-table-search="true" placeholder="Filter rows"${disabled}></label><div class="lw-meta-note">${esc(filteredRows.length)} / ${esc(rows.length)} rows</div></div><div class="lw-table-wrap"><table class="lw-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div></div>`;
      }
      function renderList(data, readOnly){
        const items = Array.isArray(data.items) ? data.items : Array.isArray(data.options) ? data.options : [];
        if(!items.length) return '<div class="lw-empty">No list items available yet.</div>';
        return `<div class="lw-list">${items.map((item, i) => {
          const x = isObj(item) ? item : {label: item};
          const key = String(x.id ?? x.key ?? x.value ?? i);
          const selected = String(data.selection ?? "") === key;
          const title = String(x.label ?? x.title ?? x.name ?? x.value ?? `Item ${i + 1}`);
          const desc = x.description ?? x.subtitle ?? x.summary ?? "";
          const click = readOnly ? "" : ` data-action="select" data-selection="${esc(key)}"`;
          return `<div class="lw-item" data-clickable="${!readOnly}" data-selected="${selected}"${click}><div class="lw-item-title">${esc(title)}</div>${desc ? `<div class="lw-item-desc">${esc(desc)}</div>` : ""}</div>`;
        }).join("")}</div>`;
      }
      function normFields(data){
        if(Array.isArray(data.fields) && data.fields.length) return data.fields.filter((field) => isObj(field));
        if(isObj(data.values)) return Object.keys(data.values).map((key) => ({ key, label: label(key), type: typeof data.values[key] === "boolean" ? "checkbox" : typeof data.values[key] === "number" ? "number" : "text" }));
        return [];
      }
      function renderForm(data, readOnly){
        const fields = normFields(data);
        const values = isObj(data.values) ? data.values : {};
        if(!fields.length) return '<div class="lw-empty">No form fields available yet.</div>';
        return `<div class="lw-form">${fields.map((field) => {
          const key = String(field.key || field.id || field.name || "");
          if(!key) return "";
          const title = String(field.label || field.title || label(key));
          const help = field.help_text || field.helpText || field.description || "";
          const type = String(field.type || "text").toLowerCase();
          const value = values[key];
          const disabled = readOnly ? " disabled" : "";
          if(type === "checkbox" || type === "boolean"){
            return `<div class="lw-field"><label>${esc(title)}</label><label class="lw-check"><input type="checkbox" data-field-key="${esc(key)}" data-field-type="checkbox"${value ? " checked" : ""}${disabled}><span>${esc(help || "Toggle")}</span></label>${readOnly ? '<div class="lw-note">Read-only because the widget is closed.</div>' : ""}</div>`;
          }
          if(type === "select"){
            const options = Array.isArray(field.options) ? field.options : [];
            const optionsHtml = options.map((option) => {
              const item = isObj(option) ? option : {value: option, label: option};
              const optionValue = String(item.value ?? item.id ?? item.label ?? "");
              const optionLabel = String(item.label ?? item.title ?? optionValue);
              return `<option value="${esc(optionValue)}"${String(value ?? "") === optionValue ? " selected" : ""}>${esc(optionLabel)}</option>`;
            }).join("");
            return `<div class="lw-field"><label>${esc(title)}</label><select data-field-key="${esc(key)}" data-field-type="select"${disabled}>${optionsHtml}</select>${help ? `<div class="lw-field-help">${esc(help)}</div>` : ""}${readOnly ? '<div class="lw-note">Read-only because the widget is closed.</div>' : ""}</div>`;
          }
          if(type === "textarea"){
            return `<div class="lw-field"><label>${esc(title)}</label><textarea data-field-key="${esc(key)}" data-field-type="textarea"${disabled}>${esc(value ?? "")}</textarea>${help ? `<div class="lw-field-help">${esc(help)}</div>` : ""}${readOnly ? '<div class="lw-note">Read-only because the widget is closed.</div>' : ""}</div>`;
          }
          const inputType = type === "number" ? "number" : "text";
          return `<div class="lw-field"><label>${esc(title)}</label><input type="${esc(inputType)}" value="${esc(value ?? "")}" data-field-key="${esc(key)}" data-field-type="${esc(inputType)}"${disabled}>${help ? `<div class="lw-field-help">${esc(help)}</div>` : ""}${readOnly ? '<div class="lw-note">Read-only because the widget is closed.</div>' : ""}</div>`;
        }).join("")}</div>`;
      }
      function chartContainer(data){
        if(Array.isArray(data)) return {points: data};
        if(!isObj(data)) return {};
        const nestedCandidates = [data.chart, data.data, data.payload, data.config, data.state, data.spec, data.chart_spec, data.chartSpec];
        for(const candidate of nestedCandidates){
          if(Array.isArray(candidate)) return {points: candidate, chart_type: data.chart_type || data.chartType || data.type || data.kind, title: data.title};
          if(isObj(candidate) && (
            Array.isArray(candidate.labels) ||
            Array.isArray(candidate.datasets) ||
            isObj(candidate.datasets) ||
            Array.isArray(candidate.series) ||
            isObj(candidate.series) ||
            Array.isArray(candidate.points) ||
            Array.isArray(candidate.items) ||
            Array.isArray(candidate.entries) ||
            Array.isArray(candidate.rows) ||
            Array.isArray(candidate.values) ||
            isObj(candidate.values) ||
            Array.isArray(candidate.data) ||
            isObj(candidate.encoding) ||
            candidate.mark !== undefined ||
            Array.isArray(candidate.xAxis) ||
            isObj(candidate.xAxis)
          )){
            return {
              ...candidate,
              title: candidate.title || data.title,
              chart_type:
                candidate.chart_type
                || candidate.chartType
                || (isObj(candidate.mark) ? candidate.mark.type || candidate.mark.kind || candidate.mark.name : candidate.mark)
                || data.chart_type
                || data.chartType
                || data.type
                || data.kind,
            };
          }
        }
        if(Array.isArray(data.data)) return {points: data.data, chart_type: data.chart_type || data.chartType || data.type || data.kind, title: data.title};
        return data;
      }
      function pickChartLabelKey(row){
        const preferred = ["label", "name", "title", "x", "category", "key"];
        for(const key of preferred){
          if(row[key] !== undefined && row[key] !== null && row[key] !== "") return key;
        }
        for(const [key, value] of Object.entries(row)){
          if(numeric(value) === null) return key;
        }
        return null;
      }
      function normalizeDatasetCollection(source){
        if(Array.isArray(source.datasets)) return source.datasets;
        if(Array.isArray(source.series)) return source.series;
        if(isObj(source.datasets)){
          return Object.entries(source.datasets).map(([name, value]) => {
            if(isObj(value)) return {...value, label: value.label || value.name || name};
            return {label: name, data: Array.isArray(value) ? value : [value]};
          });
        }
        if(isObj(source.series)){
          return Object.entries(source.series).map(([name, value]) => {
            if(isObj(value)) return {...value, label: value.label || value.name || name};
            return {label: name, data: Array.isArray(value) ? value : [value]};
          });
        }
        return [];
      }
      function classifyChartKind(chartType){
        const raw = String(chartType || "").toLowerCase();
        if(raw.includes("donut") || raw.includes("pie") || raw.includes("ring")) return "donut";
        if(raw.includes("line") || raw.includes("trend") || raw.includes("spark")) return "line";
        if(raw.includes("area")) return "area";
        return "bar";
      }
      function chartSpecRows(source){
        if(Array.isArray(source.values)) return source.values;
        if(isObj(source.data) && Array.isArray(source.data.values)) return source.data.values;
        return [];
      }
      function normalizeChartPayload(data){
        const source = chartContainer(data);
        let labels = Array.isArray(source.labels) ? source.labels.map((item) => String(item)) : [];
        let datasets = normalizeDatasetCollection(source).map((item) => isObj(item) ? item : {data: Array.isArray(item) ? item : [item]});
        const xAxis = isObj(source.xAxis) ? source.xAxis : Array.isArray(source.xAxis) && isObj(source.xAxis[0]) ? source.xAxis[0] : {};
        if(!labels.length && Array.isArray(xAxis.data)){
          labels = xAxis.data.map((item) => String(item));
        }
        if(!datasets.length && Array.isArray(source.values)){
          datasets = [{label: source.title || "Series 1", data: source.values}];
        }
        if(!datasets.length && isObj(source.values)){
          const valuePairs = Object.entries(source.values).filter(([, value]) => numeric(value) !== null);
          if(valuePairs.length){
            labels = valuePairs.map(([key]) => label(key));
            datasets = [{label: source.title || "Series 1", data: valuePairs.map(([, value]) => numeric(value) ?? 0)}];
          }
        }
        if(!datasets.length && isObj(source.encoding)){
          const rows = chartSpecRows(source).filter((row) => isObj(row));
          const encoding = source.encoding;
          const xEncoding = isObj(encoding.x) ? encoding.x : {};
          const thetaEncoding = isObj(encoding.theta) ? encoding.theta : {};
          const yEncoding = isObj(encoding.y) ? encoding.y : {};
          const radiusEncoding = isObj(encoding.radius) ? encoding.radius : {};
          const colorEncoding = isObj(encoding.color) ? encoding.color : {};
          const labelField = String(xEncoding.field || thetaEncoding.field || "").trim();
          const valueField = String(yEncoding.field || radiusEncoding.field || "").trim();
          const seriesField = String(colorEncoding.field || "").trim();
          if(rows.length && valueField){
            if(seriesField){
              const grouped = {};
              labels = [];
              rows.forEach((row, idx) => {
                const labelValue = String((labelField && row[labelField] !== undefined) ? row[labelField] : `Item ${idx + 1}`);
                const seriesName = String(row[seriesField] ?? source.title ?? "Series 1");
                const pointValue = numeric(row[valueField]);
                if(pointValue === null) return;
                if(!labels.includes(labelValue)) labels.push(labelValue);
                if(!grouped[seriesName]) grouped[seriesName] = {};
                grouped[seriesName][labelValue] = pointValue;
              });
              datasets = Object.entries(grouped).map(([seriesName, values]) => ({
                label: seriesName,
                data: labels.map((labelValue) => numeric(values[labelValue]) ?? 0),
              }));
            } else {
              labels = rows.map((row, idx) => String((labelField && row[labelField] !== undefined) ? row[labelField] : `Item ${idx + 1}`));
              datasets = [{
                label: source.title || label(valueField) || "Series 1",
                data: rows.map((row) => numeric(row[valueField]) ?? 0),
              }];
            }
          }
        }
        if(!datasets.length){
          const pointLike = Array.isArray(source.points) ? source.points : Array.isArray(source.items) ? source.items : Array.isArray(source.entries) ? source.entries : Array.isArray(source.data) ? source.data : [];
          const grouped = {};
          pointLike.forEach((point, idx) => {
            const item = isObj(point) ? point : {value: point};
            const seriesName = String(item.series || item.dataset || item.group || source.title || "Series 1");
            const pointLabel = String(item.label ?? item.name ?? item.x ?? item.category ?? item.key ?? `Item ${idx + 1}`);
            const pointValue = numeric(item.value ?? item.y ?? item.amount ?? item.count ?? item.metric ?? item.total);
            if(pointValue === null) return;
            if(!grouped[seriesName]) grouped[seriesName] = [];
            grouped[seriesName].push({label: pointLabel, value: pointValue});
          });
          const groupedSeries = Object.values(grouped);
          if(groupedSeries.length){
            labels = groupedSeries[0].map((entry) => entry.label);
            datasets = Object.entries(grouped).map(([seriesName, entries]) => ({
              label: seriesName,
              data: labels.map((labelValue) => {
                const match = entries.find((entry) => entry.label === labelValue);
                return match ? match.value : 0;
              }),
            }));
          }
        }
        if(!datasets.length && Array.isArray(source.rows)){
          const rows = source.rows.filter((row) => row !== null && row !== undefined);
          if(rows.length && Array.isArray(rows[0])){
            const rawCols = Array.isArray(source.columns) ? source.columns : Array.isArray(source.cols) ? source.cols : [];
            const normalizedCols = rawCols.map((col, idx) => typeof col === "string" ? {key: col, label: label(col)} : {key: String(col.key || col.id || col.name || col.label || `column_${idx}`), label: String(col.label || col.title || col.name || col.key || `Column ${idx + 1}`)});
            const valueColumns = normalizedCols.slice(1);
            if(valueColumns.length){
              labels = rows.map((row, idx) => String(row[0] ?? `Item ${idx + 1}`));
              datasets = valueColumns.map((col, valueIdx) => ({
                label: col.label,
                data: rows.map((row) => numeric(row[valueIdx + 1]) ?? 0),
              }));
            }
          } else if(rows.length && isObj(rows[0])){
            const labelKey = pickChartLabelKey(rows[0]);
            const explicitSeriesKeys = Array.isArray(source.series_keys) ? source.series_keys : Array.isArray(source.seriesKeys) ? source.seriesKeys : [];
            let valueKeys = explicitSeriesKeys.filter((key) => typeof key === "string" && key);
            if(!valueKeys.length){
              const candidateKeys = Object.keys(rows[0]).filter((key) => key !== labelKey);
              valueKeys = candidateKeys.filter((key) => rows.some((row) => numeric(row[key]) !== null));
            }
            if(valueKeys.length){
              labels = rows.map((row, idx) => String((labelKey && row[labelKey] !== undefined) ? row[labelKey] : `Item ${idx + 1}`));
              datasets = valueKeys.map((key) => ({
                label: label(key),
                data: rows.map((row) => numeric(row[key]) ?? 0),
              }));
            }
          }
        }
        const series = datasets.map((item, i) => {
          let rawData = Array.isArray(item.data) ? item.data : Array.isArray(item.values) ? item.values : Array.isArray(item.points) ? item.points : [];
          if(!rawData.length && isObj(item.data)){
            const pairs = Object.entries(item.data).filter(([, value]) => numeric(value) !== null);
            if(pairs.length){
              if(!labels.length) labels = pairs.map(([key]) => label(key));
              rawData = pairs.map(([, value]) => numeric(value) ?? 0);
            }
          }
          const inferredLabels = [];
          const values = rawData.map((value, idx) => {
            if(isObj(value)){
              inferredLabels.push(String(value.label ?? value.name ?? value.x ?? value.category ?? value.key ?? `Item ${idx + 1}`));
              return numeric(value.y ?? value.value ?? value.amount ?? value.count ?? value.metric ?? value.total);
            }
            inferredLabels.push(labels[idx] ?? `Item ${idx + 1}`);
            return numeric(value);
          }).filter((value) => value !== null);
          if(!labels.length && inferredLabels.length) labels = inferredLabels;
          return {
            label: String(item.label || item.name || `Series ${i + 1}`),
            color: item.color || item.backgroundColor || item.borderColor || palette[i % palette.length],
            data: values.map((value) => value ?? 0),
          };
        }).filter((item) => item.data.length);
        const maxLen = series.reduce((acc, item) => Math.max(acc, item.data.length), 0);
        if(!labels.length && maxLen > 0){
          labels = Array.from({length: maxLen}, (_, idx) => `Item ${idx + 1}`);
        }
        if(labels.length && series.length){
          series.forEach((item) => {
            if(item.data.length < labels.length){
              while(item.data.length < labels.length) item.data.push(0);
            } else if(item.data.length > labels.length){
              item.data = item.data.slice(0, labels.length);
            }
          });
        }
        const ui = chartUi(data);
        const hiddenSeries = Array.isArray(ui.hidden_series) ? ui.hidden_series.map((item) => String(item)) : [];
        const explicitChartTypeOptions = Array.isArray(source.chart_types)
          ? source.chart_types
          : Array.isArray(source.chart_type_options)
            ? source.chart_type_options
            : Array.isArray(data.chart_types)
              ? data.chart_types
              : Array.isArray(data.chart_type_options)
                ? data.chart_type_options
                : [];
        const chartTypeOptions = explicitChartTypeOptions.length ? explicitChartTypeOptions : ["bar", "line", "area", "donut"];
        const markType = isObj(source.mark) ? source.mark.type || source.mark.kind || source.mark.name : source.mark;
        const firstSeriesType = Array.isArray(source.series) && isObj(source.series[0]) ? source.series[0].type || source.series[0].chart_type : "";
        const chartType = String(ui.chart_type || source.chart_type || source.chartType || markType || firstSeriesType || data.chart_type || data.chartType || "chart");
        const filteredSeries = series.filter((item) => !hiddenSeries.includes(item.label));
        return {
          source,
          labels,
          allSeries: series,
          series: filteredSeries,
          hiddenSeries,
          chartTypeOptions: chartTypeOptions.map(normalizeOption),
          chartType,
          chartKind: classifyChartKind(chartType),
        };
      }
      function buildChartStats(normalized){
        const values = normalized.series.flatMap((item) => item.data.filter((value) => numeric(value) !== null));
        if(!values.length) return "";
        const total = values.reduce((acc, value) => acc + value, 0);
        const peak = Math.max(...values);
        const latest = normalized.series.reduce((acc, item) => acc + (numeric(item.data[item.data.length - 1]) ?? 0), 0);
        const stats = [
          {label: "Series", value: normalized.series.length},
          {label: "Categories", value: normalized.labels.length},
          {label: "Peak", value: formatNumber(peak)},
          {label: normalized.series.length > 1 ? "Latest Sum" : "Total", value: formatNumber(normalized.series.length > 1 ? latest : total)},
        ];
        return `<div class="lw-chart-stats">${stats.map((item) => `<div class="lw-stat"><div class="lw-stat-label">${esc(item.label)}</div><div class="lw-stat-value">${esc(item.value)}</div></div>`).join("")}</div>`;
      }
      function sampledLabelIndexes(length, maxLabels = 8){
        const step = Math.max(1, Math.ceil(length / maxLabels));
        return Array.from({length}, (_, idx) => idx).filter((idx) => idx % step === 0 || idx === length - 1);
      }
      function renderBarChartSvg(normalized){
        const width = 920;
        const height = 340;
        const margin = {top: 18, right: 24, bottom: 58, left: 56};
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        const values = normalized.series.flatMap((item) => item.data);
        const minValue = Math.min(0, ...values);
        const maxValue = Math.max(0, ...values);
        const safeMax = minValue === maxValue ? maxValue + 1 : maxValue;
        const span = safeMax - minValue || 1;
        const yFor = (value) => margin.top + ((safeMax - value) / span) * innerHeight;
        const zeroY = yFor(0);
        const ticks = Array.from({length: 5}, (_, idx) => safeMax - ((safeMax - minValue) * idx / 4));
        const groupWidth = innerWidth / Math.max(normalized.labels.length, 1);
        const clusterWidth = Math.min(groupWidth * 0.8, 96);
        const barWidth = Math.max(8, Math.min(36, clusterWidth / Math.max(normalized.series.length, 1)));
        const labelIndexes = new Set(sampledLabelIndexes(normalized.labels.length, normalized.labels.length > 12 ? 6 : 10));
        const grid = ticks.map((tick) => {
          const y = yFor(tick);
          return `<g><line class="lw-grid-line" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"></line><text class="lw-axis-text" x="${margin.left - 10}" y="${y + 4}" text-anchor="end">${esc(formatNumber(tick))}</text></g>`;
        }).join("");
        const axis = `<line class="lw-axis-line" x1="${margin.left}" y1="${zeroY}" x2="${width - margin.right}" y2="${zeroY}"></line>`;
        const bars = normalized.series.map((seriesItem, seriesIdx) => seriesItem.data.map((value, idx) => {
          const x = margin.left + idx * groupWidth + Math.max(0, (groupWidth - barWidth * normalized.series.length) / 2) + seriesIdx * barWidth;
          const yValue = yFor(value);
          const y = value >= 0 ? yValue : zeroY;
          const barHeight = Math.max(2, Math.abs(yValue - zeroY));
          const valueLabel = normalized.labels.length <= 8 && normalized.series.length <= 2 ? `<text class="lw-axis-text" x="${x + barWidth / 2}" y="${value >= 0 ? y - 8 : y + barHeight + 16}" text-anchor="middle">${esc(formatNumber(value))}</text>` : "";
          return `<g><rect x="${x}" y="${y}" width="${Math.max(4, barWidth - 4)}" height="${barHeight}" rx="10" fill="${esc(seriesItem.color)}"></rect>${valueLabel}</g>`;
        }).join("")).join("");
        const xLabels = normalized.labels.map((labelValue, idx) => {
          if(!labelIndexes.has(idx)) return "";
          const x = margin.left + idx * groupWidth + groupWidth / 2;
          return `<text class="lw-axis-text" x="${x}" y="${height - 18}" text-anchor="middle">${esc(truncateText(labelValue, 14))}</text>`;
        }).join("");
        return `<svg class="lw-chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">${grid}${axis}${bars}${xLabels}</svg>`;
      }
      function renderLineChartSvg(normalized, fillArea){
        const width = 920;
        const height = 340;
        const margin = {top: 18, right: 24, bottom: 58, left: 56};
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        const values = normalized.series.flatMap((item) => item.data);
        const minValue = Math.min(0, ...values);
        const maxValue = Math.max(0, ...values);
        const safeMax = minValue === maxValue ? maxValue + 1 : maxValue;
        const span = safeMax - minValue || 1;
        const yFor = (value) => margin.top + ((safeMax - value) / span) * innerHeight;
        const xFor = (idx) => normalized.labels.length === 1 ? margin.left + innerWidth / 2 : margin.left + (innerWidth * idx / (normalized.labels.length - 1));
        const ticks = Array.from({length: 5}, (_, idx) => safeMax - ((safeMax - minValue) * idx / 4));
        const labelIndexes = new Set(sampledLabelIndexes(normalized.labels.length, normalized.labels.length > 12 ? 6 : 10));
        const grid = ticks.map((tick) => {
          const y = yFor(tick);
          return `<g><line class="lw-grid-line" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"></line><text class="lw-axis-text" x="${margin.left - 10}" y="${y + 4}" text-anchor="end">${esc(formatNumber(tick))}</text></g>`;
        }).join("");
        const xLabels = normalized.labels.map((labelValue, idx) => {
          if(!labelIndexes.has(idx)) return "";
          return `<text class="lw-axis-text" x="${xFor(idx)}" y="${height - 18}" text-anchor="middle">${esc(truncateText(labelValue, 14))}</text>`;
        }).join("");
        const seriesMarkup = normalized.series.map((seriesItem) => {
          const points = seriesItem.data.map((value, idx) => ({x: xFor(idx), y: yFor(value)}));
          const path = points.map((point, idx) => `${idx === 0 ? "M" : "L"}${point.x},${point.y}`).join(" ");
          const areaPath = fillArea ? `${path} L ${points[points.length - 1].x},${margin.top + innerHeight} L ${points[0].x},${margin.top + innerHeight} Z` : "";
          const dots = points.map((point) => `<circle class="lw-point" cx="${point.x}" cy="${point.y}" r="4.5" fill="${esc(seriesItem.color)}"></circle>`).join("");
          return `<g>${fillArea ? `<path class="lw-area-path" d="${areaPath}" fill="${esc(colorWithAlpha(seriesItem.color, 0.18))}"></path>` : ""}<path class="lw-line-path" d="${path}" stroke="${esc(seriesItem.color)}"></path>${dots}</g>`;
        }).join("");
        return `<svg class="lw-chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">${grid}${seriesMarkup}${xLabels}</svg>`;
      }
      function renderDonutChart(normalized){
        const slices = (normalized.series.length === 1 ? normalized.labels.map((labelValue, idx) => ({
          label: labelValue,
          value: normalized.series[0].data[idx] ?? 0,
          color: palette[idx % palette.length],
        })) : normalized.labels.map((labelValue, idx) => ({
          label: labelValue,
          value: normalized.series.reduce((acc, item) => acc + (numeric(item.data[idx]) ?? 0), 0),
          color: palette[idx % palette.length],
        }))).filter((item) => numeric(item.value) !== null && item.value > 0);
        if(!slices.length) return "";
        const total = slices.reduce((acc, item) => acc + item.value, 0);
        let cursor = 0;
        const gradient = slices.map((slice) => {
          const start = cursor / total * 100;
          cursor += slice.value;
          const end = cursor / total * 100;
          return `${slice.color} ${start}% ${end}%`;
        }).join(",");
        const legend = slices.map((slice) => `<div class="lw-donut-item"><span class="lw-dot" style="background:${esc(slice.color)}"></span><div><div class="lw-donut-value">${esc(truncateText(slice.label, 22))}</div><div class="lw-donut-share">${esc(formatNumber((slice.value / total) * 100))}% share</div></div><div class="lw-donut-value">${esc(formatNumber(slice.value))}</div></div>`).join("");
        return `<div class="lw-donut"><div class="lw-ring" style="background:conic-gradient(${gradient})"><div class="lw-ring-hole"><strong>${esc(formatNumber(total))}</strong><span>Total</span></div></div><div class="lw-donut-legend">${legend}</div></div>`;
      }
      function renderChartPreview(source){
        if(Array.isArray(source.rows)) return renderTable(source);
        const pointLike = Array.isArray(source.points) ? source.points : Array.isArray(source.items) ? source.items : Array.isArray(source.entries) ? source.entries : Array.isArray(source.data) ? source.data : [];
        if(pointLike.length){
          return renderTable({
            columns: ["Label", "Value"],
            rows: pointLike.map((point, idx) => {
              if(isObj(point)) return [point.label ?? point.name ?? point.x ?? point.category ?? point.key ?? `Item ${idx + 1}`, point.value ?? point.y ?? point.amount ?? point.count ?? point.metric ?? point.total ?? JSON.stringify(point)];
              return [`Item ${idx + 1}`, point];
            }),
          });
        }
        if(isObj(source.values)){
          return renderTable({columns: ["Label", "Value"], rows: Object.entries(source.values).map(([key, value]) => [label(key), value])});
        }
        const scalarEntries = Object.entries(source).filter(([, value]) => value === null || ["string", "number", "boolean"].includes(typeof value));
        if(scalarEntries.length){
          return renderTable({columns: ["Field", "Value"], rows: scalarEntries.map(([key, value]) => [label(key), value])});
        }
        return renderRawDisclosure(source);
      }
      function htmlState(data){
        if(!isObj(data)) return {html: "", caption: "", height: 560};
        const html = String(data.html || data.document || data.content || data.srcdoc || "");
        const caption = String(data.caption || data.note || data.description || "");
        const explicitHeight = numeric(data.height) ?? numeric(data.min_height) ?? numeric(data.minHeight);
        const height = Math.max(260, Math.min(960, explicitHeight ?? 560));
        return {html, caption, height};
      }
      function renderHtmlWidget(data){
        const cfg = htmlState(data);
        if(!cfg.html.trim()){
          return '<div class="lw-empty">No HTML content available yet.</div>';
        }
        return `<div class="lw-html-shell">${cfg.caption ? `<div class="lw-html-caption">${esc(cfg.caption)}</div>` : ""}<iframe class="lw-html-frame" sandbox="allow-scripts allow-forms allow-modals allow-downloads" referrerpolicy="no-referrer" style="height:${cfg.height}px" srcdoc="${esc(cfg.html)}"></iframe></div>`;
      }
      function renderChartToolbar(normalized, readOnly){
        const disabled = readOnly ? " disabled" : "";
        const chartTypeButtons = normalized.chartTypeOptions.map((option) => {
          const optionKind = classifyChartKind(option.value);
          return `<button type="button" data-chart-type="${esc(option.value)}" data-active="${optionKind === normalized.chartKind}"${disabled}>${esc(label(option.label || option.value))}</button>`;
        }).join("");
        const legendButtons = normalized.allSeries.map((item) => `<button type="button" class="lw-legend-item-button" data-series-toggle="${esc(item.label)}" data-hidden="${normalized.hiddenSeries.includes(item.label)}"${disabled}><span class="lw-dot" style="background:${esc(item.color)}"></span>${esc(item.label)}</button>`).join("");
        const resetButton = normalized.hiddenSeries.length ? `<button type="button" class="lw-legend-item-button" data-series-reset="true"${disabled}>Show all</button>` : "";
        if(!chartTypeButtons && !legendButtons && !resetButton) return "";
        return `<div class="lw-chart-toolbar">${chartTypeButtons ? `<div class="lw-segment">${chartTypeButtons}</div>` : ""}${legendButtons || resetButton ? `<div class="lw-chip-wrap">${legendButtons}${resetButton}</div>` : ""}</div>`;
      }
      function renderChart(data, readOnly = false){
        const normalized = normalizeChartPayload(data);
        const labels = normalized.labels;
        const series = normalized.series;
        const toolbar = renderChartToolbar(normalized, readOnly);
        if(!series.length && normalized.allSeries.length && labels.length){
          return `<div class="lw-chart"><div class="lw-chart-top"><div><div class="lw-chart-type">${esc(label(normalized.chartType))}</div><div class="lw-chart-sub">All series are hidden. Re-enable one to continue.</div></div></div>${toolbar}<div class="lw-empty">This chart still has data, but every series is currently hidden.</div></div>`;
        }
        if(!series.length || !labels.length){
          if(isObj(normalized.source) && Object.keys(normalized.source).length){
            return `<div class="lw-chart"><div class="lw-chart-top"><div><div class="lw-chart-type">${esc(label(normalized.chartType))}</div><div class="lw-chart-sub">Showing a data preview because this chart payload could not be normalized yet.</div></div></div>${toolbar}<div class="lw-fallback">${renderChartPreview(normalized.source)}</div></div>`;
          }
          return '<div class="lw-empty">No chart data available yet.</div>';
        }
        let surface = renderBarChartSvg(normalized);
        if(normalized.chartKind === "line" || normalized.chartKind === "area") surface = renderLineChartSvg(normalized, normalized.chartKind === "area");
        if(normalized.chartKind === "donut"){
          const donut = renderDonutChart(normalized);
          surface = donut || renderBarChartSvg(normalized);
        }
        return `<div class="lw-chart"><div class="lw-chart-top"><div><div class="lw-chart-type">${esc(label(normalized.chartType))}</div><div class="lw-chart-sub">${esc(series.length > 1 ? `${series.length} series across ${labels.length} categories` : `${labels.length} categories`)}</div></div></div>${toolbar}${buildChartStats(normalized)}<div class="lw-chart-surface">${surface}</div></div>`;
      }
      function renderDashboard(data, readOnly = false){
        const panels = Array.isArray(data.panels) ? data.panels : [];
        if(!panels.length){
          const fallbackMetrics = Object.entries(data).filter(([key, value]) => key !== "panels" && (numeric(value) !== null || typeof value === "string"));
          if(fallbackMetrics.length){
            return `<div class="lw-dash"><div class="lw-metrics">${fallbackMetrics.slice(0, 6).map(([key, value]) => `<div class="lw-metric"><div class="lw-metric-title">${esc(label(key))}</div><div class="lw-metric-value">${esc(typeof value === "number" ? formatNumber(value) : value)}</div></div>`).join("")}</div></div>`;
          }
          return '<div class="lw-empty">No dashboard panels available yet.</div>';
        }
        const metrics = [];
        const details = [];
        panels.forEach((panel, idx) => {
          const item = isObj(panel) ? panel : {value: panel};
          const type = String(item.type || item.widget_type || item.kind || "").toLowerCase();
          const nested = isObj(item.data) ? item.data : isObj(item.state) ? item.state : item;
          const plainMetric = item.value !== undefined && !nested.rows && !nested.datasets && !nested.items && !nested.fields;
          if(type === "metric" || type === "stat" || type === "summary" || plainMetric){
            metrics.push(`<div class="lw-metric"><div class="lw-metric-title">${esc(item.title || item.label || "Metric")}</div><div class="lw-metric-value">${esc(item.value ?? item.metric ?? item.total ?? "-")}</div>${item.delta || item.change || item.description ? `<div class="lw-metric-delta">${esc(item.delta || item.change || item.description)}</div>` : ""}</div>`);
            return;
          }
          let content = renderJson(nested);
          if(type === "table") content = renderTable(nested, true);
          else if(type === "chart") content = renderChart(nested, true);
          else if(type === "list") content = renderList(nested, true);
          else if(type === "form") content = renderForm(nested, true);
          details.push(`<div class="lw-panel"><div class="lw-panel-title">${esc(item.title || item.label || `Panel ${idx + 1}`)}</div>${content}</div>`);
        });
        return `<div class="lw-dash">${metrics.length ? `<div class="lw-metrics">${metrics.join("")}</div>` : ""}${details.join("")}</div>`;
      }
      function renderBody(){
        meta();
        if(!state.data){el.body.innerHTML='<div class="lw-empty">Connecting widget…</div>';resize();return;}
        const type = String(cfg.widget.widget_type || "").toLowerCase();
        const readOnly = state.status === "closed";
        const resolved = resolveInteractiveData(state.data);
        const payload = resolved.payload;
        const controlsHtml = renderControls(resolved.controls, resolved.controlValues, readOnly);
        let bodyHtml = renderJson(payload);
        if(type === "table") bodyHtml = renderTable(payload, readOnly);
        else if(type === "chart") bodyHtml = renderChart(payload, readOnly);
        else if(type === "dashboard") bodyHtml = renderDashboard(payload, readOnly);
        else if(type === "form") bodyHtml = renderForm(payload, readOnly);
        else if(type === "list") bodyHtml = renderList(payload, readOnly);
        else if(type === "html" || type === "iframe" || type === "micro_app") bodyHtml = renderHtmlWidget(payload);
        el.body.innerHTML = `${controlsHtml}${bodyHtml}`;
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
      function sendPatch(patch){
        if(!patch || state.status === "closed" || !ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({type:"user_state_patch", patch}));
      }
      function collectValues(){
        const values = {};
        el.body.querySelectorAll("[data-field-key]").forEach((input) => {
          const key = input.getAttribute("data-field-key");
          const type = input.getAttribute("data-field-type");
          if(!key) return;
          values[key] = inputValue(input, type);
        });
        return values;
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
      el.body.addEventListener("click", (event) => {
        const rawTarget = event.target;
        if(!(rawTarget instanceof Element)) return;
        const controlOption = rawTarget.closest("[data-control-option='true']");
        if(controlOption){
          const key = controlOption.getAttribute("data-control-key");
          if(key) setControlValue(key, controlOption.getAttribute("data-control-value") ?? "");
          return;
        }
        const sortButton = rawTarget.closest("[data-sort-key]");
        if(sortButton && isObj(state.data)){
          const sortKey = String(sortButton.getAttribute("data-sort-key") || "");
          if(sortKey){
            const currentUi = tableUi(state.data);
            const nextDirection = currentUi.sort_by === sortKey && currentUi.sort_dir !== "desc" ? "desc" : "asc";
            setTableUi({sort_by: sortKey, sort_dir: nextDirection});
          }
          return;
        }
        const chartTypeButton = rawTarget.closest("[data-chart-type]");
        if(chartTypeButton){
          const nextChartType = chartTypeButton.getAttribute("data-chart-type");
          if(nextChartType) setChartUi({chart_type: nextChartType});
          return;
        }
        const seriesButton = rawTarget.closest("[data-series-toggle]");
        if(seriesButton){
          const seriesLabel = seriesButton.getAttribute("data-series-toggle");
          if(seriesLabel) toggleChartSeries(seriesLabel);
          return;
        }
        const resetSeriesButton = rawTarget.closest("[data-series-reset='true']");
        if(resetSeriesButton){
          setChartUi({hidden_series: []});
          return;
        }
        const target = rawTarget.closest("[data-action='select']");
        if(target && isObj(state.data)){
          const selection = target.getAttribute("data-selection");
          state.data = {...state.data, selection};
          renderBody();
          sendPatch({selection});
        }
      });
      el.body.addEventListener("input", (event) => {
        const target = event.target;
        if(!(target instanceof HTMLElement)) return;
        if(target.hasAttribute("data-table-search")){
          const nextQuery = target.value || "";
          setTableUi({query: nextQuery});
          const nextSearch = el.body.querySelector("[data-table-search='true']");
          if(nextSearch && typeof nextSearch.focus === "function"){
            nextSearch.focus();
            if(typeof nextSearch.setSelectionRange === "function"){
              const cursor = String(nextQuery).length;
              nextSearch.setSelectionRange(cursor, cursor);
            }
          }
        }
      });
      el.body.addEventListener("change", (event) => {
        const target = event.target;
        if(!(target instanceof HTMLElement)) return;
        if(target.hasAttribute("data-control-key")){
          const key = target.getAttribute("data-control-key");
          if(key) setControlValue(key, inputValue(target, target.getAttribute("data-control-type")), target.getAttribute("data-control-type") === "range" ? 80 : 150);
          return;
        }
        if(!target.hasAttribute("data-field-key")) return;
        const values = collectValues();
        const base = isObj(state.data) ? state.data : {};
        state.data = {...base, values};
        window.clearTimeout(patchTimer);
        patchTimer = window.setTimeout(() => sendPatch({values}), 150);
      });
      window.addEventListener("beforeunload", () => {
        destroyed = true;
        window.clearTimeout(reconnectTimer);
        window.clearTimeout(patchTimer);
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
    widget_type = str(widget.get("widget_type") or "").lower()
    if widget_type == "dashboard":
        return 800
    if widget_type == "chart":
        return 760
    if widget_type in {"html", "iframe", "micro_app"}:
        return 820
    if widget_type == "form":
        return 660
    if widget_type == "table":
        return 680
    if widget_type == "list":
        return 600
    return 640


def render_live_widgets(
    message_metadata: dict,
    *,
    message_key: str = "",
    auto_mount: bool = False,
):
    """Render live widgets from assistant message metadata.

    Historical widgets stay lightweight until opened, which prevents
    conversation-load reruns from mounting every widget iframe at once.
    """
    if not message_metadata:
        return

    widgets = message_metadata.get("live_widgets") or []
    if not widgets:
        return

    import streamlit.components.v1 as _stc

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
    """Render images from agent responses as thumbnails (Tavily, Image Generator)"""
    if not message_metadata:
        return

    images = message_metadata.get("images") or []
    if not images:
        return

    image_items: list[dict[str, Any]] = []
    for idx, image in enumerate(images, start=1):
        if not isinstance(image, dict):
            continue

        url_value = image.get("url")
        data_value = image.get("data")
        image_name = (
            image.get("name") or image.get("description") or image.get("caption") or f"Image {idx}"
        )

        if isinstance(url_value, str) and url_value:
            image_items.append(
                {
                    "url": url_value,
                    "name": image_name,
                }
            )
        elif isinstance(data_value, str) and data_value:
            payload = data_value.strip()
            mime = image.get("mime", "image/png")

            if payload.startswith("data:"):
                header, _, raw_payload = payload.partition(",")
                if raw_payload:
                    payload = raw_payload.strip()
                header_mime = header[5:].split(";")[0].strip() if header else ""
                if "/" in header_mime:
                    mime = header_mime

            image_items.append(
                {
                    "data": payload,
                    "mime": mime,
                    "name": image_name,
                }
            )

    if not image_items:
        return

    import streamlit.components.v1 as _stc

    # Collect image sources for the JS-based lightbox
    thumb_entries: list[dict[str, str]] = []
    for _, item in enumerate(image_items):
        caption = item.get("name") or ""
        if item.get("url"):
            src = item["url"]
        else:
            data_b64 = item.get("data")
            if not isinstance(data_b64, str) or not data_b64.strip():
                continue
            mime = item.get("mime", "image/png")
            src = f"data:{mime};base64,{data_b64}"
        thumb_entries.append({"src": src, "caption": html.escape(caption)})

    if not thumb_entries:
        return

    # Build thumbnail <img> tags with fixed-size cells and broken-image handling
    thumbs_html = ""
    for i, entry in enumerate(thumb_entries):
        # Truncate caption for display (keep full text in title attribute)
        short_cap = entry["caption"]
        if len(short_cap) > 40:
            short_cap = html.escape(entry["caption"][:37] + "...")
        else:
            short_cap = entry["caption"]  # already escaped

        cap_html = (
            f'<div class="agent-thumb-caption" title="{entry["caption"]}">{short_cap}</div>'
            if entry["caption"]
            else ""
        )
        thumbs_html += (
            f'<div class="agent-thumb-cell">'
            f'  <img src="{entry["src"]}" alt="{short_cap}" '
            f'       title="Click to view full size" data-idx="{i}" '
            f'       class="agent-thumb-img" '
            f"       onerror=\"this.parentElement.classList.add('broken')\" />"
            f"  {cap_html}"
            f"</div>"
        )

    # The JS injects a lightbox overlay into the TOP-LEVEL document (parent
    # of the Streamlit iframe) so it covers the entire browser window
    # including the sidebar.  Clicking the overlay closes it.
    # We JSON-encode the sources list for safe embedding.
    import json as _json

    sources_json = _json.dumps([e["src"] for e in thumb_entries])

    component_html = f"""
    <style>
      #thumb-gallery {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: flex-start;
      }}
      .agent-thumb-cell {{
        width: 150px;
        flex-shrink: 0;
        text-align: center;
        border-radius: 8px;
        overflow: hidden;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        transition: box-shadow .2s;
      }}
      .agent-thumb-cell:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,.12);
      }}
      /* Hide entire cell when image is broken */
      .agent-thumb-cell.broken {{
        display: none !important;
      }}
      .agent-thumb-img {{
        width: 150px;
        height: 120px;
        object-fit: cover;
        display: block;
        cursor: zoom-in;
        border-radius: 8px 8px 0 0;
        transition: opacity .2s;
      }}
      .agent-thumb-img:hover {{
        opacity: 0.82;
      }}
      .agent-thumb-caption {{
        font-size: .75em;
        color: #64748b;
        padding: 4px 6px;
        line-height: 1.3;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 150px;
      }}
    </style>
    <div id="thumb-gallery">
      {thumbs_html}
    </div>
    <script>
    (function() {{
      var sources = {sources_json};

      // Find the top-level document (escape iframe)
      var topDoc = window.top.document;

      // Ensure overlay exists in top document (create once)
      var OVERLAY_ID = '__agent_img_lightbox';
      var overlay = topDoc.getElementById(OVERLAY_ID);
      if (!overlay) {{
        overlay = topDoc.createElement('div');
        overlay.id = OVERLAY_ID;
        overlay.style.cssText = (
          'display:none;position:fixed;z-index:999999;left:0;top:0;'
          + 'width:100vw;height:100vh;background:rgba(0,0,0,.88);'
          + 'align-items:center;justify-content:center;cursor:zoom-out;'
        );
        var img = topDoc.createElement('img');
        img.id = OVERLAY_ID + '_img';
        img.style.cssText = (
          'max-width:90vw;max-height:90vh;border-radius:8px;'
          + 'box-shadow:0 0 40px rgba(0,0,0,.6);'
        );
        overlay.appendChild(img);
        overlay.addEventListener('click', function() {{
          overlay.style.display = 'none';
        }});
        topDoc.body.appendChild(overlay);
      }}

      // Attach click handlers to thumbnails
      var gallery = document.getElementById('thumb-gallery');
      gallery.addEventListener('click', function(e) {{
        var t = e.target;
        if (t.tagName === 'IMG' && t.hasAttribute('data-idx')) {{
          var idx = parseInt(t.getAttribute('data-idx'), 10);
          var src = sources[idx];
          if (src) {{
            var topOverlay = window.top.document.getElementById(OVERLAY_ID);
            var topImg = window.top.document.getElementById(OVERLAY_ID + '_img');
            topImg.src = src;
            topOverlay.style.display = 'flex';
          }}
        }}
      }});

      // After load, shrink iframe to actual content height to remove blank space
      requestAnimationFrame(function() {{
        var h = document.getElementById('thumb-gallery').offsetHeight;
        if (h > 0) {{
          document.body.style.margin = '0';
          document.body.style.overflow = 'hidden';
          var frame = window.frameElement;
          if (frame) frame.style.height = h + 'px';
        }}
      }});
    }})();
    </script>
    """

    cols_per_row = 4
    row_count = (len(thumb_entries) + cols_per_row - 1) // cols_per_row
    estimated_height = row_count * 160 + 4
    _stc.html(component_html, height=estimated_height, scrolling=False)


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


def _reset_stream_trace_state(expanded: bool = True) -> None:
    _ensure_stream_trace_state()
    st.session_state.stream_trace_items = []
    st.session_state.stream_tool_index = {}
    st.session_state.stream_trace_expanded = expanded


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

    header_html = ""
    if live:
        header_html = """
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
        """
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
    else:
        body_html = (
            f'<div class="thinking-content-rendered">{sanitize_message_content(content)}</div>'
        )

    st.markdown(
        f'<div class="thinking-container">{header_html}{body_html}</div>',
        unsafe_allow_html=True,
    )


def _build_tool_trace_items_from_artifacts(
    tool_artifacts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not isinstance(tool_artifacts, list):
        return []

    trace_items: list[dict[str, Any]] = []
    for index, artifact in enumerate(tool_artifacts, start=1):
        if not isinstance(artifact, dict):
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
                "name": artifact.get("tool", "unknown_tool"),
                "phase": "end" if state != "running" else "start",
                "state": state,
                "args": artifact.get("args"),
                "result": artifact.get("output"),
                "render": artifact.get("render"),
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
    if isinstance(render, dict):
        st.markdown(
            '<div class="trace-preview-label">Result Preview</div>',
            unsafe_allow_html=True,
        )
        if not render_tool_render_payload(render, fallback_output=result):
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
    if isinstance(error_message, str) and error_message.strip():
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
    if not any(
        [
            isinstance(thinking_content, str) and thinking_content.strip(),
            isinstance(reasoning_summary, str) and reasoning_summary.strip(),
            normalized_tools,
        ]
    ):
        return

    with st.expander("Thought Process", expanded=expanded):
        if (
            isinstance(reasoning_summary, str)
            and reasoning_summary.strip()
            or (isinstance(thinking_content, str) and thinking_content.strip())
        ):
            st.markdown('<div class="trace-section-title">Thinking</div>', unsafe_allow_html=True)
            if isinstance(reasoning_summary, str) and reasoning_summary.strip():
                _render_trace_text_block(reasoning_summary, label="Reasoning Summary")
            if isinstance(thinking_content, str) and thinking_content.strip():
                thinking_label = None if live else "Thinking Summary"
                _render_trace_text_block(
                    thinking_content,
                    label=thinking_label,
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


def _resolve_stream_tool_trace_id(tool_event: dict[str, Any]) -> str:
    explicit_id = tool_event.get("tool_call_id")
    if explicit_id:
        return str(explicit_id)

    phase = normalize_tool_phase(tool_event.get("phase") or tool_event.get("status"))
    trace_items = st.session_state.get("stream_trace_items") or []

    if phase == "end":
        for item in trace_items:
            if item.get("kind") == "tool" and str(item.get("state")).lower() == "running":
                fallback_id = item.get("tool_call_id")
                if fallback_id:
                    return str(fallback_id)

    next_index = sum(1 for item in trace_items if item.get("kind") == "tool") + 1
    return f"tool_{next_index}"


def _upsert_stream_tool_trace(tool_event: dict[str, Any]) -> None:
    _ensure_stream_trace_state()
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

    if thinking_item is None and not tool_items:
        trace_placeholder.empty()
        return

    with trace_placeholder.container():
        render_trace_panel(
            thinking_content=thinking_item.get("content") if thinking_item else None,
            tool_items=tool_items,
            expanded=bool(st.session_state.get("stream_trace_expanded", True)),
            live=True,
        )


def render_tool_artifacts(tool_artifacts: list[dict[str, Any]]):
    """
    Render tool execution artifacts as collapsible sections.
    Shows tool name, status, arguments, and results.
    """
    if not tool_artifacts:
        return

    st.markdown("#### Tool Executions", unsafe_allow_html=True)

    for idx, artifact in enumerate(tool_artifacts, start=1):
        tool_name = artifact.get("tool", "unknown_tool")
        artifact_status = str(
            artifact.get("status") or ("error" if artifact.get("error") is not None else "success")
        ).lower()
        has_error = artifact_status == "error" or artifact.get("error") is not None

        # Status badge styling
        if artifact_status == "rejected":
            status_badge_md = ":material/block: Rejected"
            status_badge_label = "Rejected"
            status_icon_name = "block"
            badge_color = COLORS["warning"]
        elif has_error:
            status_badge_md = ":material/error: Error"
            status_badge_label = "Error"
            status_icon_name = "error"
            badge_color = COLORS["error"]
        else:
            status_badge_md = ":material/check_circle: Success"
            status_badge_label = "Success"
            status_icon_name = "check_circle"
            badge_color = COLORS["success"]

        status_badge_html = (
            f'<span class="material-symbols-outlined" aria-hidden="true">{status_icon_name}</span>'
            f" {html.escape(status_badge_label)}"
        )

        # Create expander for each tool
        with st.expander(f"**[{idx}] {tool_name}** - {status_badge_md}", expanded=False):
            # Show execution status
            st.markdown(
                f'<div style="background-color: {badge_color}15; padding: 8px; border-radius: 4px; margin-bottom: 8px;">'
                f'<strong style="color: {badge_color};">Status:</strong> {status_badge_html}'
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
            render = artifact.get("render")
            if output is not None or isinstance(render, dict):
                st.markdown("**Output:**")
                if isinstance(render, dict) and render_tool_render_payload(
                    render,
                    fallback_output=output,
                ):
                    pass
                elif isinstance(output, (dict, list)):
                    render_json_output(output, label=f"{tool_name} Output", expanded=False)
                elif output is not None:
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
                st.caption(f":material/timer: Execution time: {execution_time:.2f}s")


def render_citations(message_metadata: dict[str, Any], msg_id: str | None = None):
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
                    if doc_id and doc_id in img_name or source and source in img_name:
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
                            '<div style="display: flex; flex-wrap: wrap; gap: 4px; margin: 4px 0;">'
                            + "".join(gallery_parts)
                            + "</div>",
                            unsafe_allow_html=True,
                        )


def render_thinking_summary(message_metadata: dict[str, Any]):
    """Render thinking summary from message metadata in a styled collapsible section."""
    thinking_summary = message_metadata.get("thinking_summary")
    if thinking_summary:
        with st.expander("Thought Process", expanded=False):
            # Convert markdown to HTML using the markdown library for reliable rendering
            formatted_html = _markdown.markdown(thinking_summary, extensions=["nl2br"])

            # Use custom styled container for thinking content
            st.markdown(
                f'<div class="thinking-container"><div class="thinking-content-rendered">{formatted_html}</div></div>',
                unsafe_allow_html=True,
            )


def render_reasoning_summary(message_metadata: dict[str, Any]):
    """Render OpenAI reasoning summary (not chain-of-thought) when available."""
    reasoning_summary = message_metadata.get("reasoning_summary")
    if not reasoning_summary:
        return

    title = "Reasoning (summary)"
    tokens = message_metadata.get("reasoning_tokens")
    if isinstance(tokens, int) and tokens >= 0:
        title = f"{title} - {tokens} tokens"

    with st.expander(title, expanded=False):
        # Convert markdown to HTML using the markdown library for reliable rendering
        formatted_html = _markdown.markdown(str(reasoning_summary), extensions=["nl2br"])

        st.markdown(
            f'<div class="thinking-container"><div class="thinking-content-rendered">{formatted_html}</div></div>',
            unsafe_allow_html=True,
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
            render_message_trace(message_metadata, expanded=False)

            provider = message_metadata.get("provider")
            model = message_metadata.get("model")
            if provider or model:
                if provider and model:
                    st.caption(f"{provider}:{model}")
                else:
                    st.caption(str(model or provider))

        st.markdown(content_text)  # Native markdown with LaTeX support

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
        render_agent_images(get_message_metadata(msg))

    # Show canvas artifact for assistant messages (HTML/SVG live preview)
    if not is_user:
        render_canvas_artifact(get_message_metadata(msg))

    # Show live widget cards for assistant messages
    if not is_user:
        render_live_widgets(
            get_message_metadata(msg),
            message_key=str(msg.get("id", "")),
            auto_mount=auto_mount_live_widgets,
        )

    # Show citations for assistant messages
    if not is_user:
        render_citations(get_message_metadata(msg), str(msg.get("id", "")))

    # Show feedback for assistant messages
    if not is_user:
        render_message_feedback_inline(msg)


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
            default_val = param_info.get("default", {} if param_type == "object" else [])
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
                                import time

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
                                import time

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
            for server in servers:
                server_name = server.get("name", "Unknown")
                enabled = server.get("enabled", False)
                tool_count = server.get("toolCount", 0)
                transport = server.get("transport", "unknown")
                description = server.get("description", "No description")

                status_icon = ":material/check_circle:" if enabled else ":material/cancel:"
                status_text = "Enabled" if enabled else "Disabled"

                col1, col2, col3 = st.columns([3, 1, 1])

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

    # Display tools as selectbox
    selected_tool_name = st.selectbox(
        "Select a tool to test",
        options=[tool.get("name") for tool in filtered_tools],
        format_func=lambda x: (
            f"{x} ({next((t.get('serverName', '') for t in filtered_tools if t.get('name') == x), '')})"
        ),
    )

    if not selected_tool_name:
        return

    # Get selected tool details
    selected_tool = next((t for t in filtered_tools if t.get("name") == selected_tool_name), None)

    if not selected_tool:
        return

    # Display tool details
    st.markdown("---")
    st.markdown(f"## {selected_tool.get('name')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Server:** `{selected_tool.get('serverName', 'Unknown')}`")
    with col2:
        st.markdown("**Type:** Tool")

    st.markdown(f"**Description:** {selected_tool.get('description', 'No description available')}")

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
        execute_button = st.form_submit_button("Execute Tool", width="stretch")

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
        if st.button("Clear Result"):
            st.session_state.tool_execution_result = None
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
            use_container_width=True,
        ):
            st.rerun()
    with col_reload:
        if st.button(
            "Reload from Disk",
            icon=":material/sync:",
            key="skills_reload",
            use_container_width=True,
        ):
            with st.spinner("Rescanning repo skills folder..."):
                result = reload_skills()
                if result:
                    msg = result.get("message", "Skills reloaded")
                    st.success(msg, icon=":material/check_circle:")
                    import time

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

    # Render each skill as a card
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

            col_view, col_spacer = st.columns([1, 1])

            with col_view:
                if st.button(
                    "View Content",
                    key=f"skill_view_{skill_name}",
                    icon=":material/visibility:",
                    use_container_width=True,
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

    # Key all pending decisions under the interrupt_id to avoid cross-contamination
    # on reload or when multiple interrupts occur in a single session.
    decisions_key = f"pending_decisions_{interrupt_id}" if interrupt_id else "pending_decisions"

    if interrupt_message:
        st.warning(f"**{interrupt_message}**", icon=":material/pause_circle:")

    if not action_requests:
        if st.button("Cancel"):
            st.session_state.pop("pending_interrupt", None)
            st.rerun()
        return

    # Initialize decisions in session state if not present for this interrupt
    if decisions_key not in st.session_state:
        st.session_state[decisions_key] = {}

    for idx, action_request in enumerate(action_requests):
        tool_name = (
            action_request.get("action")
            or action_request.get("tool")
            or action_request.get("name")
            or str(idx + 1)
        )
        tool_args = action_request.get("args", {})
        description = action_request.get("description", "")
        tool_call_id = (
            action_request.get("tool_call_id")
            or action_request.get("toolCallId")
            or action_request.get("id")
        )
        task_id = action_request.get("task_id") or action_request.get("taskId") or tool_call_id

        # Determine which decision types are allowed for this tool
        allowed_raw = action_request.get("allowed_decisions") or action_request.get(
            "allowedDecisions"
        )
        if allowed_raw:
            allowed_decisions = {str(d).strip().lower() for d in allowed_raw if isinstance(d, str)}
        else:
            allowed_decisions = {"approve", "edit", "reject"}

        # Check if this tool already has a decision
        current_decision = st.session_state[decisions_key].get(task_id)

        st.markdown(f"### Tool {idx + 1}: `{tool_name}`")
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
            if st.button("Change decision", key=f"change_{idx}"):
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
                ):
                    st.session_state[decisions_key][task_id] = {
                        "type": "approve",
                        "task_id": task_id,
                        "action": tool_name,
                        "args": None,
                    }
                    st.rerun()

            with col2:
                if "edit" in allowed_decisions and st.button(
                    "Edit Args", key=f"edit_{idx}", width="stretch"
                ):
                    st.session_state[f"editing_tool_{idx}"] = True
                    st.rerun()

            with col3:
                if "reject" in allowed_decisions and st.button(
                    "Reject", key=f"reject_{idx}", width="stretch"
                ):
                    st.session_state[decisions_key][task_id] = {
                        "type": "reject",
                        "task_id": task_id,
                        "action": tool_name,
                        "args": {},
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
                            "Save & Approve",
                            width="stretch",
                            type="primary",
                        ):
                            try:
                                edited_args = json.loads(edited_args_text)
                                st.session_state[decisions_key][task_id] = {
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
                        if st.form_submit_button("Cancel", width="stretch"):
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

    with col_submit:
        if st.button(
            "Submit Decisions",
            width="stretch",
            type="primary",
            disabled=not all_decided,
        ):
            submit_resume = True

    with col_approve_all:
        if st.button("Approve All", width="stretch"):
            # Auto-approve all remaining tools
            for req in action_requests:
                task_id = (
                    req.get("task_id")
                    or req.get("taskId")
                    or req.get("tool_call_id")
                    or req.get("toolCallId")
                    or req.get("id")
                )
                if task_id and task_id not in st.session_state[decisions_key]:
                    st.session_state[decisions_key][task_id] = {
                        "type": "approve",
                        "task_id": task_id,
                        "action": req.get("action"),
                        "args": None,
                    }
            submit_resume = True

    with col_cancel:
        if st.button("Cancel All", width="stretch"):
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
                    st.session_state[decisions_key][task_id] = {
                        "type": "reject",
                        "task_id": task_id,
                        "action": req.get("action"),
                        "args": {},
                    }
            submit_resume = True

    if submit_resume:
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
    }

    with st.status("Resuming execution...", expanded=True) as status:
        next_interrupt = None
        resume_error = None
        trace_placeholder = st.empty()
        response_placeholder = st.empty()
        accumulated_content = ""
        accumulated_thinking = ""

        _reset_stream_trace_state(expanded=True)

        for event in make_streaming_request("/messages/resume-interrupt", resume_payload):
            event_type = event.get("type")

            if event_type == "agent_selected":
                agent_name = event.get("agent", "unknown")
                status.update(
                    label=f"{agent_name.replace('_', ' ').title()} is processing...",
                    state="running",
                )
                continue

            if event_type == "thinking":
                content = event.get("content", "")
                accumulated_thinking += content
                _upsert_stream_thinking_trace(accumulated_thinking)
                st.session_state.stream_trace_expanded = True
                render_live_trace_panel(trace_placeholder)
                status.update(label="Thinking...", state="running")
                continue

            if event_type == "tool":
                _upsert_stream_tool_trace(event)
                render_live_trace_panel(trace_placeholder)
                tool_name = event.get("name", "unknown")
                tool_phase = event.get("phase") or event.get("status") or "unknown"
                status.update(
                    label=f"Tool: {tool_name} ({tool_phase})",
                    state="running",
                )
                continue

            if event_type == "token":
                content = event.get("content", "")
                accumulated_content += content
                if st.session_state.get("stream_trace_items"):
                    st.session_state.stream_trace_expanded = False
                    render_live_trace_panel(trace_placeholder)
                response_placeholder.markdown(accumulated_content)
                status.update(label="Resuming response...", state="running")
                continue

            if event_type == "interrupt":
                next_interrupt = event.get("interrupt")
                interrupt_message = extract_interrupt_message(next_interrupt)
                if interrupt_message:
                    status.update(label=interrupt_message, state="running")
                break

            if event_type == "error":
                resume_error = event.get("error") or "Failed to resume execution"
                status.update(label=f"Error: {resume_error}", state="error")
                break

            if event_type == "complete":
                status.update(label="Resume completed", state="complete")
                break

        _clear_inflight_state()

        # Clear decisions for this interrupt
        st.session_state.pop(decisions_key, None)
        for idx in range(len(action_requests)):
            st.session_state.pop(f"editing_tool_{idx}", None)

        if resume_error:
            st.error(resume_error)
            return

        if next_interrupt:
            st.session_state.pending_interrupt = next_interrupt
            st.session_state.interrupt_conversation_id = conversation_id
            next_interrupt_message = extract_interrupt_message(next_interrupt)
            if next_interrupt_message:
                st.toast(next_interrupt_message, icon=":material/warning:")
            st.rerun()

        st.session_state.pop("pending_interrupt", None)
        st.session_state.pop("interrupt_conversation_id", None)
        st.session_state.conversation_messages_page = 0
        st.rerun()


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
                            st.session_state.pending_interrupt = _interrupt_data
                            st.session_state.interrupt_conversation_id = conversation_id
                    break  # only check the most recent assistant message

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
                completed = status.get("completedTasks", 0)
                total = status.get("totalTasks", 0)

                with st.container():
                    # Progress bar and summary
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.progress(progress_pct / 100.0)
                        if next_task:
                            st.success(
                                f":material/checklist: **{completed}/{total}** tasks complete | "
                                f"**Current:** {next_task.get('description', 'N/A')[:50]}..."
                                if len(next_task.get("description", "")) > 50
                                else f":material/checklist: **{completed}/{total}** tasks complete | "
                                f"**Current:** {next_task.get('description', 'N/A')}"
                            )
                        else:
                            st.success(f":material/check_circle: **All {total} tasks completed!**")
                    with col2:
                        if st.button(
                            ":material/checklist: View All",
                            key="goto_planning_from_chat",
                            help="View Full Task List",
                        ):
                            st.session_state.active_view = "planning"
                            st.rerun()

                    # Show task list in expander
                    tasks = get_task_plans(conversation_id, include_completed=True)
                    if tasks:
                        task_list = tasks.get("data", []) if isinstance(tasks, dict) else tasks
                        if task_list:
                            with st.expander(":material/list_alt: Task Progress", expanded=False):
                                for task in task_list:
                                    task_status = task.get("status", "pending")
                                    task_desc = task.get("description", "No description")

                                    if task_status == "completed":
                                        st.markdown(f":material/check_circle: ~~{task_desc}~~")
                                    elif task_status == "in_progress":
                                        st.markdown(f":material/refresh: **{task_desc}** (current)")
                                    elif task_status == "skipped":
                                        st.markdown(
                                            f":material/skip_next: ~~{task_desc}~~ (skipped)"
                                        )
                                    else:
                                        st.markdown(f":material/pending: {task_desc}")

    elif conversation_id == "pending_new":
        st.markdown("# New Chat")
        queued_persona = st.session_state.get("pending_persona_prompt", "")
        if queued_persona:
            st.info(f"**Instructions queued:** {persona_preview(queued_persona, 100)}")
    else:
        st.markdown("# Welcome!")
        st.info("Select a conversation from the sidebar or create a new chat to get started.")
        return

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

        # File uploader
        file_uploader_key = f"chat_image_uploader_{conversation_id}" if conversation_id else None

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
        # ── Handle interrupted stream on rerun (Phase 2 of two-phase stop) ──
        if (
            st.session_state.get("stream_inflight")
            and st.session_state.get("stream_user_message_id")
            and conversation_id
        ):
            _handle_stop_rerun(str(conversation_id))
            return

        # Check for pending suggestion from suggestion buttons
        pending_suggestion = st.session_state.pop("pending_suggestion", "")

        # ── Message form ──
        # Capture form values first; heavy processing (streaming) happens AFTER
        # the form context exits so we can freely use st.button() etc.
        _form_send = False
        _form_attach = False
        _form_message = ""

        with st.form("message_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([6, 1, 1])

            with col1:
                _form_message = st.text_area(
                    "Message",
                    value=pending_suggestion,
                    placeholder="Type your message...",
                    height=100,
                    label_visibility="collapsed",
                    key=f"msg_input_{conversation_id}",
                )

            with col2:
                _form_send = st.form_submit_button(
                    "\nSend", use_container_width=True, type="primary"
                )

            with col3:
                _form_attach = st.form_submit_button("Attach", use_container_width=True)

        # Placeholder for the "Stop generating" button lives OUTSIDE the form
        # but directly below it, so it appears next to Send / Attach.
        stop_button_placeholder = st.empty()

        # ── Process form actions OUTSIDE the form context ──
        if _form_attach:
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
                }

                if pending_attachments:
                    message_data["attachments"] = pending_attachments

                # Use streaming endpoint for real-time response
                with st.status("Sending message...", expanded=True) as status:
                    # Create placeholder for streaming response
                    trace_placeholder = st.empty()
                    response_placeholder = st.empty()
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
                            status.update(
                                label=f"{selected_agent.replace('_', ' ').title()} is processing...",
                                state="running",
                            )

                        elif event_type == "thinking":
                            content = event.get("content", "")
                            accumulated_thinking += content
                            st.session_state.stream_partial_thinking = accumulated_thinking
                            _upsert_stream_thinking_trace(accumulated_thinking)
                            st.session_state.stream_trace_expanded = True
                            render_live_trace_panel(trace_placeholder)
                            status.update(label="Thinking...", state="running")

                        elif event_type == "token":
                            content = event.get("content", "")
                            accumulated_content += content  # Append each token chunk
                            st.session_state.stream_partial_text = accumulated_content
                            if st.session_state.get("stream_trace_items"):
                                st.session_state.stream_trace_expanded = False
                                render_live_trace_panel(trace_placeholder)
                            # Display with native markdown for LaTeX support
                            response_placeholder.markdown(accumulated_content)

                        elif event_type == "tool":
                            _upsert_stream_tool_trace(event)
                            render_live_trace_panel(trace_placeholder)
                            tool_name = event.get("name", "unknown")
                            tool_status = event.get("phase") or event.get("status") or "running"
                            status.update(
                                label=f"Tool: {tool_name} ({tool_status})",
                                state="running",
                            )

                        elif event_type == "interrupt":
                            # Workflow paused for human approval
                            event.get("thread_id")
                            event.get("pending_tool_calls") or []
                            # Extract the full interrupt response data
                            interrupt_data = event.get("interrupt")
                            interrupt_message = extract_interrupt_message(interrupt_data)
                            if interrupt_message:
                                status.update(label=interrupt_message, state="running")

                            # Store interrupt state in session for the approval UI
                            if interrupt_data:
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
                        use_container_width=True,
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
                width="stretch",
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
                st.session_state.persona_editor_pending_value = sanitized
                st.session_state.persona_editor_pending = True
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
                st.session_state.persona_editor_pending_value = ""
                st.session_state.persona_editor_pending = True
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
        tab_mcp,
        tab_skills,
    ) = st.tabs(
        [
            ":material/chat: Chat",
            ":material/checklist: Planning",
            ":material/description: Documents",
            ":material/settings: Instructions",
            ":material/smart_toy: Models",
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

    with tab_mcp:
        render_tools_tab()

    with tab_skills:
        render_skills_tab()


if __name__ == "__main__":
    main()
