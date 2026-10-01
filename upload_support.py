import os
from typing import Any

import requests
import streamlit as st  # type: ignore
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.ui.sidecar_session import SidecarSession

API_BASE_URL = os.environ.get("CHATBOT_API_BASE_URL", "http://127.0.0.1:8100")
REQUEST_TIMEOUT = (5, 30)


@st.cache_resource(show_spinner=False)
def get_http_session() -> requests.Session:
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
        pool_connections=10,
        pool_maxsize=20,
    )
    # Sends the sidecar's launch token (X-Kani-Client) and re-reads it after a restart.
    session = SidecarSession(API_BASE_URL)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _bump_api_cache_version() -> None:
    """Invalidate cached GET responses after mutations."""
    st.session_state.api_cache_version = int(st.session_state.get("api_cache_version", 0)) + 1


def upload_documents(uploaded_files: list[Any]) -> dict[str, Any] | None:
    """Upload one or more files via the canonical batch endpoint."""
    if not uploaded_files:
        return None

    try:
        files = [
            ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
            for uploaded_file in uploaded_files
        ]

        data: dict[str, str] = {}
        if (
            st.session_state.get("current_conversation_id")
            and st.session_state.current_conversation_id != "pending_new"
        ):
            data["conversation_id"] = st.session_state.current_conversation_id

        headers: dict[str, str] = {}
        if st.session_state.get("auth_token"):
            headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

        response = get_http_session().post(
            f"{API_BASE_URL}/documents/uploads",
            files=files,
            data=data,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code in {201, 207, 409}:
            result = response.json()
            accepted_count = int((result.get("data") or {}).get("accepted_count") or 0)
            if accepted_count:
                _bump_api_cache_version()
            return result
        st.error(f"Upload failed: {response.status_code} - {response.text}")
        return None

    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None


@st.cache_data(show_spinner=False, ttl=10, max_entries=200)
def _cached_conversation_documents(
    conversation_id: str, auth_token: str, cache_version: int
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    response = get_http_session().get(
        f"{API_BASE_URL}/documents/conversation/{conversation_id}",
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 200:
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    return {}


def get_uploaded_documents() -> dict[str, Any]:
    """Get list of uploaded documents for current conversation"""
    try:
        # Get documents for the specific conversation
        if (
            st.session_state.get("current_conversation_id")
            and st.session_state.current_conversation_id != "pending_new"
        ):
            return _cached_conversation_documents(
                conversation_id=str(st.session_state.current_conversation_id),
                auth_token=str(st.session_state.get("auth_token") or ""),
                cache_version=int(st.session_state.get("api_cache_version", 0)),
            )
        else:
            return {}

    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return {}


def delete_document(document_id: str) -> bool:
    """Delete a document by ID"""
    try:
        headers = {}
        if st.session_state.get("auth_token"):
            headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

        response = get_http_session().delete(
            f"{API_BASE_URL}/documents/{document_id}",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 200:
            st.success("Document deleted successfully")
            _bump_api_cache_version()
            st.cache_data.clear()
            return True
        else:
            st.error(f"Delete failed: {response.status_code}")
            return False

    except Exception as e:
        st.error(f"Delete error: {str(e)}")
        return False
