import time
from typing import Any

import requests
import streamlit as st  # type: ignore
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE_URL = "http://localhost:8000"
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
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _bump_api_cache_version() -> None:
    """Invalidate cached GET responses after mutations."""
    st.session_state.api_cache_version = int(st.session_state.get("api_cache_version", 0)) + 1


def render_upload_section():
    """Render file upload section - only when conversation is selected"""
    # Only show upload section if a conversation is selected (not pending or None)
    if (
        st.session_state.get("current_conversation_id")
        and st.session_state.current_conversation_id != "pending_new"
    ):
        st.markdown("### :material/upload_file: Upload Documents")
        st.caption("Upload to current conversation")

        uploaded_file = st.file_uploader(
            "Choose a file",
            # Mirror app/api/documents.py::SUPPORTED_UPLOAD_EXTENSIONS.
            type=["txt", "pdf", "docx", "pptx", "xlsx", "html", "md"],
            help="Upload documents for this conversation",
            key=f"uploader_{st.session_state.current_conversation_id}",  # Unique key per conversation
        )

        if uploaded_file is not None:
            # Display file info
            st.info(
                f"{uploaded_file.name} ({uploaded_file.size} bytes)",
                icon=":material/description:",
            )

            if st.button(
                "Upload File",
                icon=":material/upload:",
                width="stretch",
                key=f"upload_btn_{st.session_state.current_conversation_id}",
            ):
                upload_result = upload_document(uploaded_file)
                if upload_result:
                    st.success("File uploaded successfully!", icon=":material/check_circle:")
                    # Clear cache to refresh data
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Upload failed", icon=":material/cancel:")


def upload_document(uploaded_file) -> dict[str, Any] | None:
    """
    Upload document to the API with conversation context

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Dict containing upload response or None if failed
    """
    try:
        # Prepare file for upload
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        # Prepare form data with conversation ID
        data = {}
        if (
            st.session_state.get("current_conversation_id")
            and st.session_state.current_conversation_id != "pending_new"
        ):
            data["conversation_id"] = st.session_state.current_conversation_id

        # Get auth token from session
        headers = {}
        if st.session_state.get("auth_token"):
            headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

        # Make upload request
        response = get_http_session().post(
            f"{API_BASE_URL}/documents/upload",
            files=files,
            data=data,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 201:
            result = response.json()
            _bump_api_cache_version()
            return result
        else:
            st.error(f"Upload failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None


def poll_document_status(document_id: str):
    """
    Poll document status and display real-time updates

    Args:
        document_id: ID of the document to monitor
    """
    status_placeholder = st.empty()
    start_time = time.time()
    max_wait_time = 300
    poll_interval = 3

    try:
        while True:
            elapsed_time = time.time() - start_time

            # Check timeout
            if elapsed_time > max_wait_time:
                status_placeholder.warning("Processing timeout - please refresh manually")
                break

            # Get current status
            doc_status = get_document_status(document_id)

            if doc_status:
                status_code = doc_status.get("status")
                filename = doc_status.get("filename", "Unknown")

                # Status: 1=Processing, 2=Ready, 3=Failed
                if status_code == 1:
                    status_placeholder.info(
                        f"Processing '{filename}'... ({int(elapsed_time)}s elapsed)",
                        icon=":material/schedule:",
                    )
                elif status_code == 2:
                    status_placeholder.success(
                        f"'{filename}' is ready! Processing completed in {int(elapsed_time)}s",
                        icon=":material/check_circle:",
                    )
                    time.sleep(2)  # Show success message briefly
                    status_placeholder.empty()
                    break
                elif status_code == 3:
                    status_placeholder.error(
                        f"'{filename}' processing failed", icon=":material/cancel:"
                    )
                    break
                else:
                    status_placeholder.warning(
                        f"Unknown status for '{filename}'", icon=":material/help:"
                    )
                    break
            else:
                status_placeholder.warning(
                    "Unable to fetch document status", icon=":material/warning:"
                )
                break

            # Wait before next poll
            time.sleep(poll_interval)

    except Exception as e:
        status_placeholder.error(f"Error monitoring status: {str(e)}")


def get_document_status(document_id: str) -> dict[str, Any] | None:
    """
    Get status of a single document

    Args:
        document_id: ID of the document

    Returns:
        Dict containing document info or None if failed
    """
    try:
        headers = {}
        if st.session_state.get("auth_token"):
            headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

        response = get_http_session().get(
            f"{API_BASE_URL}/documents/{document_id}",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None

    except Exception:
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


def render_document_list():
    """Render list of uploaded documents for current conversation with status"""
    # Only show document list if a conversation is selected
    if (
        st.session_state.get("current_conversation_id")
        and st.session_state.current_conversation_id != "pending_new"
    ):
        with st.expander(":material/menu_book: Conversation Documents", expanded=False):
            # Add refresh button
            if st.button("Refresh Status", icon=":material/refresh:", key="refresh_docs"):
                st.cache_data.clear()
                st.rerun()

            docs_response = get_uploaded_documents()

            if docs_response and docs_response.get("data"):
                doc_list = docs_response.get("data", {})
                documents = doc_list.get("documents", [])

                if documents:
                    st.caption(f"Total: {doc_list.get('total', 0)} document(s)")

                    for doc in documents:
                        # Status mapping
                        status_map = {
                            1: {
                                "icon": ":material/schedule:",
                                "text": "Processing",
                                "color": "orange",
                            },
                            2: {
                                "icon": ":material/check_circle:",
                                "text": "Ready",
                                "color": "green",
                            },
                            3: {
                                "icon": ":material/cancel:",
                                "text": "Failed",
                                "color": "red",
                            },
                        }

                        status_info = status_map.get(
                            doc.get("status"),
                            {
                                "icon": ":material/help:",
                                "text": "Unknown",
                                "color": "gray",
                            },
                        )

                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"{status_info['icon']} {doc.get('filename', 'Unknown')}")
                            st.caption(f"Uploaded: {doc.get('upload_time', 'N/A')[:16]}")
                        with col2:
                            st.markdown(f":{status_info['color']}[**{status_info['text']}**]")
                        with col3:
                            if st.button(
                                "Delete",
                                icon=":material/delete:",
                                key=f"del_{doc.get('id')}",
                                help="Delete document",
                            ) and delete_document(doc.get("id")):
                                st.cache_data.clear()
                                st.rerun()

                        st.divider()
                else:
                    st.info("No documents in this conversation")
            else:
                st.info("No documents uploaded yet")


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
