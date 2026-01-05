import streamlit as st # type: ignore
import requests
import time
from typing import Dict, Any, Optional

API_BASE_URL = "http://localhost:8000"


def render_upload_section():
    """Render file upload section - only when conversation is selected"""
    # Only show upload section if a conversation is selected (not pending or None)
    if (
        st.session_state.get("current_conversation_id")
        and st.session_state.current_conversation_id != "pending_new"
    ):
        st.markdown("### 📁 Upload Documents")
        st.caption(f"Upload to current conversation")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["txt", "pdf", "docx", "md"],
            help="Upload documents for this conversation",
            key=f"uploader_{st.session_state.current_conversation_id}",  # Unique key per conversation
        )

        if uploaded_file is not None:
            # Display file info
            st.info(f"📄 {uploaded_file.name} ({uploaded_file.size} bytes)")

            if st.button(
                "Upload File",
                use_container_width=True,
                key=f"upload_btn_{st.session_state.current_conversation_id}",
            ):
                upload_result = upload_document(uploaded_file)
                if upload_result:
                    st.success("✅ File uploaded successfully!")
                    # Clear cache to refresh data
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("❌ Upload failed")


def upload_document(uploaded_file) -> Optional[Dict[str, Any]]:
    """
    Upload document to the API with conversation context

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Dict containing upload response or None if failed
    """
    try:
        # Prepare file for upload
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

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
        response = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files=files,
            data=data,
            headers=headers,
        )

        if response.status_code == 201:
            result = response.json()
            # Start polling for status updates
            if result and result.get("id"):
                poll_document_status(result["id"])
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
                status_placeholder.warning(
                    "Processing timeout - please refresh manually"
                )
                break

            # Get current status
            doc_status = get_document_status(document_id)

            if doc_status:
                status_code = doc_status.get("status")
                filename = doc_status.get("filename", "Unknown")

                # Status: 1=Processing, 2=Ready, 3=Failed
                if status_code == 1:
                    status_placeholder.info(
                        f"⏳ Processing '{filename}'... ({int(elapsed_time)}s elapsed)"
                    )
                elif status_code == 2:
                    status_placeholder.success(
                        f"✅ '{filename}' is ready! Processing completed in {int(elapsed_time)}s"
                    )
                    time.sleep(2)  # Show success message briefly
                    status_placeholder.empty()
                    break
                elif status_code == 3:
                    status_placeholder.error(f"❌ '{filename}' processing failed")
                    break
                else:
                    status_placeholder.warning(f"❓ Unknown status for '{filename}'")
                    break
            else:
                status_placeholder.warning("⚠️ Unable to fetch document status")
                break

            # Wait before next poll
            time.sleep(poll_interval)

    except Exception as e:
        status_placeholder.error(f"Error monitoring status: {str(e)}")


def get_document_status(document_id: str) -> Optional[Dict[str, Any]]:
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

        response = requests.get(
            f"{API_BASE_URL}/documents/{document_id}",
            headers=headers,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None

    except Exception as e:
        return None


def get_uploaded_documents() -> Dict[str, Any]:
    """Get list of uploaded documents for current conversation"""
    try:
        headers = {}
        if st.session_state.get("auth_token"):
            headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

        # Get documents for the specific conversation
        if (
            st.session_state.get("current_conversation_id")
            and st.session_state.current_conversation_id != "pending_new"
        ):
            response = requests.get(
                f"{API_BASE_URL}/documents/conversation/{st.session_state.current_conversation_id}",
                headers=headers,
            )
        else:
            return {}

        if response.status_code == 200:
            return response.json()
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
        with st.expander("📚 Conversation Documents", expanded=False):
            # Add refresh button
            if st.button("🔄 Refresh Status", key="refresh_docs"):
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
                            1: {"icon": "⏳", "text": "Processing", "color": "orange"},
                            2: {"icon": "✅", "text": "Ready", "color": "green"},
                            3: {"icon": "❌", "text": "Failed", "color": "red"},
                        }

                        status_info = status_map.get(
                            doc.get("status"),
                            {"icon": "❓", "text": "Unknown", "color": "gray"},
                        )

                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.text(
                                f"{status_info['icon']} {doc.get('filename', 'Unknown')}"
                            )
                            st.caption(
                                f"Uploaded: {doc.get('upload_time', 'N/A')[:16]}"
                            )
                        with col2:
                            st.markdown(
                                f":{status_info['color']}[**{status_info['text']}**]"
                            )
                        with col3:
                            if st.button(
                                "🗑️",
                                key=f"del_{doc.get('id')}",
                                help="Delete document",
                            ):
                                if delete_document(doc.get("id")):
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

        response = requests.delete(
            f"{API_BASE_URL}/documents/{document_id}",
            headers=headers,
        )

        if response.status_code == 200:
            st.success("Document deleted successfully")
            st.cache_data.clear()
            return True
        else:
            st.error(f"Delete failed: {response.status_code}")
            return False

    except Exception as e:
        st.error(f"Delete error: {str(e)}")
        return False
