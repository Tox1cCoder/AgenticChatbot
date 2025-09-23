"""
Document Upload Support for Chatbot Demo

This module provides file upload functionality that was present in the original demo.
"""

import streamlit as st
import requests
from typing import Dict, Any, Optional


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
            f"{st.session_state.get('API_BASE_URL', 'http://localhost:8000')}/documents/upload",
            files=files,
            data=data,
            headers=headers,
        )

        if response.status_code == 201:
            return response.json()
        else:
            st.error(f"Upload failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None


def get_uploaded_documents() -> Dict[str, Any]:
    """Get list of uploaded documents for current user"""
    try:
        headers = {}
        if st.session_state.get("auth_token"):
            headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

        response = requests.get(
            f"{st.session_state.get('API_BASE_URL', 'http://localhost:8000')}/documents/",
            headers=headers,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {}

    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return {}


def render_document_list():
    """Render list of uploaded documents for current conversation"""
    # Only show document list if a conversation is selected
    if (
        st.session_state.get("current_conversation_id")
        and st.session_state.current_conversation_id != "pending_new"
    ):
        with st.expander("📚 Conversation Documents", expanded=False):
            docs = get_uploaded_documents()
            if docs and docs.get("data"):
                # Filter documents for current conversation if conversation_id is available
                conversation_docs = []
                for doc in docs["data"]:
                    # If document has conversation_id, only show if it matches current conversation
                    doc_conv_id = doc.get("conversation_id")
                    if (
                        doc_conv_id is None
                        or doc_conv_id == st.session_state.current_conversation_id
                    ):
                        conversation_docs.append(doc)

                if conversation_docs:
                    for doc in conversation_docs:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(doc.get("filename", "Unknown"))
                            if doc.get("conversation_id"):
                                st.caption("📎 Linked to conversation")
                            else:
                                st.caption("📄 Global document")
                        with col2:
                            if st.button(
                                "🗑️",
                                key=f"del_{doc.get('id')}_{st.session_state.current_conversation_id}",
                                help="Delete",
                            ):
                                delete_document(doc.get("id"))
                                st.rerun()
                else:
                    st.text("No documents in this conversation")
            else:
                st.text("No documents uploaded yet")


def delete_document(document_id: str) -> bool:
    """Delete a document by ID"""
    try:
        headers = {}
        if st.session_state.get("auth_token"):
            headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

        response = requests.delete(
            f"{st.session_state.get('API_BASE_URL', 'http://localhost:8000')}/documents/{document_id}",
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
