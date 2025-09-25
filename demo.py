import streamlit as st
import requests
import json
from typing import Dict, Optional, Any, List
from upload_support import render_upload_section, render_document_list
from datetime import datetime, timedelta
from dateutil import parser

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ChatBot", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
    .main-container { max-width: 1200px; margin: 0 auto; }
    .chat-messages-container {
        border: 1px solid #e2e8f0; border-radius: 18px; background: #ffffff;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }
    .chat-input-container {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }
    .user-message {
        background: #ffffff; color: #1f2937; border: 1px solid #93c5fd;
        padding: 12px 16px; border-radius: 18px 18px 4px 18px; margin: 8px 0 8px auto;
        max-width: 72%; word-wrap: break-word; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.16);
    }
    .bot-message {
        background: #ffffff; color: #1f2937; border: 1px solid #86efac;
        padding: 12px 16px; border-radius: 18px 18px 18px 4px; margin: 8px auto 8px 0;
        max-width: 72%; word-wrap: break-word; box-shadow: 0 8px 20px rgba(34, 197, 94, 0.16);
    }
    .sidebar-conversation {
        background: #f8f9fa; border-radius: 8px; padding: 10px; margin: 5px 0;
        cursor: pointer; border: 1px solid #dee2e6; transition: all 0.3s ease;
    }
    .sidebar-conversation:hover { background: #e9ecef; transform: translateX(5px); }
    .sidebar-conversation.active {
        background: linear-gradient(135deg, #28a745, #1e7e34); color: white; border-color: #1e7e34;
    }
    .login-container {
        max-width: 400px; margin: 0 auto; padding: 40px 20px; background: white;
        border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .message-timestamp { font-size: 0.8em; color: #64748b; margin-top: 5px; }
    div:empty { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = None
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations_list" not in st.session_state:
    st.session_state.conversations_list = []
if "show_login" not in st.session_state:
    st.session_state.show_login = (
        "auth_token" not in st.session_state or not st.session_state.get("auth_token")
    )
if "show_conversation_manager" not in st.session_state:
    st.session_state.show_conversation_manager = False
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "conversation_messages_meta" not in st.session_state:
    st.session_state.conversation_messages_meta = None
if "conversation_messages_page" not in st.session_state:
    st.session_state.conversation_messages_page = 0
if "has_more_messages" not in st.session_state:
    st.session_state.has_more_messages = True


def reset_conversation_state() -> None:
    st.session_state.messages = []
    st.session_state.conversation_messages_meta = None
    st.session_state.conversation_messages_page = 0
    st.session_state.has_more_messages = True


def make_api_request(method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    if st.session_state.get("auth_token"):
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

    try:
        response = getattr(requests, method.lower())(url, json=data, headers=headers)
        response_data = response.json()

        if not response_data.get("success"):
            error_code = response_data.get("code", "unknown_error")
            error_message = response_data.get("message", "An unknown error occurred.")

            if error_code == "unauthenticated":
                st.error(
                    f"🔒 Authentication required. Please log in. (Error: {error_message})"
                )
                st.session_state.auth_token = None
                st.session_state.current_user_id = None
                st.session_state.show_login = True
            else:
                st.error(f"❌ API Error ({error_code}): {error_message}")
            return {}

        return response_data
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to API. Make sure FastAPI server is running on localhost:8000"
        )
        return {}
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {}


@st.cache_data(show_spinner=False)
def get_user(user_id: str) -> Dict[str, Any]:
    response = make_api_request("GET", f"/users/{user_id}")
    return response.get("data", {})


@st.cache_data(show_spinner=False)
def get_conversations(page: int = 1, limit: int = 20) -> Dict[str, Any]:
    """Get paginated conversations"""
    response = make_api_request("GET", f"/conversations/?page={page}&limit={limit}")
    return response  # Returns full response with meta and items


@st.cache_data(show_spinner=False)
def get_messages(
    conversation_id: str,
    page: int = 1,
    limit: int = 10,
    order_by: str = "created_at",
    order_direction: str = "desc",
) -> Dict[str, Any]:
    """Get paginated conversation messages"""
    endpoint = (
        f"/conversations/{conversation_id}/messages"
        f"?page={page}&limit={limit}&orderBy={order_by}&orderDirection={order_direction}"
    )
    response = make_api_request("GET", endpoint)
    return response  # Returns full response with meta and items


@st.cache_data(show_spinner=False)
def get_user_messages_paginated(
    page: int = 1,
    limit: int = 10,
    order_by: str = "created_at",
    order_direction: str = "asc",
) -> Dict[str, Any]:
    """Get paginated user messages"""
    response = make_api_request(
        "GET",
        f"/messages/?page={page}&limit={limit}&order_by={order_by}&order_direction={order_direction}",
    )
    return response  # Returns full response with meta and items


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
                        st.success("✅ Signed in successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")

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
                        st.error("❌ Please fill out all fields")
                    elif password != confirm_password:
                        st.error("❌ Passwords do not match")
                    else:
                        user_data = {
                            "username": username,
                            "email": email,
                            "password": password,
                        }
                        result = make_api_request("POST", "/auth/signup", user_data)
                        if result:
                            st.cache_data.clear()
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
                                    "✅ Account created and signed in successfully!"
                                )
                                st.rerun()
                            else:
                                st.success("✅ Account created! Please sign in.")
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

        st.divider()

        if (
            not st.session_state.conversations_list
            and st.session_state.current_user_id
            and st.session_state.auth_token
        ):
            conversations_response = get_conversations()
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

                if st.button(
                    conv["title"],
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    if conv["id"] != st.session_state.current_conversation_id:
                        st.session_state.current_conversation_id = conv["id"]
                        reset_conversation_state()
                        st.rerun()

        # Document upload and list section for selected conversation
        # render_upload_section()
        # render_document_list()

        st.divider()

        if st.session_state.current_user_id:
            user = get_user(st.session_state.current_user_id)
            if user:
                st.markdown(f"**👤 {user['username']}**")
                if st.button("🚪 Sign Out", use_container_width=True):
                    st.session_state.current_user_id = None
                    st.session_state.current_conversation_id = None
                    reset_conversation_state()
                    st.session_state.conversations_list = []
                    st.session_state.auth_token = None
                    st.session_state.show_login = True
                    st.rerun()


def render_conversation_manager():
    if st.session_state.show_conversation_manager:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("### Conversation Manager")

            search_term = st.text_input(
                "🔍 Search conversations:", placeholder="Type to search..."
            )

            if st.session_state.conversations_list:
                filtered_convs = st.session_state.conversations_list

                if search_term:
                    filtered_convs = []
                    for conv in st.session_state.conversations_list:
                        messages_response = get_messages(conv["id"])
                        if messages_response and messages_response.get("data"):
                            messages = messages_response["data"]["items"]
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
                        with st.expander(f"💬 {conv['title']}", expanded=False):
                            messages_response = get_messages(conv["id"])
                            if messages_response and messages_response.get("data"):
                                messages = messages_response["data"]["items"]
                                if messages:
                                    recent_messages = sorted(
                                        messages,
                                        key=lambda x: x.get("createdAt", ""),
                                        reverse=True,
                                    )[:3]
                                    for msg in recent_messages:
                                        sender_value = msg.get("sender")
                                        sender_icon = (
                                            "👤"
                                            if sender_value in (1, "user")
                                            else "🤖"
                                        )
                                        st.markdown(
                                            f"{sender_icon} **{msg['sender']}:** {msg['content'][:100]}..."
                                        )

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
                                        st.cache_data.clear()
                                        st.success(f"✅ Deleted '{conv['title']}'")
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
        conversations_response = get_conversations()
        if conversations_response and conversations_response.get("data"):
            st.session_state.conversations_list = conversations_response["data"][
                "items"
            ]

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

            existing_messages = {msg["id"]: msg for msg in st.session_state.messages}
            for item in items:
                existing_messages[item["id"]] = item

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

            current_page = meta.get("current_page", page)
            last_page = meta.get("last_page", current_page)
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
    elif conversation_id == "pending_new":
        st.markdown("# New Chat - Start typing to begin!")
    else:
        st.markdown("# Welcome! Select a conversation to view messages")

    if conversation_id and conversation_id not in (None, "pending_new"):
        if st.session_state.has_more_messages:
            if st.button("⬆️ Load older messages", key="load_more_messages"):
                next_page = st.session_state.conversation_messages_page + 1
                load_messages_page(next_page, show_spinner=True)
        elif st.session_state.conversation_messages_page > 0:
            st.caption("All caught up — showing the entire thread.")

    st.markdown(
        """<div class="chat-messages-container" style="height: 70vh; overflow-y: auto; padding: 20px; margin-bottom: 20px;">""",
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
            "<div style='text-align:center; color:#94a3b8;'>No messages yet — send the first one!</div>",
            unsafe_allow_html=True,
        )

    for msg in messages_to_display:
        if msg["sender"] == 1:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px 0; align-items: flex-start; gap: 10px;">
                    <div class="user-message">
                        {msg["content"].replace('<', '&lt;').replace('>', '&gt;')}
                        <div class="message-timestamp">You • {format_time(msg.get("createdAt", "now"))}</div>
                    </div>
                    <div style="border: 2px solid #007bff; color: #007bff; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">👤</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0; align-items: flex-start; gap: 10px;">
                    <div style="border: 2px solid #28a745; color: #28a745; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">🤖</div>
                    <div class="bot-message">
                        {msg["content"].replace('<', '&lt;').replace('>', '&gt;')}
                        <div class="message-timestamp">Assistant • {format_time(msg.get("createdAt", "now"))}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    if conversation_id:
        st.markdown(
            '<div class="chat-input-container" style="background: white; padding: 25px; border-radius: 18px; border: 1px solid #e2e8f0; box-shadow: 0 10px 24px rgba(15,23,42,0.08); margin-top: 20px;">',
            unsafe_allow_html=True,
        )

        with st.form("message_form", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "Upload a document for context", type=["pdf", "txt", "docx"]
            )

            col1, col2 = st.columns([4, 1])
            with col1:
                message_content = st.text_area(
                    "Message",
                    placeholder="Type your message here...",
                    height=80,
                    label_visibility="collapsed",
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                send_button = st.form_submit_button("Send", use_container_width=True)

            if send_button:
                if uploaded_file is not None:
                    with st.spinner("Processing uploaded file..."):
                        files = {"file": uploaded_file}
                        doc_response = make_api_request(
                            "POST", "/documents/", files=files
                        )
                        if doc_response and doc_response.get("data"):
                            st.success(f"✅ Document uploaded: {uploaded_file.name}")
                        else:
                            st.error("❌ Failed to upload document")
                            return

                if message_content.strip():
                    if conversation_id == "pending_new":
                        conversation_data = {
                            "title": (
                                message_content[:50] + "..."
                                if len(message_content) > 50
                                else message_content
                            )
                        }
                        conv_response = make_api_request(
                            "POST", "/conversations/", conversation_data
                        )
                        if conv_response and conv_response.get("data"):
                            st.session_state.current_conversation_id = conv_response[
                                "data"
                            ]["id"]
                            st.cache_data.clear()
                            reset_conversation_state()
                        else:
                            st.error("Failed to create conversation")
                            return

                    message_data = {
                        "content": message_content,
                        "conversation_id": st.session_state.current_conversation_id,
                    }

                    with st.spinner("Thinking..."):
                        response = make_api_request("POST", "/messages/", message_data)

                    if response and response.get("data"):
                        st.cache_data.clear()
                        reset_conversation_state()
                        load_messages_page(1)
                        st.rerun()
                    else:
                        st.error("Failed to send message")

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

    if st.session_state.show_conversation_manager:
        render_conversation_manager()
        return

    render_chat_interface()


if __name__ == "__main__":
    main()
