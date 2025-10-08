import streamlit as st
import requests
import json
import html
import re
from typing import Dict, Optional, Any, List
from upload_support import render_upload_section, render_document_list
from datetime import datetime, timedelta
from dateutil import parser

API_BASE_URL = "http://localhost:8000"

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def sanitize_message_content(content: Any) -> str:
    """Strip HTML tags, apply markdown formatting, and escape text for safe display."""
    if not isinstance(content, str):
        return ""

    without_tags = _HTML_TAG_RE.sub("", content)
    normalized = html.unescape(without_tags).replace("\r\n", "\n").replace("\r", "\n")

    # Convert markdown bold syntax to HTML
    markdown_processed = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", normalized)
    # Convert markdown italic syntax to HTML
    markdown_processed = re.sub(r"\*(.*?)\*", r"<em>\1</em>", markdown_processed)
    # Convert markdown code syntax to HTML
    markdown_processed = re.sub(r"`(.*?)`", r"<code>\1</code>", markdown_processed)

    safe_text = html.escape(markdown_processed.strip(), quote=False)
    # Re-apply HTML formatting that was escaped
    safe_text = safe_text.replace("&lt;strong&gt;", "<strong>").replace(
        "&lt;/strong&gt;", "</strong>"
    )
    safe_text = safe_text.replace("&lt;em&gt;", "<em>").replace("&lt;/em&gt;", "</em>")
    safe_text = safe_text.replace("&lt;code&gt;", "<code>").replace(
        "&lt;/code&gt;", "</code>"
    )

    return safe_text.replace("\n", "<br>")


st.set_page_config(
    page_title="ChatBot", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
    .main-container { max-width: 1200px; margin: 0 auto; }
    .chat-wrapper {
        display: flex; flex-direction: column; gap: 20px;
        height: calc(100vh - 220px); min-height: 460px;
    }
    .chat-messages-container {
        border: 1px solid #e2e8f0; border-radius: 18px; background: #ffffff;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        flex: 1 1 auto; overflow-y: auto; padding: 20px; margin-bottom: 0;
    }
    .chat-messages-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-messages-container::-webkit-scrollbar-thumb {
        background: #cbd5f5; border-radius: 4px;
    }
    .chat-input-container {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        padding: 25px;
    }
    .chat-input-sticky {
        position: sticky; bottom: 0; z-index: 5;
    }
    .user-message {
        background: #ffffff; color: #1f2937; border: 1px solid #93c5fd;
        padding: 12px 16px; border-radius: 18px 18px 4px 18px; margin: 8px 0 8px auto;
        max-width: 72%; word-wrap: break-word; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.16);
    }
    .bot-message {
        background: #ffffff; color: #1f2937; border: 1px solid #86efac;
        padding: 12px 16px; border-radius: 18px 18px 18px 4px; margin: 8px auto 8px 0;
        max-width: 60%; white-space: normal; line-height: 1.5; word-wrap: break-word; overflow-wrap: anywhere;
        box-shadow: 0 8px 20px rgba(34, 197, 94, 0.16);
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
    .message-timestamp { font-size: 0.8em; color: #64748b; margin-top: 5px; white-space: normal; display: block; max-width: 100%; overflow-wrap: anywhere; }
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
if "API_BASE_URL" not in st.session_state:
    st.session_state.API_BASE_URL = API_BASE_URL


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
        f"?page={page}&limit={limit}&orderBy={order_by}&orderDirection={order_direction}"
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


@st.cache_data(show_spinner=False)
def get_feedback(message_id: str) -> Optional[Dict[str, Any]]:
    """Get feedback for a specific message (1-1 relationship)"""
    response = make_api_request("GET", f"/messages/{message_id}/feedbacks")
    if not response:
        return None

    data = response.get("data")

    # API may return a single feedback object or wrap it in a list – normalise to a dict.
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
                st.markdown(f"**👤 {user['username']}**")
                if st.button("Sign Out", use_container_width=True):
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
                "🔍 Search conversations:", placeholder="Type to search..."
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
                        with st.expander(f"💬 {conv['title']}", expanded=False):
                            # Check if this conversation already has messages from include_messages
                            if conv.get("messages"):
                                # Use the messages that were already included
                                for msg in conv["messages"]:
                                    sender_value = msg.get("sender")
                                    sender_icon = (
                                        "👤" if sender_value in (1, "user") else "🤖"
                                    )
                                    sender_name = (
                                        "user"
                                        if sender_value in (1, "user")
                                        else "assistant"
                                    )
                                    st.markdown(
                                        f"{sender_icon} **{sender_name}:** {msg['content'][:100]}..."
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
        # Load conversations without messages (messages only needed in conversation manager)
        conversations_response = get_conversations(include_messages=False)
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

    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages-container">', unsafe_allow_html=True)

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
        sender_value = msg.get("sender")
        is_user_message = sender_value in (1, "user", "USER", "User")

        if is_user_message:
            # User message with no feedback option
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px 0; align-items: flex-start; gap: 10px;">
                    <div class="user-message">
                        {sanitize_message_content(msg.get("content"))}
                        <div class="message-timestamp">You • {format_time(msg.get("createdAt", "now"))}</div>
                    </div>
                    <div style="border: 2px solid #007bff; color: #007bff; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">👤</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Assistant message with feedback option
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0; align-items: flex-start; gap: 10px;">
                    <div style="border: 2px solid #28a745; color: #28a745; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px; flex-shrink: 0;">🤖</div>
                    <div style="display: flex; flex-direction: column;">
                        <div class="bot-message" style="margin: 0;">
                            {sanitize_message_content(msg.get("content"))}
                            <div class="message-timestamp">Assistant • {format_time(msg.get("createdAt", "now"))}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Add feedback section below the message
            with st.container():
                st.markdown(
                    '<div style="margin-left: 45px;">', unsafe_allow_html=True
                )  # Align with message content

                with st.popover("💭", help="Give feedback"):
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
                                st.success("✅ Feedback submitted!")
                                st.cache_data.clear()
                                st.rerun()

                # Show existing feedback directly beneath the bot message
                feedback = get_feedback(msg["id"])
                if isinstance(feedback, dict) and feedback:
                    rating = feedback.get("rating")
                    comment_text = feedback.get("comment")

                    if rating is not None:
                        st.markdown(f"⭐ {rating}/5")

                    if comment_text:
                        preview = comment_text[:100]
                        suffix = "..." if len(comment_text) > 100 else ""
                        st.markdown(f'💭 *"{preview}{suffix}"*')

                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if conversation_id:
        st.markdown(
            '<div class="chat-input-container chat-input-sticky">',
            unsafe_allow_html=True,
        )

        with st.form("message_form", clear_on_submit=True):

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
                        "conversationId": st.session_state.current_conversation_id,
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
