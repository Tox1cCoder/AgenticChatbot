import streamlit as st
import requests
import json
from typing import Dict, Optional, Any, List

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(page_title="Chatbot API Demo", page_icon="🤖", layout="wide")

# Initialize session state
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = None
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "users_list" not in st.session_state:
    st.session_state.users_list = []
if "conversations_list" not in st.session_state:
    st.session_state.conversations_list = []


def make_api_request(method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
    """Make API request and handle errors"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)

        if response.status_code >= 400:
            st.error(f"API Error {response.status_code}: {response.text}")
            return {}

        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to API. Make sure FastAPI server is running on localhost:8000"
        )
        return {}
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {}


# --- Cached API GET functions ---
@st.cache_data(show_spinner=False)
def get_users() -> List[Dict[str, Any]]:
    return make_api_request("GET", "/users/") or []


@st.cache_data(show_spinner=False)
def get_conversations(user_id: str) -> List[Dict[str, Any]]:
    return make_api_request("GET", f"/conversations/user/{user_id}") or []


@st.cache_data(show_spinner=False)
def get_health_api() -> Dict:
    return make_api_request("GET", "/health/")


@st.cache_data(show_spinner=False)
def get_health_db() -> Dict:
    return make_api_request("GET", "/health/db")


@st.cache_data(show_spinner=False)
def get_messages(conversation_id: str, user_id: str) -> List[Dict[str, Any]]:
    return (
        make_api_request(
            "GET", f"/messages/conversation/{conversation_id}/thread?user_id={user_id}"
        )
        or []
    )


@st.cache_data(show_spinner=False)
def get_feedbacks(message_id: str) -> List[Dict[str, Any]]:
    return make_api_request("GET", f"/feedback/message/{message_id}") or []


@st.cache_data(show_spinner=False)
def get_feedback_stats(message_id: str) -> Dict:
    return make_api_request("GET", f"/feedback/message/{message_id}/stats") or {}


# Header
st.title("Chatbot API Demo")

# Sidebar
with st.sidebar:
    st.header("👤 User Management")

    # Create User
    with st.expander("Create User"):
        with st.form("create_user"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            full_name = st.text_input("Full Name")

            if st.form_submit_button("Create User"):
                user_data = {
                    "username": username,
                    "email": email,
                    "password": password,
                    "full_name": full_name,
                }
                result = make_api_request("POST", "/users/", user_data)
                if result:
                    st.session_state.current_user_id = result.get("id")
                    st.session_state.users_list = []  # Force reload
                    st.success("✅ User created and selected!")
                    st.rerun()

    # Select User
    if not st.session_state.users_list:
        users = get_users()
        if users:
            st.session_state.users_list = users

    if st.session_state.users_list:
        user_options = {
            f"{user['username']} ({user['email']})": user["id"]
            for user in st.session_state.users_list
        }
        selected_user_display = st.selectbox(
            "Select User:", options=list(user_options.keys())
        )

        if selected_user_display:
            selected_user_id = user_options[selected_user_display]
            if st.session_state.current_user_id != selected_user_id:
                st.session_state.current_user_id = selected_user_id
                st.session_state.current_conversation_id = None
                st.session_state.conversations_list = []
                st.session_state.messages = []
                st.rerun()

    # Show current user
    if st.session_state.current_user_id:
        current_user = next(
            (
                u
                for u in st.session_state.users_list
                if u["id"] == st.session_state.current_user_id
            ),
            None,
        )
        if current_user:
            st.success(f"👤 {current_user['username']}")

    st.divider()

    # Health Check
    st.subheader("🏥 Health")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("API", key="health_api"):
            health = get_health_api()
            st.json(health)
            if health:
                st.success("✅ API OK")
    with col2:
        if st.button("DB", key="health_db"):
            health = get_health_db()
            st.json(health)
            if health:
                st.success("✅ DB OK")

# Main Content
if not st.session_state.current_user_id:
    st.warning("👈 Please create or select a user from the sidebar")
    st.stop()

# Conversation Management
col1, col2 = st.columns([1, 1])

with col1:
    # Create Conversation
    with st.form("create_conversation"):
        conv_title = st.text_input("New Conversation Title:")
        if st.form_submit_button("➕ Create"):
            conv_data = {"title": conv_title}
            result = make_api_request(
                "POST",
                f"/conversations/user/{st.session_state.current_user_id}",
                conv_data,
            )
            if result:
                st.session_state.current_conversation_id = result.get("id")
                st.session_state.conversations_list = []  # Force reload
                st.success("✅ Conversation created!")
                st.rerun()

with col2:
    # Load and Select Conversation
    if not st.session_state.conversations_list:
        conversations = get_conversations(st.session_state.current_user_id)
        if conversations:
            st.session_state.conversations_list = conversations

    if st.session_state.conversations_list:
        conv_options = {
            conv["title"]: conv["id"] for conv in st.session_state.conversations_list
        }
        selected_conv_title = st.selectbox(
            "Select Conversation:", options=list(conv_options.keys())
        )

        if (
            selected_conv_title
            and conv_options[selected_conv_title]
            != st.session_state.current_conversation_id
        ):
            st.session_state.current_conversation_id = conv_options[selected_conv_title]
            # Load messages
            messages = get_messages(
                st.session_state.current_conversation_id,
                st.session_state.current_user_id,
            )
            st.session_state.messages = messages or []
            st.rerun()

# Chat Interface
if st.session_state.current_conversation_id:
    current_conv = next(
        (
            c
            for c in st.session_state.conversations_list
            if c["id"] == st.session_state.current_conversation_id
        ),
        None,
    )
    if current_conv:
        st.subheader(f"💬 {current_conv['title']}")

    # Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            # Rating and comment for assistant messages
            if msg["role"] == "assistant":
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    rating = st.selectbox(
                        "Rate:", [1, 2, 3, 4, 5], key=f"rating_{msg['id']}"
                    )
                with col2:
                    comment = st.text_input(
                        "Comment:", value="", key=f"comment_{msg['id']}"
                    )
                    if st.button("👍", key=f"rate_{msg['id']}"):
                        feedback_data = {
                            "message_id": msg["id"],
                            "rating": rating,
                            "comment": comment,
                        }
                        result = make_api_request(
                            "POST",
                            f"/feedback/user/{st.session_state.current_user_id}",
                            feedback_data,
                        )
                        if result:
                            st.cache_data.clear()
                            st.success(f"Rated {rating}⭐ with comment!")
                with col3:
                    if st.button("📊", key=f"stats_{msg['id']}"):
                        stats = get_feedback_stats(msg["id"])
                        if stats:
                            st.json(stats)
                # Show feedback history (if available)
                feedbacks = get_feedbacks(msg["id"])
                if feedbacks:
                    for fb in feedbacks:
                        st.caption(f"Rated {fb['rating']}⭐: {fb.get('comment', '')}")

    # Send Message
    with st.form("send_message", clear_on_submit=True):
        message_content = st.text_area("Type your message:", height=100)
        if st.form_submit_button("📤 Send"):
            if message_content.strip():
                message_data = {
                    "conversation_id": st.session_state.current_conversation_id,
                    "content": message_content,
                    "role": "user",
                }
                result = make_api_request("POST", "/messages/", message_data)
                if result:
                    st.cache_data.clear()
                    # Reload messages
                    messages = get_messages(
                        st.session_state.current_conversation_id,
                        st.session_state.current_user_id,
                    )
                    st.session_state.messages = messages or []
                    st.rerun()

else:
    st.info("👆 Create or select a conversation to start chatting")
