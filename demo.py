import streamlit as st
import requests
import json
from typing import Dict, Optional, Any, List
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern CSS styling
st.markdown(
    """
<style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .chat-container {
        height: 60vh;
        overflow-y: auto;
        padding: 20px;
        background: transparent;
        border-radius: 10px;
        border: none;
        margin-bottom: 20px;
    }
    
    .user-message {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        color: #212529;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px auto;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #007bff;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        color: #212529;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px auto 8px 0;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
    }
    
    .sidebar-conversation {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .sidebar-conversation:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    
    .sidebar-conversation.active {
        background: linear-gradient(135deg, #28a745, #1e7e34);
        color: white;
        border-color: #1e7e34;
    }
    
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 40px 20px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: white;
        color: #333;
        border: 1px solid #ddd;
        border-radius: 25px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #f8f9fa;
        border-color: #aaa;
        transform: translateY(-1px);
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .feedback-modal {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 1px solid #dee2e6;
    }
    
    .conversation-manager {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        max-height: 70vh;
        overflow-y: auto;
    }
    
    .search-container {
        margin-bottom: 20px;
    }
    
    .message-timestamp {
        font-size: 0.8em;
        color: #6c757d;
        margin-top: 5px;
    }
    
    .rating-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Hide Streamlit markdown containers causing white boxes */
    [data-testid="stMarkdownContainer"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Hide login container visual artifacts */
    .login-container [data-testid="stMarkdownContainer"] {
        display: none !important;
    }
    
    /* Remove all floating container elements */
    div[data-testid="stMarkdownContainer"]:empty {
        display: none !important;
    }
    
    /* Force transparent background for all markdown containers */
    .stMarkdown > div {
        background: transparent !important;
    }
    
    /* Hide conversation-manager and other empty containers */
    .conversation-manager:empty {
        display: none !important;
    }
    
    .login-container:empty {
        display: none !important;
    }
    
    /* Universal container cleanup - hide all empty divs */
    div:empty {
        display: none !important;
    }
    
    /* Force hide blank wrapper containers */
    div[class*="container"]:empty {
        display: none !important;
    }
    
    /* Additional container hiding for persistent elements */
    .stContainer > div:empty {
        display: none !important;
    }
    }
</style>
""",
    unsafe_allow_html=True,
)

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
if "show_login" not in st.session_state:
    st.session_state.show_login = True
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False
if "show_feedback_modal" not in st.session_state:
    st.session_state.show_feedback_modal = False
if "show_conversation_manager" not in st.session_state:
    st.session_state.show_conversation_manager = False
if "selected_message_for_feedback" not in st.session_state:
    st.session_state.selected_message_for_feedback = None


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
            "GET",
            f"/messages/conversations/{conversation_id}/messages/thread?user_id={user_id}",
        )
        or []
    )


@st.cache_data(show_spinner=False)
def get_feedbacks(message_id: str) -> List[Dict[str, Any]]:
    return make_api_request("GET", f"/messages/{message_id}/feedback") or []


@st.cache_data(show_spinner=False)
def get_feedback_stats(message_id: str) -> Dict:
    return make_api_request("GET", f"/messages/{message_id}/feedback/stats") or {}


def render_login_page():
    """Render modern login/signup page"""

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("# 🤖 ChatBot")
        st.markdown("### Welcome back!")

        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

        with tab1:
            # Add user selection window for demo
            with st.expander("🔍 Demo User Selector", expanded=False):
                st.markdown("**Quick login for demo testing**")
                users = get_users()
                if users:
                    for user in users[:5]:  # Show first 5 users
                        if st.button(
                            f"Login as: {user.get('email', user.get('username', 'Unknown'))}",
                            key=f"quick_login_{user['id']}",
                            use_container_width=True,
                        ):
                            st.session_state.current_user_id = user["id"]
                            st.session_state.show_login = False
                            st.success(
                                f"✅ Signed in as {user.get('email', user.get('username', 'User'))}"
                            )
                            st.rerun()
                else:
                    st.info("No users found in database")

            with st.form("login_form"):
                st.markdown("#### Sign in to your account")
                email = st.text_input("Email", placeholder="Enter your email")
                password = st.text_input(
                    "Password", type="password", placeholder="Enter your password"
                )

                if st.form_submit_button("Sign In", use_container_width=True):
                    # For demo purposes, we'll just find user by email
                    users = get_users()
                    user = next((u for u in users if u.get("email") == email), None)
                    if user:
                        st.session_state.current_user_id = user["id"]
                        st.session_state.show_login = False
                        st.success("✅ Signed in successfully!")
                        st.rerun()
                    else:
                        st.error("❌ User not found")

        with tab2:
            with st.form("signup_form"):
                st.markdown("#### Create new account")
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
                        result = make_api_request("POST", "/users/", user_data)
                        if result:
                            st.cache_data.clear()
                            st.session_state.current_user_id = result.get("id")
                            st.session_state.show_login = False
                            st.success("✅ Account created successfully!")
                            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def render_conversation_sidebar():
    """Render modern conversation sidebar"""
    with st.sidebar:
        st.markdown("### 💬 Conversations")

        # New conversation button
        if st.button("➕ New Chat", use_container_width=True):
            conv_data = {"title": f"New Chat {datetime.now().strftime('%H:%M')}"}
            result = make_api_request(
                "POST",
                f"/conversations/user/{st.session_state.current_user_id}",
                conv_data,
            )
            if result:
                st.session_state.current_conversation_id = result.get("id")
                st.session_state.conversations_list = []  # Force reload
                st.session_state.messages = []
                st.rerun()

        # Conversation manager button
        if st.button("📁 Manage Conversations", use_container_width=True):
            st.session_state.show_conversation_manager = True
            st.rerun()

        st.divider()

        # Load conversations
        if not st.session_state.conversations_list and st.session_state.current_user_id:
            conversations = get_conversations(st.session_state.current_user_id)
            if conversations:
                st.session_state.conversations_list = conversations

        # Display conversations
        if st.session_state.conversations_list:
            for conv in st.session_state.conversations_list:
                is_active = conv["id"] == st.session_state.current_conversation_id

                # Create clickable conversation item
                container = st.container()
                with container:
                    if st.button(
                        conv["title"],
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        if conv["id"] != st.session_state.current_conversation_id:
                            st.session_state.current_conversation_id = conv["id"]
                            # Load all messages at once for the conversation
                            messages = get_messages(
                                conv["id"], st.session_state.current_user_id
                            )
                            st.session_state.messages = messages or []
                            st.rerun()

        st.divider()

        # User info and logout
        if st.session_state.current_user_id:
            users = get_users()
            current_user = next(
                (u for u in users if u["id"] == st.session_state.current_user_id), None
            )
            if current_user:
                st.markdown(f"**👤 {current_user['username']}**")
                if st.button("🚪 Sign Out", use_container_width=True):
                    st.session_state.current_user_id = None
                    st.session_state.current_conversation_id = None
                    st.session_state.messages = []
                    st.session_state.conversations_list = []
                    st.session_state.show_login = True
                    st.rerun()


def render_feedback_modal(message_id: str):
    """Render modern feedback modal window as popup overlay"""
    if (
        st.session_state.show_feedback_modal
        and st.session_state.selected_message_for_feedback == message_id
    ):
        # Create overlay modal using columns for centering
        with st.container():
            # Add overlay background
            st.markdown(
                """
                <style>
                .feedback-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.5);
                    z-index: 1000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .feedback-popup {
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 15px 40px rgba(0,0,0,0.3);
                    border: 1px solid #dee2e6;
                    max-width: 500px;
                    width: 90%;
                    position: relative;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Create centered modal content
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.markdown('<div class="feedback-popup">', unsafe_allow_html=True)
                st.markdown("### 📝 Provide Feedback")

                with st.form(f"feedback_form_{message_id}"):
                    rating = st.select_slider(
                        "Rate this response:",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        format_func=lambda x: "⭐" * x,
                    )

                    comment = st.text_area(
                        "Additional comments (optional):",
                        placeholder="Share your thoughts about this response...",
                    )

                    col_submit, col_cancel = st.columns(2)
                    with col_submit:
                        if st.form_submit_button(
                            "Submit Feedback", use_container_width=True
                        ):
                            feedback_data = {
                                "message_id": message_id,
                                "rating": rating,
                                "comment": comment,
                            }
                            result = make_api_request(
                                "POST",
                                f"/messages/{message_id}/feedback?user_id={st.session_state.current_user_id}",
                                feedback_data,
                            )
                            if result:
                                st.cache_data.clear()
                                st.session_state.show_feedback_modal = False
                                st.session_state.selected_message_for_feedback = None
                                st.success("✅ Feedback submitted!")
                                st.rerun()

                    with col_cancel:
                        if st.form_submit_button("Cancel", use_container_width=True):
                            st.session_state.show_feedback_modal = False
                            st.session_state.selected_message_for_feedback = None
                            st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)


def render_conversation_manager():
    """Render conversation manager window"""
    if st.session_state.show_conversation_manager:
        st.markdown('<div class="conversation-manager">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("### 📁 Conversation Manager")

            # Search functionality
            st.markdown('<div class="search-container">', unsafe_allow_html=True)
            search_term = st.text_input(
                "🔍 Search conversations by message content:",
                placeholder="Type to search through all your conversations...",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Display conversations with search functionality
            if st.session_state.conversations_list:
                filtered_convs = st.session_state.conversations_list

                if search_term:
                    # Search through messages in conversations
                    filtered_convs = []
                    for conv in st.session_state.conversations_list:
                        messages = get_messages(
                            conv["id"], st.session_state.current_user_id
                        )
                        for msg in messages:
                            if search_term.lower() in msg.get("content", "").lower():
                                if conv not in filtered_convs:
                                    filtered_convs.append(conv)
                                break

                if filtered_convs:
                    for conv in filtered_convs:
                        with st.expander(f"💬 {conv['title']}", expanded=False):
                            # Show preview of recent messages
                            messages = get_messages(
                                conv["id"], st.session_state.current_user_id
                            )
                            if messages:
                                for msg in messages[-3:]:  # Show last 3 messages
                                    sender_icon = (
                                        "👤" if msg["sender"] == "user" else "🤖"
                                    )
                                    st.markdown(
                                        f"{sender_icon} **{msg['sender']}:** {msg['content'][:100]}..."
                                    )

                            col_open, col_delete = st.columns(2)
                            with col_open:
                                if st.button(f"Select", key=f"open_{conv['id']}"):
                                    st.session_state.current_conversation_id = conv[
                                        "id"
                                    ]
                                    st.session_state.messages = (
                                        get_messages(
                                            conv["id"], st.session_state.current_user_id
                                        )
                                        or []
                                    )
                                    st.session_state.show_conversation_manager = False
                                    st.rerun()

                            with col_delete:
                                if st.button(
                                    f"Delete",
                                    key=f"delete_{conv['id']}",
                                    type="secondary",
                                ):
                                    # Implement conversation deletion
                                    result = make_api_request(
                                        "DELETE",
                                        f"/conversations/{conv['id']}?user_id={st.session_state.current_user_id}",
                                    )
                                    if result:
                                        st.session_state.conversations_list = (
                                            []
                                        )  # Force reload
                                        if (
                                            st.session_state.current_conversation_id
                                            == conv["id"]
                                        ):
                                            st.session_state.current_conversation_id = (
                                                None
                                            )
                                            st.session_state.messages = []
                                        st.cache_data.clear()  # Clear cache to refresh data
                                        st.success(f"✅ Deleted '{conv['title']}'")
                                        st.rerun()
                else:
                    st.info("No conversations found matching your search.")
            else:
                st.info("No conversations available.")

            if st.button("Close Manager", use_container_width=True):
                st.session_state.show_conversation_manager = False
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def render_chat_interface():
    """Render modern chat interface"""
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
            st.markdown(f"# 💬 {current_conv['title']}")

        # Chat container with modern styling
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        # Display all messages with proper alignment
        for msg in st.session_state.messages:
            if msg["sender"] == "user":
                # User message (right aligned with avatar)
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin: 10px 0; align-items: flex-start; gap: 10px;">
                        <div class="user-message">
                            {msg["content"]}
                            <div class="message-timestamp">You • {msg.get("created_at", "now")}</div>
                        </div>
                        <div style="background: #007bff; color: white; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">👤</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                # Bot message (left aligned with avatar and small feedback button)
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin: 10px 0; align-items: flex-start; gap: 10px;">
                        <div style="background: #6c757d; color: white; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">🤖</div>
                        <div style="display: flex; flex-direction: column; gap: 5px; max-width: 70%;">
                            <div class="bot-message">
                                {msg["content"]}
                                <div class="message-timestamp">Assistant • {msg.get("created_at", "now")}</div>
                            </div>
                        </div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
                # Small feedback button positioned right next to bot message
                if st.button(
                    "💭", key=f"feedback_btn_{msg['id']}", help="Give feedback"
                ):
                    st.session_state.show_feedback_modal = True
                    st.session_state.selected_message_for_feedback = msg["id"]
                    st.rerun()

                # Show existing feedback
                feedbacks = get_feedbacks(msg["id"])
                if feedbacks:
                    avg_rating = sum(fb.get("rating", 0) for fb in feedbacks) / len(
                        feedbacks
                    )
                    st.markdown(f"⭐ {avg_rating:.1f} ({len(feedbacks)} reviews)")

                    # Show individual feedback comments
                    with st.expander("View Feedback", expanded=False):
                        for fb in feedbacks:
                            if fb.get("comment"):
                                st.markdown(f"**{fb['rating']}⭐** - {fb['comment']}")
                            else:
                                st.markdown(f"**{fb['rating']}⭐**")

        st.markdown("</div>", unsafe_allow_html=True)

        # Message input with modern styling
        with st.form("message_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                message_content = st.text_area(
                    "Message",
                    placeholder="Type your message here...",
                    height=100,
                    label_visibility="collapsed",
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                send_button = st.form_submit_button("📤 Send", use_container_width=True)

            if send_button and message_content.strip():
                # Send user message
                message_data = {
                    "conversation_id": st.session_state.current_conversation_id,
                    "content": message_content,
                    "sender": "user",
                }
                result = make_api_request("POST", "/messages/", message_data)
                if result:
                    st.cache_data.clear()
                    # Reload all messages for the conversation
                    messages = get_messages(
                        st.session_state.current_conversation_id,
                        st.session_state.current_user_id,
                    )
                    st.session_state.messages = messages or []
                    st.rerun()
    else:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                # Welcome to Chatbot Demo
                
                ### Start a conversation by:
                - 🆕 Creating a new chat from the sidebar
                - 📁 Opening an existing conversation
                
                Select or create a conversation to begin.
            """
            )


# Main application logic
def main():
    # Show login page if not authenticated
    if st.session_state.show_login and not st.session_state.current_user_id:
        render_login_page()
        return

    # Render conversation sidebar
    render_conversation_sidebar()

    # Show conversation manager if requested
    if st.session_state.show_conversation_manager:
        render_conversation_manager()
        return

    # Show feedback modal if requested
    if st.session_state.show_feedback_modal:
        render_feedback_modal(st.session_state.selected_message_for_feedback)

    # Main chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()
