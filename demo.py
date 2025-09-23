import streamlit as st
import requests
import json
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from dateutil import parser

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="ChatBot",
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

# Initialize session state with session persistence
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
    # Only show login if no auth token exists
    st.session_state.show_login = (
        "auth_token" not in st.session_state or not st.session_state.get("auth_token")
    )
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False
if "show_conversation_manager" not in st.session_state:
    st.session_state.show_conversation_manager = False
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None


def make_api_request(method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
    """Make API request and handle errors"""
    url = f"{API_BASE_URL}{endpoint}"

    # Add JWT authorization header for authentication
    headers = {}
    if "auth_token" in st.session_state and st.session_state.auth_token:
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)

        response_data = response.json()

        if not response_data.get("success"):
            # Handle API errors based on the new format
            error_code = response_data.get("code", "unknown_error")
            error_message = response_data.get("message", "An unknown error occurred.")
            error_details = response_data.get("error")

            if error_code == "unauthenticated":
                st.error(
                    f"🔒 Authentication required. Please log in. (Error: {error_message})"
                )
                st.session_state.auth_token = None
                st.session_state.current_user_id = None
                st.session_state.show_login = True
            elif error_code == "invalid_input":
                st.error(
                    f"❌ Validation Error: {error_message}. Details: {error_details}"
                )
            else:
                st.error(f"❌ API Error ({error_code}): {error_message}")
            return {}  # Return empty dict on error

        return response_data  # Return the full response data on success

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
def get_user(user_id: str) -> Dict[str, Any]:
    response = make_api_request("GET", f"/users/{user_id}")
    if response and response.get("data") is not None:
        return response["data"]
    return {}


@st.cache_data(show_spinner=False)
def get_conversations() -> List[Dict[str, Any]]:
    response = make_api_request("GET", f"/conversations/")
    if response and response.get("data") is not None:
        return response["data"]
    return []


@st.cache_data(show_spinner=False)
def get_health_api() -> Dict:
    response = make_api_request("GET", "/health/")
    if response and response.get("data") is not None:
        return response["data"]
    return {}


@st.cache_data(show_spinner=False)
def get_health_db() -> Dict:
    response = make_api_request("GET", "/health/db")
    if response and response.get("data") is not None:
        return response["data"]
    return {}


@st.cache_data(show_spinner=False)
def get_messages(conversation_id: str) -> List[Dict[str, Any]]:
    response = make_api_request(
        "GET",
        f"/conversations/{conversation_id}/messages",
    )
    if response and response.get("data") is not None:
        return response["data"]
    return []


@st.cache_data(show_spinner=False)
def create_feedback(message_id: str, rating: int, comment: str) -> Dict[str, Any]:
    feedback_data = {
        "rating": rating,
        "comment": comment,
    }
    response = make_api_request(
        "POST", f"/messages/{message_id}/feedbacks", feedback_data
    )
    if response and response.get("data") is not None:
        return response["data"]
    return {}


@st.cache_data(show_spinner=False)
def get_feedback_for_message(message_id: str, user_id: str) -> Dict[str, Any]:
    response = make_api_request(
        "GET", f"/messages/{message_id}/feedbacks/user/{user_id}"
    )
    if response and response.get("data") is not None:
        return response["data"]
    return {}


@st.cache_data(show_spinner=False)
def get_feedback_stats(message_id: str) -> Dict:
    response = make_api_request("GET", f"/messages/{message_id}/feedbacks/stats")
    if response and response.get("data") is not None:
        return response["data"]
    return {}


@st.cache_data(show_spinner=False)
def get_feedbacks(message_id: str) -> List[Dict[str, Any]]:
    """Get all feedbacks for a message"""
    response = make_api_request("GET", f"/messages/{message_id}/feedbacks")
    if response and response.get("data") is not None:
        return response["data"]
    return []


def render_login_page():
    """Render login/signup page"""

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
                    # Use auth API for login
                    login_data = {"email": email, "password": password}
                    auth_response = make_api_request("POST", "/auth/login", login_data)

                    if auth_response and "data" in auth_response:
                        st.session_state.auth_token = auth_response["data"][
                            "accessToken"
                        ]
                        st.session_state.current_user_id = auth_response["data"].get(
                            "userId"
                        )
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
                            # After successful signup, authenticate the user
                            login_data = {"email": email, "password": password}
                            auth_response = make_api_request(
                                "POST", "/auth/login", login_data
                            )

                            if auth_response and "data" in auth_response:
                                st.session_state.auth_token = auth_response["data"][
                                    "accessToken"
                                ]
                                st.session_state.current_user_id = auth_response[
                                    "data"
                                ].get("userId")
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
    """Render modern conversation sidebar"""
    with st.sidebar:
        st.markdown("### Conversations")

        # New conversation button - will create conversation on first message
        if st.button("New Chat", use_container_width=True):
            st.session_state.current_conversation_id = (
                "pending_new"  # Special state for new conversation
            )
            st.session_state.messages = []
            st.rerun()

        # Conversation manager button
        if st.button("Manage Conversations", use_container_width=True):
            st.session_state.show_conversation_manager = True
            st.rerun()

        st.divider()

        # Load conversations - only if user is authenticated
        if (
            not st.session_state.conversations_list
            and st.session_state.current_user_id
            and st.session_state.auth_token
        ):
            conversations = get_conversations()
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
                            messages = get_messages(conv["id"])
                            st.session_state.messages = messages or []
                            st.rerun()

        st.divider()

        # User info and logout
        if st.session_state.current_user_id:
            user = get_user(st.session_state.current_user_id)
            if user:
                st.markdown(f"**👤 {user['username']}**")
                if st.button("🚪 Sign Out", use_container_width=True):
                    st.session_state.current_user_id = None
                    st.session_state.current_conversation_id = None
                    st.session_state.messages = []
                    st.session_state.conversations_list = []
                    st.session_state.auth_token = None  # Clear auth token on logout
                    st.session_state.show_login = True
                    st.rerun()


def render_conversation_manager():
    """Render conversation manager window"""
    if st.session_state.show_conversation_manager:
        st.markdown('<div class="conversation-manager">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("### Conversation Manager")

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
                        messages = get_messages(conv["id"])
                        for msg in messages:
                            if search_term.lower() in msg.get("content", "").lower():
                                if conv not in filtered_convs:
                                    filtered_convs.append(conv)
                                break

                if filtered_convs:
                    for conv in filtered_convs:
                        with st.expander(f"💬 {conv['title']}", expanded=False):
                            # Show preview of recent messages
                            messages = get_messages(conv["id"])
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
                                        get_messages(conv["id"]) or []
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
                                        f"/conversations/{conv['id']}",
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
    # Handle conversation title display
    if (
        st.session_state.current_conversation_id
        and st.session_state.current_conversation_id != "pending_new"
    ):
        # Ensure conversations list is loaded
        if not st.session_state.conversations_list and st.session_state.current_user_id:
            conversations = get_conversations()
            if conversations:
                st.session_state.conversations_list = conversations

        current_conv = next(
            (
                c
                for c in st.session_state.conversations_list
                if c["id"] == st.session_state.current_conversation_id
            ),
            None,
        )

        if current_conv:
            st.markdown(f"# {current_conv['title']}")
        else:
            st.markdown("# Conversation")

    elif st.session_state.current_conversation_id == "pending_new":
        st.markdown("# New Chat - Start typing to begin!")
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
        return

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    def format_time(iso_string: str) -> str:
        try:
            dt = parser.isoparse(iso_string)
            now = datetime.now(dt.tzinfo)

            # If same day → just show time
            if dt.date() == now.date():
                return dt.strftime("%H:%M")

            # If within this week → show weekday + time
            if now - timedelta(days=7) < dt <= now:
                return dt.strftime("%a %H:%M")

            # Else → full date + time
            return dt.strftime("%b %d, %Y %H:%M")
        except Exception:
            return iso_string

    # Display all messages with proper alignment
    for msg in st.session_state.messages:
        if msg["sender"] == 1:  # User messages (MessageRole.user = 1)
            # User message
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px 0; align-items: flex-start; gap: 10px;">
                    <div class="user-message" style="background: #f1f1f1; padding: 10px; border-radius: 10px; max-width: 70%;">
                        {msg["content"]}
                        <div class="message-timestamp" style="font-size: 12px; color: gray; margin-top: 5px;">
                            You • {format_time(msg.get("createdAt", "now"))}
                        </div>
                    </div>
                    <div style="border: 2px solid #007bff; color: #007bff; border-radius: 50%; 
                                width: 35px; height: 35px; display: flex; align-items: center; 
                                justify-content: center; font-weight: bold; font-size: 14px;">
                        👤
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Bot message
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0; align-items: flex-start; gap: 10px;">
                    <div style="border: 2px solid #6c757d; color: #6c757d; border-radius: 50%; 
                                width: 35px; height: 35px; display: flex; align-items: center; 
                                justify-content: center; font-weight: bold; font-size: 14px;">
                        🤖
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 5px; max-width: 70%;">
                        <div class="bot-message" style="background: #e9ecef; padding: 10px; border-radius: 10px;">
                            {msg["content"]}
                            <div class="message-timestamp" style="font-size: 12px; color: gray; margin-top: 5px;">
                                Assistant • {format_time(msg.get("createdAt", "now"))}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.popover("💭", help="Give feedback"):
                st.markdown("### Provide Feedback")

                with st.form(f"feedback_form_{msg['id']}"):
                    rating = st.selectbox("Rating", [1, 2, 3, 4, 5], index=4)
                    comment = st.text_area("Comment (optional)", height=100)

                    if st.form_submit_button(
                        "Submit Feedback", use_container_width=True
                    ):
                        feedback_data = {
                            "message_id": msg[
                                "id"
                            ],  # Include message_id in request body
                            "rating": rating,
                            "comment": comment,
                        }
                        make_api_request(
                            "POST",
                            f"/messages/{msg['id']}/feedbacks",
                            feedback_data,
                        )
                        st.success("✅ Feedback submitted!")
                        st.cache_data.clear()  # Clear cache to refresh feedback data
                        st.rerun()
            # Show existing feedback
            feedbacks = get_feedbacks(msg["id"])
            for fb in feedbacks:
                rating = fb.get("rating", 0)

                # Show individual feedback comments
                with st.expander("View Feedback", expanded=False):
                    for fb in feedbacks:
                        if fb.get("comment"):
                            st.markdown(f"**{fb['rating']}⭐** - {fb['comment']}")
                        else:
                            st.markdown(f"**{fb['rating']}⭐**")

    st.markdown("</div>", unsafe_allow_html=True)

    # Message input form
    if st.session_state.current_conversation_id:
        with st.form("message_form", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "Upload a document for context",
                type=["pdf", "txt", "docx"],
            )

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
                send_button = st.form_submit_button("Send", use_container_width=True)

            if send_button:
                if uploaded_file is not None:
                    # Process the uploaded file
                    with st.spinner("Processing uploaded file..."):
                        try:
                            # Create form data for file upload
                            files = {"file": uploaded_file}

                            # Upload file to backend for RAG processing
                            upload_response = requests.post(
                                f"{API_BASE_URL}/documents",
                                files=files,
                                headers={
                                    "Authorization": f"Bearer {st.session_state.get('auth_token', '')}"
                                },
                            )

                            if upload_response.status_code == 200:
                                upload_result = upload_response.json()
                                # Handle ApiResponse wrapper format
                                if upload_result.get("data") is not None:
                                    data = upload_result["data"]
                                else:
                                    data = upload_result
                                st.success(
                                    f"✅ File '{uploaded_file.name}' uploaded and processed successfully!"
                                )
                                st.info(
                                    f"Added {data.get('chunks_created', 0)} knowledge chunks to the database."
                                )
                            else:
                                st.error(
                                    f"❌ Failed to upload file: {upload_response.text}"
                                )

                        except Exception as e:
                            st.error(f"❌ Error uploading file: {str(e)}")

                if message_content.strip():
                    if st.session_state.current_conversation_id == "pending_new":
                        # Create new conversation on first message
                        conv_data = {
                            "title": f"New Chat {datetime.now().strftime('%H:%M')}"
                        }
                        conv_result = make_api_request(
                            "POST",
                            f"/conversations/",
                            conv_data,
                        )
                        if conv_result and conv_result.get("data") is not None:
                            st.session_state.current_conversation_id = conv_result[
                                "data"
                            ]["id"]
                            st.session_state.conversations_list = []  # Force reload
                        else:
                            st.error("Failed to create conversation")
                            return

                    # Send user message
                    message_data = {
                        "conversation_id": st.session_state.current_conversation_id,
                        "content": message_content,
                    }
                    result = make_api_request("POST", "/messages/", message_data)
                    if result and "data" in result:
                        # Clear cache and force reload of conversations list to reflect new conversation
                        st.cache_data.clear()
                        # Reload all messages from backend to get both user message and bot response
                        messages = get_messages(
                            st.session_state.current_conversation_id
                        )
                        st.session_state.messages = messages or []
                        # Force refresh conversations list to include new conversation
                        st.session_state.conversations_list = []
                        st.rerun()


# Main application logic
def main():
    # Show login page if not authenticated - require BOTH user_id AND auth_token
    if (
        st.session_state.show_login
        or not st.session_state.current_user_id
        or not st.session_state.auth_token
    ):
        render_login_page()
        return

    # Render conversation sidebar
    render_conversation_sidebar()

    # Show conversation manager if requested
    if st.session_state.show_conversation_manager:
        render_conversation_manager()
        return

    # Main chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()
