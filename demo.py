import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(page_title="Chatbot API Demo", page_icon="🤖", layout="wide")

st.title("🤖 Chatbot API Demo")
st.markdown("A simple interface to test the FastAPI chatbot backend")

# Initialize session state
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = None
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


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
            "❌ Cannot connect to API. Make sure the FastAPI server is running on localhost:8000"
        )
        return {}
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {}


# Sidebar for user management
st.sidebar.header("👤 User Management")

# Create User Section
with st.sidebar.expander("Create New User"):
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
                st.success(f"✅ User created! ID: {result.get('id')}")
                st.session_state.current_user_id = result.get("id")

# Select Existing User
with st.sidebar.expander("Select User"):
    if st.button("Load Users"):
        users = make_api_request("GET", "/users/")
        if users:
            for user in users:
                if st.button(
                    f"{user['username']} ({user['email']})", key=f"user_{user['id']}"
                ):
                    st.session_state.current_user_id = user["id"]
                    st.success(f"Selected user: {user['username']}")

# Display current user
if st.session_state.current_user_id:
    st.sidebar.success(f"✅ Current User ID: {st.session_state.current_user_id[:8]}...")
else:
    st.sidebar.warning("⚠️ No user selected")

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat Interface")

    # Conversation Management
    if st.session_state.current_user_id:
        st.subheader("Conversations")

        # Create new conversation
        with st.expander("Create New Conversation"):
            with st.form("create_conversation"):
                conv_title = st.text_input("Conversation Title")
                if st.form_submit_button("Create Conversation"):
                    conv_data = {"title": conv_title}
                    result = make_api_request(
                        "POST",
                        f"/conversations/user/{st.session_state.current_user_id}",
                        conv_data,
                    )
                    if result:
                        st.success(f"✅ Conversation created!")
                        st.session_state.current_conversation_id = result.get("id")

        # Load conversations
        if st.button("Load My Conversations"):
            conversations = make_api_request(
                "GET", f"/conversations/user/{st.session_state.current_user_id}"
            )
            if conversations:
                for conv in conversations:
                    if st.button(f"📁 {conv['title']}", key=f"conv_{conv['id']}"):
                        st.session_state.current_conversation_id = conv["id"]
                        # Load messages for this conversation
                        messages = make_api_request(
                            "GET",
                            f"/messages/conversation/{conv['id']}/thread?user_id={st.session_state.current_user_id}",
                        )
                        if messages:
                            st.session_state.messages = messages

        # Display current conversation
        if st.session_state.current_conversation_id:
            st.info(
                f"📝 Current Conversation: {st.session_state.current_conversation_id[:8]}..."
            )

            # Chat messages
            st.subheader("Messages")

            # Display messages
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    st.caption(f"ID: {msg['id'][:8]}... | {msg['created_at']}")

                    # Feedback section for assistant messages
                    if msg["role"] == "assistant":
                        col_rate1, col_rate2, col_rate3 = st.columns([1, 1, 2])
                        with col_rate1:
                            rating = st.selectbox(
                                "Rate", [1, 2, 3, 4, 5], key=f"rating_{msg['id']}"
                            )
                        with col_rate2:
                            if st.button("👍 Rate", key=f"rate_btn_{msg['id']}"):
                                feedback_data = {
                                    "message_id": msg["id"],
                                    "rating": rating,
                                    "comment": "Rated via demo",
                                }
                                result = make_api_request(
                                    "POST",
                                    f"/feedback/user/{st.session_state.current_user_id}",
                                    feedback_data,
                                )
                                if result:
                                    st.success("Feedback submitted!")
                        with col_rate3:
                            if st.button("📊 View Stats", key=f"stats_{msg['id']}"):
                                stats = make_api_request(
                                    "GET", f"/feedback/message/{msg['id']}/stats"
                                )
                                if stats:
                                    st.json(stats)

            # Send new message
            st.subheader("Send Message")
            with st.form("send_message"):
                message_content = st.text_area("Your message:")
                if st.form_submit_button("Send Message"):
                    message_data = {
                        "conversation_id": st.session_state.current_conversation_id,
                        "content": message_content,
                        "role": "user",
                    }
                    result = make_api_request("POST", "/messages/", message_data)
                    if result:
                        st.success("✅ Message sent!")
                        # Reload messages
                        messages = make_api_request(
                            "GET",
                            f"/messages/conversation/{st.session_state.current_conversation_id}/thread?user_id={st.session_state.current_user_id}",
                        )
                        if messages:
                            st.session_state.messages = messages
                            st.rerun()
        else:
            st.warning("⚠️ Please create or select a conversation first")
    else:
        st.warning("⚠️ Please create or select a user first")

with col2:
    st.header("🔧 API Tools")

    # Health Check
    st.subheader("Health Check")
    if st.button("Check API Health"):
        health = make_api_request("GET", "/health/")
        if health:
            st.success("✅ API is healthy!")
            st.json(health)

    if st.button("Check DB Health"):
        db_health = make_api_request("GET", "/health/db")
        if db_health:
            st.success("✅ Database is healthy!")
            st.json(db_health)

    # Raw API Testing
    st.subheader("Raw API Testing")
    with st.expander("Test Custom Endpoint"):
        method = st.selectbox("Method", ["GET", "POST", "PUT", "DELETE"])
        endpoint = st.text_input("Endpoint (e.g., /users/)")

        if method in ["POST", "PUT"]:
            json_data = st.text_area("JSON Data", "{}")
            try:
                data = json.loads(json_data) if json_data.strip() else None
            except:
                st.error("Invalid JSON")
                data = None
        else:
            data = None

        if st.button("Send Request"):
            result = make_api_request(method, endpoint, data)
            if result:
                st.json(result)

    # Quick Stats
    st.subheader("📊 Quick Stats")
    if st.button("Get All Users"):
        users = make_api_request("GET", "/users/")
        if users:
            st.write(f"Total Users: {len(users)}")
            for user in users[:5]:  # Show first 5
                st.write(f"- {user['username']} ({user['email']})")

    if st.session_state.current_user_id and st.button("Get My Conversations"):
        conversations = make_api_request(
            "GET", f"/conversations/user/{st.session_state.current_user_id}"
        )
        if conversations:
            st.write(f"Total Conversations: {len(conversations)}")
            for conv in conversations[:5]:  # Show first 5
                st.write(f"- {conv['title']}")

# Footer
st.markdown("---")
st.markdown(
    "🚀 **FastAPI Chatbot Demo** | Make sure your API server is running on `localhost:8000`"
)
