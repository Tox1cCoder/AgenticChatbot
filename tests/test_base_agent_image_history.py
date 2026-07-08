from app.ai.agents.chat_agent import ChatAgent
from app.ai.schemas import AgentMessage, MessageRole
from app.ai.token_instrumentation import estimate_agent_message_tokens


def test_base_agent_converts_user_history_images_to_multimodal_content():
    agent = ChatAgent()
    history = [
        AgentMessage(
            role=MessageRole.USER,
            content="previous image",
            attachments=[{"name": "a.png", "mime": "image/png", "data": "abc"}],
        )
    ]

    converted = agent._convert_history_to_langchain_messages(history)

    assert converted[0].content == [
        {"type": "text", "text": "previous image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]


def test_token_estimator_counts_attachment_overhead():
    with_image = AgentMessage(
        role=MessageRole.USER,
        content="short",
        attachments=[{"name": "a.png", "mime": "image/png", "data": "abc"}],
    )
    without_image = AgentMessage(role=MessageRole.USER, content="short")

    assert estimate_agent_message_tokens(with_image) > estimate_agent_message_tokens(without_image)
