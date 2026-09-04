
from langchain_core.messages import HumanMessage

from app.ai.graph import MultiAgentWorkflow

ATTACHMENTS = [{"name": "screen.png", "mime": "image/png", "data": "abc"}]


def test_workflow_applies_current_turn_images_to_last_human_message():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    result, has_images = workflow._build_turn_messages_with_attachments(
        [HumanMessage(content="inspect this")],
        ATTACHMENTS,
    )

    assert has_images is True
    assert result[0].content == [
        {"type": "text", "text": "inspect this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]


def test_workflow_leaves_turn_plain_without_valid_images():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    result, has_images = workflow._build_turn_messages_with_attachments(
        [HumanMessage(content="hello")],
        [{"mime": "text/plain", "data": "abc"}],
    )

    assert has_images is False
    assert result[0].content == "hello"


def test_workflow_marks_image_response_metadata_once():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    class Response:
        metadata = None

    response = Response()

    workflow._mark_response_has_images(response, True)

    assert response.metadata == {"has_images": True}




