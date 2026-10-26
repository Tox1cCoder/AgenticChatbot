"""Guard schema parity between the service and AI-layer request models.

Regression: the AI-layer ``WorkflowExecutionRequest`` was missing the
``inline_rich_response_v1`` field, so ``AIService._to_ai_request`` round-tripped
the request through ``model_validate`` and silently dropped the capability flag.
The graph then read ``getattr(request, "inline_rich_response_v1", False)`` as
``False`` for every turn, leaving the entire inline rich-response pipeline
(marker parsing, widget/image placement, auto-placement) dormant — widgets fell
back to append-after-body and unresolved markers leaked into the answer text.
"""

from app.ai.schemas import WorkflowExecutionRequest as AIWorkflowExecutionRequest
from app.schemas.workflow import WorkflowExecutionRequest
from app.schemas.workflow import WorkflowExecutionRequest as ServiceWorkflowExecutionRequest
from app.services.ai_service import AIService


def test_to_ai_request_preserves_inline_rich_response_flag():
    req = WorkflowExecutionRequest(message="hi", conversation_id="c1", inline_rich_response_v1=True)
    ai_req = AIService._to_ai_request(req)
    assert ai_req.inline_rich_response_v1 is True


def test_to_ai_request_defaults_flag_false():
    req = WorkflowExecutionRequest(message="hi", conversation_id="c1")
    ai_req = AIService._to_ai_request(req)
    assert ai_req.inline_rich_response_v1 is False


def test_to_ai_request_preserves_answer_visual_capability():
    req = WorkflowExecutionRequest(message="hi", conversation_id="c1", answer_visuals_v1=True)
    ai_req = AIService._to_ai_request(req)
    assert ai_req.answer_visuals_v1 is True


def test_graph_context_carries_answer_visual_capability():
    from app.ai.graph import MultiAgentWorkflow

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    for requested in (True, False):
        state = workflow._build_initial_state_from_request(
            AIWorkflowExecutionRequest(message="hi", answer_visuals_v1=requested)
        )
        assert state["context"]["answer_visuals_v1"] is requested


def test_message_service_forwards_answer_visual_capability():
    import inspect

    from app.services.message_service import MessageService

    source = inspect.getsource(MessageService._build_user_message_workflow_request)
    assert 'getattr(message_create_data, "answer_visuals_v1", False)' in source


def test_workflow_request_conversion_preserves_attachments():
    attachments = [{"name": "a.png", "mime": "image/png", "data": "abc"}]
    request = ServiceWorkflowExecutionRequest(message="see image", attachments=attachments)

    ai_request = AIWorkflowExecutionRequest.model_validate(request.model_dump(mode="python"))

    assert ai_request.attachments == attachments
