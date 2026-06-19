"""Guard schema parity between the service and AI-layer request models.

Regression: the AI-layer ``WorkflowExecutionRequest`` was missing the
``inline_rich_response_v1`` field, so ``AIService._to_ai_request`` round-tripped
the request through ``model_validate`` and silently dropped the capability flag.
The graph then read ``getattr(request, "inline_rich_response_v1", False)`` as
``False`` for every turn, leaving the entire inline rich-response pipeline
(marker parsing, widget/image placement, auto-placement) dormant — widgets fell
back to append-after-body and unresolved markers leaked into the answer text.
"""

from app.schemas.workflow import WorkflowExecutionRequest
from app.services.ai_service import AIService


def test_to_ai_request_preserves_inline_rich_response_flag():
    req = WorkflowExecutionRequest(message="hi", conversation_id="c1", inline_rich_response_v1=True)
    ai_req = AIService._to_ai_request(req)
    assert ai_req.inline_rich_response_v1 is True


def test_to_ai_request_defaults_flag_false():
    req = WorkflowExecutionRequest(message="hi", conversation_id="c1")
    ai_req = AIService._to_ai_request(req)
    assert ai_req.inline_rich_response_v1 is False
