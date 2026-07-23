"""Guard against silent field drift between the two WorkflowExecutionRequest schemas.

``AIService._to_ai_request`` rebuilds the AI-layer request from the service-layer
request via ``model_validate(model_dump(...))``. Pydantic ignores unknown keys, so
any field added to one schema but not mirrored in the other is silently dropped at
that boundary. This test fails loudly when the field sets diverge.
"""

from app.ai.schemas import WorkflowExecutionRequest as AIWorkflowExecutionRequest
from app.schemas.workflow import WorkflowExecutionRequest as ServiceWorkflowExecutionRequest


def test_workflow_request_field_sets_match():
    service_fields = set(ServiceWorkflowExecutionRequest.model_fields)
    ai_fields = set(AIWorkflowExecutionRequest.model_fields)

    only_service = service_fields - ai_fields
    only_ai = ai_fields - service_fields

    assert not only_service and not only_ai, (
        "WorkflowExecutionRequest schemas drifted; _to_ai_request would silently "
        f"drop fields. service-only={sorted(only_service)} ai-only={sorted(only_ai)}"
    )
