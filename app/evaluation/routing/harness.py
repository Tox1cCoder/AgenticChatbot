"""The fixed inventory and context stubs one evaluation run is measured against.

Both CLIs import this. If each built its own inventory they could drift, and
the release gate's `inventory_version_mismatch` check — the thing that stops a
good result on one agent line-up from releasing another — would be comparing a
value to itself.

The inventory is deliberately **one snapshot for the whole run**, not one per
case. `inventory_version` is part of the tuple a report is pinned to, so a
per-case inventory would leave the report unable to name what it measured.
The two custom agents exist because the dataset routes to them by name; they
are fixtures of the evaluation, not of the product.

Context is derived from each case's own `context` field rather than from a
live conversation. That is what makes a run reproducible: a builder reading
the database would score differently tomorrow for reasons that have nothing to
do with the router.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.ai.workflow.inventory import RoutingInventory, build_routing_inventory
from app.ai.workflow.routing import RoutingContextRequest
from app.evaluation.routing.contracts import RoutingEvalCase

__all__ = [
    "EVALUATION_CUSTOM_AGENTS",
    "EvaluationDocumentRepository",
    "EvaluationHistoryProvider",
    "build_context_request",
    "build_evaluation_inventory",
]

#: Custom agents the golden dataset addresses by name. Their capability text
#: is what the router actually reads, so it is written the way a real user
#: would configure one rather than as a placeholder.
EVALUATION_CUSTOM_AGENTS: dict[str, dict[str, Any]] = {
    "custom_agent:data-analyst": {
        "display_name": "Data Analyst",
        "capability_description": (
            "Analyses product and business metrics: churn, retention, active users, "
            "cohort and funnel questions."
        ),
    },
    "custom_agent:legal-reviewer": {
        "display_name": "Legal Reviewer",
        "capability_description": (
            "Reviews contract language: clauses, liability, indemnity, termination and "
            "compliance wording."
        ),
    },
}

_BASE_AGENT_IDS = (
    "chat_agent",
    "rag_agent",
    "search_agent",
    "image_generator_agent",
    "planning_agent",
    "canvas_agent",
)


def build_evaluation_inventory() -> RoutingInventory:
    """The one inventory every case in a run is routed against."""
    return build_routing_inventory(
        base_agent_ids=_BASE_AGENT_IDS,
        custom_agents=EVALUATION_CUSTOM_AGENTS,
        attached_custom_agent_ids=set(EVALUATION_CUSTOM_AGENTS),
    )


class EvaluationHistoryProvider:
    """Replays exactly what a case declares, and nothing else.

    A case with no ``previous_agent_id`` is a genuinely new turn, so the router
    gets no prior agent to be sticky about. That matters: stickiness is the
    single easiest way to score well on follow-ups without classifying them.
    """

    def __init__(self, previous_agent_id: str | None) -> None:
        self._previous_agent_id = previous_agent_id

    async def build_context(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(memory=None, messages=[])

    async def get_previous_final_agent_id(self, **_kwargs: Any) -> str | None:
        return self._previous_agent_id


class EvaluationDocumentRepository:
    """One synthetic attached document when the case says there is one.

    Metadata only, which is all the real repository gives the router — a
    filename and a status, never content. A document_qa case the router cannot
    see an attachment for is unanswerable, and would be measuring the harness.
    """

    def __init__(self, has_documents: bool) -> None:
        self._has_documents = has_documents

    async def aget_routing_descriptors(self, _conversation_id: str, _limit: int) -> list[dict]:
        if not self._has_documents:
            return []
        return [
            {
                "document_id": "eval-doc-1",
                "filename": "attached-document.pdf",
                "file_type": "application/pdf",
                "status": "indexed",
                "upload_time": "2026-09-01T00:00:00Z",
            }
        ]


def build_context_request(
    case: RoutingEvalCase, inventory: RoutingInventory, *, user_id: str
) -> RoutingContextRequest:
    """Turn one labelled case into the request the production builder reads."""
    context = case.context or {}
    active_canvas = None
    if context.get("canvas_open"):
        active_canvas = {
            "artifact_id": "canvas:main",
            "revision": 1,
            "title": "Working document",
            "is_latest_assistant": True,
        }

    return RoutingContextRequest(
        message=case.message,
        inventory=inventory,
        conversation_id=f"eval-{case.case_id}",
        user_id=user_id,
        user_message_id=case.case_id,
        active_canvas=active_canvas,
        attachment_count=1 if context.get("has_documents") else 0,
        locale=None if case.language == "mixed" else case.language,
    )
