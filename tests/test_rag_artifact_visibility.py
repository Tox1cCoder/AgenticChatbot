from unittest.mock import AsyncMock

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


class FakeRAGAgent:
    async def process_message(
        self,
        message,
        conversation_id,
        *,
        internal_tools=None,
        handoff_target_descriptions=None,
    ):
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="The retrieved chunks show revenue increased.",
            ),
            metadata={"agentic_mode": True},
        )


def _make_workflow():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.rag_agent = FakeRAGAgent()
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow._get_state_attachments = lambda _state: []
    return workflow


