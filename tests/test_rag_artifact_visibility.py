import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage, ToolMessage

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


class FakeRAGAgent:
    async def process_message(self, message, conversation_id):
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


def test_rag_node_merges_existing_search_document_artifacts_into_response():
    workflow = _make_workflow()
    artifact = {
        "tool_call_id": "chunk-call",
        "tool": "search_documents",
        "args": {"action": "search_chunks", "query": "revenue"},
        "output": "SEARCH RESULTS:\n\n[1] report.pdf\nchunk evidence",
        "error": None,
        "status": "success",
    }
    state = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {"tool_artifacts": [artifact]},
        "messages": [
            HumanMessage(content="What changed in revenue?"),
            ToolMessage(
                content="SEARCH RESULTS:\n\n[1] report.pdf\nchunk evidence",
                tool_call_id="chunk-call",
                name="search_documents",
            ),
        ],
    }

    asyncio.run(workflow._rag_node(state))

    assert state["response"].tool_artifacts == [artifact]
    assert state["response"].message.content == "The retrieved chunks show revenue increased."
