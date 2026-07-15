import asyncio
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.ai.agents.rag_agent import RAGAgent
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def test_rag_agent_includes_user_attachments_in_multimodal_query():
    agent = object.__new__(RAGAgent)
    agent.tools = []
    agent._build_skills_suffix = lambda **_kwargs: ""
    agent._get_tools_for_binding = lambda **_kwargs: []

    async def init_tools():
        agent.tools = []

    agent._init_tools = init_tools

    agent._resolve_runtime_model_config = lambda *_args, **_kwargs: SimpleNamespace(
        provider="openai",
        model="gpt-4o",
        capabilities={"supports_vision": True},
        fallback_config=None,
        warnings=[],
    )

    captured = {}

    async def fake_invoke(**kwargs):
        captured["messages"] = kwargs["messages"]
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="ok"),
            metadata={},
        )

    agent._invoke_agentic_rag_model = fake_invoke

    asyncio.run(
        agent._process_message_agentic(
            AgentMessage(
                role=MessageRole.USER,
                content="compare image to docs",
                attachments=[{"name": "screen.png", "mime": "image/png", "data": "abc"}],
                metadata={"original_query": "compare image to docs"},
            ),
            conversation_id="conv-1",
        ),
    )

    human_messages = [m for m in captured["messages"] if isinstance(m, HumanMessage)]
    assert any(
        isinstance(m.content, list)
        and {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        in m.content
        for m in human_messages
    )


def test_rag_agent_keeps_prior_user_image_attachments_in_history():
    agent = object.__new__(RAGAgent)
    agent.tools = []
    agent._build_skills_suffix = lambda **_kwargs: ""
    agent._get_tools_for_binding = lambda **_kwargs: []

    async def init_tools():
        agent.tools = []

    agent._init_tools = init_tools
    agent._resolve_runtime_model_config = lambda *_args, **_kwargs: SimpleNamespace(
        provider="openai",
        model="gpt-4o",
        capabilities={"supports_vision": True},
        fallback_config=None,
        warnings=[],
    )
    captured = {}

    async def fake_invoke(**kwargs):
        captured["messages"] = kwargs["messages"]
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="ok"),
            metadata={},
        )

    agent._invoke_agentic_rag_model = fake_invoke
    history = [
        AgentMessage(
            role=MessageRole.USER,
            content="earlier screenshot",
            attachments=[{"name": "earlier.png", "mime": "image/png", "data": "abc"}],
        )
    ]

    asyncio.run(
        agent._process_message_agentic(
            AgentMessage(
                role=MessageRole.USER,
                content="compare it with the report",
                metadata={
                    "original_query": "compare it with the report",
                    "history": history,
                },
            ),
            conversation_id="conv-1",
        ),
    )

    human_messages = [m for m in captured["messages"] if isinstance(m, HumanMessage)]
    assert any(
        isinstance(m.content, list)
        and {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        in m.content
        for m in human_messages
    )
