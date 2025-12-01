import asyncio
import logging
from typing import Optional, TYPE_CHECKING, List, Dict, Any
from uuid import UUID

from langgraph.graph import StateGraph, END, START
from langgraph.types import Command, interrupt
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .schemas import (
    GraphState,
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
    InterruptDecision,
    InterruptDecisionType,
)
from .agents.router import Router
from .agents.chat_agent import ChatAgent
from .agents.rag_agent import RAGAgent
from .agents.search_agent import SearchAgent
from .agents.image_generator_agent import ImageGeneratorAgent
from .memory import get_memory_manager
from ..core.config import settings
from .hitl_config import build_interrupt_response, requires_human_approval
from .utils import normalize_tool_call, coerce_response_text
import json

if TYPE_CHECKING:
    from ..repositories.document import DocumentRepository

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        document_repository: Optional["DocumentRepository"] = None,
    ):
        self.qdrant_client = qdrant_client
        self.router = Router()
        self.chat_agent = ChatAgent()
        self.rag_agent = RAGAgent(
            settings=settings,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            collection_name=settings.qdrant_collection_name,
        )
        self.search_agent = SearchAgent()
        self.image_generator_agent = ImageGeneratorAgent()
        self.agents = {
            "chat_agent": self.chat_agent,
            "rag_agent": self.rag_agent,
            "search_agent": self.search_agent,
            "image_generator_agent": self.image_generator_agent,
        }

        self.checkpointer = checkpointer
        self.document_repository = document_repository

        self.graph = self._build_graph()
        self._cleanup_agents = [
            self.chat_agent,
            self.search_agent,
            self.rag_agent,
            self.image_generator_agent,
        ]

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        workflow.add_node("route", self._route_node)
        workflow.add_node("chat_agent", self._chat_node)
        workflow.add_node("rag_agent", self._rag_node)
        workflow.add_node("search_agent", self._search_node)
        workflow.add_node("image_generator_agent", self._image_generator_node)
        workflow.add_node("approval", self._approval_node)
        workflow.add_node("tools", self._tool_node)

        workflow.add_edge(START, "route")

        workflow.add_conditional_edges(
            "route",
            self._should_continue,
            {
                "chat_agent": "chat_agent",
                "rag_agent": "rag_agent",
                "search_agent": "search_agent",
                "image_generator_agent": "image_generator_agent",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "chat_agent",
            self._should_call_tools,
            {
                "approval": "approval",
                "tools": "tools",
                "end": END,
            },
        )

        workflow.add_edge("rag_agent", END)

        workflow.add_conditional_edges(
            "search_agent",
            self._should_call_tools,
            {
                "approval": "approval",
                "tools": "tools",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "image_generator_agent",
            self._should_call_tools,
            {
                "approval": "approval",
                "tools": "tools",
                "end": END,
            },
        )

        workflow.add_edge("approval", "tools")

        workflow.add_conditional_edges(
            "tools",
            self._route_tool_output,
            {
                "chat_agent": "chat_agent",
                "search_agent": "search_agent",
                "image_generator_agent": "image_generator_agent",
                "end": END,  # Fallback
            },
        )

        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()

    async def _tool_node(self, state: GraphState) -> GraphState:
        """Execute pending tool calls."""
        messages = state.get("messages", [])
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        selected_agent_name = state.get("selected_agent")
        agent = self.agents.get(selected_agent_name)
        if not agent:
            return state

        if not hasattr(agent, "tools") or not agent.tools:
            if hasattr(agent, "_init_mcp"):
                await agent._init_mcp()
            elif hasattr(agent, "_init_tools"):
                await agent._init_tools()

        if not hasattr(agent, "tools") or not agent.tools:
            return state

        tool_map = {t.name: t for t in agent.tools}
        tool_outputs = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool = tool_map.get(tool_name)
            if tool:
                try:
                    result = (
                        await tool.ainvoke(tool_args)
                        if tool.coroutine
                        else tool.invoke(tool_args)
                    )
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": str(result),
                        }
                    )
                except Exception as e:
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Error: {e}",
                        }
                    )
            else:
                tool_outputs.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": f"Error: Tool {tool_name} not found",
                    }
                )

        new_messages = []
        for output in tool_outputs:
            new_messages.append(
                ToolMessage(
                    content=output["content"],
                    tool_call_id=output["tool_call_id"],
                    name=output["name"],
                )
            )

        state.setdefault("messages", []).extend(new_messages)
        return state

    async def _approval_node(self, state: GraphState) -> GraphState:
        """
        Human-in-the-loop approval node.

        Pauses execution using interrupt() for tools that require human approval.
        The interrupt payload contains tool details for the frontend to display.
        Resume with Command(resume=decisions) where decisions contain approve/reject/edit info.
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        # Build action requests for the interrupt payload
        action_requests = [
            normalize_tool_call(tool_call) for tool_call in last_message.tool_calls
        ]

        # Pause execution and wait for human decision
        # The resume value will be a list of InterruptDecision objects
        human_decisions = interrupt(
            {
                "action_requests": action_requests,
                "message": "Tool execution requires human approval",
            }
        )

        # Process decisions from human review
        if not human_decisions:
            # No decisions provided, reject all tools
            rejection_messages = [
                ToolMessage(
                    content="Tool execution cancelled: No approval provided",
                    tool_call_id=tool_call.get("id"),
                    name=tool_call.get("name"),
                )
                for tool_call in last_message.tool_calls
            ]
            # Replace AI message (no tool calls) and add rejection messages
            state["messages"] = (
                messages[:-1]
                + [AIMessage(content=last_message.content)]
                + rejection_messages
            )
            return state

        # Parse decisions and update state accordingly
        decisions = (
            human_decisions if isinstance(human_decisions, list) else [human_decisions]
        )
        decision_map: Dict[str, Any] = {}
        for d in decisions:
            if isinstance(d, dict):
                task_id = d.get("task_id") or d.get("tool_call_id")
                if task_id:
                    decision_map[task_id] = d

        tool_calls_to_keep = []
        rejection_messages = []

        for tool_call in last_message.tool_calls:
            tool_call_id = tool_call.get("id")
            tool_name = tool_call.get("name")
            decision = decision_map.get(tool_call_id, {})
            decision_type = decision.get("type", "reject")

            if decision_type in ("accept", "approve"):
                tool_calls_to_keep.append(tool_call)
            elif decision_type == "edit":
                modified_args = decision.get("args", tool_call.get("args", {}))
                tool_calls_to_keep.append(
                    {
                        "name": tool_name,
                        "args": modified_args,
                        "id": tool_call_id,
                    }
                )
            else:  # reject, respond, or unknown
                feedback = decision.get("args", {}).get(
                    "message", "Tool execution rejected by user"
                )
                rejection_messages.append(
                    ToolMessage(
                        content=feedback,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

        # Update the AI message with only approved tool calls
        if tool_calls_to_keep:
            new_ai_message = AIMessage(
                content=last_message.content,
                tool_calls=tool_calls_to_keep,
            )
            # Replace the last message with updated tool calls, add any rejection messages
            state["messages"] = messages[:-1] + [new_ai_message] + rejection_messages
        else:
            # All tools rejected, remove tool calls from AI message and add rejection messages
            state["messages"] = (
                messages[:-1]
                + [AIMessage(content=last_message.content)]
                + rejection_messages
            )

        return state

    def _should_call_tools(self, state: GraphState) -> str:
        """Check if the last message has tool calls and whether they need approval."""
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        # Check if any tool requires human approval
        tool_names = [
            normalize_tool_call(tc).get("name") for tc in last_message.tool_calls
        ]
        if requires_human_approval(tool_names):
            return "approval"

        return "tools"

    def _route_tool_output(self, state: GraphState) -> str:
        """Route back to the selected agent after tool execution."""
        return state.get("selected_agent", "end")

    def _build_interrupt_agent_response(
        self,
        state_snapshot: Any,
        thread_id: Optional[str],
        fallback_conversation_id: Optional[str] = None,
    ) -> Optional[AgentResponse]:
        """Create an AgentResponse containing interrupt metadata from graph state."""
        if not state_snapshot or not thread_id:
            return None

        values = getattr(state_snapshot, "values", {})
        messages = values.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return None

        action_requests = [
            normalize_tool_call(tool_call) for tool_call in last_message.tool_calls
        ]

        conversation_id = (
            values.get("conversation_id") or fallback_conversation_id or ""
        )
        interrupt_response = build_interrupt_response(
            {"action_requests": action_requests},
            thread_id,
            conversation_id,
        )

        selected_agent = values.get("selected_agent", "search_agent")
        agent_type_map = {
            "chat_agent": AgentType.CHAT,
            "rag_agent": AgentType.RAG,
            "search_agent": AgentType.SEARCH,
            "image_generator_agent": AgentType.IMAGE_GENERATOR,
        }
        agent_type = agent_type_map.get(selected_agent, AgentType.SEARCH)

        return AgentResponse(
            agent_type=agent_type,
            agent_id=selected_agent or "search_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="Tool execution requires approval",
            ),
            metadata={"interrupt": interrupt_response},
        )

    async def _route_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
        conversation_id = state.get("conversation_id")
        has_documents = self._conversation_has_documents(conversation_id)

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"persona": state.get("persona")},
        )

        selected_agent = await self.router.route_message(
            agent_msg, list(self.agents.keys()), has_documents=has_documents
        )

        if selected_agent == "rag_agent" and not has_documents:
            selected_agent = "chat_agent"

        state["selected_agent"] = selected_agent
        return state

    def _conversation_has_documents(self, conversation_id: Optional[str]) -> bool:
        if not conversation_id or not self.document_repository:
            return False
        try:
            return (
                self.document_repository.count_by_conversation(UUID(conversation_id))
                > 0
            )
        except (ValueError, Exception):
            return False

    async def _chat_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        conversation_history = []
        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            memory_manager = get_memory_manager()
            conv_memory = await memory_manager.get_memory(
                UUID(conversation_id), UUID(user_id), force_refresh=True
            )
            history_limit = (
                settings.chat_history_max_messages
                if settings.chat_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        context = state.get("context", {})
        last_human_idx = next(
            (
                idx
                for idx in range(len(messages) - 1, -1, -1)
                if isinstance(messages[idx], HumanMessage)
            ),
            None,
        )

        has_tool_context = any(
            isinstance(m, (AIMessage, ToolMessage))
            and (
                isinstance(m, ToolMessage)
                or (hasattr(m, "tool_calls") and m.tool_calls)
            )
            for m in messages[last_human_idx + 1 :]
            if last_human_idx is not None
        )

        if has_tool_context:
            response = await self.chat_agent.invoke_model_with_history(
                messages, conversation_history, state.get("persona"), conversation_id
            )
        else:
            user_content = (
                messages[last_human_idx].content
                if last_human_idx is not None
                and hasattr(messages[last_human_idx], "content")
                else content
            )

            agent_msg = AgentMessage(
                role=MessageRole.USER,
                content=user_content,
                metadata={
                    "history": conversation_history,
                    "persona": state.get("persona"),
                },
                attachments=context.get("attachments"),
            )

            response = await self.chat_agent.invoke_model(agent_msg, conversation_id)

        state["response"] = response

        ai_kwargs = {"content": response.message.content}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        return state

    async def _rag_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        conversation_history = []
        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            memory_manager = get_memory_manager()
            conv_memory = await memory_manager.get_memory(
                UUID(conversation_id), UUID(user_id), force_refresh=True
            )
            history_limit = (
                settings.rag_history_max_messages
                if settings.rag_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        context = state.get("context", {})
        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history, "persona": state.get("persona")},
            attachments=context.get("attachments"),
        )

        response = await self.rag_agent.process_message(agent_msg, conversation_id)
        state["response"] = response
        state.setdefault("messages", []).append(
            AIMessage(content=response.message.content)
        )

        return state

    async def _search_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        last_human_idx = next(
            (
                idx
                for idx in range(len(messages) - 1, -1, -1)
                if isinstance(messages[idx], HumanMessage)
            ),
            None,
        )

        has_tool_context = any(
            isinstance(m, (AIMessage, ToolMessage))
            and (
                isinstance(m, ToolMessage)
                or (hasattr(m, "tool_calls") and m.tool_calls)
            )
            for m in messages[last_human_idx + 1 :]
            if last_human_idx is not None
        )

        conversation_history = []
        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            memory_manager = get_memory_manager()
            conv_memory = await memory_manager.get_memory(
                UUID(conversation_id), UUID(user_id), force_refresh=True
            )
            history_limit = (
                settings.search_history_max_messages
                if settings.search_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        if has_tool_context:
            response = await self.search_agent.invoke_model_with_history(
                messages, conversation_history, state.get("persona"), conversation_id
            )
        else:
            user_content = (
                messages[last_human_idx].content
                if last_human_idx is not None
                and hasattr(messages[last_human_idx], "content")
                else content
            )

            context = state.get("context", {})
            agent_msg = AgentMessage(
                role=MessageRole.USER,
                content=user_content,
                metadata={
                    "history": conversation_history,
                    "persona": state.get("persona"),
                },
                attachments=context.get("attachments"),
            )

            response = await self.search_agent.process_message(
                agent_msg, conversation_id
            )

        state["response"] = response

        ai_kwargs = {"content": response.message.content}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        return state

    async def _image_generator_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_human_idx = next(
            (
                idx
                for idx in range(len(messages) - 1, -1, -1)
                if isinstance(messages[idx], HumanMessage)
            ),
            None,
        )

        last_human_message = (
            messages[last_human_idx] if last_human_idx is not None else None
        )
        content = last_human_message.content if last_human_message else ""

        has_tool_context = any(
            isinstance(m, (AIMessage, ToolMessage))
            and (
                isinstance(m, ToolMessage)
                or (hasattr(m, "tool_calls") and m.tool_calls)
            )
            for m in messages[last_human_idx + 1 :]
            if last_human_idx is not None
        )

        conversation_history = []
        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            memory_manager = get_memory_manager()
            conv_memory = await memory_manager.get_memory(
                UUID(conversation_id), UUID(user_id), force_refresh=True
            )
            history_limit = (
                settings.chat_history_max_messages
                if settings.chat_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        if has_tool_context:
            response = await self.image_generator_agent.invoke_model_with_history(
                messages, conversation_history, state.get("persona"), conversation_id
            )
        else:
            full_history = conversation_history + messages[:-1]
            context = state.get("context", {})

            agent_msg = AgentMessage(
                role=MessageRole.USER,
                content=content,
                metadata={"history": full_history, "persona": state.get("persona")},
                attachments=context.get("attachments"),
            )

            response = await self.image_generator_agent.invoke_model(
                agent_msg, conversation_id
            )

        state["response"] = response

        ai_kwargs = {"content": response.message.content}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        return state

    def _should_continue(self, state: GraphState) -> str:
        selected_agent = state.get("selected_agent")
        if selected_agent in self.agents:
            return selected_agent
        return "end"

    async def execute(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        persona: Optional[str] = None,
        attachments: Optional[list] = None,
    ) -> Optional[AgentResponse]:

        initial_state: GraphState = {
            "messages": [HumanMessage(content=message)],
            "context": {},
        }

        if conversation_id is not None:
            initial_state["conversation_id"] = conversation_id
        if user_id is not None:
            initial_state["user_id"] = user_id
        initial_state["selected_agent"] = None
        initial_state["response"] = None
        initial_state["persona"] = persona
        initial_state["reasoning_steps"] = None
        initial_state["tool_results"] = None
        initial_state["iteration_count"] = None

        # Store attachments in context for agent access
        if attachments:
            initial_state["context"]["attachments"] = attachments

        config = (
            {"configurable": {"thread_id": thread_id}}
            if self.checkpointer and thread_id
            else None
        )
        result = await self.graph.ainvoke(initial_state, config=config)

        if self.checkpointer and thread_id:
            state_snapshot = await self.graph.aget_state(config)
            if state_snapshot.next and len(state_snapshot.next) > 0:
                interrupt_agent_response = self._build_interrupt_agent_response(
                    state_snapshot, thread_id, conversation_id
                )
                if interrupt_agent_response:
                    return interrupt_agent_response

        agent_response = result.get("response")
        if (
            agent_response
            and isinstance(agent_response.metadata, dict)
            and "interrupt" in agent_response.metadata
        ):
            return agent_response

        return agent_response

    async def resume_execution(
        self, thread_id: str, resume_value: Any
    ) -> Optional[AgentResponse]:
        """Resume execution after handling interrupts."""
        if not self.checkpointer:
            raise RuntimeError("Checkpointing must be enabled for resume_execution")

        config = {"configurable": {"thread_id": thread_id}}
        result = await self.graph.ainvoke(Command(resume=resume_value), config=config)
        return result.get("response")

    async def resume(
        self,
        thread_id: str,
        user_input: Optional[str] = None,
    ) -> Optional[AgentResponse]:
        """
        Resume the workflow from an interrupt with a simple approval.

        For more complex decision handling (edit/reject), use resume_with_decisions().
        """
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot resume.")

        config = {"configurable": {"thread_id": thread_id}}

        # Get current state to extract tool calls for auto-approval
        state_snapshot = await self.graph.aget_state(config)
        messages = state_snapshot.values.get("messages", [])

        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            # Auto-approve all tool calls
            resume_data = [
                {
                    "task_id": tc.get("id"),
                    "tool_call_id": tc.get("id"),
                    "type": "accept",
                    "args": None,
                }
                for tc in messages[-1].tool_calls
            ]
        else:
            resume_data = user_input

        result = await self.graph.ainvoke(Command(resume=resume_data), config=config)

        # Check for further interrupts
        final_snapshot = await self.graph.aget_state(config)
        if final_snapshot.next and len(final_snapshot.next) > 0:
            if "approval" in final_snapshot.next:
                interrupt_response = self._build_interrupt_agent_response(
                    final_snapshot,
                    thread_id,
                    final_snapshot.values.get("conversation_id"),
                )
                if interrupt_response:
                    return interrupt_response

        response = result.get("response")
        if not response:
            response = (await self.graph.aget_state(config)).values.get("response")

        return response

    async def resume_with_decisions(
        self,
        thread_id: str,
        decisions: List[InterruptDecision],
        interrupt_id: Optional[str] = None,
    ) -> Optional[AgentResponse]:
        """
        Resume workflow execution with user decisions on tool execution.

        Args:
            thread_id: The thread ID to resume
            decisions: List of decisions for each tool (accept/edit/reject)
            interrupt_id: Optional interrupt identifier (for validation)

        Returns:
            AgentResponse with the bot's response after processing decisions
        """
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot resume.")

        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = await self.graph.aget_state(config)

        # Validate interrupt state
        if not state_snapshot.next or len(state_snapshot.next) == 0:
            raise ValueError("Workflow is not in interrupted state")
        if "approval" not in state_snapshot.next:
            raise ValueError(
                f"Unexpected interrupt state: next nodes are {state_snapshot.next}"
            )

        # Convert decisions to format expected by approval node
        resume_data = [
            {
                "task_id": d.task_id,
                "tool_call_id": d.task_id,  # task_id maps to tool_call_id
                "type": d.type.value if hasattr(d.type, "value") else d.type,
                "args": d.args,
            }
            for d in decisions
        ]

        result = await self.graph.ainvoke(Command(resume=resume_data), config=config)

        # Check if workflow paused again for additional tools
        final_snapshot = await self.graph.aget_state(config)
        if final_snapshot.next and len(final_snapshot.next) > 0:
            if "approval" in final_snapshot.next:
                interrupt_agent_response = self._build_interrupt_agent_response(
                    final_snapshot,
                    thread_id,
                    final_snapshot.values.get("conversation_id"),
                )
                if interrupt_agent_response:
                    return interrupt_agent_response

        # Extract response from result
        response = result.get("response")
        if not response:
            final_state = await self.graph.aget_state(config)
            response = final_state.values.get("response")

        return response

    async def execute_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        persona: Optional[str] = None,
        attachments: Optional[list] = None,
    ):
        """
        Execute the workflow with streaming support.
        Uses a hybrid approach:
        - For RAG agent: calls agent's stream_message() directly for native Gemini streaming with thinking
        - For other agents: uses LangGraph's astream_events for LangChain streaming
        """
        initial_state: GraphState = {
            "messages": [HumanMessage(content=message)],
            "context": {},
        }

        if conversation_id is not None:
            initial_state["conversation_id"] = conversation_id
        if user_id is not None:
            initial_state["user_id"] = user_id
        initial_state["selected_agent"] = None
        initial_state["response"] = None
        initial_state["persona"] = persona
        initial_state["reasoning_steps"] = None
        initial_state["tool_results"] = None
        initial_state["iteration_count"] = None

        if attachments:
            initial_state["context"]["attachments"] = attachments

        config = (
            {"configurable": {"thread_id": thread_id}}
            if self.checkpointer and thread_id
            else None
        )

        try:
            routed_state = await self._route_node(initial_state)
            selected_agent = routed_state.get("selected_agent")
            initial_state["selected_agent"] = selected_agent
        except Exception as e:
            yield {"type": "error", "error": str(e)}
            return

        yield {"type": "agent_selected", "agent": selected_agent}

        if selected_agent == "rag_agent":
            conversation_history = []
            if conversation_id and user_id:
                try:
                    memory_manager = get_memory_manager()
                    conv_memory = await memory_manager.get_memory(
                        UUID(conversation_id), UUID(user_id), force_refresh=True
                    )
                    history_limit = (
                        settings.rag_history_max_messages
                        if settings.rag_history_max_messages > 0
                        else None
                    )
                    conversation_history = conv_memory.get_recent_messages(
                        limit=history_limit, exclude_last=1
                    )
                except Exception:
                    pass

            agent_msg = AgentMessage(
                role=MessageRole.USER,
                content=message,
                metadata={"history": conversation_history, "persona": persona},
                attachments=attachments,
            )

            try:
                async for event in self.rag_agent.stream_message(
                    agent_msg, conversation_id
                ):
                    event_type = event.get("type")
                    if event_type in ["thinking", "token", "tool_start", "tool_end"]:
                        yield event
                    elif event_type == "complete":
                        if event.get("response"):
                            yield {
                                "type": "complete",
                                "response": event.get("response"),
                            }
                        return
                    elif event_type == "error":
                        yield event
                        return
                yield {"type": "error", "error": "RAG agent stream ended unexpectedly"}
            except Exception as e:
                yield {"type": "error", "error": str(e)}
            return

        accumulated_content = ""
        try:
            async for event in self.graph.astream_events(
                initial_state, config=config, version="v1"
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        # Use coerce_response_text to handle various content formats
                        # including Anthropic's content blocks with 'text' key
                        content = coerce_response_text(chunk.content)

                        if content:
                            accumulated_content += content
                            additional_kwargs = getattr(chunk, "additional_kwargs", {})
                            if additional_kwargs.get(
                                "thought"
                            ) or additional_kwargs.get("thinking"):
                                yield {"type": "thinking", "content": content}
                            else:
                                yield {"type": "token", "content": content}

                elif kind == "on_tool_start":
                    yield {"type": "tool_start", "name": event["name"]}
                elif kind == "on_tool_end":
                    yield {"type": "tool_end", "name": event["name"]}

        except Exception as e:
            yield {"type": "error", "error": str(e)}
            return

        if self.checkpointer and thread_id:
            try:
                snapshot = await self.graph.aget_state(config)

                if snapshot.next and len(snapshot.next) > 0:
                    messages = snapshot.values.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if (
                            isinstance(last_msg, AIMessage)
                            and hasattr(last_msg, "tool_calls")
                            and last_msg.tool_calls
                        ):
                            pending_tool_calls = [
                                normalize_tool_call(tc) for tc in last_msg.tool_calls
                            ]
                            interrupt_response = build_interrupt_response(
                                {"action_requests": pending_tool_calls},
                                thread_id,
                                conversation_id or "",
                            )
                            yield {
                                "type": "interrupt",
                                "next": snapshot.next,
                                "thread_id": thread_id,
                                "pending_tool_calls": pending_tool_calls,
                                "interrupt": interrupt_response,
                            }
                            return

                response = snapshot.values.get("response")
                if response:
                    yield {"type": "complete", "response": response}
                elif accumulated_content:
                    response = AgentResponse(
                        agent_type=(
                            AgentType.CHAT
                            if selected_agent == "chat_agent"
                            else AgentType.SEARCH
                        ),
                        agent_id=selected_agent or "unknown",
                        message=AgentMessage(
                            role=MessageRole.ASSISTANT, content=accumulated_content
                        ),
                        metadata={
                            "model": (
                                settings.chat_agent_model
                                if selected_agent == "chat_agent"
                                else settings.search_agent_model
                            )
                        },
                    )
                    yield {"type": "complete", "response": response}
                else:
                    yield {"type": "error", "error": "No response generated"}
            except Exception as e:
                yield {"type": "error", "error": str(e)}
        else:
            if accumulated_content:
                response = AgentResponse(
                    agent_type=(
                        AgentType.CHAT
                        if selected_agent == "chat_agent"
                        else AgentType.SEARCH
                    ),
                    agent_id=selected_agent or "unknown",
                    message=AgentMessage(
                        role=MessageRole.ASSISTANT, content=accumulated_content
                    ),
                    metadata={},
                )
                yield {"type": "complete", "response": response}
            else:
                yield {"type": "error", "error": "No response generated"}

    async def get_state(self, thread_id: str) -> dict:
        """Get the current state of a workflow."""
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot get state.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.graph.aget_state(config)

        pending_tool_calls = None
        messages = snapshot.values.get("messages", [])
        if messages:
            last_message = messages[-1]
            if (
                isinstance(last_message, AIMessage)
                and hasattr(last_message, "tool_calls")
                and last_message.tool_calls
            ):
                pending_tool_calls = last_message.tool_calls

        return {
            "interrupted": bool(snapshot.next),
            "next": snapshot.next,
            "values": snapshot.values,
            "pending_tool_calls": pending_tool_calls,
            "selected_agent": snapshot.values.get("selected_agent"),
        }

    async def cleanup(self):
        """Cleanup resources from agents that use MCP tools."""
        for agent in self._cleanup_agents:
            if hasattr(agent, "cleanup"):
                try:
                    await agent.cleanup()
                except Exception:
                    pass


def create_workflow(
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    document_repository: Optional["DocumentRepository"] = None,
) -> MultiAgentWorkflow:
    """
    Create multi-agent workflow with required shared dependencies.
    """
    return MultiAgentWorkflow(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        checkpointer=checkpointer,
        document_repository=document_repository,
    )
