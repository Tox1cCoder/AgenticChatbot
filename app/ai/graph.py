import json
import logging
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING, List, Dict, Any, Tuple
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
    TodoStatus,
)
from .agents.router import Router
from .agents.chat_agent import ChatAgent
from .agents.rag_agent import RAGAgent
from .agents.search_agent import SearchAgent
from .agents.image_generator_agent import ImageGeneratorAgent
from .agents.planning_agent import PlanningAgent
from .summarization_middleware import summarize_for_state
from .memory import get_memory_manager
from ..core.config import settings
from .hitl_config import build_interrupt_response, requires_human_approval
from .rag_tool_actions import execute_search_documents_action
from .todo_actions import apply_write_todos_action
from .utils import normalize_tool_call, coerce_response_text, make_json_safe, extract_content_from_result
from ..core.response_constants import NO_RESPONSE_GENERATED

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..repositories.document import DocumentRepository


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
        self.planning_agent = PlanningAgent()
        self.agents = {
            "chat_agent": self.chat_agent,
            "rag_agent": self.rag_agent,
            "search_agent": self.search_agent,
            "image_generator_agent": self.image_generator_agent,
            "planning_agent": self.planning_agent,
        }

        self.checkpointer = checkpointer
        self.document_repository = document_repository

        # Conversation history cache: conversation_id -> (history, timestamp)
        self._history_cache: Dict[str, Tuple[List, datetime]] = {}
        self._history_cache_ttl_seconds: int = 60  # Cache for 60 seconds

        self.graph = self._build_graph()
        self._cleanup_agents = [
            self.chat_agent,
            self.search_agent,
            self.rag_agent,
            self.image_generator_agent,
        ]
        self._initialized = False

    def _get_current_turn_messages(self, messages: List) -> List:
        if not messages:
            return messages

        # Find the last HumanMessage index
        last_human_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                last_human_idx = idx
                break

        if last_human_idx is None:
            return messages

        # Return only messages from the current turn
        return messages[last_human_idx:]

    # ============================================================
    # Shared Helper Methods 
    # ============================================================

    async def _get_conversation_history(
        self, conversation_id: Optional[str], user_id: Optional[str]
    ) -> List:
        """
        Get conversation history with caching.

        Caches history per conversation_id with TTL to avoid
        redundant database queries within a single graph execution.
        """
        if not conversation_id or not user_id:
            return []

        # Check cache first
        cache_key = conversation_id
        now = datetime.now(timezone.utc)

        if cache_key in self._history_cache:
            cached_history, cached_time = self._history_cache[cache_key]
            age_seconds = (now - cached_time).total_seconds()

            if age_seconds < self._history_cache_ttl_seconds:
                return cached_history

        try:
            memory_manager = get_memory_manager()
            conv_memory = await memory_manager.get_memory(
                UUID(conversation_id), UUID(user_id), force_refresh=True
            )
            history = conv_memory.get_recent_messages(limit=None, exclude_last=1)

            # Cache the result
            self._history_cache[cache_key] = (history, now)

            return history
        except Exception:
            return []

    def invalidate_history_cache(self, conversation_id: str) -> None:
        """Invalidate cached history for a conversation (call when new messages added)."""
        self._history_cache.pop(conversation_id, None)

    def _find_last_human_message_index(self, messages: List) -> Optional[int]:
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                return idx
        return None

    def _has_tool_context(self, messages: List, last_human_idx: Optional[int]) -> bool:
        if last_human_idx is None:
            return False

        return any(
            isinstance(m, (AIMessage, ToolMessage))
            and (
                isinstance(m, ToolMessage)
                or (hasattr(m, "tool_calls") and m.tool_calls)
            )
            for m in messages[last_human_idx + 1 :]
        )

    def _merge_tool_artifacts(
        self, state: GraphState, response: AgentResponse, append_images: bool = False
    ) -> None:
        context = state.get("context", {})
        tool_artifacts = context.get("tool_artifacts", [])
        tool_images = context.get("tool_images", [])

        if tool_artifacts:
            response.tool_artifacts = tool_artifacts

        if tool_images:
            if not response.metadata:
                response.metadata = {}
            if append_images:
                existing_images = response.metadata.get("images", [])
                existing_images.extend(tool_images)
                response.metadata["images"] = existing_images
            else:
                response.metadata["images"] = tool_images

    def _finalize_agent_response(
        self, state: GraphState, response: AgentResponse
    ) -> GraphState:
        state["response"] = response

        ai_kwargs = {"content": response.message.content}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        return state

    def _build_initial_state(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        persona: Optional[str] = None,
        attachments: Optional[list] = None,
        current_task: Optional[Dict[str, Any]] = None,
        all_tasks: Optional[List[Dict[str, Any]]] = None,
        planning_mode_enabled: bool = False,
        has_existing_plan: bool = False,
        existing_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> GraphState:
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
        initial_state["iteration_count"] = None

        if current_task:
            initial_state["current_task"] = current_task
            initial_state["task_plan_id"] = current_task.get("id")
        if all_tasks:
            initial_state["all_tasks"] = all_tasks

        if attachments:
            initial_state["context"]["attachments"] = attachments

        initial_state["context"]["planning_mode_enabled"] = planning_mode_enabled
        initial_state["context"]["has_existing_plan"] = has_existing_plan
        if existing_tasks:
            initial_state["context"]["existing_tasks"] = existing_tasks
            initial_state["todos"] = existing_tasks
            initial_state["current_task_index"] = self._find_first_pending_task(
                existing_tasks
            )

        # Initialize planning call count for budget tracking
        initial_state["planning_call_count"] = 0

        # Default to "planning" phase - only switch to "executing" when user requests
        initial_state["planning_phase"] = "planning"

        return initial_state

    def _find_first_pending_task(self, tasks: List[Dict[str, Any]]) -> Optional[int]:
        """Find the first pending or in-progress task index in the task list."""
        for i, task in enumerate(tasks):
            status = task.get("status", "pending")
            if status in ("pending", "in_progress"):
                return i
        return None

    async def initialize(self) -> None:
        if self._initialized:
            return

        for agent in [
            self.chat_agent,
            self.search_agent,
            self.image_generator_agent,
            self.rag_agent,
            self.planning_agent,
        ]:
            if hasattr(agent, "_init_tools"):
                await agent._init_tools()
            elif hasattr(agent, "_init_mcp"):
                await agent._init_mcp()

        self._initialized = True

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        # Add summarization node first - runs ONCE at start of each request
        workflow.add_node("summarize", self._summarization_node)
        workflow.add_node("route", self._route_node)
        workflow.add_node("chat_agent", self._chat_node)
        workflow.add_node("rag_agent", self._rag_node)
        workflow.add_node("search_agent", self._search_node)
        workflow.add_node("image_generator_agent", self._image_generator_node)
        workflow.add_node("planning_agent", self._planning_node)
        workflow.add_node("planning_tools", self._planning_tools_node)
        workflow.add_node("rag_tools", self._rag_tools_node)
        workflow.add_node("approval", self._approval_node)
        workflow.add_node("tools", self._tool_node)

        # Route: START -> summarize -> route -> agents
        # Summarization runs ONCE before routing, not on every agent iteration
        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "route")

        workflow.add_conditional_edges(
            "route",
            self._should_continue,
            {
                "chat_agent": "chat_agent",
                "rag_agent": "rag_agent",
                "search_agent": "search_agent",
                "image_generator_agent": "image_generator_agent",
                "planning_agent": "planning_agent",
                "end": END,
            },
        )

        # Consolidate conditional edges for agents that use standard tool calling
        tool_calling_agents = ["chat_agent", "search_agent", "image_generator_agent"]
        for agent_name in tool_calling_agents:
            workflow.add_conditional_edges(
                agent_name,
                self._should_call_tools,
                {
                    "approval": "approval",
                    "tools": "tools",
                    "end": END,
                },
            )

        # RAG agent: direct to END if not agentic, or rag_tools loop if agentic
        workflow.add_conditional_edges(
            "rag_agent",
            self._should_call_rag_tools,
            {
                "rag_tools": "rag_tools",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "rag_tools",
            self._should_continue_rag,
            {
                "rag_agent": "rag_agent",
                "end": END,
            },
        )

        # Planning agent ReAct loop: planning_agent → planning_tools → planning_agent OR end
        workflow.add_conditional_edges(
            "planning_agent",
            self._should_call_planning_tools,
            {
                "planning_tools": "planning_tools",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "planning_tools",
            self._should_continue_planning,
            {
                "planning_agent": "planning_agent",
                "end": END,
            },
        )

        workflow.add_edge("approval", "tools")

        # Dynamic tool routing map based on agent registry
        tool_routing_map = {
            agent_name: agent_name
            for agent_name in self.agents.keys()
            if agent_name != "rag_agent"
        }
        tool_routing_map["end"] = END

        workflow.add_conditional_edges(
            "tools",
            self._route_tool_output,
            tool_routing_map,
        )

        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        return workflow.compile()

    async def _tool_node(self, state: GraphState) -> GraphState:
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
        tool_artifacts = []
        all_images = []

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
                    # Extract actual content from Content objects
                    result = extract_content_from_result(result)
                    result_str = str(result)
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": result_str,
                        }
                    )

                    # Track tool artifact
                    tool_artifacts.append(
                        {
                            "tool_call_id": tool_id,
                            "tool": tool_name,
                            "args": tool_args,
                            "output": (
                                result_str[:1000]
                                if len(result_str) > 1000
                                else result_str
                            ),
                            "error": None,
                        }
                    )

                    try:
                        parsed_result = json.loads(result_str)
                        if (
                            isinstance(parsed_result, dict)
                            and "images" in parsed_result
                        ):
                            for img in parsed_result["images"]:
                                if isinstance(img, dict):
                                    img_url = img.get("url")
                                    img_desc = img.get("description", "")
                                    if img_url:
                                        all_images.append(
                                            {
                                                "url": img_url,
                                                "description": img_desc,
                                            }
                                        )
                    except (json.JSONDecodeError, TypeError):
                        pass

                except Exception as e:
                    error_msg = f"Error: {e}"
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": error_msg,
                        }
                    )
                    tool_artifacts.append(
                        {
                            "tool": tool_name,
                            "args": tool_args,
                            "output": None,
                            "error": str(e),
                        }
                    )
            else:
                error_msg = f"Error: Tool {tool_name} not found"
                tool_outputs.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": error_msg,
                    }
                )
                tool_artifacts.append(
                    {
                        "tool_call_id": tool_id,
                        "tool": tool_name,
                        "args": tool_args,
                        "output": None,
                        "error": f"Tool {tool_name} not found",
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

        current_iteration = state.get("iteration_count") or 0
        state["iteration_count"] = current_iteration + 1

        context = state.get("context", {})
        if tool_artifacts:
            existing_artifacts = context.get("tool_artifacts", [])
            existing_artifacts.extend(tool_artifacts)
            context["tool_artifacts"] = existing_artifacts
        if all_images:
            existing_images = context.get("tool_images", [])
            existing_images.extend(all_images)
            context["tool_images"] = existing_images
        state["context"] = context

        return state

    async def _approval_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        action_requests = [
            normalize_tool_call(tool_call) for tool_call in last_message.tool_calls
        ]

        human_decisions = interrupt(
            {
                "action_requests": action_requests,
                "message": "Tool execution requires human approval",
            }
        )

        if not human_decisions:
            rejection_messages = [
                ToolMessage(
                    content="Tool execution cancelled: No approval provided",
                    tool_call_id=tool_call.get("id"),
                    name=tool_call.get("name"),
                )
                for tool_call in last_message.tool_calls
            ]
            state["messages"] = (
                messages[:-1]
                + [AIMessage(content=last_message.content)]
                + rejection_messages
            )
            return state

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
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        tool_names = [
            normalize_tool_call(tc).get("name") for tc in last_message.tool_calls
        ]
        if requires_human_approval(tool_names):
            return "approval"

        return "tools"

    def _route_tool_output(self, state: GraphState) -> str:
        iteration_count = state.get("iteration_count", 0)
        max_iterations = getattr(settings, "react_agent_max_iterations")

        if iteration_count >= max_iterations:
            messages = state.get("messages", [])
            if messages:
                context = state.get("context", {})
                context["max_iterations_reached"] = True
                state["context"] = context
            return "end"

        selected_agent = state.get("selected_agent", "end")

        if selected_agent != "end" and selected_agent not in self.agents:
            logger.warning(
                f"Selected agent '{selected_agent}' not found in registry, ending conversation"
            )
            return "end"

        return selected_agent

    def _build_interrupt_agent_response(
        self,
        state_snapshot: Any,
        thread_id: Optional[str],
        fallback_conversation_id: Optional[str] = None,
    ) -> Optional[AgentResponse]:
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

        agent = self.agents.get(selected_agent)
        agent_type = (
            agent.agent_type
            if agent and hasattr(agent, "agent_type")
            else AgentType.SEARCH
        )

        return AgentResponse(
            agent_type=agent_type,
            agent_id=selected_agent or "search_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="Tool execution requires approval",
            ),
            metadata={"interrupt": interrupt_response},
        )

    async def _summarization_node(self, state: GraphState) -> GraphState:
        """
        Summarization node that runs ONCE at the start of each user request.
        """
        try:
            # This will check thresholds and apply summarization if needed
            # The function mutates state by replacing old messages with summary
            state = await summarize_for_state(state)
        except Exception as e:
            # Log but don't fail the request if summarization errors
            logger.warning(f"Summarization node error (continuing anyway): {e}")

        return state

    async def _route_node(self, state: GraphState) -> GraphState:
        if state.get("selected_agent"):
            return state

        messages = state.get("messages", [])
        if not messages:
            return state

        context = state.get("context", {})

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
        conversation_id = state.get("conversation_id")
        has_documents = self._conversation_has_documents(conversation_id)

        planning_mode_enabled = context.get("planning_mode_enabled", False)
        has_existing_plan = context.get("has_existing_plan", False)

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"persona": state.get("persona")},
        )

        available_agents = list(self.agents.keys())

        selected_agent = await self.router.route_message(
            agent_msg,
            available_agents,
            has_documents=has_documents,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
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

    def _build_graph_config(
        self, thread_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        config: Dict[str, Any] = {}
        recursion_limit = getattr(settings, "react_agent_recursion_limit", None)
        if recursion_limit and recursion_limit > 0:
            config["recursion_limit"] = recursion_limit

        if self.checkpointer and thread_id:
            config.setdefault("configurable", {})["thread_id"] = thread_id

        return config or None

    async def _chat_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id
        )

        current_turn_messages = self._get_current_turn_messages(messages)

        response = await self.chat_agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
        )

        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

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

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id
        )

        context = state.get("context", {})

        last_human_idx = self._find_last_human_message_index(messages)
        original_query = (
            messages[last_human_idx].content
            if last_human_idx is not None
            else content
        )

        tool_context = []
        if last_human_idx is not None:
            for msg in messages[last_human_idx + 1 :]:
                if isinstance(msg, ToolMessage):
                    tool_context.append(msg.content)

        metadata = {
            "persona": state.get("persona"),
            "history": conversation_history,
            "original_query": original_query,
            "tool_context": tool_context,
            "agentic_images": context.get("agentic_images", []),  # Pass images for multimodal LLM
        }

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=original_query,
            metadata=metadata,
            attachments=context.get("attachments"),
        )

        response = await self.rag_agent.process_message(agent_msg, conversation_id)
        state["response"] = response

        ai_kwargs = {"content": response.message.content or ""}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        return state

    async def _rag_tools_node(self, state: GraphState) -> GraphState:
        """
        Execute RAG document exploration tools (agentic mode).

        - SCAN_ALL: Preview all documents in conversation
        - READ_DOCUMENT: Full content of specific document
        - SEARCH_CHUNKS: Vector search
        - GREP_DOCUMENT: Regex search
        - LIST_DOCUMENTS: List available documents
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        conversation_id = state.get("conversation_id")
        tool_outputs = []
        context = state.get("context", {})
        tool_artifacts: List[Dict[str, Any]] = []
        all_images: List[Dict[str, str]] = []
        max_agentic_images = getattr(settings, "agentic_rag_max_images", 6)

        # Track agentic iteration count
        agentic_iteration = context.get("agentic_rag_iteration", 0) + 1
        context["agentic_rag_iteration"] = agentic_iteration

        normalized_tool_calls = [
            normalize_tool_call(tool_call) for tool_call in last_message.tool_calls
        ]

        non_search_tool_calls = [
            tc for tc in normalized_tool_calls if tc.get("name") != "search_documents"
        ]
        non_search_outputs_by_id: Dict[str, str] = {}

        tool_map: Dict[str, Any] = {}
        if non_search_tool_calls:
            tool_calls_to_execute = list(non_search_tool_calls)

            tool_names = [tc.get("name") for tc in non_search_tool_calls]
            if requires_human_approval(tool_names):
                human_decisions = interrupt(
                    {
                        "action_requests": non_search_tool_calls,
                        "message": "Tool execution requires human approval",
                    }
                )

                if not human_decisions:
                    for tc in non_search_tool_calls:
                        non_search_outputs_by_id[tc.get("id")] = (
                            "Tool execution cancelled: No approval provided"
                        )
                    tool_calls_to_execute = []
                else:
                    decisions = (
                        human_decisions
                        if isinstance(human_decisions, list)
                        else [human_decisions]
                    )
                    decision_map: Dict[str, Any] = {}
                    for d in decisions:
                        if isinstance(d, dict):
                            task_id = d.get("task_id") or d.get("tool_call_id")
                            if task_id:
                                decision_map[task_id] = d

                    tool_calls_to_execute = []
                    for tc in non_search_tool_calls:
                        tool_call_id = tc.get("id")
                        tool_name = tc.get("name")
                        decision = decision_map.get(tool_call_id, {})
                        decision_type = decision.get("type", "reject")

                        if decision_type in ("accept", "approve"):
                            tool_calls_to_execute.append(tc)
                        elif decision_type == "edit":
                            modified_args = decision.get("args", tc.get("args", {}))
                            tool_calls_to_execute.append(
                                {"name": tool_name, "args": modified_args, "id": tool_call_id}
                            )
                        else:
                            feedback = decision.get("args", {}).get(
                                "message", "Tool execution rejected by user"
                            )
                            non_search_outputs_by_id[tool_call_id] = feedback

            if tool_calls_to_execute:
                selected_agent_name = state.get("selected_agent")
                agent = (
                    self.agents.get(selected_agent_name) if selected_agent_name else None
                )
                if agent:
                    if not hasattr(agent, "tools") or not agent.tools:
                        if hasattr(agent, "_init_mcp"):
                            await agent._init_mcp()
                        elif hasattr(agent, "_init_tools"):
                            await agent._init_tools()
                    if hasattr(agent, "tools") and agent.tools:
                        tool_map = {t.name: t for t in agent.tools}

                for tc in tool_calls_to_execute:
                    tool_name = tc.get("name")
                    tool_id = tc.get("id")
                    tool_args = tc.get("args", {})

                    tool = tool_map.get(tool_name) if tool_map else None
                    if tool:
                        try:
                            result = (
                                await tool.ainvoke(tool_args)
                                if tool.coroutine
                                else tool.invoke(tool_args)
                            )
                            result = extract_content_from_result(result)
                            result_str = str(result)
                            non_search_outputs_by_id[tool_id] = result_str
                            tool_artifacts.append(
                                {
                                    "tool_call_id": tool_id,
                                    "tool": tool_name,
                                    "args": tool_args,
                                    "output": (
                                        result_str[:1000]
                                        if len(result_str) > 1000
                                        else result_str
                                    ),
                                    "error": None,
                                }
                            )
                            try:
                                parsed_result = json.loads(result_str)
                                if (
                                    isinstance(parsed_result, dict)
                                    and "images" in parsed_result
                                ):
                                    for img in parsed_result["images"]:
                                        if isinstance(img, dict):
                                            img_url = img.get("url")
                                            img_desc = img.get("description", "")
                                            if img_url:
                                                all_images.append(
                                                    {
                                                        "url": img_url,
                                                        "description": img_desc,
                                                    }
                                                )
                            except (json.JSONDecodeError, TypeError):
                                pass
                        except Exception as e:
                            error_msg = f"Error: {e}"
                            non_search_outputs_by_id[tool_id] = error_msg
                            tool_artifacts.append(
                                {
                                    "tool_call_id": tool_id,
                                    "tool": tool_name,
                                    "args": tool_args,
                                    "output": None,
                                    "error": str(e),
                                }
                            )
                    else:
                        error_msg = f"Error: Tool {tool_name} not found"
                        non_search_outputs_by_id[tool_id] = error_msg
                        tool_artifacts.append(
                            {
                                "tool_call_id": tool_id,
                                "tool": tool_name,
                                "args": tool_args,
                                "output": None,
                                "error": f"Tool {tool_name} not found",
                            }
                        )

        for tool_call_data in normalized_tool_calls:
            tool_name = tool_call_data.get("name")
            tool_id = tool_call_data.get("id")
            tool_args = tool_call_data.get("args", {})

            if tool_name != "search_documents":
                tool_outputs.append(
                    {
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": non_search_outputs_by_id.get(
                            tool_id, f"Error: Tool {tool_name} not found"
                        ),
                    }
                )
                continue

            result, _ = await execute_search_documents_action(
                rag_agent=self.rag_agent,
                conversation_id=conversation_id,
                tool_args=tool_args,
                context=context,
                max_agentic_images=max_agentic_images,
            )

            tool_outputs.append({
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": result,
            })

        # Add tool messages to state
        for output in tool_outputs:
            state.setdefault("messages", []).append(
                ToolMessage(
                    content=output["content"],
                    tool_call_id=output["tool_call_id"],
                    name=output["name"],
                )
            )

        state["context"] = context

        # Increment iteration count
        current_iteration = state.get("iteration_count") or 0
        state["iteration_count"] = current_iteration + 1

        if tool_artifacts:
            existing_artifacts = context.get("tool_artifacts", [])
            existing_artifacts.extend(tool_artifacts)
            context["tool_artifacts"] = existing_artifacts
        if all_images:
            existing_images = context.get("tool_images", [])
            existing_images.extend(all_images)
            context["tool_images"] = existing_images

        return state

    def _should_call_rag_tools(self, state: GraphState) -> str:
        if not settings.agentic_rag_enabled:
            return "end"

        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "rag_tools"

        return "end"

    def _should_continue_rag(self, state: GraphState) -> str:
        """
        Determine if RAG agentic loop should continue or end.
        """
        context = state.get("context", {})
        agentic_iteration = context.get("agentic_rag_iteration", 0)

        # Check iteration limit
        max_iterations = settings.agentic_max_iterations
        if agentic_iteration >= max_iterations:
            logger.warning(
                f"RAG agentic loop reached max iterations ({max_iterations}), forcing end"
            )
            return "end"

        # Continue to RAG agent for more processing
        return "rag_agent"

    async def _search_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id
        )

        current_turn_messages = self._get_current_turn_messages(messages)

        response = await self.search_agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
        )

        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

    async def _image_generator_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id
        )

        current_turn_messages = self._get_current_turn_messages(messages)

        response = await self.image_generator_agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
        )

        self._merge_tool_artifacts(state, response, append_images=True)
        return self._finalize_agent_response(state, response)

    async def _planning_node(self, state: GraphState) -> GraphState:
        """
        Planning agent node with tool-calling ReAct pattern.

        Uses invoke_model_with_history to get responses that may include tool calls
        for the write_todos tool. Supports both initial plan creation and
        ongoing task management.
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        # Get conversation context
        conversation_history = []
        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            memory_manager = get_memory_manager()
            conv_memory = await memory_manager.get_memory(
                UUID(conversation_id), UUID(user_id), force_refresh=True
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=None, exclude_last=1
            )

        context = state.get("context", {})
        existing_tasks = context.get("existing_tasks", [])

        # Get planning phase and check if we need to generate plan response
        planning_phase = state.get("planning_phase", "planning")
        should_generate_plan_response = context.get("generate_plan_response", False)

        # Clear the flag after reading
        if should_generate_plan_response:
            context["generate_plan_response"] = False
            state["context"] = context

        # Get current todos from state (may have been updated by planning_tools)
        todos = state.get("todos", [])
        current_task_index = state.get("current_task_index")

        # Increment planning call count for budget tracking
        planning_call_count = (state.get("planning_call_count") or 0) + 1
        state["planning_call_count"] = planning_call_count

        persona = state.get("persona")

        # Get only current turn messages for the model
        current_turn_messages = self._get_current_turn_messages(messages)

        # Call the planning agent with history
        response = await self.planning_agent.invoke_model_with_history(
            messages=current_turn_messages,
            conversation_history=conversation_history,
            persona=persona,
            conversation_id=conversation_id,
            todos=todos,
            current_task_index=current_task_index,
            planning_phase=planning_phase,
            should_describe_plan=should_generate_plan_response,
        )

        state["response"] = response

        # Check if agent switched to executing phase via response metadata
        if response.metadata.get("planning_phase"):
            state["planning_phase"] = response.metadata["planning_phase"]

        # Add AI message to state (with tool calls if present)
        ai_kwargs = {"content": response.message.content or ""}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        if response.metadata.get("todos"):
            state["todos"] = response.metadata["todos"]

        # Mark that final summary was generated if this was a summary call
        if context.get("generate_final_summary") and not response.message.tool_calls:
            context["final_summary_generated"] = True
            state["context"] = context

        return state

    async def _planning_tools_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        todos = list(state.get("todos", []))  # Make a copy
        current_task_index = state.get("current_task_index")
        tool_outputs = []
        write_todos_actions: List[str] = []

        # Track errors for circuit breaker
        context = state.get("context", {})
        had_error = False
        max_todos = getattr(settings, "max_todos_per_plan", 50)

        for tool_call in last_message.tool_calls:
            tool_call_data = normalize_tool_call(tool_call)
            tool_name = tool_call_data.get("name")
            tool_id = tool_call_data.get("id")
            tool_args = tool_call_data.get("args", {})

            if tool_name != "write_todos":
                try:
                    tool = None
                    if self.planning_agent and hasattr(self.planning_agent, "tools"):
                        for t in self.planning_agent.tools:
                            if t.name == tool_name:
                                tool = t
                                break

                    if tool:
                        result = await tool.ainvoke(tool_args)
                        tool_outputs.append(
                            {
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": (
                                    str(result)
                                    if result
                                    else "Tool executed successfully"
                                ),
                            }
                        )
                    else:
                        tool_outputs.append(
                            {
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": f"Tool not found: {tool_name}",
                            }
                        )
                except Exception as e:
                    logger.error(f"Error executing MCP tool {tool_name}: {e}")
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": f"Error executing tool: {str(e)}",
                        }
                    )
                    had_error = True  # Mark error for circuit breaker
                continue

            try:
                todos, current_task_index, result, action = apply_write_todos_action(
                    todos=todos,
                    current_task_index=current_task_index,
                    tool_args=tool_args,
                    max_todos=max_todos,
                )
                write_todos_actions.append(action)

                if action == "set_todos" and result.startswith(
                    "Error: Plan exceeds maximum"
                ):
                    requested = len(tool_args.get("todos", []) or [])
                    logger.warning(
                        "Rejected plan with %d todos (max: %d)", requested, max_todos
                    )

            except Exception as e:
                raw_action = tool_args.get("action")
                action = raw_action.value if hasattr(raw_action, "value") else raw_action
                result = f"Error executing {action}: {str(e)}"
                had_error = True  # Mark error for circuit breaker

            tool_outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result,
                }
            )

        # Add tool messages to state
        for output in tool_outputs:
            state.setdefault("messages", []).append(
                ToolMessage(
                    content=output["content"],
                    tool_call_id=output["tool_call_id"],
                    name=output["name"],
                )
            )

        # Update state with new todos
        state["todos"] = todos
        state["current_task_index"] = current_task_index

        # Check if plan was modified (SET_TODOS, ADD_TODO, UPDATE_TODO, REMOVE_TODO)
        # and mark context so we can return to agent for response generation
        context = state.get("context", {})
        plan_modifying_actions = {"set_todos", "add_todo", "update_todo", "remove_todo"}
        execution_actions = {"start_todo", "complete_todo"}

        for action in write_todos_actions:
            if action in plan_modifying_actions:
                context["plan_just_modified"] = True
            if action in execution_actions:
                state["planning_phase"] = "executing"
                logger.debug(f"Switched to executing phase due to {action} action")
        state["context"] = context

        # Add tool artifacts to response for UI visibility
        response = state.get("response")
        if response and tool_outputs:
            if response.tool_artifacts is None:
                response.tool_artifacts = []
            for output in tool_outputs:
                response.tool_artifacts.append(
                    {
                        "tool": output["name"],
                        "result": output["content"],
                    }
                )
            state["response"] = response

        # Increment iteration count for budget tracking
        current_iteration = state.get("iteration_count") or 0
        state["iteration_count"] = current_iteration + 1

        # Update consecutive_errors counter for circuit breaker
        if had_error:
            context["consecutive_errors"] = context.get("consecutive_errors", 0) + 1
            logger.warning(
                f"Planning consecutive errors: {context['consecutive_errors']}"
            )
        else:
            # Reset on successful iteration
            context["consecutive_errors"] = 0
        state["context"] = context

        return state

    def _should_call_planning_tools(self, state: GraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        if last_message.tool_calls:
            return "planning_tools"

        return "end"

    def _should_continue_planning(self, state: GraphState) -> str:
        planning_call_count = state.get("planning_call_count", 0)
        max_iterations = getattr(settings, "planning_max_iterations", 20)

        # Check iteration budget
        if planning_call_count >= max_iterations:
            context = state.get("context", {})
            context["planning_budget_reached"] = True
            context["pause_reason"] = "max_iterations_reached"
            state["context"] = context
            logger.warning(
                f"Planning budget exceeded: {planning_call_count} >= {max_iterations}"
            )
            return "end"

        # Circuit breaker: check consecutive errors
        context = state.get("context", {})
        consecutive_errors = context.get("consecutive_errors", 0)
        max_consecutive_errors = getattr(
            settings, "planning_consecutive_errors_limit", 3
        )

        if consecutive_errors >= max_consecutive_errors:
            context["planning_budget_reached"] = True
            context["pause_reason"] = "consecutive_errors_limit"
            state["context"] = context
            logger.warning(
                f"Planning circuit breaker triggered: {consecutive_errors} consecutive errors"
            )
            return "end"

        context = state.get("context", {})
        planning_phase = state.get("planning_phase", "planning")

        # If plan was just created/modified, return to agent for confirmation response
        if context.get("plan_just_modified"):
            context["plan_just_modified"] = False
            context["generate_plan_response"] = True
            state["context"] = context
            return "planning_agent"

        # In planning phase, don't auto-execute - just end after plan response
        if planning_phase == "planning":
            return "end"

        # === EXECUTING PHASE LOGIC ===
        todos = state.get("todos", [])
        if todos:
            pending_count = sum(
                1
                for t in todos
                if t.get("status")
                in (
                    TodoStatus.PENDING.value,
                    "pending",
                    TodoStatus.IN_PROGRESS.value,
                    "in_progress",
                )
            )
            if pending_count == 0:
                context["all_tasks_completed"] = True
                state["context"] = context

                # Check if we've already generated the final summary
                if context.get("final_summary_generated"):
                    return "end"

                # Need one more iteration to generate completion summary
                context["generate_final_summary"] = True
                state["context"] = context
                return "planning_agent"

        return "planning_agent"

    def _should_continue(self, state: GraphState) -> str:
        selected_agent = state.get("selected_agent")
        if selected_agent in self.agents:
            return selected_agent
        return "end"

    def _get_agent_type(self, selected_agent: Optional[str]) -> AgentType:
        agent_type_map = {
            "chat_agent": AgentType.CHAT,
            "rag_agent": AgentType.RAG,
            "search_agent": AgentType.SEARCH,
            "image_generator_agent": AgentType.IMAGE_GENERATOR,
            "planning_agent": AgentType.PLANNING,
        }
        return agent_type_map.get(selected_agent, AgentType.CHAT)

    async def execute(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        persona: Optional[str] = None,
        attachments: Optional[list] = None,
        current_task: Optional[Dict[str, Any]] = None,
        all_tasks: Optional[List[Dict[str, Any]]] = None,
        planning_mode_enabled: bool = False,
        has_existing_plan: bool = False,
        existing_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[AgentResponse]:

        initial_state = self._build_initial_state(
            message=message,
            conversation_id=conversation_id,
            user_id=user_id,
            persona=persona,
            attachments=attachments,
            current_task=current_task,
            all_tasks=all_tasks,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
            existing_tasks=existing_tasks,
        )

        config = self._build_graph_config(thread_id)
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

        final_todos = result.get("todos", [])
        if agent_response and final_todos:
            if agent_response.metadata is None:
                agent_response.metadata = {}
            agent_response.metadata["todos"] = final_todos
            context = result.get("context", {})
            if context.get("all_tasks_completed"):
                agent_response.metadata["all_tasks_completed"] = True
            if context.get("planning_budget_reached"):
                agent_response.metadata["planning_budget_reached"] = True
            agent_response.metadata["planning_call_count"] = result.get(
                "planning_call_count", 0
            )

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
        if not self.checkpointer:
            raise RuntimeError("Checkpointing must be enabled for resume_execution")

        config = self._build_graph_config(thread_id)
        result = await self.graph.ainvoke(Command(resume=resume_value), config=config)
        return result.get("response")

    async def resume(
        self,
        thread_id: str,
        user_input: Optional[str] = None,
    ) -> Optional[AgentResponse]:
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot resume.")

        config = self._build_graph_config(thread_id)

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
    ) -> Optional[AgentResponse]:
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot resume.")

        config = self._build_graph_config(thread_id)
        state_snapshot = await self.graph.aget_state(config)

        if not state_snapshot.next or len(state_snapshot.next) == 0:
            raise ValueError("Workflow is not in interrupted state")
        if "approval" not in state_snapshot.next:
            raise ValueError(
                f"Unexpected interrupt state: next nodes are {state_snapshot.next}"
            )

        resume_data = [
            {
                "task_id": d.task_id,
                "tool_call_id": d.task_id,
                "type": d.type.value if hasattr(d.type, "value") else d.type,
                "args": d.args,
            }
            for d in decisions
        ]

        result = await self.graph.ainvoke(Command(resume=resume_data), config=config)

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
        current_task: Optional[Dict[str, Any]] = None,
        all_tasks: Optional[List[Dict[str, Any]]] = None,
        planning_mode_enabled: bool = False,
        has_existing_plan: bool = False,
        existing_tasks: Optional[List[Dict[str, Any]]] = None,
    ):
        initial_state = self._build_initial_state(
            message=message,
            conversation_id=conversation_id,
            user_id=user_id,
            persona=persona,
            attachments=attachments,
            current_task=current_task,
            all_tasks=all_tasks,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
            existing_tasks=existing_tasks,
        )

        config = self._build_graph_config(thread_id)

        try:
            routed_state = await self._route_node(initial_state)
            selected_agent = routed_state.get("selected_agent")
            initial_state["selected_agent"] = selected_agent
        except Exception as e:
            yield {"type": "error", "error": str(e)}
            return

        yield {"type": "agent_selected", "agent": selected_agent}

        # For traditional RAG we use the agent's custom streaming implementation.
        # For agentic RAG we must stream the LangGraph execution so tool calls can run.
        if selected_agent == "rag_agent" and not settings.agentic_rag_enabled:
            conversation_history = []
            if conversation_id and user_id:
                try:
                    memory_manager = get_memory_manager()
                    conv_memory = await memory_manager.get_memory(
                        UUID(conversation_id), UUID(user_id), force_refresh=True
                    )
                    # Get all messages from history without limit
                    conversation_history = conv_memory.get_recent_messages(
                        limit=None, exclude_last=1
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

        # Handle planning agent - now with streaming support
        if selected_agent == "planning_agent":
            # Convert existing_tasks to todos format for state
            todos = []
            if existing_tasks:
                for i, task in enumerate(existing_tasks):
                    todos.append(
                        {
                            "id": task.get("id", str(i)),
                            "description": task.get("description", ""),
                            "status": task.get("status", "pending"),
                            "order": task.get("task_order", i),
                        }
                    )

            # Set up initial state for graph execution
            initial_state["todos"] = todos
            # Find the first pending or in-progress task as current
            current_task_index = None
            for i, todo in enumerate(todos):
                status = todo.get("status", "pending")
                if status in ("pending", "in_progress"):
                    current_task_index = i
                    break
            initial_state["current_task_index"] = current_task_index
            initial_state["planning_call_count"] = 0
            # Don't return early - fall through to the common streaming logic below

        accumulated_content = ""
        accumulated_thinking = ""  # Track thinking content for non-RAG agents
        current_tool_calls = {}  # Track tool call chunks by index
        emitted_tool_call_ids = (
            set()
        )  # Track which tool calls have had tool_start emitted

        def _consume_text_chunk(text_chunk: str) -> Optional[str]:
            """
            Return the incremental delta to emit, updating accumulated_content in-place.
            """
            nonlocal accumulated_content

            if not text_chunk:
                return None

            chunk_text = coerce_response_text(text_chunk)
            if not chunk_text:
                return None

            if accumulated_content:
                # Exact repeat of what we've already accumulated.
                if chunk_text == accumulated_content:
                    return None

                # Cumulative chunk: new chunk starts with what we already have.
                if chunk_text.startswith(accumulated_content):
                    delta = chunk_text[len(accumulated_content) :]
                    accumulated_content = chunk_text
                    return delta or None

                # Duplicate tail chunk.
                if accumulated_content.endswith(chunk_text):
                    return None

            accumulated_content += chunk_text
            return chunk_text

        try:
            # - "messages": Stream LLM tokens with metadata (includes tool_call_chunks)
            # - "updates": Stream state updates after each node (includes completed messages)
            async for chunk in self.graph.astream(
                initial_state, config=config, stream_mode=["messages", "updates"]
            ):
                # Handle tuple format from multiple stream modes
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    mode, data = chunk

                    if mode == "messages":
                        # LLM token streaming - data is (message_chunk, metadata)
                        message_chunk, metadata = data

                        # Skip ToolMessage - tool results are handled in updates mode
                        if isinstance(message_chunk, ToolMessage):
                            continue

                        # Handle text content using content_blocks (latest pattern)
                        if (
                            hasattr(message_chunk, "content_blocks")
                            and message_chunk.content_blocks
                        ):
                            for block in message_chunk.content_blocks:
                                block_type = block.get("type")

                                if block_type == "text":
                                    text_content = block.get("text", "")
                                    delta = _consume_text_chunk(text_content)
                                    if delta:
                                        yield {"type": "token", "content": delta}

                                # Handle thinking block type
                                elif block_type == "thinking":
                                    thinking_content = block.get(
                                        "thinking", ""
                                    ) or block.get("text", "")
                                    if thinking_content:
                                        accumulated_thinking += thinking_content
                                        yield {
                                            "type": "thinking",
                                            "content": thinking_content,
                                        }

                                # Handle reasoning block type (LangChain Google GenAI)
                                elif block_type == "reasoning":
                                    reasoning_content = block.get(
                                        "reasoning", ""
                                    ) or block.get("text", "")
                                    if reasoning_content:
                                        accumulated_thinking += reasoning_content
                                        yield {
                                            "type": "thinking",
                                            "content": reasoning_content,
                                        }

                                elif block_type == "tool_call_chunk":
                                    # Stream tool call chunks as they arrive
                                    tool_index = block.get("index", 0)
                                    tool_id = block.get("id")
                                    tool_name = block.get("name")
                                    tool_args = block.get("args", "")

                                    # Initialize or update tool call tracking
                                    if tool_index not in current_tool_calls:
                                        current_tool_calls[tool_index] = {
                                            "id": tool_id,
                                            "name": tool_name,
                                            "args": "",
                                        }

                                    # Accumulate args
                                    if tool_args:
                                        current_tool_calls[tool_index][
                                            "args"
                                        ] += tool_args

                                    # Update name/id if present
                                    if (
                                        tool_name
                                        and not current_tool_calls[tool_index]["name"]
                                    ):
                                        current_tool_calls[tool_index][
                                            "name"
                                        ] = tool_name
                                    if (
                                        tool_id
                                        and not current_tool_calls[tool_index]["id"]
                                    ):
                                        current_tool_calls[tool_index]["id"] = tool_id

                        # Handle content as list (when include_thoughts=True)
                        # LangChain returns content as list with thinking/reasoning and text parts
                        elif hasattr(message_chunk, "content") and isinstance(
                            message_chunk.content, list
                        ):
                            for part in message_chunk.content:
                                if isinstance(part, dict):
                                    part_type = part.get("type", "")

                                    if part_type == "thinking":
                                        thinking_content = part.get(
                                            "thinking", ""
                                        ) or part.get("text", "")
                                        if thinking_content:
                                            accumulated_thinking += thinking_content
                                            yield {
                                                "type": "thinking",
                                                "content": thinking_content,
                                            }
                                    elif part_type == "reasoning":
                                        reasoning_content = part.get(
                                            "reasoning", ""
                                        ) or part.get("text", "")
                                        if reasoning_content:
                                            accumulated_thinking += reasoning_content
                                            yield {
                                                "type": "thinking",
                                                "content": reasoning_content,
                                            }
                                    elif part_type == "text":
                                        text_content = part.get("text", "")
                                        delta = _consume_text_chunk(text_content)
                                        if delta:
                                            yield {"type": "token", "content": delta}
                                elif isinstance(part, str) and part:
                                    delta = _consume_text_chunk(part)
                                    if delta:
                                        yield {"type": "token", "content": delta}

                        # Fallback: Handle legacy string content attribute
                        elif (
                            hasattr(message_chunk, "content")
                            and message_chunk.content
                            and isinstance(message_chunk.content, str)
                        ):
                            content = coerce_response_text(message_chunk.content)
                            delta = _consume_text_chunk(content)
                            if delta:
                                yield {"type": "token", "content": delta}

                        # Check for chunk completion and emit complete tool calls
                        if (
                            hasattr(message_chunk, "chunk_position")
                            and message_chunk.chunk_position == "last"
                        ):
                            # Emit accumulated tool calls
                            for tool_call in current_tool_calls.values():
                                if tool_call["name"]:  # Only emit if we have a name
                                    tool_call_id = tool_call["id"]
                                    # Skip if already emitted
                                    if (
                                        tool_call_id
                                        and tool_call_id in emitted_tool_call_ids
                                    ):
                                        continue
                                    if tool_call_id:
                                        emitted_tool_call_ids.add(tool_call_id)

                                    try:
                                        # Parse args if it's a JSON string
                                        args = (
                                            json.loads(tool_call["args"])
                                            if tool_call["args"]
                                            else {}
                                        )
                                    except:
                                        args = tool_call["args"]

                                    yield {
                                        "type": "tool_start",
                                        "name": tool_call["name"],
                                        "tool_call_id": tool_call_id,
                                        "args": make_json_safe(args),
                                    }
                            # Clear for next message
                            current_tool_calls = {}

                    elif mode == "updates":
                        # State updates - check for completed messages with thinking/tools
                        for node_name, node_state in data.items():
                            # Emit node completion event for planning nodes
                            if node_name in ("planning_agent", "planning_tools"):
                                # Extract relevant info from the node state
                                node_info = {"node": node_name}

                                # For planning_agent, include tool call info
                                if (
                                    node_name == "planning_agent"
                                    and "messages" in node_state
                                ):
                                    messages = node_state.get("messages", [])
                                    if messages:
                                        last_msg = (
                                            messages[-1]
                                            if isinstance(messages, list)
                                            else messages
                                        )
                                        if (
                                            isinstance(last_msg, AIMessage)
                                            and hasattr(last_msg, "tool_calls")
                                            and last_msg.tool_calls
                                        ):
                                            node_info["tool_calls"] = [
                                                {
                                                    "name": tc.get("name"),
                                                    "id": tc.get("id"),
                                                    "args": make_json_safe(
                                                        tc.get("args", {})
                                                    ),
                                                }
                                                for tc in last_msg.tool_calls
                                            ]

                                # For planning_tools, include execution results
                                if node_name == "planning_tools":
                                    todos = node_state.get("todos", [])
                                    if todos:
                                        node_info["todos_count"] = len(todos)
                                        node_info["current_task_index"] = (
                                            node_state.get("current_task_index")
                                        )

                                yield {"type": "node_complete", **node_info}

                            if "messages" in node_state:
                                messages = node_state["messages"]
                                if messages:
                                    last_msg = (
                                        messages[-1]
                                        if isinstance(messages, list)
                                        else messages
                                    )

                                    # Handle AIMessage - extract tool_calls only
                                    if isinstance(last_msg, AIMessage):
                                        # Handle tool calls
                                        if (
                                            hasattr(last_msg, "tool_calls")
                                            and last_msg.tool_calls
                                        ):
                                            for tool_call in last_msg.tool_calls:
                                                tool_call_id = tool_call.get("id")
                                                # Only emit if not already emitted from messages mode
                                                if (
                                                    tool_call_id
                                                    and tool_call_id
                                                    not in emitted_tool_call_ids
                                                ):
                                                    emitted_tool_call_ids.add(
                                                        tool_call_id
                                                    )
                                                    yield {
                                                        "type": "tool_start",
                                                        "name": tool_call.get(
                                                            "name", "unknown"
                                                        ),
                                                        "tool_call_id": tool_call_id,
                                                        "args": make_json_safe(
                                                            tool_call.get("args", {})
                                                        ),
                                                    }

                                    # Handle ToolMessage (result)
                                    elif isinstance(last_msg, ToolMessage):
                                        yield {
                                            "type": "tool_end",
                                            "name": getattr(
                                                last_msg, "name", "unknown"
                                            ),
                                            "tool_call_id": getattr(
                                                last_msg, "tool_call_id", None
                                            ),
                                            "result": make_json_safe(last_msg.content),
                                        }
                else:
                    # Single mode or legacy format - try to handle gracefully
                    logger.warning(f"Unexpected stream chunk format: {type(chunk)}")

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
                    if accumulated_thinking and not response.metadata.get(
                        "thinking_summary"
                    ):
                        response.metadata["thinking_summary"] = accumulated_thinking

                    if accumulated_content and not (response.message.content or "").strip():
                        response.message.content = accumulated_content

                    # For planning agent, include todos in metadata
                    if selected_agent == "planning_agent":
                        final_todos = snapshot.values.get("todos", [])
                        if final_todos:
                            response.metadata["todos"] = final_todos
                        response.metadata["planning_call_count"] = snapshot.values.get(
                            "planning_call_count", 0
                        )
                        # Check context for completion status
                        context = snapshot.values.get("context", {})
                        if context.get("all_tasks_completed"):
                            response.metadata["all_tasks_completed"] = True
                        if context.get("planning_budget_reached"):
                            response.metadata["planning_budget_reached"] = True

                    yield {"type": "complete", "response": response}
                elif accumulated_content:
                    metadata_model = (
                        settings.chat_agent_model
                        if selected_agent == "chat_agent"
                        else settings.search_agent_model
                    )
                    metadata = {"model": metadata_model}

                    # Include thinking summary in metadata
                    if accumulated_thinking:
                        metadata["thinking_summary"] = accumulated_thinking

                    # For planning agent, include todos in metadata
                    if selected_agent == "planning_agent":
                        final_todos = snapshot.values.get("todos", [])
                        if final_todos:
                            metadata["todos"] = final_todos
                        metadata["planning_call_count"] = snapshot.values.get(
                            "planning_call_count", 0
                        )

                    agent_type = self._get_agent_type(selected_agent)
                    response = AgentResponse(
                        agent_type=agent_type,
                        agent_id=selected_agent or "unknown",
                        message=AgentMessage(
                            role=MessageRole.ASSISTANT, content=accumulated_content
                        ),
                        metadata=metadata,
                    )
                    yield {"type": "complete", "response": response}
                else:
                    yield {"type": "error", "error": NO_RESPONSE_GENERATED}
            except Exception as e:
                yield {"type": "error", "error": str(e)}
        else:
            if accumulated_content:
                metadata = {}
                # Include thinking summary in metadata
                if accumulated_thinking:
                    metadata["thinking_summary"] = accumulated_thinking

                agent_type = self._get_agent_type(selected_agent)
                response = AgentResponse(
                    agent_type=agent_type,
                    agent_id=selected_agent or "unknown",
                    message=AgentMessage(
                        role=MessageRole.ASSISTANT, content=accumulated_content
                    ),
                    metadata=metadata,
                )
                yield {"type": "complete", "response": response}
            else:
                yield {"type": "error", "error": NO_RESPONSE_GENERATED}

    async def get_state(self, thread_id: str) -> dict:
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot get state.")

        config = self._build_graph_config(thread_id)
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
