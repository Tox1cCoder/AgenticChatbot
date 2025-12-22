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
)
from .agents.router import Router
from .agents.chat_agent import ChatAgent
from .agents.rag_agent import RAGAgent
from .agents.search_agent import SearchAgent
from .agents.image_generator_agent import ImageGeneratorAgent
from .agents.planning_agent import PlanningAgent
from .memory import get_memory_manager
from ..core.config import settings
from .hitl_config import build_interrupt_response, requires_human_approval
from .utils import normalize_tool_call, coerce_response_text, make_json_safe

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

        self.graph = self._build_graph()
        self._cleanup_agents = [
            self.chat_agent,
            self.search_agent,
            self.rag_agent,
            self.image_generator_agent,
        ]
        self._initialized = False

    def _get_current_turn_messages(self, messages: List) -> List:
        """
        Extract only messages from the current turn (from last HumanMessage onwards).
        """
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

    async def initialize(self) -> None:
        """
        Eagerly initialize all agent tools.
        """
        if self._initialized:
            return

        # Pre-initialize MCP tools for all agents that use them
        for agent in [
            self.chat_agent,
            self.search_agent,
            self.image_generator_agent,
            self.rag_agent,
        ]:
            if hasattr(agent, "_init_tools"):
                await agent._init_tools()
            elif hasattr(agent, "_init_mcp"):
                await agent._init_mcp()

        self._initialized = True

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        workflow.add_node("route", self._route_node)
        workflow.add_node("chat_agent", self._chat_node)
        workflow.add_node("rag_agent", self._rag_node)
        workflow.add_node("search_agent", self._search_node)
        workflow.add_node("image_generator_agent", self._image_generator_node)
        workflow.add_node("planning_agent", self._planning_node)
        workflow.add_node("planning_tools", self._planning_tools_node)
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
                "planning_agent": "planning_agent",
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
        tool_artifacts = []  # Track tool execution artifacts
        all_images = []  # Track images from tool results

        import json  # Import for parsing JSON results

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

                    # Extract images from tool result (e.g., Tavily search results)
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
                        pass  # Not JSON or no images

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

        # Increment iteration count to track tool-calling loops
        current_iteration = state.get("iteration_count") or 0
        state["iteration_count"] = current_iteration + 1

        # Store tool artifacts and images in context for later extraction
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
        # Check iteration count to prevent infinite loops
        iteration_count = state.get("iteration_count", 0)
        max_iterations = getattr(settings, "react_agent_max_iterations")

        # If we've exceeded max iterations, end the conversation
        if iteration_count >= max_iterations:
            # Add a message indicating we've reached the limit
            messages = state.get("messages", [])
            if messages:
                context = state.get("context", {})
                context["max_iterations_reached"] = True
                state["context"] = context
            return "end"

        return state.get("selected_agent", "end")

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
        agent_type_map = {
            "chat_agent": AgentType.CHAT,
            "rag_agent": AgentType.RAG,
            "search_agent": AgentType.SEARCH,
            "image_generator_agent": AgentType.IMAGE_GENERATOR,
            "planning_agent": AgentType.PLANNING,
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

        context = state.get("context", {})

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
        conversation_id = state.get("conversation_id")
        has_documents = self._conversation_has_documents(conversation_id)

        # Extract planning context from state
        planning_mode_enabled = context.get("planning_mode_enabled", False)
        has_existing_plan = context.get("has_existing_plan", False)

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"persona": state.get("persona")},
        )

        # Include planning_agent in available agents
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

    def _build_plan_context_string(
        self,
        current_task: Optional[Dict[str, Any]],
        all_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        if not current_task or not current_task.get("description"):
            return None

        parts = ["=== TASK EXECUTION CONTEXT ==="]

        if all_tasks:
            completed = [t for t in all_tasks if t.get("status") == "completed"]
            pending = [t for t in all_tasks if t.get("status") == "pending"]
            in_progress = [t for t in all_tasks if t.get("status") == "in_progress"]

            parts.append(
                f"\nPlan Progress: {len(completed)}/{len(all_tasks)} tasks completed"
            )

            if completed:
                parts.append("\n✅ COMPLETED TASKS:")
                for t in completed:
                    order = t.get("task_order", t.get("order", 0))
                    parts.append(
                        f"  {order + 1}. {t.get('description', 'No description')}"
                    )

            current_order = current_task.get("order", current_task.get("task_order", 0))
            parts.append(f"\n🔄 CURRENT TASK (#{current_order + 1}):")
            parts.append(f"  {current_task.get('description', '')}")
            parts.append(
                "\n  ** You are executing THIS task now. Focus on completing it. **"
            )

            remaining = [
                t
                for t in pending
                if t.get("task_order", t.get("order", 0)) > current_order
            ]
            if remaining:
                parts.append("\n⏳ REMAINING TASKS:")
                for t in remaining:
                    order = t.get("task_order", t.get("order", 0))
                    parts.append(
                        f"  {order + 1}. {t.get('description', 'No description')}"
                    )

        else:
            order = current_task.get("order", current_task.get("task_order", 0))
            parts.append(f"\n🔄 CURRENT TASK (#{order + 1}):")
            parts.append(f"  {current_task.get('description', '')}")
            parts.append("\n  ** Focus on completing this specific task. **")

        parts.append("\n=== END TASK CONTEXT ===\n")
        return "\n".join(parts)

    async def _chat_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        current_task = state.get("current_task")
        all_tasks = state.get("all_tasks")
        plan_context_str = self._build_plan_context_string(current_task, all_tasks)

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
            # Get all messages from history without limit
            conversation_history = conv_memory.get_recent_messages(
                limit=None, exclude_last=1
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
            # Only pass current turn messages to prevent old tool calls from being repeated
            current_turn_messages = self._get_current_turn_messages(messages)
            response = await self.chat_agent.invoke_model_with_history(
                current_turn_messages,
                conversation_history,
                state.get("persona"),
                conversation_id,
            )
        else:
            user_content = (
                messages[last_human_idx].content
                if last_human_idx is not None
                and hasattr(messages[last_human_idx], "content")
                else content
            )

            enriched_content = user_content
            if plan_context_str:
                enriched_content = f"{plan_context_str}\n\n{user_content}"

            metadata = {
                "history": conversation_history,
                "persona": state.get("persona"),
            }
            if current_task:
                metadata["task_context"] = current_task

            agent_msg = AgentMessage(
                role=MessageRole.USER,
                content=enriched_content,
                metadata=metadata,
                attachments=context.get("attachments"),
            )

            response = await self.chat_agent.invoke_model(agent_msg, conversation_id)

        # Merge tool artifacts and images from context into response
        context = state.get("context", {})
        tool_artifacts = context.get("tool_artifacts", [])
        tool_images = context.get("tool_images", [])

        if tool_artifacts:
            response.tool_artifacts = tool_artifacts
        if tool_images:
            if not response.metadata:
                response.metadata = {}
            response.metadata["images"] = tool_images

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

        current_task = state.get("current_task")
        all_tasks = state.get("all_tasks")
        plan_context_str = self._build_plan_context_string(current_task, all_tasks)

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
            # Get all messages from history without limit
            conversation_history = conv_memory.get_recent_messages(
                limit=None, exclude_last=1
            )

        context = state.get("context", {})

        enriched_content = content
        if plan_context_str:
            enriched_content = f"{plan_context_str}\n\n{content}"

        metadata = {"history": conversation_history, "persona": state.get("persona")}
        if current_task:
            metadata["task_context"] = current_task

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=enriched_content,
            metadata=metadata,
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

        current_task = state.get("current_task")
        all_tasks = state.get("all_tasks")
        plan_context_str = self._build_plan_context_string(current_task, all_tasks)

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
            # Get all messages from history without limit
            conversation_history = conv_memory.get_recent_messages(
                limit=None, exclude_last=1
            )

        if has_tool_context:
            # Only pass current turn messages to prevent old tool calls from being repeated
            current_turn_messages = self._get_current_turn_messages(messages)
            response = await self.search_agent.invoke_model_with_history(
                current_turn_messages,
                conversation_history,
                state.get("persona"),
                conversation_id,
            )
        else:
            user_content = (
                messages[last_human_idx].content
                if last_human_idx is not None
                and hasattr(messages[last_human_idx], "content")
                else content
            )

            enriched_content = user_content
            if plan_context_str:
                enriched_content = f"{plan_context_str}\n\n{user_content}"

            context = state.get("context", {})
            metadata = {
                "history": conversation_history,
                "persona": state.get("persona"),
            }
            if current_task:
                metadata["task_context"] = current_task

            agent_msg = AgentMessage(
                role=MessageRole.USER,
                content=enriched_content,
                metadata=metadata,
                attachments=context.get("attachments"),
            )

            response = await self.search_agent.process_message(
                agent_msg, conversation_id
            )

        # Merge tool artifacts and images from context into response
        context = state.get("context", {})
        tool_artifacts = context.get("tool_artifacts", [])
        tool_images = context.get("tool_images", [])

        if tool_artifacts:
            response.tool_artifacts = tool_artifacts
        if tool_images:
            if not response.metadata:
                response.metadata = {}
            response.metadata["images"] = tool_images

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

        current_task = state.get("current_task")
        all_tasks = state.get("all_tasks")
        plan_context_str = self._build_plan_context_string(current_task, all_tasks)

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
            # Get all messages from history without limit
            conversation_history = conv_memory.get_recent_messages(
                limit=None, exclude_last=1
            )

        if has_tool_context:
            # Only pass current turn messages to prevent old tool calls from being repeated
            current_turn_messages = self._get_current_turn_messages(messages)
            response = await self.image_generator_agent.invoke_model_with_history(
                current_turn_messages,
                conversation_history,
                state.get("persona"),
                conversation_id,
            )
        else:
            full_history = conversation_history + messages[:-1]
            context = state.get("context", {})

            enriched_content = content
            if plan_context_str:
                enriched_content = f"{plan_context_str}\n\n{content}"

            metadata = {"history": full_history, "persona": state.get("persona")}
            if current_task:
                metadata["task_context"] = current_task

            agent_msg = AgentMessage(
                role=MessageRole.USER,
                content=enriched_content,
                metadata=metadata,
                attachments=context.get("attachments"),
            )

            response = await self.image_generator_agent.invoke_model(
                agent_msg, conversation_id
            )

        # Merge tool artifacts and images from context into response
        context = state.get("context", {})
        tool_artifacts = context.get("tool_artifacts", [])
        tool_images = context.get("tool_images", [])

        if tool_artifacts:
            response.tool_artifacts = tool_artifacts
        if tool_images:
            if not response.metadata:
                response.metadata = {}
            # Append to existing images from image generator
            existing_images = response.metadata.get("images", [])
            existing_images.extend(tool_images)
            response.metadata["images"] = existing_images

        state["response"] = response

        ai_kwargs = {"content": response.message.content}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        return state

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
        )

        state["response"] = response
        
        # Add AI message to state (with tool calls if present)
        ai_kwargs = {"content": response.message.content or ""}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        # Sync todos from response metadata if plan was generated/modified
        if response.metadata.get("todos"):
            state["todos"] = response.metadata["todos"]

        return state

    async def _planning_tools_node(self, state: GraphState) -> GraphState:
        """
        Execute write_todos tool calls from the planning agent.
        
        Handles the actual todo state updates based on tool call arguments.
        """
        from .schemas import TodoAction, TodoStatus
        
        messages = state.get("messages", [])
        if not messages:
            return state
            
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state
        
        todos = list(state.get("todos", []))  # Make a copy
        current_task_index = state.get("current_task_index")
        tool_outputs = []
        
        for tool_call in last_message.tool_calls:
            tool_call_data = normalize_tool_call(tool_call)
            tool_name = tool_call_data.get("name")
            tool_id = tool_call_data.get("id")
            tool_args = tool_call_data.get("args", {})
            
            if tool_name != "write_todos":
                # Execute MCP tool
                try:
                    # Find the tool in the planning agent's tools
                    tool = None
                    if self.planning_agent and hasattr(self.planning_agent, 'tools'):
                        for t in self.planning_agent.tools:
                            if t.name == tool_name:
                                tool = t
                                break
                    
                    if tool:
                        # Execute the tool
                        result = await tool.ainvoke(tool_args)
                        tool_outputs.append({
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": str(result) if result else "Tool executed successfully",
                        })
                    else:
                        tool_outputs.append({
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": f"Tool not found: {tool_name}",
                        })
                except Exception as e:
                    logger.error(f"Error executing MCP tool {tool_name}: {e}")
                    tool_outputs.append({
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": f"Error executing tool: {str(e)}",
                    })
                continue
            
            action = tool_args.get("action")
            result = ""
            
            try:
                if action == TodoAction.SET_TODOS.value or action == "set_todos":
                    # Replace all todos
                    new_todos = tool_args.get("todos", [])
                    todos = new_todos
                    current_task_index = 0 if todos else None
                    result = f"Set {len(todos)} todos in the plan."
                
                elif action == TodoAction.ADD_TODO.value or action == "add_todo":
                    # Add a single todo
                    new_todo = tool_args.get("todo", {})
                    if new_todo:
                        new_todo["order"] = len(todos)
                        todos.append(new_todo)
                        result = f"Added todo: {new_todo.get('description', 'unknown')}"
                    else:
                        result = "Error: No todo provided for ADD_TODO"
                
                elif action == TodoAction.COMPLETE_TODO.value or action == "complete_todo":
                    # Mark todo as completed
                    todo_id = tool_args.get("todo_id")
                    for i, todo in enumerate(todos):
                        if todo.get("id") == todo_id:
                            todo["status"] = TodoStatus.COMPLETED.value
                            result = f"Completed todo: {todo.get('description', todo_id)}"
                            # Move to next task
                            if current_task_index is not None and i == current_task_index:
                                current_task_index = self._find_next_ready_task(todos, i)
                            break
                    else:
                        result = f"Todo with id {todo_id} not found"
                
                elif action == TodoAction.START_TODO.value or action == "start_todo":
                    # Mark todo as in progress
                    todo_id = tool_args.get("todo_id")
                    for i, todo in enumerate(todos):
                        if todo.get("id") == todo_id:
                            todo["status"] = TodoStatus.IN_PROGRESS.value
                            current_task_index = i
                            result = f"Started todo: {todo.get('description', todo_id)}"
                            break
                    else:
                        result = f"Todo with id {todo_id} not found"
                
                elif action == TodoAction.UPDATE_TODO.value or action == "update_todo":
                    # Update a todo
                    updated_todo = tool_args.get("todo", {})
                    todo_id = updated_todo.get("id")
                    for i, todo in enumerate(todos):
                        if todo.get("id") == todo_id:
                            todos[i] = {**todo, **updated_todo}
                            result = f"Updated todo: {todo_id}"
                            break
                    else:
                        result = f"Todo with id {todo_id} not found"
                
                elif action == TodoAction.REMOVE_TODO.value or action == "remove_todo":
                    # Remove a todo
                    todo_id = tool_args.get("todo_id")
                    for i, todo in enumerate(todos):
                        if todo.get("id") == todo_id:
                            todos.pop(i)
                            result = f"Removed todo: {todo_id}"
                            # Adjust current task index if needed
                            if current_task_index is not None:
                                if i < current_task_index:
                                    current_task_index -= 1
                                elif i == current_task_index:
                                    current_task_index = self._find_next_ready_task(todos, max(0, i - 1))
                            break
                    else:
                        result = f"Todo with id {todo_id} not found"
                
                else:
                    result = f"Unknown action: {action}"
                    
            except Exception as e:
                result = f"Error executing {action}: {str(e)}"
            
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
        
        # Update state with new todos
        state["todos"] = todos
        state["current_task_index"] = current_task_index
        
        # Increment iteration count for budget tracking
        current_iteration = state.get("iteration_count") or 0
        state["iteration_count"] = current_iteration + 1
        
        return state

    def _find_next_ready_task(
        self, todos: list, start_index: int = 0
    ) -> Optional[int]:
        """Find the next task that is ready to execute (pending with all deps completed)."""
        from .schemas import TodoStatus
        
        completed_ids = {
            t.get("id") for t in todos 
            if t.get("status") == TodoStatus.COMPLETED.value
        }
        
        for i in range(start_index, len(todos)):
            todo = todos[i]
            if todo.get("status") == TodoStatus.PENDING.value:
                deps = todo.get("dependencies", [])
                if all(dep in completed_ids for dep in deps):
                    return i
        return None

    def _should_call_planning_tools(self, state: GraphState) -> str:
        """Determine if planning agent made tool calls that need execution."""
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"
        
        # Route to planning_tools for ANY tool call (write_todos or MCP tools)
        if last_message.tool_calls:
            return "planning_tools"
        
        # No tool calls - end
        return "end"

    def _should_continue_planning(self, state: GraphState) -> str:
        """Determine if planning agent should continue after tool execution.
        
        Continues until:
        - All tasks are completed, OR
        - Max iterations reached (respects LLM rate limits), OR
        - Agent decides to stop (no more tool calls)
        """
        from .schemas import TodoStatus
        
        # Check iteration/call budget (higher limit for agentic behavior)
        planning_call_count = state.get("planning_call_count", 0)
        max_iterations = getattr(settings, "planning_max_iterations", 15)
        
        # Check if we've exceeded the budget
        if planning_call_count >= max_iterations:
            context = state.get("context", {})
            context["planning_budget_reached"] = True
            context["pause_reason"] = "max_iterations_reached"
            state["context"] = context
            return "end"
        
        # Check if all tasks are completed
        todos = state.get("todos", [])
        if todos:
            pending_count = sum(1 for t in todos if t.get("status") in (
                TodoStatus.PENDING.value, "pending",
                TodoStatus.IN_PROGRESS.value, "in_progress"
            ))
            if pending_count == 0:
                # All tasks completed - stop naturally
                context = state.get("context", {})
                context["all_tasks_completed"] = True
                state["context"] = context
                return "end"
        
        # Continue to planning agent for more processing
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

        # Task planning context
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
        interrupt_id: Optional[str] = None,
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

        config = self._build_graph_config(thread_id)

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

        # Handle planning agent - uses ainvoke (not streaming) internally
        if selected_agent == "planning_agent":
            # Convert existing_tasks to todos format for state
            todos = []
            if existing_tasks:
                for i, task in enumerate(existing_tasks):
                    todos.append({
                        "id": task.get("id", str(i)),
                        "description": task.get("description", ""),
                        "status": task.get("status", "pending"),
                        "order": task.get("task_order", i),
                        "dependencies": task.get("dependencies", []),
                        "complexity": task.get("estimated_complexity"),
                    })

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

            try:
                # Use ainvoke for planning agent since it doesn't stream internally
                result = await self.graph.ainvoke(initial_state, config=config)
                
                response = result.get("response")
                final_todos = result.get("todos", [])
                
                # Ensure we have a valid response with todos in metadata
                if response:
                    if response.metadata is None:
                        response.metadata = {}
                    if final_todos:
                        response.metadata["todos"] = final_todos
                    response.metadata["planning_call_count"] = result.get("planning_call_count", 0)
                    
                    # Check context for completion status
                    context = result.get("context", {})
                    if context.get("all_tasks_completed"):
                        response.metadata["all_tasks_completed"] = True
                    if context.get("planning_budget_reached"):
                        response.metadata["planning_budget_reached"] = True
                    
                    # If response content is empty, try to extract from last AI message
                    if not response.message.content:
                        messages = result.get("messages", [])
                        for msg in reversed(messages):
                            if isinstance(msg, AIMessage) and msg.content:
                                response.message.content = msg.content
                                break
                
                # Yield the content as tokens for UI compatibility
                if response and response.message and response.message.content:
                    yield {"type": "token", "content": response.message.content}
                
                if response:
                    yield {"type": "complete", "response": response}
                else:
                    yield {"type": "error", "error": "No response generated"}
            except Exception as e:
                yield {"type": "error", "error": str(e)}
            return


        accumulated_content = ""
        accumulated_thinking = ""  # Track thinking content for non-RAG agents
        try:
            async for event in self.graph.astream_events(
                initial_state, config=config, version="v1"
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        content = coerce_response_text(chunk.content)

                        if content:
                            additional_kwargs = getattr(chunk, "additional_kwargs", {})
                            if additional_kwargs.get(
                                "thought"
                            ) or additional_kwargs.get("thinking"):
                                accumulated_thinking += content
                                yield {"type": "thinking", "content": content}
                            else:
                                accumulated_content += content
                                yield {"type": "token", "content": content}

                elif kind == "on_tool_start":
                    data = event.get("data") or {}
                    tool_call_id = event.get("run_id") or event.get("id")
                    tool_input = data.get("input") if isinstance(data, dict) else None
                    yield {
                        "type": "tool_start",
                        "name": event.get("name", "unknown"),
                        "tool_call_id": str(tool_call_id) if tool_call_id else None,
                        "args": make_json_safe(tool_input),
                    }
                elif kind == "on_tool_end":
                    data = event.get("data") or {}
                    tool_call_id = event.get("run_id") or event.get("id")
                    tool_output = data.get("output") if isinstance(data, dict) else None
                    yield {
                        "type": "tool_end",
                        "name": event.get("name", "unknown"),
                        "tool_call_id": str(tool_call_id) if tool_call_id else None,
                        "result": make_json_safe(tool_output),
                    }

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
                    yield {"type": "complete", "response": response}
                elif accumulated_content:
                    metadata = {
                        "model": (
                            settings.chat_agent_model
                            if selected_agent == "chat_agent"
                            else settings.search_agent_model
                        )
                    }
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
                    yield {"type": "error", "error": "No response generated"}
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
                yield {"type": "error", "error": "No response generated"}

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
