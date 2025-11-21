import asyncio
import logging
from typing import Optional, TYPE_CHECKING, List, Dict, Any
from uuid import UUID

from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .schemas import (
    GraphState,
    AgentMessage,
    AgentResponse,
    MessageRole,
    InterruptDecision,
    InterruptDecisionType,
)
from typing import Any
from .agents.router import Router
from .agents.chat_agent import ChatAgent
from .agents.rag_agent import RAGAgent
from .agents.search_agent import SearchAgent
from .agents.image_generator_agent import ImageGeneratorAgent
from .memory import get_memory_manager
from ..core.config import settings

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
        logger.info(
            f"Multi-agent workflow initialized (checkpointing: {'enabled' if checkpointer else 'disabled'})"
        )

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

        # Add generic tool execution node
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

        workflow.add_edge("chat_agent", END)
        workflow.add_edge("rag_agent", END)

        # Search and Image Generator agents can return tool calls
        workflow.add_conditional_edges(
            "search_agent",
            self._should_call_tools,
            {
                "tools": "tools",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "image_generator_agent",
            self._should_call_tools,
            {
                "tools": "tools",
                "end": END,
            },
        )

        # After tools, go back to the agent that called them
        workflow.add_conditional_edges(
            "tools",
            self._route_tool_output,
            {
                "search_agent": "search_agent",
                "image_generator_agent": "image_generator_agent",
                "end": END,  # Fallback
            },
        )

        # Compile with checkpointer and interrupts
        # We interrupt BEFORE the "tools" node to allow human approval
        if self.checkpointer:
            return workflow.compile(
                checkpointer=self.checkpointer, interrupt_before=["tools"]
            )
        else:
            return workflow.compile(interrupt_before=["tools"])

    async def _tool_node(self, state: GraphState) -> GraphState:
        """
        Execute pending tool calls.
        """
        messages = state.get("messages", [])
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            logger.warning("Tool node called but no tool calls found in last message")
            return state

        # We need to execute the tools.
        # Since we don't have a global tool registry easily accessible here (agents have their own tools),
        # we need to find the right tools.
        # The state has "selected_agent".
        selected_agent_name = state.get("selected_agent")
        agent = self.agents.get(selected_agent_name)

        if not agent:
            logger.error(f"Agent {selected_agent_name} not found")
            return state

        # Ensure agent tools are initialized
        if not hasattr(agent, "tools") or not agent.tools:
            logger.info(f"Initializing tools for agent {selected_agent_name}")
            # Initialize tools based on agent type
            if hasattr(agent, "_init_mcp"):
                await agent._init_mcp()
            elif hasattr(agent, "_init_tools"):
                await agent._init_tools()
            else:
                logger.error(
                    f"Agent {selected_agent_name} does not have tool initialization method"
                )
                return state

        if not hasattr(agent, "tools") or not agent.tools:
            logger.error(
                f"Agent {selected_agent_name} has no tools available after initialization"
            )
            return state

        # Create a map of tool name to tool instance
        tool_map = {t.name: t for t in agent.tools}

        logger.info(f"Available tools: {list(tool_map.keys())}")

        tool_outputs = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool = tool_map.get(tool_name)
            if tool:
                try:
                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    # Execute tool
                    # Note: Some tools might be async
                    if tool.coroutine:
                        result = await tool.ainvoke(tool_args)
                    else:
                        result = tool.invoke(tool_args)

                    logger.info(f"Tool {tool_name} result: {str(result)[:200]}")
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": str(result),
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Error executing tool {tool_name}: {e}", exc_info=True
                    )
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Error: {str(e)}",
                        }
                    )
            else:
                logger.error(
                    f"Tool {tool_name} not found in available tools: {list(tool_map.keys())}"
                )
                tool_outputs.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": f"Error: Tool {tool_name} not found",
                    }
                )

        # Append tool outputs to messages
        # In LangChain, tool outputs are ToolMessage
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

    def _should_call_tools(self, state: GraphState) -> str:
        """Check if the last message has tool calls."""
        messages = state.get("messages", [])
        if not messages:
            logger.info("_should_call_tools: No messages in state")
            return "end"

        last_message = messages[-1]
        has_tool_calls = isinstance(last_message, AIMessage) and last_message.tool_calls

        logger.info(
            f"_should_call_tools: Last message type: {type(last_message)}, has tool_calls: {has_tool_calls}"
        )
        if has_tool_calls:
            logger.info(
                f"_should_call_tools: Tool calls found: {last_message.tool_calls}"
            )
            return "tools"

        return "end"

    def _route_tool_output(self, state: GraphState) -> str:
        """Route back to the selected agent after tool execution."""
        return state.get("selected_agent", "end")

    async def _route_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            logger.error("No messages to route")
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        # Check if conversation has documents available
        conversation_id = state.get("conversation_id")
        has_documents = self._conversation_has_documents(conversation_id)

        persona = state.get("persona")
        agent_msg = AgentMessage(
            role=MessageRole.USER, content=content, metadata={"persona": persona}
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
            conversation_uuid = UUID(conversation_id)
        except ValueError:
            logger.warning(
                "Invalid conversation_id '%s' encountered while checking documents",
                conversation_id,
            )
            return False

        try:
            return self.document_repository.count_by_conversation(conversation_uuid) > 0
        except Exception as exc:
            logger.error(
                "Failed to determine document availability for conversation %s: %s",
                conversation_id,
                exc,
            )
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
        memory_manager = get_memory_manager()

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            conv_id_uuid = UUID(conversation_id)
            user_id_uuid = UUID(user_id)

            conv_memory = await memory_manager.get_memory(
                conv_id_uuid, user_id_uuid, force_refresh=True
            )
            history_limit = (
                settings.chat_history_max_messages
                if settings.chat_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        persona = state.get("persona")
        context = state.get("context", {})
        attachments = context.get("attachments")

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history, "persona": persona},
            attachments=attachments,
        )

        response = await self.chat_agent.process_message(agent_msg, conversation_id)

        state["response"] = response
        state.setdefault("messages", []).append(
            AIMessage(content=response.message.content)
        )

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
        memory_manager = get_memory_manager()

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            conv_id_uuid = UUID(conversation_id)
            user_id_uuid = UUID(user_id)

            conv_memory = await memory_manager.get_memory(
                conv_id_uuid, user_id_uuid, force_refresh=True
            )
            history_limit = (
                settings.rag_history_max_messages
                if settings.rag_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        persona = state.get("persona")
        context = state.get("context", {})
        attachments = context.get("attachments")

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history, "persona": persona},
            attachments=attachments,
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
            logger.error("No messages in state for search agent")
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        # Keep the original user request even after tool calls
        last_human_idx = next(
            (
                idx
                for idx in range(len(messages) - 1, -1, -1)
                if isinstance(messages[idx], HumanMessage)
            ),
            None,
        )

        user_content = (
            messages[last_human_idx].content
            if last_human_idx is not None
            and hasattr(messages[last_human_idx], "content")
            else content
        )

        # Surface tool outputs to the LLM when resuming after tool execution
        tool_messages = (
            [m for m in messages[last_human_idx + 1 :] if isinstance(m, ToolMessage)]
            if last_human_idx is not None
            else [m for m in messages if isinstance(m, ToolMessage)]
        )

        # If tool messages exist, this is the second invocation after tool execution
        # We need to generate the final response using tool results
        has_tool_results = len(tool_messages) > 0

        if has_tool_results:
            tool_summaries = "\n".join(
                f"{getattr(tool_msg, 'name', 'tool')}: {tool_msg.content}"
                for tool_msg in tool_messages
            )
            user_content = (
                f"{user_content}\n\nTool results:\n{tool_summaries}\n"
                "Based on these tool results, provide a final comprehensive answer. Do NOT request additional tool calls."
            )

        conversation_history = []
        memory_manager = get_memory_manager()

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            conv_id_uuid = UUID(conversation_id)
            user_id_uuid = UUID(user_id)

            conv_memory = await memory_manager.get_memory(
                conv_id_uuid, user_id_uuid, force_refresh=True
            )
            history_limit = (
                settings.search_history_max_messages
                if settings.search_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        persona = state.get("persona")
        context = state.get("context", {})
        attachments = context.get("attachments")

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=user_content,
            metadata={"history": conversation_history, "persona": persona},
            attachments=attachments,
        )

        response = await self.search_agent.process_message(agent_msg, conversation_id)

        logger.info(
            f"Search agent response - has tool_calls: {response.message.tool_calls is not None}"
        )
        if response.message.tool_calls:
            logger.info(f"Search agent tool calls: {response.message.tool_calls}")

        state["response"] = response
        ai_message_kwargs = {"content": response.message.content}
        if response.message.tool_calls:
            ai_message_kwargs["tool_calls"] = response.message.tool_calls
            logger.info(
                f"Adding tool_calls to AIMessage: {response.message.tool_calls}"
            )

        state.setdefault("messages", []).append(AIMessage(**ai_message_kwargs))

        return state

    async def _image_generator_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            logger.error("No messages in state for image generator agent")
            return state

        last_human_message = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        content = last_human_message.content if last_human_message else ""

        # Check for tool results (second invocation after tool execution)
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_messages:
            tool_summaries = "\n".join(
                f"{getattr(tool_msg, 'name', 'tool')}: {tool_msg.content}"
                for tool_msg in tool_messages
            )
            content = (
                f"{content}\n\nTool results:\n{tool_summaries}\n"
                "Use these results to enhance the image generation prompt. Generate the final image now."
            )

        conversation_history = []
        memory_manager = get_memory_manager()

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            conv_id_uuid = UUID(conversation_id)
            user_id_uuid = UUID(user_id)

            conv_memory = await memory_manager.get_memory(
                conv_id_uuid, user_id_uuid, force_refresh=True
            )
            history_limit = (
                settings.chat_history_max_messages
                if settings.chat_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        # Combine history
        current_turn_messages = messages
        full_history = conversation_history + current_turn_messages[:-1]

        persona = state.get("persona")
        context = state.get("context", {})
        attachments = context.get("attachments")

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": full_history, "persona": persona},
            attachments=attachments,
        )

        response = await self.image_generator_agent.invoke_model(
            agent_msg, conversation_id
        )

        state["response"] = response

        if response.message.tool_calls:
            ai_msg = AIMessage(
                content=response.message.content, tool_calls=response.message.tool_calls
            )
            state.setdefault("messages", []).append(ai_msg)
        else:
            state.setdefault("messages", []).append(
                AIMessage(content=response.message.content)
            )

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

        config = None
        if self.checkpointer and thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        result = await self.graph.ainvoke(initial_state, config=config)

        agent_response = result.get("response")
        if agent_response and isinstance(agent_response.metadata, dict):
            if "interrupt" in agent_response.metadata:
                logger.info("MultiAgentWorkflow detected interrupt in agent response")
                return agent_response

        return agent_response

    async def resume_execution(
        self,
        thread_id: str,
        resume_value: Any,
    ) -> Optional[AgentResponse]:
        """
        Resume execution after handling interrupts.
        """
        if not self.checkpointer:
            raise RuntimeError("Checkpointing must be enabled for resume_execution")

        config = {"configurable": {"thread_id": thread_id}}
        resume_payload = resume_value

        command = Command(resume=resume_payload)
        result = await self.graph.ainvoke(command, config=config)

        return result.get("response")

    async def resume(
        self,
        thread_id: str,
        user_input: Optional[str] = None,  # Approval or rejection or modification
        rejection_messages: Optional[
            List
        ] = None,  # Tool rejection messages to add to state
    ) -> Optional[AgentResponse]:
        """
        Resume the workflow from an interrupt.

        Args:
            thread_id: The thread ID to resume
            user_input: Optional user input (not currently used)
            rejection_messages: Optional list of ToolMessages indicating tool rejection
        """
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot resume.")

        config = {"configurable": {"thread_id": thread_id}}

        logger.info(f"Resuming workflow for thread_id={thread_id}")

        # Check current state before resume
        state_snapshot = await self.graph.aget_state(config)
        logger.info(
            f"Current state - next nodes: {state_snapshot.next}, interrupted: {len(state_snapshot.next) > 0 if state_snapshot.next else False}"
        )

        # Get the selected agent from state
        selected_agent = state_snapshot.values.get("selected_agent")

        # If rejection messages provided, add them to state and route back to agent (skip tools)
        if rejection_messages:
            logger.info(
                f"Rejection: Adding {len(rejection_messages)} rejection messages to state and routing to {selected_agent}"
            )
            # Update the state to add rejection messages
            await self.graph.aupdate_state(
                config=config,
                values={"messages": rejection_messages},
                as_node="tools",  # Pretend the tools node added these messages
            )

            # Now resume execution - it will route to the selected agent
            logger.info(f"Resuming to route back to {selected_agent}")
            result = await self.graph.ainvoke(None, config=config)
        else:
            # Approval: Resume execution normally (will execute tools node)
            logger.info("Approval: Resuming workflow to execute tools")
            result = await self.graph.ainvoke(None, config=config)

        # Check if response was generated
        response = result.get("response")
        if not response:
            logger.warning("No response after resume, checking final state...")
            final_state = await self.graph.aget_state(config)
            logger.info(f"Final state - next nodes: {final_state.next}")
            response = final_state.values.get("response")

        return response

    async def resume_with_decisions(
        self,
        thread_id: str,
        decisions: List[InterruptDecision],
        interrupt_id: Optional[str] = None,
    ) -> Optional[AgentResponse]:
        """
        Resume workflow execution with user decisions on tool execution.
        
        This method properly handles accept/edit/reject decisions:
        - ACCEPT/APPROVE: Allow tool to execute with original args
        - EDIT: Modify tool arguments before execution  
        - REJECT/RESPOND: Skip tool execution and provide feedback to the agent
        
        Args:
            thread_id: The thread ID to resume
            decisions: List of decisions for each tool (accept/edit/reject)
            interrupt_id: Optional interrupt identifier
            
        Returns:
            AgentResponse with the bot's response after processing decisions
        """
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot resume.")

        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Resuming with decisions for thread_id={thread_id}")
        
        # Get current state to examine pending tool calls
        state_snapshot = await self.graph.aget_state(config)
        logger.info(
            f"Current state - next nodes: {state_snapshot.next}, "
            f"interrupted: {len(state_snapshot.next) > 0 if state_snapshot.next else False}"
        )
        
        # Extract the last AI message with tool calls
        messages = state_snapshot.values.get("messages", [])
        if not messages:
            logger.error("No messages in state, cannot resume")
            return None
            
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            logger.error("Last message is not AIMessage with tool calls")
            return None
        
        # Build a mapping of task_id to decision
        decision_map: Dict[str, InterruptDecision] = {}
        for decision in decisions:
            task_id = decision.task_id
            if task_id:
                decision_map[task_id] = decision
        
        logger.info(f"Decision map has {len(decision_map)} entries")
        
        # Process each tool call based on decisions
        tool_calls_to_execute = []
        rejection_messages_list = []
        modified_tool_calls = []
        
        for tool_call in last_message.tool_calls:
            tool_call_id = tool_call.get("id")
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            # Find the decision for this tool call
            decision = decision_map.get(tool_call_id)
            
            if not decision:
                logger.warning(f"No decision found for tool call {tool_call_id}, defaulting to reject")
                # Default to rejection if no decision provided
                rejection_messages_list.append(
                    ToolMessage(
                        content=f"Tool execution rejected: No decision provided",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )
                continue
            
            decision_type = decision.type
            
            if decision_type in (InterruptDecisionType.ACCEPT, InterruptDecisionType.APPROVE):
                # Accept: allow execution with original args
                logger.info(f"Tool {tool_name} ({tool_call_id}) accepted for execution")
                tool_calls_to_execute.append(tool_call)
                
            elif decision_type == InterruptDecisionType.EDIT:
                # Edit: modify arguments before execution
                logger.info(f"Tool {tool_name} ({tool_call_id}) modified with new args")
                modified_args = decision.args or tool_args
                modified_tool_call = {
                    "name": tool_name,
                    "args": modified_args,
                    "id": tool_call_id,
                }
                tool_calls_to_execute.append(modified_tool_call)
                modified_tool_calls.append(tool_call_id)
                
            elif decision_type in (InterruptDecisionType.REJECT, InterruptDecisionType.RESPOND):
                # Reject: skip execution and provide feedback message
                feedback_msg = "User rejected tool execution"
                if decision.args and "message" in decision.args:
                    feedback_msg = decision.args["message"]
                    
                logger.info(f"Tool {tool_name} ({tool_call_id}) rejected with message: {feedback_msg}")
                rejection_messages_list.append(
                    ToolMessage(
                        content=feedback_msg,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )
            else:
                logger.warning(f"Unknown decision type {decision_type}, rejecting tool")
                rejection_messages_list.append(
                    ToolMessage(
                        content=f"Tool execution rejected: Unknown decision type",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )
        
        # Now handle the resume based on what was decided
        if rejection_messages_list and not tool_calls_to_execute:
            # ALL tools were rejected - add rejection messages and route back to agent (skip tools node)
            logger.info(f"All {len(rejection_messages_list)} tools rejected, routing back to agent")
            
            selected_agent = state_snapshot.values.get("selected_agent")
            
            # Update state with rejection messages as if tools node executed them
            await self.graph.aupdate_state(
                config=config,
                values={"messages": rejection_messages_list},
                as_node="tools",
            )
            
            # Resume - will route back to the selected agent
            result = await self.graph.ainvoke(None, config=config)
            
        elif tool_calls_to_execute and not rejection_messages_list:
            # ALL tools were accepted/edited - proceed with execution
            logger.info(f"All {len(tool_calls_to_execute)} tools approved for execution")
            
            # If any tools were edited, update the last message with modified tool calls
            if modified_tool_calls:
                logger.info(f"Updating {len(modified_tool_calls)} modified tool calls in state")
                # Create a new AI message with the modified tool calls
                new_ai_message = AIMessage(
                    content=last_message.content,
                    tool_calls=tool_calls_to_execute,
                )
                # Replace the last message
                updated_messages = messages[:-1] + [new_ai_message]
                await self.graph.aupdate_state(
                    config=config,
                    values={"messages": updated_messages},
                    as_node=state_snapshot.values.get("selected_agent", "search_agent"),
                )
            
            # Resume execution - will execute tools node
            logger.info("Resuming to execute approved tools")
            result = await self.graph.ainvoke(None, config=config)
            
        elif tool_calls_to_execute and rejection_messages_list:
            # MIXED decisions - some approved, some rejected
            logger.info(
                f"Mixed decisions: {len(tool_calls_to_execute)} approved, "
                f"{len(rejection_messages_list)} rejected"
            )
            
            # Update the AI message to only include approved tool calls
            new_ai_message = AIMessage(
                content=last_message.content,
                tool_calls=tool_calls_to_execute,
            )
            
            # Replace the last message with modified version
            updated_messages = messages[:-1] + [new_ai_message]
            selected_agent = state_snapshot.values.get("selected_agent")
            
            await self.graph.aupdate_state(
                config=config,
                values={"messages": updated_messages},
                as_node=selected_agent,
            )
            
            # Resume to execute approved tools
            logger.info("Executing approved tools...")
            result = await self.graph.ainvoke(None, config=config)
            
            # After execution, add rejection messages for rejected tools
            logger.info("Adding rejection messages for rejected tools...")
            current_state = await self.graph.aget_state(config)
            current_messages = current_state.values.get("messages", [])
            
            await self.graph.aupdate_state(
                config=config,
                values={"messages": current_messages + rejection_messages_list},
                as_node="tools",
            )
            
            # Continue execution to let agent process all tool results
            logger.info("Resuming to process all tool results...")
            result = await self.graph.ainvoke(None, config=config)
            
        else:
            # No tools to execute at all (shouldn't happen)
            logger.error("No tools to execute and no rejections - unexpected state")
            return None
        
        # Extract response from result
        response = result.get("response")
        if not response:
            logger.warning("No response after resume, checking final state...")
            final_state = await self.graph.aget_state(config)
            logger.info(f"Final state - next nodes: {final_state.next}")
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
        Yields token-level events by calling agent streaming methods directly.
        """
        logger.info(
            f"execute_stream started - thread_id: {thread_id}, checkpointer enabled: {self.checkpointer is not None}"
        )

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

        config = None
        if self.checkpointer and thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        # Track if we've hit an interrupt
        interrupted = False

        # Stream events from the graph
        try:
            async for event in self.graph.astream_events(
                initial_state, config=config, version="v1"
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        # Ensure content is a string, not a list or other type
                        content = chunk.content
                        if isinstance(content, list):
                            content = "".join(str(item) for item in content)
                        elif not isinstance(content, str):
                            content = str(content)
                        yield {"type": "token", "content": content}

                elif kind == "on_tool_start":
                    # We can filter tools if needed, but for now expose all
                    yield {"type": "tool_start", "name": event["name"]}

                elif kind == "on_tool_end":
                    yield {"type": "tool_end", "name": event["name"]}

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

        # Check final state for response or interrupt
        if self.checkpointer and thread_id:
            snapshot = await self.graph.aget_state(config)
            logger.info(
                f"Final state check - next: {snapshot.next}, has response: {snapshot.values.get('response') is not None}, messages count: {len(snapshot.values.get('messages', []))}"
            )

            # Check if interrupted (next nodes exist means we're paused)
            if snapshot.next and len(snapshot.next) > 0:
                logger.info(f"Workflow interrupted before nodes: {snapshot.next}")
                # Extract pending tool calls from the last AI message
                messages = snapshot.values.get("messages", [])
                logger.info(f"Number of messages in state: {len(messages)}")

                pending_tool_calls = None
                if messages:
                    last_msg = messages[-1]
                    logger.info(
                        f"Last message type: {type(last_msg)}, has tool_calls attr: {hasattr(last_msg, 'tool_calls')}"
                    )

                    if isinstance(last_msg, AIMessage):
                        if hasattr(last_msg, "tool_calls"):
                            logger.info(
                                f"Tool calls on last message: {last_msg.tool_calls}"
                            )
                            if last_msg.tool_calls:
                                # Convert tool calls to serializable dictionaries
                                pending_tool_calls = []
                                for tc in last_msg.tool_calls:
                                    if isinstance(tc, dict):
                                        pending_tool_calls.append(tc)
                                    else:
                                        # Convert object to dict
                                        pending_tool_calls.append(
                                            {
                                                "name": getattr(tc, "name", "unknown"),
                                                "args": getattr(tc, "args", {}),
                                                "id": getattr(tc, "id", None),
                                            }
                                        )
                                logger.info(
                                    f"Converted {len(pending_tool_calls)} tool calls to dicts"
                                )
                        else:
                            logger.warning(
                                "Last AI message has no tool_calls attribute"
                            )
                    else:
                        logger.warning(
                            f"Last message is not AIMessage: {type(last_msg)}"
                        )
                else:
                    logger.warning("No messages in state")

                yield {
                    "type": "interrupt",
                    "next": snapshot.next,
                    "thread_id": thread_id,
                    "pending_tool_calls": pending_tool_calls,
                }
                interrupted = True
            else:
                # Not interrupted, workflow completed
                interrupted = False

            # Only yield complete if we have a response and we're not interrupted
            if not interrupted:
                response = snapshot.values.get("response")
                if response:
                    logger.info("Yielding complete event with response")
                    yield {"type": "complete", "response": response}
                else:
                    # No response - this is an error
                    logger.error("Workflow completed without generating a response")
                    logger.error(
                        f"Final state values keys: {list(snapshot.values.keys())}"
                    )
                    yield {"type": "error", "error": "No response generated"}
        else:
            # If no checkpointer, we can't detect interrupts
            logger.error(
                f"No checkpointer or thread_id - checkpointer: {self.checkpointer is not None}, thread_id: {thread_id}"
            )
            yield {"type": "error", "error": "Checkpointing not configured"}

    async def get_state(self, thread_id: str) -> dict:
        """
        Get the current state of a workflow.

        Args:
            thread_id: The thread ID to check

        Returns:
            Dictionary containing state information including:
            - interrupted: Whether the workflow is currently interrupted
            - next: The next node(s) to execute
            - values: The current state values
            - pending_tool_calls: Any pending tool calls awaiting approval
        """
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot get state.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.graph.aget_state(config)

        # Extract pending tool calls if any
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
            "interrupted": len(snapshot.next) > 0 if snapshot.next else False,
            "next": snapshot.next,
            "values": snapshot.values,
            "pending_tool_calls": pending_tool_calls,
            "selected_agent": snapshot.values.get("selected_agent"),
        }

    async def cleanup(self):
        """Cleanup resources from agents that use MCP tools"""
        logger.info("Cleaning up MultiAgentWorkflow resources...")
        for agent in self._cleanup_agents:
            if hasattr(agent, "cleanup"):
                try:
                    await agent.cleanup()
                except Exception as e:
                    logger.error(
                        f"Error cleaning up agent {agent.__class__.__name__}: {e}"
                    )
        logger.info("MultiAgentWorkflow cleanup completed")


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
