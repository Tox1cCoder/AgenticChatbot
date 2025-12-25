import logging
from typing import Optional, List, Dict, Any, AsyncIterator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from .base_agent import BaseAgent
from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_search_prompt, SEARCH_SYSTEM_PROMPT, TOOL_CONTEXT_SUFFIX
from ..utils import coerce_response_text
from ...core.config import settings
from ..mcp_integration import get_global_mcp_manager

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):

    def __init__(self):
        super().__init__(model_name=settings.search_agent_model)

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SEARCH

    @property
    def agent_id(self) -> str:
        return "search_agent"

    def _get_base_system_prompt(self) -> str:
        return SEARCH_SYSTEM_PROMPT

    async def invoke_model(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        # Initialize MCP tools if needed
        if self.mcp_manager is None:
            await self._init_tools()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        # Check if this invocation includes tool results (post-tool-execution)
        # This happens when the graph routes back after tool execution
        has_tool_results = "Tool results:" in message.content

        # Build prompt
        prompt = build_search_prompt(
            message.content,
            conversation_history,
            persona=persona,
            has_tool_results=has_tool_results,
        )

        # Configure tool calling; allow follow-up tool planning when needed
        llm_with_tools = self._get_llm_with_tools()

        # Invoke model
        response = await llm_with_tools.ainvoke([HumanMessage(content=prompt)])

        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = response.tool_calls

        # Create response metadata
        search_metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "tools_available": len(self.tools),
            "agent_type": "tool_calling",
            "persona_used": persona,
        }

        return AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id="search_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content=coerce_response_text(response.content),
                tool_calls=tool_calls if tool_calls else None,
            ),
            metadata=search_metadata,
        )

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        return await self.invoke_model(message, conversation_id)

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        # Initialize MCP tools if needed
        if self.mcp_manager is None:
            await self._init_tools()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        # Build prompt
        prompt = build_search_prompt(
            message.content, conversation_history, persona=persona
        )

        llm_with_tools = self.langchain_model.bind_tools(self.tools)
        accumulated_content = ""
        accumulated_thinking = ""
        current_tool_calls = {}  # Track tool call chunks

        try:
            # Use astream_events for LangChain model streaming

            async for event in llm_with_tools.astream_events(
                [HumanMessage(content=prompt)], version="v2"
            ):
                event_type = event.get("event")

                # Handle LLM token streaming
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue

                    # Use content_blocks for better content handling
                    if hasattr(chunk, "content_blocks") and chunk.content_blocks:
                        for block in chunk.content_blocks:
                            block_type = block.get("type")

                            if block_type == "text":
                                token = block.get("text", "")
                                if not token:
                                    continue

                                # Check for thinking/reasoning markers
                                additional_kwargs = getattr(
                                    chunk, "additional_kwargs", {}
                                )
                                is_thinking = additional_kwargs.get(
                                    "thought"
                                ) or additional_kwargs.get("thinking")

                                if is_thinking:
                                    accumulated_thinking += token
                                    yield {"type": "thinking", "content": token}
                                else:
                                    accumulated_content += token
                                    yield {"type": "token", "content": token}
                            
                            # Handle explicit thinking block type
                            elif block_type == "thinking":
                                thinking_content = block.get("thinking", "")
                                if thinking_content:
                                    accumulated_thinking += thinking_content
                                    yield {"type": "thinking", "content": thinking_content}

                            elif block_type == "tool_call_chunk":
                                # Accumulate tool call chunks
                                tool_index = block.get("index", 0)
                                if tool_index not in current_tool_calls:
                                    current_tool_calls[tool_index] = {
                                        "id": block.get("id"),
                                        "name": block.get("name"),
                                        "args": "",
                                    }
                                if block.get("args"):
                                    current_tool_calls[tool_index]["args"] += block.get(
                                        "args"
                                    )
                                if (
                                    block.get("name")
                                    and not current_tool_calls[tool_index]["name"]
                                ):
                                    current_tool_calls[tool_index]["name"] = block.get(
                                        "name"
                                    )
                                if (
                                    block.get("id")
                                    and not current_tool_calls[tool_index]["id"]
                                ):
                                    current_tool_calls[tool_index]["id"] = block.get(
                                        "id"
                                    )

                    elif hasattr(chunk, "content"):
                        token = coerce_response_text(chunk.content)
                        if not token:
                            continue

                        # Check for thinking/reasoning markers
                        additional_kwargs = getattr(chunk, "additional_kwargs", {})
                        is_thinking = additional_kwargs.get(
                            "thought"
                        ) or additional_kwargs.get("thinking")

                        if is_thinking:
                            accumulated_thinking += token
                            yield {"type": "thinking", "content": token}
                        else:
                            accumulated_content += token
                            yield {"type": "token", "content": token}

                    # Check for chunk completion
                    if (
                        hasattr(chunk, "chunk_position")
                        and chunk.chunk_position == "last"
                    ):
                        # Emit accumulated tool calls
                        for tool_call in current_tool_calls.values():
                            if tool_call["name"]:
                                try:
                                    import json

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
                                    "tool_call_id": tool_call["id"],
                                    "args": args,
                                }
                        current_tool_calls = {}

                # Handle tool execution events
                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    tool_output = event.get("data", {}).get("output")
                    tool_call_id = event.get("run_id")
                    yield {
                        "type": "tool_end",
                        "name": tool_name,
                        "tool_call_id": str(tool_call_id) if tool_call_id else None,
                        "result": tool_output,
                    }

        except Exception as e:
            yield {"type": "error", "error": str(e)}

    async def cleanup(self):
        await super().cleanup()
