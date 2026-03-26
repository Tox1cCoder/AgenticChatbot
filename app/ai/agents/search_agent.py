import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage

from ..prompts import SEARCH_SYSTEM_PROMPT, build_search_prompt
from ..schemas import AgentMessage, AgentResponse, AgentType
from ..utils import normalize_tool_call
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_config_key="search")

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
        conversation_id: str | None = None,
    ) -> AgentResponse:
        # Initialize MCP tools if needed
        if self.mcp_manager is None:
            await self._init_tools()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")
        message_content = message.content or ""
        request_user_id = message.metadata.get("user_id")
        request_device_id = message.metadata.get("device_id")
        model_request = message.metadata.get("model_request")
        history_summary = message.metadata.get("history_summary")

        # Check if this invocation includes tool results (post-tool-execution)
        # This happens when the graph routes back after tool execution
        has_tool_results = "Tool results:" in message_content

        # Build prompt
        prompt = build_search_prompt(
            message_content,
            conversation_history,
            persona=persona,
            has_tool_results=has_tool_results,
        )

        response = await self.invoke_model_with_history(
            [HumanMessage(content=prompt)],
            conversation_history,
            persona,
            conversation_id,
            user_id=request_user_id,
            device_id=request_device_id,
            model_request=model_request,
            history_summary=history_summary,
        )
        response.metadata["conversation_id"] = conversation_id
        response.metadata["context_messages"] = len(conversation_history)
        response.metadata["agent_type"] = "tool_calling"
        response.metadata["persona_used"] = persona
        return response

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: str | None = None,
    ) -> AgentResponse:
        return await self.invoke_model(message, conversation_id)

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        response = await self.invoke_model(message, conversation_id)

        for tool_call in response.message.tool_calls or []:
            normalized_tool_call = normalize_tool_call(tool_call)
            yield {
                "type": "tool_start",
                "name": normalized_tool_call.get("name", "unknown"),
                "tool_call_id": normalized_tool_call.get("id"),
                "args": normalized_tool_call.get("args"),
            }

        thinking_summary = str(response.metadata.get("thinking_summary") or "").strip()
        if thinking_summary:
            yield {"type": "thinking", "content": thinking_summary}

        yield {"type": "token", "content": response.message.content or ""}
        yield {"type": "complete", "response": response}

    async def cleanup(self):
        await super().cleanup()
