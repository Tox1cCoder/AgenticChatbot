import logging

from langchain_core.messages import HumanMessage

from ...interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ..prompts import SEARCH_SYSTEM_PROMPT, build_search_prompt
from ..schemas import AgentMessage, AgentResponse, AgentType
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    def __init__(self, runtime_model_resolver: IRuntimeModelResolver | None = None):
        super().__init__(
            agent_config_key="search",
            runtime_model_resolver=runtime_model_resolver,
        )

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

    async def cleanup(self):
        await super().cleanup()
