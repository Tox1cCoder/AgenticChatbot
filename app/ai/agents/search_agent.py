import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from ...interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ..prompts import SEARCH_SYSTEM_PROMPT, build_search_prompt
from ..schemas import AgentMessage, AgentResponse, AgentType
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...usage.recorder import ModelUsageRecorder
    from ..workflow.specialists import SpecialistDefinition


class SearchAgent(BaseAgent):
    def __init__(
        self,
        runtime_model_resolver: IRuntimeModelResolver | None = None,
        recorder: "ModelUsageRecorder | None" = None,
    ):
        super().__init__(
            agent_config_key="search",
            runtime_model_resolver=runtime_model_resolver,
            recorder=recorder,
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
        )
        response.metadata["conversation_id"] = conversation_id
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


def _has_tool_context(messages: list) -> bool:
    """Whether this turn already carries tool results.

    Prompts differ before and after tools have run, so the flag is computed
    from the messages rather than tracked as loop state.
    """
    for message in messages or []:
        if getattr(message, "type", None) == "tool":
            return True
        if getattr(message, "tool_calls", None):
            return True
        additional = getattr(message, "additional_kwargs", None)
        if isinstance(additional, dict) and additional.get("tool_calls"):
            return True
    return False


def build_search_specialist_definition(agent: "SearchAgent") -> "SpecialistDefinition":
    """Declare search_agent as configuration for a ``create_agent`` subgraph.

    The prompt and tool set stay here because they are this agent's domain
    knowledge; the model/tool loop belongs to the framework.
    """
    from ..workflow.specialists import SpecialistDefinition

    async def system_prompt_factory(request) -> str:
        return agent._build_system_prompt(
            request.persona,
            _has_tool_context(request.messages),
            user_id=request.user_id,
            device_id=request.device_id,
            **request.extras.get("system_prompt_kwargs", {}),
        )

    async def tool_factory(request) -> list:
        if request.extras.get("disable_tools"):
            # A forced final response must not be able to call another tool.
            return []
        await agent._init_tools()
        return agent._get_tools_for_binding(
            conversation_id=request.conversation_id,
            internal_tools=request.extras.get("internal_tools"),
            user_id=request.user_id,
            device_id=request.device_id,
            include_hand_off=request.extras.get("include_hand_off"),
            excluded_tool_names=request.extras.get("excluded_tool_names"),
        )

    return SpecialistDefinition(
        agent_id="search_agent",
        agent_type=AgentType.SEARCH,
        model_config_key="search",
        system_prompt_factory=system_prompt_factory,
        tool_factory=tool_factory,
        agent=agent,
        output_policy_ids=("public_content", "artifact_provenance", "tool_message_pairing"),
    )
