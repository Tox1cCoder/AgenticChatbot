import logging
import json
from typing import Optional, List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_search_prompt, SEARCH_SYSTEM_PROMPT
from ..utils import coerce_response_text, extract_agent_execution_info, get_error_recovery_hint
from ...core.config import settings
from ...core.exceptions.mcp import ServerNotFoundError
from ..mcp_integration import MCPManager

logger = logging.getLogger(__name__)


class SearchAgent:

    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.langchain_model = None
        self.mcp_manager = None
        self.tools = []
        self._init_gemini()

    def _init_gemini(self):
        api_key = settings.gemini_api_key

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()
        self.langchain_model = ChatGoogleGenerativeAI(
            model=self.model_name, google_api_key=api_key, temperature=0.0
        )

    async def _init_mcp(self):
        """Initialize MCP manager and load external tool suites"""
        if self.mcp_manager is None:
            try:
                self.mcp_manager = MCPManager()
                await self.mcp_manager.initialize()

                combined_tools: Dict[str, BaseTool] = {}

                preferred_servers = ["tavily", "time"]

                for server_name in preferred_servers:
                    server_tools = await self.mcp_manager.get_server_tools(
                        server_name
                    )

                    for tool in server_tools:
                        combined_tools[tool.name] = tool

                if not combined_tools:
                    for tool in await self.mcp_manager.get_tools():
                        combined_tools[tool.name] = tool

                self.tools = list(combined_tools.values())

            except Exception as e:
                logger.error(f"Failed to initialize MCP manager: {e}", exc_info=True)
                self.tools = []

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process a search query with tool calling"""

        # Initialize MCP tools if needed
        if self.mcp_manager is None:
            await self._init_mcp()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        # Build prompt
        prompt = build_search_prompt(
            message.content, conversation_history, persona=persona
        )

        error_message: Optional[str] = None
        try:
            # Create agent executor
            agent_executor = self._create_agent_executor(self.tools, SEARCH_SYSTEM_PROMPT)
            
            # Invoke agent with the user message
            agent_response = await agent_executor.ainvoke({
                "messages": [HumanMessage(content=prompt)]
            })
            
            # Extract execution info
            execution_info = extract_agent_execution_info(agent_response)
            
            response_text = execution_info["response_text"]
            tools_used = execution_info["tools_used"]
            tool_artifacts = execution_info["tool_artifacts"]
            
            extracted_images = []
            for artifact in tool_artifacts:
                if artifact.get("tool") == "tavily_search" and artifact.get("output"):
                    images = self._extract_images_from_tavily(artifact["output"])
                    if images:
                        extracted_images.extend(images)

        except Exception as e:
            logger.error(f"Error invoking search agent: {e}", exc_info=True)
            error_message = f"{type(e).__name__}: {e}"
            recovery_hint = get_error_recovery_hint(e, "search_agent", {})
            tools_used = []
            tool_artifacts = [
                {
                    "tool": "search_agent",
                    "args": {},
                    "error": error_message,
                    "hint": recovery_hint,
                }
            ]
            extracted_images = []

        response_text = coerce_response_text(response_text)
        
        # Create response metadata
        search_metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "tools_available": len(self.tools),
            "agent_type": "tool_calling",
            "persona_used": persona,
        }

        # Add tool usage metadata
        if tools_used:
            search_metadata["tools_used"] = tools_used
            search_metadata["tool_calls_count"] = len(tools_used)

        # Add images to metadata if any were extracted
        if extracted_images:
            search_metadata["images"] = extracted_images
        if tool_artifacts:
            search_metadata["tool_artifacts"] = tool_artifacts
            if not error_message:
                error_entries = [art["error"] for art in tool_artifacts if art.get("error")]
                if error_entries:
                    error_message = error_entries[0]
                    search_metadata["error"] = error_message
        elif error_message:
            search_metadata["error"] = error_message

        # Create response message
        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=response_text
        )

        return AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id="search_agent",
            message=response_message,
            metadata=search_metadata,
            tool_artifacts=tool_artifacts if tool_artifacts else None,
            error=error_message,
        )

    def _create_agent_executor(self, tools: List[BaseTool], system_prompt: str):
        """Create agent executor with proper tool binding configuration."""
        if not self.langchain_model:
            raise RuntimeError("SearchAgent language model not initialized")

        # Configure tool calling based on settings
        tool_choice = settings.tool_choice_mode if hasattr(settings, 'tool_choice_mode') else "auto"
        
        # Configure model with tool binding
        llm_with_tools = self.langchain_model.bind_tools(
            tools, 
            tool_config={
                "function_calling_config": {
                    "mode": tool_choice.upper() if tool_choice in ["auto", "any", "none"] else "AUTO"
                }
            }
        )
        
        agent = create_agent(
            model=llm_with_tools,
            tools=tools,
            system_prompt=system_prompt
        )
        
        return agent

    def _extract_images_from_tavily(self, tool_result: Any) -> List[Dict[str, str]]:
        """Extract images from Tavily search results"""
        try:
            # Parse tool result as JSON if it's a string
            if isinstance(tool_result, str):
                result_data = json.loads(tool_result)
            else:
                result_data = tool_result

            # Extract images array from response
            images = result_data.get("images", [])

            # Return images with url and description
            return [
                {"url": img.get("url", ""), "description": img.get("description", "")}
                for img in images
                if img.get("url")
            ]
        except Exception as e:
            logger.error(f"Failed to extract images from Tavily response: {e}")
            return []

    async def cleanup(self):
        """Cleanup MCP resources"""
        if self.mcp_manager:
            try:
                await self.mcp_manager.cleanup()
                logger.info("Search Agent MCP manager cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up Search Agent: {e}")
