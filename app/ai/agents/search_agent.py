import logging
import json
from typing import Optional, List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_search_prompt
from ...core.config import settings
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
        if not api_key:
            logger.error("Gemini API key not configured")
            return

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        self.langchain_model = ChatGoogleGenerativeAI(
            model=self.model_name, google_api_key=api_key, temperature=0.7
        )

    async def _init_mcp(self):
        """Initialize MCP manager and load Tavily tools"""
        if self.mcp_manager is None:
            try:
                self.mcp_manager = MCPManager()
                await self.mcp_manager.initialize()

                # Get tools from Tavily server
                self.tools = await self.mcp_manager.get_server_tools("tavily")

            except Exception as e:
                logger.error(f"Failed to initialize MCP manager: {e}", exc_info=True)
                self.tools = []

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process a search query using tool calling"""

        if self.mcp_manager is None:
            await self._init_mcp()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        # Build prompt
        prompt = build_search_prompt(
            message.content, conversation_history, persona=persona
        )

        try:
            # Bind tools to model for this request
            llm_with_tools = self.langchain_model.bind_tools(
                self.tools, tool_config={"function_calling_config": {"mode": "AUTO"}}
            )

            # Create initial message
            messages = [HumanMessage(content=prompt)]

            # Agent loop: model -> tool calls -> model -> response
            max_iterations = 5
            extracted_images = []

            for iteration in range(max_iterations):
                # Invoke model
                ai_message = await llm_with_tools.ainvoke(messages)
                messages.append(ai_message)

                # Check if model wants to use tools
                if not ai_message.tool_calls:
                    response_text = ai_message.content
                    break

                # Execute tool calls
                for tool_call in ai_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    # Find and execute the tool
                    tool_result = None
                    for tool in self.tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = await tool.ainvoke(tool_args)

                                # Extract images from Tavily response
                                if tool_name == "tavily_search":
                                    images = self._extract_images_from_tavily(
                                        tool_result
                                    )
                                    if images:
                                        extracted_images.extend(images)

                            except Exception as e:
                                tool_result = f"Error executing tool: {str(e)}"
                            break

                    # Add tool result to messages
                    if tool_result is None:
                        tool_result = f"Tool {tool_name} not found"

                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                    )
            else:
                # Max iterations reached
                response_text = "Search completed but max iterations reached."

        except Exception as e:
            logger.error(f"Error invoking search agent: {e}", exc_info=True)
            response_text = "An error occurred while processing your search request."

        # Create response metadata
        search_metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "tools_available": len(self.tools),
            "agent_type": "tool_calling",
            "persona_used": persona,
        }

        # Add images to metadata if any were extracted
        if extracted_images:
            search_metadata["images"] = extracted_images

        # Create response message
        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=response_text
        )

        return AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id="search_agent",
            message=response_message,
            metadata=search_metadata,
        )

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
