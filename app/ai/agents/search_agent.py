import logging
from typing import Optional

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_search_prompt, SEARCH_SYSTEM_PROMPT
from ...core.config import settings
from ..mcp_integration import MCPManager

logger = logging.getLogger(__name__)


class SearchAgent:

    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.gemini_client = None
        self.langchain_model = None
        self.mcp_manager = None
        self.agent_executor = None
        self.tools = []
        self._init_gemini()

    def _init_gemini(self):
        api_key = settings.gemini_api_key
        if not api_key:
            logger.error("Gemini API key not configured")
            return

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        self.gemini_client = genai.Client(api_key=api_key)
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

    async def _init_agent(self):
        """Initialize tool calling agent with AgentExecutor"""
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SEARCH_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Bind tools to the model with function calling config
        llm_with_tools = self.langchain_model.bind_tools(
            self.tools, tool_config={"function_calling_config": {"mode": "AUTO"}}
        )

        # Create tool calling agent
        agent = create_tool_calling_agent(llm_with_tools, self.tools, prompt)

        # Wrap in AgentExecutor
        self.agent_executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=False
        )

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process a search query using tool calling agent"""

        if self.mcp_manager is None:
            await self._init_mcp()

        if self.agent_executor is None:
            await self._init_agent()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])

        # Build prompt
        prompt = build_search_prompt(message.content, conversation_history)

        try:
            # Invoke agent executor
            result = await self.agent_executor.ainvoke({"input": prompt})

            # Extract response from result
            response_text = result.get("output", "No response from search agent")

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
        }

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

    async def cleanup(self):
        """Cleanup MCP resources"""
        if self.mcp_manager:
            try:
                await self.mcp_manager.cleanup()
                logger.info("Search Agent MCP manager cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up Search Agent: {e}")
