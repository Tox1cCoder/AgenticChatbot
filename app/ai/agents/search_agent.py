import logging
from typing import Optional

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_search_prompt
from ...core.config import settings
from ..mcp_integration import MCPManager

logger = logging.getLogger(__name__)


class SearchAgent:

    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.gemini_client = None
        self.langchain_model = None
        self.mcp_manager = None
        self.react_agent = None
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
        logger.info("Gemini client and LangChain model initialized for Search Agent")

    async def _init_mcp(self):
        """Initialize MCP manager and load Tavily tools"""
        if self.mcp_manager is None:
            try:
                self.mcp_manager = MCPManager()
                await self.mcp_manager.initialize()

                # Get tools from Tavily server
                self.tools = await self.mcp_manager.get_server_tools("tavily")
                logger.info(f"Loaded {len(self.tools)} tools from Tavily MCP server")
                if len(self.tools) == 0:
                    logger.warning(
                        "No tools loaded from Tavily server. Check MCP server configuration and API key."
                    )
            except Exception as e:
                logger.error(f"Failed to initialize MCP manager: {e}", exc_info=True)
                self.tools = []

    async def _init_react_agent(self):
        """Initialize ReAct agent with tools"""
        # Create prompt template for ReAct agent
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that can search the web for information.",
                ),
                ("placeholder", "{messages}"),
            ]
        )

        # Create ReAct agent with LangChain model
        self.react_agent = create_react_agent(
            model=self.langchain_model, tools=self.tools, prompt=prompt
        )
        logger.info("ReAct agent initialized for Search Agent")

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process a search query using ReAct agent with Tavily tools"""

        if self.mcp_manager is None:
            await self._init_mcp()

        if self.react_agent is None:
            await self._init_react_agent()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])

        # Build prompt
        prompt = build_search_prompt(message.content, conversation_history)

        # Invoke ReAct agent
        result = await self.react_agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )

        # Extract final AI message from result
        if "messages" in result and len(result["messages"]) > 0:
            # Get the last AI message
            ai_messages = [
                msg
                for msg in result["messages"]
                if hasattr(msg, "type") and msg.type == "ai"
            ]
            if ai_messages:
                response_text = ai_messages[-1].content
            else:
                response_text = str(result["messages"][-1].content)
        else:
            response_text = "No response from search agent"

        # Create response metadata
        search_metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "tools_used": len(self.tools),
            "agent_type": "react",
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
