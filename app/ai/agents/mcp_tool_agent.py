import logging
from typing import Optional

from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..mcp_integration import MCPManager
from ...core.config import settings

logger = logging.getLogger(__name__)


# Agent prompt template
MCP_AGENT_PROMPT = """You are a mathematical assistant with access to calculator tools.

When a user asks for mathematical calculations, use the available tools to compute accurate results.

Available tools:
{tools}

Tool names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""


class MCPToolAgent:
    """Agent that uses MCP tools for calculations"""

    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.mcp_manager: Optional[MCPManager] = None
        self.agent_executor: Optional[AgentExecutor] = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Lazy initialization of MCP tools and agent"""
        if self._initialized:
            return

        try:
            # Initialize MCP Manager
            self.mcp_manager = MCPManager()
            await self.mcp_manager.initialize()

            # Get MCP tools
            tools = await self.mcp_manager.get_tools()

            if not tools:
                logger.warning(
                    "No MCP tools available. MCP agent will have limited functionality."
                )
                self._initialized = True
                return

            # Create LLM
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=settings.gemini_api_key,
                temperature=0.1,
            )

            # Create prompt
            prompt = PromptTemplate.from_template(MCP_AGENT_PROMPT)

            # Create agent
            agent = create_react_agent(llm, tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
            )

            self._initialized = True
            logger.info(f"MCP Tool Agent initialized with {len(tools)} tools")

        except Exception as e:
            logger.error(f"Failed to initialize MCP Tool Agent: {e}")
            self._initialized = True 

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process message using MCP tools"""

        await self._ensure_initialized()

        # Execute agent with tools
        result = await self.agent_executor.ainvoke({"input": message.content})

        output = result.get("output")

        return AgentResponse(
            agent_type=AgentType.MCP_TOOL,
            agent_id="mcp_tool_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content=output,
            ),
            sources=[],
            confidence=0.9,
        )

    async def cleanup(self):
        """Cleanup MCP resources"""
        if self.mcp_manager:
            await self.mcp_manager.cleanup()
