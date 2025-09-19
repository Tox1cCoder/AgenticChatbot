from typing import List, Optional
from langchain.core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import logging


logger = loggerging.getLogger(__name__)


class BaseAgent:
    def __init__(self, system_prompt: str):
        self.llm = None
        self.tools: List[Any] = []
        self.agent_type = "base"
        self.agent_name = "BaseAgent"
        self.system_prompt = system_prompt
        self.conversation: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    def get_system_prompt(self) -> str:
        raise NotImplementedError("Subclasses must implement get_system_prompt method")

    async def initialize_tools(self):
        agent_tools = await get_tools_for_agent(self.agent_type)
        mcp_tools = await ToolHanlder.initialize_tools()
        self.tools = agent_tools + mcp_tools
        logger.info(
            f"Initialized tools for {self.agent_name}: {[tool.name for tool in self.tools]}"
        )

    async def invoke(
        self, message: HumanMessage, chat_history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        pass
