import logging
import base64
from typing import Optional, List, Dict, Any, AsyncIterator

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_chat_prompt
from ..utils import (
    coerce_response_text,
    extract_agent_execution_info,
    get_error_recovery_hint,
)
from ...core.config import settings
from ...core.exceptions.mcp import ServerNotFoundError
from ..mcp_integration import MCPManager

logger = logging.getLogger(__name__)


class ChatAgent:

    def __init__(self):
        self.model_name = "gemini-flash-latest"
        self.gemini_client = None
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

        self.gemini_client = genai.Client(api_key=api_key)

        self.langchain_model = ChatGoogleGenerativeAI(
            model=self.model_name, google_api_key=api_key, temperature=0.8
        )

    async def _init_tools(self):
        """Initialize MCP manager and load general-purpose tools"""
        if self.mcp_manager is None:
            try:
                self.mcp_manager = MCPManager()
                await self.mcp_manager.initialize()

                combined_tools: Dict[str, BaseTool] = {}

                preferred_servers = ["calculator", "time"]
                for server_name in preferred_servers:
                    try:
                        server_tools = await self.mcp_manager.get_server_tools(
                            server_name
                        )
                    except ServerNotFoundError:
                        logger.debug(
                            "Preferred MCP server '%s' not configured for ChatAgent",
                            server_name,
                        )
                        continue

                    for tool in server_tools:
                        combined_tools[tool.name] = tool

                for tool in await self.mcp_manager.get_tools():
                    combined_tools.setdefault(tool.name, tool)

                self.tools = list(combined_tools.values())

                server_status = self.mcp_manager.get_servers_status()
                active_servers = [
                    name
                    for name, status in server_status.items()
                    if status.get("enabled")
                ]
                logger.info(
                    "Loaded %d MCP tools for ChatAgent from %d servers",
                    len(self.tools),
                    len(active_servers),
                )

            except Exception as e:
                logger.error(
                    f"Failed to initialize MCP manager for ChatAgent: {e}",
                    exc_info=True,
                )
                self.tools = []

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:

        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        prompt = build_chat_prompt(
            message.content, conversation_history, persona=persona
        )

        attachments = (
            message.attachments
            if hasattr(message, "attachments") and message.attachments
            else None
        )

        response_text: str = ""
        tools_used: List[str] = []
        tool_artifacts: List[Dict[str, Any]] = []

        try:
            if attachments:
                response_text = await self._generate_with_vision(prompt, attachments)
            else:
                # Initialize tools if not done yet
                if self.mcp_manager is None:
                    await self._init_tools()

                if self.tools and self.langchain_model:
                    response_text, tools_used, tool_artifacts = (
                        await self._generate_with_tools(prompt)
                    )
                else:
                    response_text = await self._generate(prompt)
        except Exception as exc:
            logger.error(
                "Error while processing message in ChatAgent: %s", exc, exc_info=True
            )
            response_text = await self._handle_generation_error(prompt, exc)
            error_description = f"{type(exc).__name__}: {exc}"
            tool_artifacts.append(
                {
                    "tool": "chat_agent",
                    "args": {},
                    "error": error_description,
                }
            )
            # Signal that the normal tool flow failed
            tools_used = []

        response_text = coerce_response_text(response_text)

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=response_text
        )

        # Build metadata
        metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "persona_used": persona,
            "has_images": bool(attachments),
            "tools_available": len(self.tools),
        }

        # Add tool usage metadata if tools were used
        if tools_used:
            metadata["tools_used"] = tools_used
            metadata["tool_calls_count"] = len(tools_used)
        if tool_artifacts:
            metadata["tool_artifacts"] = tool_artifacts
            error_artifacts = [
                artifact for artifact in tool_artifacts if artifact.get("error")
            ]
            if error_artifacts:
                metadata["error"] = error_artifacts[0]["error"]

        return AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=response_message,
            metadata=metadata,
            tool_artifacts=tool_artifacts if tool_artifacts else None,
        )

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream message processing with token-by-token generation.
        Yields events as tokens are generated.
        """
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        prompt = build_chat_prompt(
            message.content, conversation_history, persona=persona
        )

        attachments = (
            message.attachments
            if hasattr(message, "attachments") and message.attachments
            else None
        )

        accumulated_content = ""
        tools_used: List[str] = []
        tool_artifacts: List[Dict[str, Any]] = []

        try:
            if attachments:
                # Vision with attachments - non-streaming fallback
                response_text = await self._generate_with_vision(prompt, attachments)
                # Yield as single token
                yield {"type": "token", "content": response_text}
                accumulated_content = response_text
            else:
                # Initialize tools if not done yet
                if self.mcp_manager is None:
                    await self._init_tools()

                if self.tools and self.langchain_model:
                    # Stream with tools
                    async for event in self._generate_with_tools_stream(prompt):
                        if event["type"] == "token":
                            accumulated_content += event["content"]
                            yield event
                        elif event["type"] in ["tool_start", "tool_end"]:
                            yield event
                        elif event["type"] == "result":
                            # Extract final result info
                            accumulated_content = event["response_text"]
                            tools_used = event["tools_used"]
                            tool_artifacts = event["tool_artifacts"]
                else:
                    # Stream without tools
                    async for chunk in self._generate_stream(prompt):
                        accumulated_content += chunk
                        yield {"type": "token", "content": chunk}

        except Exception as exc:
            logger.error(
                "Error while streaming message in ChatAgent: %s", exc, exc_info=True
            )
            error_text = await self._handle_generation_error(prompt, exc)
            yield {"type": "token", "content": error_text}
            accumulated_content = error_text

            error_description = f"{type(exc).__name__}: {exc}"
            tool_artifacts.append(
                {
                    "tool": "chat_agent",
                    "args": {},
                    "error": error_description,
                }
            )
            tools_used = []

        # Coerce final response
        accumulated_content = coerce_response_text(accumulated_content)

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=accumulated_content
        )

        # Build metadata
        metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "persona_used": persona,
            "has_images": bool(attachments),
            "tools_available": len(self.tools),
        }

        # Add tool usage metadata if tools were used
        if tools_used:
            metadata["tools_used"] = tools_used
            metadata["tool_calls_count"] = len(tools_used)
        if tool_artifacts:
            metadata["tool_artifacts"] = tool_artifacts
            error_artifacts = [
                artifact for artifact in tool_artifacts if artifact.get("error")
            ]
            if error_artifacts:
                metadata["error"] = error_artifacts[0]["error"]

        # Yield complete event with full response
        yield {
            "type": "complete",
            "response": AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=response_message,
                metadata=metadata,
                tool_artifacts=tool_artifacts if tool_artifacts else None,
            ),
        }

    async def _generate_with_tools_stream(
        self, prompt: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate streaming response with tool calling support.
        Yields token chunks and tool execution events.
        """
        try:
            # Create agent executor
            agent_executor = self._create_agent_executor(self.tools, prompt)

            accumulated_text = ""
            tools_used: List[str] = []

            # Stream agent execution
            async for event in agent_executor.astream_events(
                {"messages": [HumanMessage(content=prompt)]}, version="v1"
            ):
                event_type = event.get("event")

                # Extract tokens from LLM events
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # Handle both string and list content
                        token = chunk.content
                        if isinstance(token, list):
                            # If it's a list, join the string parts
                            token = "".join(str(item) for item in token if item)
                        if token:  # Only yield non-empty tokens
                            accumulated_text += token
                            yield {"type": "token", "content": token}

                # Track tool execution start
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown_tool")
                    yield {"type": "tool_start", "name": tool_name}

                # Track tool execution end
                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown_tool")
                    yield {"type": "tool_end", "name": tool_name}

            # Get final response for complete extraction
            agent_response = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}
            )

            # Extract execution info
            execution_info = extract_agent_execution_info(agent_response)

            response_text = execution_info["response_text"]
            tools_used = execution_info["tools_used"]
            tool_artifacts = execution_info["tool_artifacts"]

            # Log parallel tool execution summary
            if tools_used:
                logger.info(
                    f"ChatAgent executed {len(tools_used)} tool(s): {', '.join(tools_used)}"
                )

            # Yield result info
            yield {
                "type": "result",
                "response_text": response_text,
                "tools_used": tools_used,
                "tool_artifacts": tool_artifacts,
            }

        except Exception as exc:
            logger.error("Error in tool calling streaming: %s", exc, exc_info=True)
            raise

    async def _generate(self, prompt: str) -> str:
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")

        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    async def _generate_stream(self, prompt: str):
        """
        Generate streaming response from Gemini API.
        Yields text chunks as they arrive.
        """
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")

        try:
            response_stream = self.gemini_client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
            )

            for chunk in response_stream:
                if hasattr(chunk, "text"):
                    yield chunk.text

        except Exception as exc:
            logger.error(f"Error in streaming generation: {exc}", exc_info=True)
            raise RuntimeError(f"Gemini streaming API error: {exc}") from exc

    async def _handle_generation_error(self, prompt: str, error: Exception) -> str:
        """Ask the base LLM to craft a user-facing reply that acknowledges an internal error."""
        error_message = f"{type(error).__name__}: {error}"
        recovery_hint = get_error_recovery_hint(error, "chat_agent", {})

        fallback_prompt = (
            f"{prompt}\n\n"
            "SYSTEM NOTE FOR ASSISTANT:\n"
            "You attempted to respond to the user but encountered a system error.\n"
            f"Error details: {error_message}\n"
            f"Recovery hint: {recovery_hint}\n\n"
            "Compose a concise, empathetic reply to the user acknowledging the issue and, if possible, suggesting a next step."
        )

        fallback_response = await self._generate(fallback_prompt)
        return coerce_response_text(fallback_response)

    async def _generate_with_tools(
        self, prompt: str, streaming_callback=None
    ) -> tuple[str, List[str], List[Dict[str, Any]]]:
        """Generate response with tool calling support"""
        try:
            # Create agent executor
            agent_executor = self._create_agent_executor(self.tools, prompt)

            # If streaming callback provided, use streaming mode
            if streaming_callback:
                # Stream agent execution
                full_response = ""
                async for event in agent_executor.astream_events(
                    {"messages": [HumanMessage(content=prompt)]}, version="v1"
                ):
                    # Extract tokens from LLM events
                    if event.get("event") == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content"):
                            token = chunk.content
                            if token:
                                full_response += token
                                await streaming_callback(token)
                    # Notify about tool executions
                    elif event.get("event") == "on_tool_start":
                        tool_name = event.get("name", "unknown_tool")
                        await streaming_callback(
                            f"\n[Executing tool: {tool_name}]\n", is_tool_event=True
                        )

                # Get final response for extraction
                agent_response = await agent_executor.ainvoke(
                    {"messages": [HumanMessage(content=prompt)]}
                )
            else:
                # Invoke agent with the user message (non-streaming)
                agent_response = await agent_executor.ainvoke(
                    {"messages": [HumanMessage(content=prompt)]}
                )

            # Extract execution info
            execution_info = extract_agent_execution_info(agent_response)

            response_text = execution_info["response_text"]
            tools_used = execution_info["tools_used"]
            tool_artifacts = execution_info["tool_artifacts"]

            # Log parallel tool execution summary
            if tools_used:
                logger.info(
                    f"ChatAgent executed {len(tools_used)} tool(s): {', '.join(tools_used)}"
                )

            return response_text, tools_used, tool_artifacts

        except Exception as exc:
            logger.error("Error in tool calling flow: %s", exc, exc_info=True)
            raise

    def _create_agent_executor(self, tools: List[BaseTool], system_prompt: str):
        """Create agent executor with proper tool binding configuration."""
        # Configure tool calling based on settings
        tool_choice = (
            settings.tool_choice_mode
            if hasattr(settings, "tool_choice_mode")
            else "auto"
        )

        # Configure model with tool binding
        llm_with_tools = self.langchain_model.bind_tools(
            tools,
            tool_config={
                "function_calling_config": {
                    "mode": (
                        tool_choice.upper()
                        if tool_choice in ["auto", "any", "none"]
                        else "AUTO"
                    )
                }
            },
        )

        agent = create_agent(
            model=llm_with_tools, tools=tools, system_prompt=system_prompt
        )

        return agent

    async def _generate_with_vision(self, prompt: str, attachments: List[dict]) -> str:
        """Generate response with vision support using multimodal content"""
        parts = []

        parts.append(types.Part(text=prompt))

        # Add images from attachments
        for attachment in attachments:
            try:
                # Decode base64 image data
                image_data = base64.b64decode(attachment.get("data", ""))
                mime_type = attachment.get("mime", "image/jpeg")

                # Create image part from bytes
                parts.append(
                    types.Part.from_bytes(data=image_data, mime_type=mime_type)
                )
                logger.info(
                    f"Added image to vision request: {attachment.get('name', 'unknown')}"
                )
            except Exception as img_err:
                logger.error(f"Failed to process image attachment: {img_err}")

        # Generate response with multimodal content
        response = self.gemini_client.models.generate_content(
            model=self.model_name, contents=parts
        )
        return response.text if hasattr(response, "text") else str(response)

    async def cleanup(self):
        """Cleanup MCP resources"""
        if self.mcp_manager:
            try:
                await self.mcp_manager.cleanup()
                logger.info("ChatAgent MCP cleanup completed")
            except Exception as e:
                logger.error(f"Error cleaning up ChatAgent MCP resources: {e}")
