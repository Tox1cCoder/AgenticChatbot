import base64
import logging
from typing import Optional, List, Dict, Any, AsyncIterator

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_image_generator_prompt
from ...core.config import settings
from ..mcp_integration import MCPManager
from ..utils import extract_agent_execution_info

logger = logging.getLogger(__name__)


class ImageGeneratorAgent:
    """Agent responsible for generating images."""

    def __init__(self):
        self.model_name = settings.image_generator_model
        self.default_aspect_ratio = settings.image_generator_default_aspect_ratio
        self.max_images = max(1, settings.image_generator_max_images)
        self.enabled = settings.enable_image_generation
        self.gemini_client: Optional[genai.Client] = None
        self.langchain_model = None
        self.mcp_manager = None
        self.tools = []
        self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Gemini client"""
        if not self.enabled:
            return

        api_key = settings.gemini_api_key

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        self.gemini_client = genai.Client(api_key=api_key)

        self.langchain_model = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", google_api_key=api_key, temperature=0.8
        )

    async def _init_tools(self):
        """Initialize MCP manager and load all available tools"""
        if self.mcp_manager is None:
            try:
                self.mcp_manager = MCPManager()
                await self.mcp_manager.initialize()

                self.tools = await self.mcp_manager.get_tools()

                server_status = self.mcp_manager.get_servers_status()
                active_servers = [
                    name
                    for name, status in server_status.items()
                    if status.get("enabled")
                ]
                logger.info(
                    "Loaded %d MCP tools for ImageGeneratorAgent from %d servers",
                    len(self.tools),
                    len(active_servers),
                )

            except Exception as e:
                logger.error(
                    f"Failed to initialize MCP manager for ImageGeneratorAgent: {e}",
                    exc_info=True,
                )
                self.tools = []

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

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """Generate an image for the provided prompt."""

        if not self.enabled:
            logger.warning("Image generation requested but disabled")
            return self._build_error_response(
                "Image generation is currently disabled. Please contact the administrator.",
                conversation_id,
            )

        if not self.gemini_client:
            logger.error("Gemini client not initialized for image generation")
            return self._build_error_response(
                "Unable to generate images because the Gemini client is not configured.",
                conversation_id,
            )

        # Initialize tools if not done yet
        if self.mcp_manager is None:
            await self._init_tools()

        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        enhanced_content = message.content
        tools_used = []
        tool_artifacts = []

        if self.tools:
            try:
                enhanced_content, tools_used, tool_artifacts = (
                    await self._enhance_prompt_with_tools(
                        message.content, conversation_history, persona
                    )
                )
            except Exception as e:
                enhanced_content = message.content

        prompt = build_image_generator_prompt(
            enhanced_content,
            conversation_history,
            persona=persona,
        )

        try:
            images, narrative = await self._generate_images(prompt, enhanced_content)
        except Exception as err:
            logger.error("Image generation failed: %s", err, exc_info=True)
            return self._build_error_response(
                f"An error occurred while generating the image: {err}",
                conversation_id,
            )

        if not images:
            logger.warning("No images generated for prompt: %s", prompt)
            return self._build_error_response(
                "No image was generated for your request. Please try refining your description.",
                conversation_id,
            )

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT,
            content=narrative
            or "Here is the image I created based on your description.",
        )

        response_metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "persona_used": persona,
            "images": images,
            "tools_available": len(self.tools),
        }

        # Add tool usage metadata if tools were used
        if tools_used:
            response_metadata["tools_used"] = tools_used
            response_metadata["tool_calls_count"] = len(tools_used)
        if tool_artifacts:
            response_metadata["tool_artifacts"] = tool_artifacts

        return AgentResponse(
            agent_type=AgentType.IMAGE_GENERATOR,
            agent_id="image_generator_agent",
            message=response_message,
            metadata=response_metadata,
            tool_artifacts=tool_artifacts if tool_artifacts else None,
        )

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream message processing for image generation.
        Yields status updates and final result.
        """
        if not self.enabled:
            logger.warning("Image generation requested but disabled")
            error_msg = "Image generation is currently disabled. Please contact the administrator."
            yield {"type": "token", "content": error_msg}
            yield {
                "type": "complete",
                "response": self._build_error_response(error_msg, conversation_id),
            }
            return

        if not self.gemini_client:
            logger.error("Gemini client not initialized for image generation")
            error_msg = (
                "Unable to generate images because the Gemini client is not configured."
            )
            yield {"type": "token", "content": error_msg}
            yield {
                "type": "complete",
                "response": self._build_error_response(error_msg, conversation_id),
            }
            return

        # Initialize tools if not done yet
        if self.mcp_manager is None:
            await self._init_tools()

        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        enhanced_content = message.content
        tools_used = []
        tool_artifacts = []

        # Yield status update for prompt enhancement
        if self.tools:
            yield {"type": "token", "content": "Analyzing prompt..."}
            try:
                enhanced_content, tools_used, tool_artifacts = (
                    await self._enhance_prompt_with_tools(
                        message.content, conversation_history, persona
                    )
                )
            except Exception as e:
                enhanced_content = message.content

        prompt = build_image_generator_prompt(
            enhanced_content,
            conversation_history,
            persona=persona,
        )

        # Yield status update for image generation
        yield {"type": "token", "content": " Generating image..."}

        try:
            images, narrative = await self._generate_images(prompt, enhanced_content)
        except Exception as err:
            logger.error("Image generation failed: %s", err, exc_info=True)
            error_msg = f" Error: {err}"
            yield {"type": "token", "content": error_msg}
            yield {
                "type": "complete",
                "response": self._build_error_response(
                    f"An error occurred while generating the image: {err}",
                    conversation_id,
                ),
            }
            return

        if not images:
            logger.warning("No images generated for prompt: %s", prompt)
            error_msg = " No image was generated."
            yield {"type": "token", "content": error_msg}
            yield {
                "type": "complete",
                "response": self._build_error_response(
                    "No image was generated for your request. Please try refining your description.",
                    conversation_id,
                ),
            }
            return

        # Yield success status and narrative
        success_msg = f" Done! {narrative or 'Here is the image I created based on your description.'}"
        yield {"type": "token", "content": success_msg}

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT,
            content=narrative
            or "Here is the image I created based on your description.",
        )

        response_metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "persona_used": persona,
            "images": images,
            "tools_available": len(self.tools),
        }

        # Add tool usage metadata if tools were used
        if tools_used:
            response_metadata["tools_used"] = tools_used
            response_metadata["tool_calls_count"] = len(tools_used)
        if tool_artifacts:
            response_metadata["tool_artifacts"] = tool_artifacts

        # Yield complete event
        yield {
            "type": "complete",
            "response": AgentResponse(
                agent_type=AgentType.IMAGE_GENERATOR,
                agent_id="image_generator_agent",
                message=response_message,
                metadata=response_metadata,
                tool_artifacts=tool_artifacts if tool_artifacts else None,
            ),
        }

    async def _enhance_prompt_with_tools(
        self, original_prompt: str, conversation_history: List, persona: Optional[str]
    ) -> tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Enhance the image prompt with contextual information from tools.
        """
        if not self.tools:
            return original_prompt, [], []

        try:
            enhancement_system_prompt = f"""You are analyzing a user's image generation request to determine if external tools can provide useful context.

User request: {original_prompt}

Available tools: {', '.join(tool.name for tool in self.tools)}

If tools can provide useful context (e.g., current date/time for "today", calculations for "show me 25% of 100 items"), use them and return an enhanced prompt with the additional information.

If the request is self-contained (e.g., "draw a cat", "create a sunset scene"), return the original prompt unchanged.

Provide ONLY the enhanced prompt text, nothing else."""

            # Create agent executor
            agent_executor = self._create_agent_executor(
                self.tools, enhancement_system_prompt
            )

            # Invoke agent
            agent_response = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=original_prompt)]}
            )

            # Extract execution info
            execution_info = extract_agent_execution_info(agent_response)

            enhanced_prompt = execution_info["response_text"]
            tools_used = execution_info["tools_used"]
            tool_artifacts = execution_info["tool_artifacts"]

            return enhanced_prompt, tools_used, tool_artifacts

        except Exception as exc:
            logger.warning("Failed to enhance prompt with tools: %s", exc)
            return original_prompt, [], []

    async def _generate_images(
        self, prepared_prompt: str, original_prompt: str
    ) -> tuple[List[dict], str]:
        if not self.gemini_client:
            return [], ""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prepared_prompt)],
            )
        ]

        config_kwargs = {"response_modalities": ["IMAGE", "TEXT"]}
        image_config_cls = getattr(types, "ImageGenerationConfig", None)
        if image_config_cls is not None:
            config_kwargs["image_generation_config"] = image_config_cls(
                number_of_images=self.max_images,
                aspect_ratio=self.default_aspect_ratio,
            )

        generate_config = types.GenerateContentConfig(**config_kwargs)

        images: List[dict] = []
        narrative_parts: List[str] = []

        stream = self.gemini_client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=generate_config,
        )

        for chunk in stream:
            if not getattr(chunk, "candidates", None):
                continue

            candidate = chunk.candidates[0]
            if not candidate or not getattr(candidate, "content", None):
                continue

            for part in getattr(candidate.content, "parts", []) or []:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    encoded = self._encode_image(inline_data.data)
                    if encoded:
                        images.append(
                            {
                                "data": encoded,
                                "mime": inline_data.mime_type or "image/png",
                                "prompt": original_prompt,
                                "model": self.model_name,
                                "aspect_ratio": self.default_aspect_ratio,
                            }
                        )
                        if len(images) >= self.max_images:
                            break

                text_segment = getattr(part, "text", None)
                if text_segment:
                    narrative_parts.append(text_segment)

            if len(images) >= self.max_images:
                break

            top_level_text = getattr(candidate, "text", None)
            if top_level_text:
                narrative_parts.append(top_level_text)

        narrative = " ".join(segment.strip() for segment in narrative_parts if segment)
        return images, narrative.strip()

    @staticmethod
    def _encode_image(raw_data) -> Optional[str]:
        """Convert inline image data into base64 string."""
        if raw_data is None:
            return None

        try:
            if isinstance(raw_data, (bytes, bytearray)):
                return base64.b64encode(raw_data).decode("utf-8")
            if isinstance(raw_data, str):
                return raw_data
            if isinstance(raw_data, memoryview):
                return base64.b64encode(raw_data.tobytes()).decode("utf-8")
            return base64.b64encode(bytes(raw_data)).decode("utf-8")
        except Exception as err:
            logger.error("Failed to encode image data: %s", err, exc_info=True)
            return None

    def _build_error_response(
        self, message: str, conversation_id: Optional[str]
    ) -> AgentResponse:
        """Create standardized error responses."""
        response_message = AgentMessage(
            role=MessageRole.ASSISTANT,
            content=message,
        )

        return AgentResponse(
            agent_type=AgentType.IMAGE_GENERATOR,
            agent_id="image_generator_agent",
            message=response_message,
            metadata={
                "model": self.model_name,
                "conversation_id": conversation_id,
                "images": [],
                "error": True,
            },
            error=message,
        )

    async def cleanup(self):
        """Cleanup MCP resources"""
        if self.mcp_manager:
            try:
                await self.mcp_manager.cleanup()
                logger.info("ImageGeneratorAgent MCP cleanup completed")
            except Exception as e:
                logger.error(
                    f"Error cleaning up ImageGeneratorAgent MCP resources: {e}"
                )
