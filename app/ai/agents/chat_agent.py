import base64
import contextlib
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from google.genai import types
from langchain_core.messages import HumanMessage, SystemMessage

from ...interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ...usage import begin_usage_operation, bind_usage_context, current_usage_context
from ...usage.types import UsageOperation
from ..agent_config import build_gemini_generate_config
from ..image_context import build_multimodal_content, normalize_image_attachment
from ..prompts import CHAT_SYSTEM_PROMPT, build_chat_prompt
from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..utils import coerce_response_text
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...usage.recorder import ModelUsageRecorder


class ChatAgent(BaseAgent):
    def __init__(
        self,
        runtime_model_resolver: IRuntimeModelResolver | None = None,
        recorder: "ModelUsageRecorder | None" = None,
    ):
        super().__init__(
            agent_config_key="chat",
            runtime_model_resolver=runtime_model_resolver,
            recorder=recorder,
        )

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "chat_agent"

    def _get_base_system_prompt(self) -> str:
        return CHAT_SYSTEM_PROMPT

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: str | None = None,
    ) -> AgentResponse:
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")
        model_request = message.metadata.get("model_request")
        request_user_id = message.metadata.get("user_id")
        message_content = message.content or ""
        request_device_id = message.metadata.get("device_id")

        prompt = build_chat_prompt(message_content, conversation_history, persona=persona)

        attachments = (
            message.attachments if hasattr(message, "attachments") and message.attachments else None
        )

        try:
            if attachments:
                response_text, runtime_metadata = await self._generate_with_vision(
                    prompt,
                    attachments,
                    conversation_id=conversation_id,
                    user_id=request_user_id,
                    device_id=request_device_id,
                    model_request=model_request,
                )

                return AgentResponse(
                    agent_type=AgentType.CHAT,
                    agent_id="chat_agent",
                    message=AgentMessage(
                        role=MessageRole.ASSISTANT,
                        content=coerce_response_text(response_text),
                    ),
                    metadata={
                        **runtime_metadata,
                        "conversation_id": conversation_id,
                        "persona_used": persona,
                        "has_images": True,
                    },
                )

            return await self.invoke_model(message, conversation_id)
        except Exception as exc:
            logger.error("Error while processing message in ChatAgent: %s", exc, exc_info=True)
            return self._build_error_response(
                message="I encountered an error processing your request.",
                conversation_id=conversation_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def invoke_model(
        self,
        message: AgentMessage,
        conversation_id: str | None = None,
    ) -> AgentResponse:
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")
        message_content = message.content or ""
        request_user_id = message.metadata.get("user_id")
        request_device_id = message.metadata.get("device_id")
        model_request = message.metadata.get("model_request")

        prompt = build_chat_prompt(
            message_content,
            conversation_history,
            persona=persona,
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
        response.metadata["persona_used"] = persona
        return response

    async def _generate(self, prompt: str) -> str:
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")

        try:
            system_prompt = self._get_full_system_prompt()
            generation_config = build_gemini_generate_config(
                model_name=self.model_name,
                include_thinking=True,
                system_instruction=system_prompt,
            )
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config,
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    _MAX_VISION_FALLBACK_ATTEMPTS: int = 3

    async def _generate_with_vision(
        self,
        prompt: str,
        attachments: list[dict],
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        model_request: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        runtime_config = self._resolve_runtime_model_config(user_id, model_request)

        if not runtime_config.capabilities.get("supports_vision", False):
            fallback_runtime = self._create_fallback_runtime_config(
                runtime_config.fallback_config,
                reason="vision_not_supported",
                from_provider=runtime_config.provider,
                inherited_warnings=runtime_config.warnings,
            )
            if fallback_runtime:
                runtime_config = fallback_runtime

        attempted_providers: set[str] = set()
        last_error: Exception | None = None

        # One usage operation spans the whole fallback loop, so retries and
        # provider fallbacks for this single vision request share an operation
        # id while each real provider attempt is a distinct recorded attempt.
        with self._begin_vision_usage() as operation:
            for _attempt in range(self._MAX_VISION_FALLBACK_ATTEMPTS):
                try:
                    if runtime_config.provider == "openai":
                        llm, _ = self._create_langchain_model_from_runtime(
                            runtime_config,
                            user_id=user_id,
                            enable_reasoning_summary=False,
                        )

                        content = build_multimodal_content(prompt, attachments)

                        response = await self._ainvoke_with_retries(
                            llm,
                            [
                                SystemMessage(
                                    content=self._get_full_system_prompt(
                                        user_id=user_id,
                                        device_id=device_id,
                                    )
                                ),
                                HumanMessage(content=content),
                            ],
                            operation=operation,
                            provider=runtime_config.provider,
                            model=runtime_config.model,
                        )
                        metadata = {"conversation_id": conversation_id}
                        self._apply_runtime_metadata(metadata, runtime_config)
                        return coerce_response_text(response.content), metadata

                    parts = [types.Part(text=prompt)]
                    for attachment in attachments:
                        try:
                            normalized = normalize_image_attachment(attachment)
                            if normalized is None:
                                continue
                            image_url = normalized["url"]
                            if not image_url.startswith("data:"):
                                continue
                            raw_data = image_url.partition(",")[2]
                            if not raw_data:
                                continue

                            image_data = base64.b64decode(raw_data)
                            parts.append(
                                types.Part.from_bytes(
                                    data=image_data,
                                    mime_type=normalized["mime"],
                                )
                            )
                        except Exception as img_err:
                            logger.error("Failed to process image attachment: %s", img_err)

                    gemini_client = self._create_gemini_client_from_runtime(runtime_config)
                    if gemini_client is None:
                        raise RuntimeError(
                            "Gemini client is not available for multimodal generation"
                        )

                    system_prompt = self._get_full_system_prompt(
                        user_id=user_id,
                        device_id=device_id,
                    )
                    generation_config = build_gemini_generate_config(
                        model_name=runtime_config.model,
                        include_thinking=True,
                        system_instruction=system_prompt,
                    )
                    response = await self._invoke_vision_gemini(
                        gemini_client,
                        model=runtime_config.model,
                        parts=parts,
                        config=generation_config,
                        operation=operation,
                    )

                    metadata = {"conversation_id": conversation_id}
                    self._apply_runtime_metadata(metadata, runtime_config)
                    return (
                        response.text if hasattr(response, "text") else str(response),
                        metadata,
                    )
                except Exception as exc:
                    last_error = exc
                    attempted_providers.add(runtime_config.provider)

                    fallback_runtime = self._create_fallback_runtime_config(
                        runtime_config.fallback_config,
                        reason="provider_error",
                        from_provider=runtime_config.provider,
                        inherited_warnings=runtime_config.warnings,
                    )
                    if (
                        not fallback_runtime
                        or fallback_runtime.provider == runtime_config.provider
                        or fallback_runtime.provider in attempted_providers
                    ):
                        raise
                    runtime_config = fallback_runtime

        # Exhausted all fallback attempts
        if last_error:
            raise last_error
        raise RuntimeError("Vision generation failed after exhausting all fallback attempts")

    @contextlib.contextmanager
    def _begin_vision_usage(self) -> Iterator[UsageOperation | None]:
        """Bind a ``vision`` operation for one vision request (spans fallbacks).

        Yields ``None`` (binding nothing) when no recorder is configured so the
        recorder-less path is byte-for-byte unchanged.
        """
        if self.recorder is None:
            yield None
            return
        context = current_usage_context().child(operation="vision", agent_id=self.agent_id)
        with bind_usage_context(context), begin_usage_operation() as operation:
            yield operation

    async def _invoke_vision_gemini(
        self,
        gemini_client: Any,
        *,
        model: str,
        parts: list[Any],
        config: Any,
        operation: UsageOperation | None,
    ) -> Any:
        """Invoke the Gemini vision call once, recording the attempt when enabled.

        The SDK call is synchronous; the existing behavior (blocking on the
        event loop) is retained, only wrapped so the attempt is recorded. The
        async recorder persists off the loop via ``asyncio.to_thread``.
        """

        def _call() -> Any:
            return gemini_client.models.generate_content(
                model=model,
                contents=parts,
                config=config,
            )

        if self.recorder is None or operation is None:
            return _call()

        async def _acall() -> Any:
            return _call()

        return await self.recorder.record_one_async_attempt(
            call=_acall,
            provider="gemini",
            model=model,
            operation=operation,
        )

    async def cleanup(self):
        await super().cleanup()
