import asyncio
import hashlib
import json
from typing import TYPE_CHECKING

from google import genai

from ..usage import UsageContext, begin_usage_operation, bind_usage_context
from ..usage.types import UsageOperation
from .agent_config import AGENT_CONFIG, create_gemini_client

if TYPE_CHECKING:
    from ..usage.recorder import ModelUsageRecorder

SUGGESTION_PROMPT = """Generate useful follow-up questions for this conversation exchange.

User asked: {user_query}

Assistant responded: {response_summary}

Rules:
- Generate 0-3 COMPLETE follow-up questions (never truncate or use "...")
- Each question must be a full, grammatically correct sentence
- Keep questions concise (maximum 30 words per question)
- Match the language of the conversation (if user speaks Vietnamese, respond in Vietnamese)
- Only suggest if genuinely useful for continuing the conversation
- Return EMPTY array [] if no good suggestions (e.g., for greetings, simple acknowledgments)
- Questions should explore different aspects or go deeper into the topic

Return ONLY a JSON array of strings, nothing else. Examples:
["Tell me more about X", "How does this compare to Y?"]
[]
["What are the benefits?", "Can you give an example?", "How do I get started?"]
["Bạn có thể giải thích thêm không?", "Có ví dụ nào khác không?"]"""


class SuggestionGenerator:
    """Generates follow-up question suggestions using Gemini."""

    def __init__(self, model_name: str | None = None):
        config = AGENT_CONFIG["suggestion"]
        self.model_name = model_name or config["model"]
        self.client: genai.Client | None = None
        self._suggestion_cache: dict = {}
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Gemini client."""
        try:
            self.client = create_gemini_client()
        except Exception:
            self.client = None

    def _create_cache_key(self, user_query: str, response_content: str) -> str:
        """Create a hash-based cache key for query and response."""
        # Truncate as done in generate_suggestions for consistency
        truncated_query = user_query[:200]
        truncated_response = response_content[:500]

        # Create deterministic hash
        content = f"{truncated_query}||{truncated_response}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_suggestions(
        self,
        cache_key: str,
        prompt: str,
        recorder: "ModelUsageRecorder | None" = None,
        operation: UsageOperation | None = None,
    ) -> list[str] | None:
        """Internal cached method for LLM calls.

        Runs on a worker thread (dispatched via ``asyncio.to_thread``), so the
        provider call is recorded with the recorder's sync variant. Cache hits
        return before any provider call, so they record zero events.
        """
        _key = (cache_key, prompt)
        if _key in self._suggestion_cache:
            return self._suggestion_cache[_key]

        if not self.client:
            return None

        def _generate():
            return self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 1,
                    "max_output_tokens": 512,
                },
            )

        try:
            if recorder is not None and operation is not None:
                response = recorder.record_one_sync_attempt(
                    call=_generate,
                    provider="gemini",
                    model=self.model_name,
                    operation=operation,
                )
            else:
                response = _generate()

            if not response or not hasattr(response, "text"):
                return None

            # Parse JSON response
            text = response.text.strip()

            # Handle potential markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else ""

            suggestions = json.loads(text)

            if not isinstance(suggestions, list):
                return None

            if len(self._suggestion_cache) >= 100:
                self._suggestion_cache.pop(next(iter(self._suggestion_cache)))
            self._suggestion_cache[_key] = suggestions
            return suggestions

        except (json.JSONDecodeError, Exception):
            return None

    async def generate_suggestions(
        self,
        user_query: str,
        response_content: str,
        max_suggestions: int = 3,
        *,
        usage_context: UsageContext | None = None,
        recorder: "ModelUsageRecorder | None" = None,
    ) -> list[str]:
        """
        Generate follow-up question suggestions.

        Args:
            user_query: The user's original question
            response_content: The assistant's response
            max_suggestions: Maximum number of suggestions (default 3)
            usage_context: Attribution for the provider call, if tracking is on
            recorder: Recorder used to attribute the provider call

        Returns:
            List of 0-3 suggestion strings
        """
        if not self.client:
            return []

        try:
            # Truncate response if too long (keep first 500 chars for context)
            response_summary = response_content[:500]
            if len(response_content) > 500:
                response_summary += "..."

            prompt = SUGGESTION_PROMPT.format(
                user_query=user_query[:200],  # Truncate long queries
                response_summary=response_summary,
            )

            # Create cache key and get cached result
            cache_key = self._create_cache_key(user_query, response_content)
            if recorder is not None and usage_context is not None:
                # The bound context propagates into the worker thread, so the
                # sync recorder inside ``_get_cached_suggestions`` attributes the
                # attempt correctly; cache hits still record nothing.
                with bind_usage_context(usage_context), begin_usage_operation() as operation:
                    suggestions = await asyncio.to_thread(
                        self._get_cached_suggestions,
                        cache_key,
                        prompt,
                        recorder,
                        operation,
                    )
            else:
                suggestions = await asyncio.to_thread(
                    self._get_cached_suggestions,
                    cache_key,
                    prompt,
                )

            if suggestions is None:
                return []

            # Validate and clean suggestions
            valid_suggestions = []
            for s in suggestions[:max_suggestions]:
                if isinstance(s, str) and s.strip():
                    # Clean and validate length
                    clean = s.strip()
                    if len(clean) <= 150:  # Max 150 chars per suggestion
                        valid_suggestions.append(clean)

            return valid_suggestions

        except Exception:
            return []


# Global singleton instance
_suggestion_generator: SuggestionGenerator | None = None


def get_suggestion_generator() -> SuggestionGenerator:
    """Get or create the global suggestion generator instance."""
    global _suggestion_generator
    if _suggestion_generator is None:
        _suggestion_generator = SuggestionGenerator()
    return _suggestion_generator


async def generate_follow_up_suggestions(
    user_query: str,
    response_content: str,
    *,
    usage_context: UsageContext | None = None,
    recorder: "ModelUsageRecorder | None" = None,
) -> list[str]:
    """
    Convenience function to generate follow-up suggestions.

    Args:
        user_query: The user's original question
        response_content: The assistant's response
        usage_context: Attribution for the provider call, if tracking is on
        recorder: Recorder used to attribute the provider call

    Returns:
        List of 0-3 suggestion strings
    """
    generator = get_suggestion_generator()
    return await generator.generate_suggestions(
        user_query,
        response_content,
        usage_context=usage_context,
        recorder=recorder,
    )
