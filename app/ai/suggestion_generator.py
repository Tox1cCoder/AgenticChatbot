import json
import logging
from typing import List, Optional

from google import genai

from ..core.config import settings

SUGGESTION_PROMPT = """Based on this conversation exchange, generate follow-up questions the user might want to ask next.

User asked: {user_query}

Assistant responded: {response_summary}

Rules:
- Generate 0-3 concise follow-up questions
- Only suggest if genuinely useful for continuing the conversation
- Keep each question under 60 characters
- Make them natural and conversational
- Return EMPTY array [] if no good suggestions (e.g., for greetings, simple acknowledgments)
- Questions should explore different aspects or go deeper into the topic

Return ONLY a JSON array of strings, nothing else. Examples:
["Tell me more about X", "How does this compare to Y?"]
[]
["What are the benefits?", "Can you give an example?", "How do I get started?"]"""


class SuggestionGenerator:
    """Generates follow-up question suggestions using Gemini."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "gemini-2.0-flash"
        self.client: Optional[genai.Client] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Gemini client."""
        try:
            api_key = settings.gemini_api_key
            if not api_key:
                return

            if api_key.startswith("GEMINI_API_KEY="):
                api_key = api_key.split("=", 1)[1].strip()

            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            self.client = None

    async def generate_suggestions(
        self,
        user_query: str,
        response_content: str,
        max_suggestions: int = 3,
    ) -> List[str]:
        """
        Generate follow-up question suggestions.

        Args:
            user_query: The user's original question
            response_content: The assistant's response
            max_suggestions: Maximum number of suggestions (default 3)

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

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 256,
                },
            )

            if not response or not hasattr(response, "text"):
                return []

            # Parse JSON response
            text = response.text.strip()
            
            # Handle potential markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
            
            suggestions = json.loads(text)

            if not isinstance(suggestions, list):
                return []

            # Validate and clean suggestions
            valid_suggestions = []
            for s in suggestions[:max_suggestions]:
                if isinstance(s, str) and s.strip():
                    # Clean and validate length
                    clean = s.strip()
                    if len(clean) <= 100:  # Max 100 chars per suggestion
                        valid_suggestions.append(clean)

            return valid_suggestions

        except json.JSONDecodeError as e:
            return []
        except Exception as e:
            return []


# Global singleton instance
_suggestion_generator: Optional[SuggestionGenerator] = None


def get_suggestion_generator() -> SuggestionGenerator:
    """Get or create the global suggestion generator instance."""
    global _suggestion_generator
    if _suggestion_generator is None:
        _suggestion_generator = SuggestionGenerator()
    return _suggestion_generator


async def generate_follow_up_suggestions(
    user_query: str,
    response_content: str,
) -> List[str]:
    """
    Convenience function to generate follow-up suggestions.

    Args:
        user_query: The user's original question
        response_content: The assistant's response

    Returns:
        List of 0-3 suggestion strings
    """
    generator = get_suggestion_generator()
    return await generator.generate_suggestions(user_query, response_content)
