"""
Canvas Agent — generates interactive, renderable artifacts (HTML/CSS/JS, SVG, React CDN).

Inspired by Gemini Canvas: the agent produces a self-contained code artifact that the
frontend renders inside a sandboxed preview panel alongside the conversation.

Artifact lifecycle:
  1. User asks for something renderable ("build a calculator", "make a bar chart", …).
  2. CanvasAgent generates a complete self-contained HTML document (with inline CSS/JS).
  3. The response metadata carries a `canvas_artifact` dict that the frontend uses to
     populate the canvas preview without separate API calls.
  4. For follow-up edits ("change the background to blue", "add a reset button") the agent
     receives the previous artifact content through conversation history and returns an
     updated version — the frontend replaces the preview in place.

canvas_artifact shape (stored in AgentResponse.metadata["canvas_artifact"]):
  {
    "content":  "<full self-contained HTML string>",
    "language": "html" | "svg" | "react",
    "title":    "Short human-readable title",
    "editable": true
  }
"""

import logging
import re
from typing import Optional, List, Dict, Any, AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage as LCHumanMessage

from .base_agent import BaseAgent
from ..schemas import AgentMessage, AgentResponse, AgentType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_CANVAS_SYSTEM_PROMPT = """\
You are an expert web developer and creative coder specialised in generating \
interactive, self-contained browser artifacts.

## YOUR TASK
When the user requests an interactive piece (a game, calculator, visualisation, \
form, animation, data chart, SVG graphic, etc.) you must:

1. Write a **brief** conversational description of what you built (1-3 sentences, \
no headings, no bullet points).
2. Output the **complete, self-contained artifact** inside a single fenced code block.

## CODE QUALITY RULES
- The artifact MUST be runnable on its own inside a sandboxed iframe — zero external \
dependencies unless loaded via a public CDN.
- Prefer vanilla HTML + inline CSS + inline JavaScript for maximum compatibility.
- For React components, use the React/ReactDOM UMD CDN scripts and render into \
`<div id="root">`.
- For SVG-only artifacts (icons, logos, static illustrations) output a fenced \
`\`\`\`svg` block instead of `\`\`\`html`.
- Make the artifact visually polished: comfortable fonts, sensible colours, \
responsive layout (flexbox / grid where appropriate).
- All JavaScript must be contained inside a `<script>` tag at the bottom of `<body>`.

## CODE BLOCK FORMAT
\`\`\`html
<!DOCTYPE html>
... complete document ...
\`\`\`

## EDITING / UPDATING
If the conversation history already contains a canvas artifact, you are editing it.
Re-output the **complete updated document** — do NOT produce a diff or partial snippet.

## WHAT NOT TO DO
- Do NOT split code across multiple code blocks.
- Do NOT use Markdown headings or bullet points in your description.
- Do NOT add any text after the closing code fence.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"```(?P<lang>html|svg|jsx?|react|tsx?|javascript|js|css)?\s*\n"
    r"(?P<code>[\s\S]*?)"
    r"```",
    re.IGNORECASE,
)

_TITLE_RE = re.compile(r"<title>(?P<t>[^<]{1,120})</title>", re.IGNORECASE)


def _extract_artifact(text: str) -> Optional[Dict[str, Any]]:
    """
    Pull the first fenced code block from the LLM output and classify it.

    Returns a ``canvas_artifact`` dict, or ``None`` when no code block is found.
    """
    match = _CODE_FENCE_RE.search(text)
    if not match:
        return None

    lang_hint = (match.group("lang") or "html").lower()
    code = match.group("code").strip()

    if not code:
        return None

    # Normalise language label
    if lang_hint in {"jsx", "tsx", "react"}:
        language = "react"
    elif lang_hint == "svg":
        language = "svg"
    else:
        language = "html"

    # Try to derive a title from <title> tag; fall back to generic label
    title_match = _TITLE_RE.search(code)
    title = title_match.group("t").strip() if title_match else "Canvas"

    return {
        "content": code,
        "language": language,
        "title": title,
        "editable": True,
    }


def _strip_code_block(text: str) -> str:
    """Return the text with the code fence removed (the conversational description)."""
    return _CODE_FENCE_RE.sub("", text).strip()


def _extract_previous_artifact(conversation_history: List[Any]) -> Optional[str]:
    """
    Walk conversation history in reverse to find the most recent canvas artifact.
    Returns the raw HTML/code string so the agent can use it for iterative edits.
    """
    for msg in reversed(conversation_history):
        metadata = getattr(msg, "metadata", None) or {}
        artifact = metadata.get("canvas_artifact")
        if artifact and isinstance(artifact, dict) and artifact.get("content"):
            return artifact["content"]
    return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CanvasAgent(BaseAgent):
    """Generates and iteratively edits self-contained canvas artifacts."""

    def __init__(self) -> None:
        super().__init__(agent_config_key="canvas")

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CANVAS

    @property
    def agent_id(self) -> str:
        return "canvas_agent"

    def _get_base_system_prompt(self) -> str:
        return _CANVAS_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Core generation — wraps the base invoke_model_with_history
    # ------------------------------------------------------------------

    async def invoke_model_with_history(
        self,
        messages: List[BaseMessage],
        conversation_history: List[Any],
        persona: Optional[str],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        model_request: Optional[Dict[str, Any]] = None,
        history_summary: Optional[str] = None,
        **system_prompt_kwargs: Any,
    ) -> AgentResponse:
        # Let the base class handle LLM invocation (tool calls, provider switching, …).
        response = await super().invoke_model_with_history(
            messages=messages,
            conversation_history=conversation_history,
            persona=persona,
            conversation_id=conversation_id,
            user_id=user_id,
            model_request=model_request,
            history_summary=history_summary,
            **system_prompt_kwargs,
        )

        # If the base class returned a tool-call or error, pass it straight through.
        if response.message.tool_calls or response.error:
            return response

        raw_text = (response.message.content or "").strip()
        if not raw_text:
            return response

        # Extract the code artifact
        artifact = _extract_artifact(raw_text)
        description = _strip_code_block(raw_text)

        if artifact:
            if not response.metadata:
                response.metadata = {}
            response.metadata["canvas_artifact"] = artifact
            # Replace the raw LLM output (which contains the full code block) with
            # a clean conversational description for the chat thread.
            response.message.content = description or (
                f"Here's your **{artifact['title']}**! "
                "You can view and interact with it in the canvas panel. "
                "Let me know if you'd like any changes."
            )
        else:
            # The LLM didn't produce a fenced code block — return as-is so the user
            # sees the response and can ask again.
            logger.warning(
                "CanvasAgent: no fenced code block found in LLM output "
                "(conversation_id=%s)",
                conversation_id,
            )

        return response

    # ------------------------------------------------------------------
    # Simple (non-history) entrypoint kept for compatibility with process_message
    # ------------------------------------------------------------------

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        return await self.invoke_model_with_history(
            messages=[LCHumanMessage(content=message.content or "")],
            conversation_history=[],
            persona=message.metadata.get("persona"),
            conversation_id=conversation_id,
        )

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        result = await self.process_message(message, conversation_id)
        if result.error:
            yield {"type": "error", "error": result.error}
            return
        yield {"type": "complete", "response": result}

    async def cleanup(self) -> None:
        await super().cleanup()
