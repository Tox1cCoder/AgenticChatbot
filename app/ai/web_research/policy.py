"""Structural admission limits; search count remains model-driven."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .contracts import ResearchMode, VisualIntent

_EXPLICIT_WEB_REQUEST = re.compile(
    r"\b(?:browse|search (?:the )?web|web search|look (?:it |this )?up|"
    r"check online|find (?:the )?latest|latest news|"
    r"current (?:price|status|schedule)|today(?:'s)?)\b",
    re.IGNORECASE,
)
_CURRENT_FACT_REQUEST = re.compile(
    r"\b(?:latest|current|today|news|weather|price|schedule|law|regulation|"
    r"standard|software version|release notes)\b",
    re.IGNORECASE,
)
_PUBLIC_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def enforce_web_requirement(decision: object, message: str):
    """Prevent an explicit web request from being downgraded by a router."""

    requires_web = bool(
        _EXPLICIT_WEB_REQUEST.search(message or "")
        or _CURRENT_FACT_REQUEST.search(message or "")
        or _PUBLIC_URL.search(message or "")
    )
    if bool(getattr(decision, "requires_web", False)) or not requires_web:
        return decision
    mode = getattr(decision, "research_mode", "none")
    return decision.model_copy(
        update={"requires_web": True, "research_mode": mode if mode != "none" else "quick"}
    )


@dataclass(frozen=True)
class ResearchLimits:
    max_sources: int
    max_page_opens: int
    max_model_images: int

    @classmethod
    def for_mode(
        cls,
        mode: ResearchMode,
        *,
        visual_intent: VisualIntent = "none",
    ) -> ResearchLimits:
        value = {
            "none": cls(max_sources=0, max_page_opens=0, max_model_images=0),
            "quick": cls(max_sources=5, max_page_opens=2, max_model_images=4),
            "agentic": cls(max_sources=8, max_page_opens=4, max_model_images=4),
        }[mode]
        if visual_intent == "gallery" and mode != "none":
            return replace(value, max_model_images=6)
        if visual_intent == "none":
            return replace(value, max_model_images=0)
        return value


__all__ = ["ResearchLimits", "enforce_web_requirement"]
