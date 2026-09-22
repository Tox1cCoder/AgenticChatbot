"""Structural admission limits; search count remains model-driven."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import ResearchMode, VisualIntent, image_capacity, source_capacity

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
    #: Text sources the search provider is asked for and the registry admits.
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
        page_opens = {"none": 0, "quick": 2, "agentic": 4}[mode]
        return cls(
            max_sources=source_capacity(mode, "none"),
            max_page_opens=page_opens,
            max_model_images=image_capacity(mode, visual_intent),
        )


__all__ = ["ResearchLimits", "enforce_web_requirement"]
