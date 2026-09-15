"""Structural admission limits; search count remains model-driven."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .contracts import ResearchMode, VisualIntent


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


__all__ = ["ResearchLimits"]
