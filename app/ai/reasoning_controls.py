"""Model-aware, provider-native reasoning controls.

Provider model-list endpoints do not expose the accepted reasoning-level enum.
This module enriches their coarse capability flags with a deliberately small
registry sourced from the providers' model documentation. Unknown models remain
usable, but only with the provider default (no explicit parameter).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReasoningControl:
    supported: bool
    parameter_name: str | None
    display_label: str
    levels: tuple[str, ...] = ()
    default_level: str | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["levels"] = list(self.levels)
        return value


# Gemini 2.5 and the compatibility aliases below use the legacy numeric
# transport even though the UI exposes Google's documented named effort levels.
_GEMINI_BUDGET_RULES = (
    (
        ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"),
        ("low", "medium", "high"),
        None,
    ),
    (
        ("gemini-flash-latest", "gemini-flash-lite-latest"),
        ("low", "medium", "high"),
        None,
    ),
)


# Ordered most-specific-first so Pro variants never inherit a broader family.
_GEMINI_RULES = (
    (("gemini-3.6-flash",), ("minimal", "low", "medium", "high"), "medium"),
    (("gemini-3.5-flash-lite",), ("minimal", "low", "medium", "high"), "minimal"),
    (("gemini-3.5-flash",), ("minimal", "low", "medium", "high"), "medium"),
    (("gemini-3.1-pro-preview",), ("low", "medium", "high"), "high"),
    (("gemini-3.1-flash-lite-image",), ("minimal", "high"), "minimal"),
    (("gemini-3-flash-preview",), ("minimal", "low", "medium", "high"), "high"),
    (("gemini-3-pro-preview",), ("low", "high"), "high"),
    (("gemini-pro-latest",), ("low", "medium", "high"), None),
)

_OPENAI_RULES = (
    (("o1-pro",), ("high",), "high"),
    (("o1", "o3", "o4-mini"), ("low", "medium", "high"), "medium"),
    (("gpt-5.5-pro", "gpt-5.4-pro", "gpt-5.2-pro"), ("medium", "high", "xhigh"), "medium"),
    (("gpt-5-pro",), ("high",), "high"),
    (("gpt-5.6",), ("none", "low", "medium", "high", "xhigh", "max"), "medium"),
    (("gpt-5.5",), ("none", "low", "medium", "high", "xhigh"), "medium"),
    (("gpt-5.4", "gpt-5.2"), ("none", "low", "medium", "high", "xhigh"), "none"),
    (("gpt-5.1",), ("none", "low", "medium", "high"), "none"),
    (("gpt-5",), ("minimal", "low", "medium", "high"), None),
)

_GEMINI_LEVEL_BUDGETS = {
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}


def _matches(model: str, prefixes: tuple[str, ...]) -> bool:
    return any(model == prefix or model.startswith(f"{prefix}-") for prefix in prefixes)


def resolve_reasoning_control(
    provider: str,
    model: str,
    *,
    supports_reasoning: bool | None = None,
) -> ReasoningControl:
    provider_key = str(provider or "").strip().lower()
    model_key = str(model or "").strip().lower()

    if provider_key == "gemini":
        for prefixes, levels, default in _GEMINI_BUDGET_RULES:
            if _matches(model_key, prefixes):
                return ReasoningControl(
                    supported=True,
                    parameter_name="thinking_budget",
                    display_label="Thinking level",
                    levels=levels,
                    default_level=default,
                    source="official_registry",
                )
        rules = _GEMINI_RULES
        label = "Thinking level"
        parameter = "thinking_level"
    elif provider_key == "openai":
        rules = _OPENAI_RULES
        label = "Reasoning effort"
        parameter = "reasoning.effort"
    else:
        return ReasoningControl(False, None, "Reasoning")

    for prefixes, levels, default in rules:
        if _matches(model_key, prefixes):
            return ReasoningControl(
                supported=True,
                parameter_name=parameter,
                display_label=label,
                levels=levels,
                default_level=default,
                source="official_registry",
            )

    return ReasoningControl(
        supported=bool(supports_reasoning),
        parameter_name=None,
        display_label=label,
        source="provider_api" if supports_reasoning else "unknown",
    )


def validate_reasoning_effort(
    provider: str,
    model: str,
    value: object,
    *,
    supports_reasoning: bool | None = None,
) -> str | None:
    """Return an exact native value, or reject it without translating it."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError("Reasoning control must be a string or null for Provider default.")

    native_value = value.strip().lower()
    control = resolve_reasoning_control(
        provider,
        model,
        supports_reasoning=supports_reasoning,
    )
    if native_value not in control.levels:
        accepted = ", ".join(control.levels) if control.levels else "Provider default only"
        raise ValueError(
            f"{control.display_label} '{native_value}' is unsupported for "
            f"{provider}:{model}. Accepted: {accepted}."
        )
    return native_value


def gemini_reasoning_kwargs(model: str, value: str) -> dict[str, Any]:
    """Build the one native Gemini transport control for a named level."""
    native_value = validate_reasoning_effort(
        "gemini",
        model,
        value,
        supports_reasoning=True,
    )
    if native_value is None:
        return {}

    control = resolve_reasoning_control(
        "gemini",
        model,
        supports_reasoning=True,
    )
    if control.parameter_name == "thinking_budget":
        return {"thinking_budget": _GEMINI_LEVEL_BUDGETS[native_value]}
    if control.parameter_name == "thinking_level":
        return {"thinking_level": native_value}
    return {}
