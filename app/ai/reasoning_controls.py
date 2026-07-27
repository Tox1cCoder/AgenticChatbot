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


# Ordered most-specific-first so Pro variants never inherit a broader family.
_GEMINI_RULES = (
    (("gemini-3.6-flash",), ("minimal", "low", "medium", "high"), "medium"),
    (("gemini-3.5-flash-lite",), ("minimal", "low", "medium", "high"), "minimal"),
    (("gemini-3.5-flash",), ("minimal", "low", "medium", "high"), "medium"),
    (("gemini-3.1-pro-preview",), ("low", "medium", "high"), "high"),
    (("gemini-3.1-flash-lite-image",), ("minimal", "high"), "minimal"),
    (("gemini-3-flash-preview",), ("minimal", "low", "medium", "high"), "high"),
    (("gemini-3-pro-preview",), ("low", "high"), "high"),
)

_OPENAI_RULES = (
    (("gpt-5.5-pro", "gpt-5.4-pro", "gpt-5.2-pro"), ("medium", "high", "xhigh"), "medium"),
    (("gpt-5-pro",), ("high",), "high"),
    (("gpt-5.6",), ("none", "low", "medium", "high", "xhigh", "max"), "medium"),
    (("gpt-5.5",), ("none", "low", "medium", "high", "xhigh"), "medium"),
    (("gpt-5.4", "gpt-5.2"), ("none", "low", "medium", "high", "xhigh"), "none"),
    (("gpt-5.1",), ("none", "low", "medium", "high"), "none"),
    (("gpt-5",), ("minimal", "low", "medium", "high"), None),
)


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
