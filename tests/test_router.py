"""Router contract guards that outlived the ``Router`` compatibility adapter.

Semantic routing behavior is covered by ``tests/test_routing_service.py``.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace


def test_structured_router_prompt_is_owned_by_the_routing_module():
    from app.ai.workflow.routing import ROUTER_SYSTEM_PROMPT

    assert "structured" in ROUTER_SYSTEM_PROMPT.lower()
    assert "untrusted" in ROUTER_SYSTEM_PROMPT.lower()
    # The instruction must not restate a phrase-matching rule.
    assert "keyword" in ROUTER_SYSTEM_PROMPT.lower()


def test_router_no_longer_exposes_stickiness_helpers():
    from app.ai.workflow import custom_agents as custom_agents_module

    source = pathlib.Path(custom_agents_module.__file__).read_text(encoding="utf-8")
    assert "_sticky_custom_agent" not in source
    assert "_match_explicit_custom_agent" not in source


def test_legacy_router_settings_still_resolve():
    from app.core.config import settings

    assert settings.router_model
    assert settings.router_provider == "gemini"
    assert isinstance(SimpleNamespace(model=settings.router_model).model, str)
