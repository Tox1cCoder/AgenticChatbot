from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from app.api.model_config import ProviderOptionsSnapshot
from app.services.provider_service import ProviderService


def _load_helpers() -> dict[str, Any]:
    source = Path("demo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"_provider_models", "_model_reasoning_options"}
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(ast.Module(body=body, type_ignores=[]), "demo.py", "exec"), namespace)
    return namespace


def test_reasoning_options_use_selected_catalog_model() -> None:
    helper = _load_helpers()["_model_reasoning_options"]
    provider = {
        "models": [
            {
                "id": "gemini-3.6-flash",
                "reasoningControl": {
                    "displayLabel": "Thinking level",
                    "levels": ["minimal", "low", "medium", "high"],
                },
            }
        ]
    }
    assert helper(provider, "gemini-3.6-flash") == (
        "Thinking level",
        [None, "minimal", "low", "medium", "high"],
    )


def test_unknown_model_only_offers_provider_default() -> None:
    helper = _load_helpers()["_model_reasoning_options"]
    assert helper({"models": []}, "custom-model") == ("Reasoning", [None])


def test_catalog_descriptor_reaches_streamlit_reasoning_options() -> None:
    service = ProviderService(provider_repository=MagicMock())
    raw_model = service._normalize_gemini_model(
        "gemini-pro-latest",
        "Gemini Pro Latest",
        ["generateContent"],
        True,
    )
    provider = ProviderOptionsSnapshot.model_validate(
        {
            "provider_type": "gemini",
            "configured": True,
            "key_source": "env",
            "sync_status": "ready",
            "models": [raw_model],
        }
    ).model_dump(by_alias=True)

    helper = _load_helpers()["_model_reasoning_options"]
    assert helper(provider, "gemini-pro-latest") == (
        "Thinking level",
        [None, "low", "medium", "high"],
    )


def test_agent_model_controls_stay_together_and_rerun_on_change() -> None:
    source = Path("demo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    render_models = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_models_view"
    )
    render_source = ast.get_source_segment(source, render_models)

    assert render_source is not None
    config_heading = 'st.subheader("Configure models and parameters")'
    assert render_source.index(config_heading) < render_source.index('"Catalog model"')
    assert 'st.form("agent_model_config_form")' not in render_source
