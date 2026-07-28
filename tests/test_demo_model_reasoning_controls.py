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


def test_cached_catalog_is_enriched_before_streamlit_builds_reasoning_options() -> None:
    service = ProviderService(provider_repository=MagicMock())
    catalog = service._normalize_catalog_metadata(
        {
            "catalog": {
                "models": [
                    {
                        "id": "gemini-3.6-flash",
                        "provider_type": "gemini",
                        "supports_reasoning": True,
                    }
                ]
            }
        }
    )
    helper = _load_helpers()["_model_reasoning_options"]

    assert helper({"models": catalog["models"]}, "gemini-3.6-flash") == (
        "Thinking level",
        [None, "minimal", "low", "medium", "high"],
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


def test_catalog_model_selector_has_no_competing_default_value() -> None:
    source = Path("demo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    render_models = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_models_view"
    )
    selectboxes = [
        node
        for node in ast.walk(render_models)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "selectbox"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Catalog model"
    ]

    assert len(selectboxes) == 1
    assert all(keyword.arg != "index" for keyword in selectboxes[0].keywords)


def test_temperature_slider_has_no_competing_default_value() -> None:
    source = Path("demo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    render_models = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_models_view"
    )
    sliders = [
        node
        for node in ast.walk(render_models)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "slider"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Temperature"
    ]

    assert len(sliders) == 1
    assert all(keyword.arg != "value" for keyword in sliders[0].keywords)


def test_custom_model_input_has_no_competing_default_value() -> None:
    source = Path("demo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    render_models = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_models_view"
    )
    inputs = [
        node
        for node in ast.walk(render_models)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "text_input"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Custom model ID"
    ]

    assert len(inputs) == 1
    assert all(keyword.arg != "value" for keyword in inputs[0].keywords)


def test_custom_model_checkbox_has_no_competing_default_value() -> None:
    source = Path("demo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    render_models = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_models_view"
    )
    checkboxes = [
        node
        for node in ast.walk(render_models)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "checkbox"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Allow custom model override"
    ]

    assert len(checkboxes) == 1
    assert all(keyword.arg != "value" for keyword in checkboxes[0].keywords)


def test_reasoning_selector_has_no_competing_default_index() -> None:
    source = Path("demo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    render_models = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_models_view"
    )
    selectors = [
        node
        for node in ast.walk(render_models)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "selectbox"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "reasoning_label"
    ]

    assert len(selectors) == 1
    assert all(keyword.arg != "index" for keyword in selectors[0].keywords)


def test_chat_prompt_has_one_native_widget_state_rule() -> None:
    source = Path("app/ai/prompts.py").read_text(encoding="utf-8")
    assert source.count("initial_state") == 1
    assert "never serialize it as a JSON string" in source
