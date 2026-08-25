"""Models tab form state must survive a workspace tab switch.

Streamlit deletes widget-keyed session state for widgets a script run does not
render, and ``_render_main_workspace_tabs`` renders only the open tab. Leaving
the Models tab therefore drops every ``model_cfg_*`` key while the snapshot
cache in ``model_config_options_cache`` survives, so returning to the tab has to
re-seed the form from the snapshot instead of falling back to the head of the
provider catalog.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEMO_SOURCE = Path("demo.py").read_text(encoding="utf-8")

_WANTED_FUNCTIONS = {
    "_normalize_provider_type",
    "_snapshot_provider_list",
    "_snapshot_provider_map",
    "_snapshot_agent_config",
    "_provider_models",
    "_provider_model_ids",
    "_sync_model_config_form_state",
    "_model_config_form_state_keys",
    "_model_config_form_state_is_intact",
}
_WANTED_ASSIGNMENTS = {
    "MODEL_CFG_AGENT_LABELS",
    "MODEL_CFG_AGENT_KEYS",
    "MODEL_CFG_FORM_STATE_PREFIXES",
}


def _load_form_state_helpers() -> dict[str, Any]:
    """Exec the snapshot/form-state helpers against a stub ``st.session_state``."""
    tree = ast.parse(DEMO_SOURCE)
    body: list[ast.stmt] = []
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in _WANTED_FUNCTIONS
        ):
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id in _WANTED_ASSIGNMENTS
                for target in targets
            ):
                body.append(node)

    namespace: dict[str, Any] = {"Any": Any, "st": SimpleNamespace(session_state={})}
    exec(compile(ast.Module(body=body, type_ignores=[]), "demo.py", "exec"), namespace)
    return namespace


def _render_models_view_source() -> str:
    tree = ast.parse(DEMO_SOURCE)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "render_models_view"
    )
    source = ast.get_source_segment(DEMO_SOURCE, node)
    assert source is not None
    return source


def _snapshot() -> dict[str, Any]:
    """A snapshot whose persisted chat model is not the catalog head."""
    return {
        "providers": [
            {
                "providerType": "gemini",
                "configured": True,
                "keySource": "env",
                "syncStatus": "ready",
                "models": [
                    {"id": "gemini-flash-latest", "recommended": True},
                    {"id": "gemini-3.7-flash", "recommended": False},
                    {"id": "gemini-3.1-pro-preview", "recommended": False},
                ],
            },
            {
                "providerType": "openai",
                "configured": False,
                "keySource": "none",
                "syncStatus": "unknown",
                "models": [],
            },
        ],
        "agentConfig": {
            "chat": {
                "provider": "gemini",
                "model": "gemini-3.7-flash",
                "temperature": 0.4,
                "source": "persisted",
                "isCustomModel": False,
                "reasoningEffort": None,
                "warnings": [],
            },
            "rag": {
                "provider": "gemini",
                "model": "gemini-3.1-pro-preview",
                "temperature": 1.0,
                "source": "persisted",
                "isCustomModel": False,
                "reasoningEffort": None,
                "warnings": [],
            },
            "search": {
                "provider": "gemini",
                "model": "gemini-flash-latest",
                "temperature": 1.0,
                "source": "default",
                "isCustomModel": False,
                "reasoningEffort": None,
                "warnings": [],
            },
            "planning": {
                "provider": "gemini",
                "model": "gemini-flash-latest",
                "temperature": 1.0,
                "source": "default",
                "isCustomModel": False,
                "reasoningEffort": None,
                "warnings": [],
            },
        },
    }


def _drop_widget_state(session_state: dict[str, Any]) -> None:
    """Reproduce Streamlit's cleanup of widget state for an unrendered tab."""
    for key in [key for key in session_state if key.startswith("model_cfg_")]:
        del session_state[key]


def test_sync_seeds_the_persisted_model_not_the_catalog_head() -> None:
    helpers = _load_form_state_helpers()
    session_state = helpers["st"].session_state

    helpers["_sync_model_config_form_state"](_snapshot())

    assert session_state["model_cfg_model_select_chat"] == "gemini-3.7-flash"
    assert session_state["model_cfg_model_select_rag"] == "gemini-3.1-pro-preview"
    assert session_state["model_cfg_temperature_chat"] == 0.4


def test_synced_form_state_is_reported_intact() -> None:
    helpers = _load_form_state_helpers()

    helpers["_sync_model_config_form_state"](_snapshot())

    assert helpers["_model_config_form_state_is_intact"]() is True


def test_state_dropped_with_the_hidden_tab_is_not_intact() -> None:
    helpers = _load_form_state_helpers()
    session_state = helpers["st"].session_state

    helpers["_sync_model_config_form_state"](_snapshot())
    _drop_widget_state(session_state)

    assert helpers["_model_config_form_state_is_intact"]() is False


def test_a_single_dropped_key_is_not_intact() -> None:
    helpers = _load_form_state_helpers()
    session_state = helpers["st"].session_state

    helpers["_sync_model_config_form_state"](_snapshot())
    del session_state["model_cfg_model_select_chat"]

    assert helpers["_model_config_form_state_is_intact"]() is False


def test_reseeding_restores_the_persisted_model_after_a_tab_switch() -> None:
    helpers = _load_form_state_helpers()
    session_state = helpers["st"].session_state
    snapshot = _snapshot()

    helpers["_sync_model_config_form_state"](snapshot)
    _drop_widget_state(session_state)
    helpers["_sync_model_config_form_state"](snapshot)

    assert session_state["model_cfg_model_select_chat"] == "gemini-3.7-flash"
    assert session_state["model_cfg_temperature_chat"] == 0.4


def test_form_state_keys_cover_every_agent_and_control() -> None:
    helpers = _load_form_state_helpers()

    keys = set(helpers["_model_config_form_state_keys"]())

    for agent_key in helpers["MODEL_CFG_AGENT_KEYS"]:
        for prefix in helpers["MODEL_CFG_FORM_STATE_PREFIXES"]:
            assert f"{prefix}{agent_key}" in keys


def test_models_view_reseeds_when_widget_state_was_dropped() -> None:
    render_source = _render_models_view_source()

    assert "_model_config_form_state_is_intact()" in render_source
    sync_call = render_source.index("_sync_model_config_form_state(snapshot)")
    guard = render_source.index("not _model_config_form_state_is_intact()")
    assert guard < sync_call


def test_models_view_renders_exactly_the_agents_the_form_state_covers() -> None:
    render_source = _render_models_view_source()

    assert "MODEL_CFG_AGENT_LABELS" in render_source
    assert '("chat", "Chat")' not in render_source
