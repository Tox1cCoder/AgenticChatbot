"""The persona editor must re-seed when Streamlit drops its widget state.

``persona_editor_origin`` is a plain session-state key, so it survives a
workspace tab switch, but the editor text lives under the ``persona_editor_value``
widget key, which Streamlit deletes for any tab it did not render. Checking the
origin alone lets the editor come back empty for the same conversation, and
"Save Persona" would then clear the saved persona.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEMO_SOURCE = Path("demo.py").read_text(encoding="utf-8")


def _load_helper() -> tuple[Any, dict[str, Any]]:
    tree = ast.parse(DEMO_SOURCE)
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_persona_editor_needs_seed"
    ]
    assert body, "_persona_editor_needs_seed is missing from demo.py"
    session_state: dict[str, Any] = {}
    namespace: dict[str, Any] = {"st": SimpleNamespace(session_state=session_state)}
    exec(compile(ast.Module(body=body, type_ignores=[]), "demo.py", "exec"), namespace)
    return namespace["_persona_editor_needs_seed"], session_state


def _settings_view_source() -> str:
    tree = ast.parse(DEMO_SOURCE)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "render_settings_view"
    )
    source = ast.get_source_segment(DEMO_SOURCE, node)
    assert source is not None
    return source


def test_seeds_when_the_editor_has_no_origin_yet() -> None:
    needs_seed, _session_state = _load_helper()

    assert needs_seed("conv-1") is True


def test_seeds_when_the_conversation_changed() -> None:
    needs_seed, session_state = _load_helper()
    session_state["persona_editor_origin"] = "conv-1"
    session_state["persona_editor_value"] = "be terse"

    assert needs_seed("conv-2") is True


def test_does_not_reseed_while_the_editor_state_is_present() -> None:
    needs_seed, session_state = _load_helper()
    session_state["persona_editor_origin"] = "conv-1"
    session_state["persona_editor_value"] = "be terse"

    assert needs_seed("conv-1") is False


def test_reseeds_after_a_tab_switch_dropped_the_widget_state() -> None:
    needs_seed, session_state = _load_helper()
    session_state["persona_editor_origin"] = "conv-1"
    session_state["persona_editor_value"] = "be terse"

    del session_state["persona_editor_value"]  # Streamlit's hidden-widget cleanup

    assert needs_seed("conv-1") is True


def test_an_empty_persona_the_user_typed_is_not_reseeded() -> None:
    needs_seed, session_state = _load_helper()
    session_state["persona_editor_origin"] = "conv-1"
    session_state["persona_editor_value"] = ""

    assert needs_seed("conv-1") is False


def test_settings_view_uses_the_seed_helper() -> None:
    assert "_persona_editor_needs_seed(conversation_id)" in _settings_view_source()


def test_editor_value_is_not_pre_created_as_a_session_default() -> None:
    """A pre-created default would mask the dropped widget state with an empty string."""
    tree = ast.parse(DEMO_SOURCE)
    defaults = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "SESSION_STATE_DEFAULTS"
    )
    assert isinstance(defaults.value, ast.Dict)
    keys = {key.value for key in defaults.value.keys if isinstance(key, ast.Constant)}

    assert "persona_editor_value" not in keys
    assert "persona_editor_origin" in keys
