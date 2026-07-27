from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


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
