from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STREAMLIT_ENTRYPOINTS = ("demo.py", "upload_support.py")


def _use_container_width_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        lines.extend(
            keyword.lineno for keyword in node.keywords if keyword.arg == "use_container_width"
        )

    return lines


def test_streamlit_calls_use_width_instead_of_use_container_width():
    offenders = {
        entrypoint: lines
        for entrypoint in _STREAMLIT_ENTRYPOINTS
        if (lines := _use_container_width_lines(_REPO_ROOT / entrypoint))
    }

    assert not offenders, (
        "Replace Streamlit use_container_width with width "
        "('stretch' for True, 'content' for False): "
        f"{offenders}"
    )
