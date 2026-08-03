"""Architecture guardrails for the production skills refactor.

These tests document the approved end state even where the runtime code has not
yet been refactored to match it.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _demo_source() -> str:
    return REPO_ROOT.joinpath("demo.py").read_text(encoding="utf-8")


def test_server_app_does_not_expose_public_skills_routes():
    main_source = REPO_ROOT.joinpath("app", "main.py").read_text(encoding="utf-8")
    assert "app.include_router(skills_router)" not in main_source


def test_demo_manages_skills_through_the_sidecar_api():
    demo_source = _demo_source()
    assert "skills_snapshot" not in demo_source
    assert 'make_api_request("GET", "/skills")' in demo_source
    assert 'make_api_request("POST", "/skills/reload")' in demo_source


def test_skill_uploads_go_to_the_sidecar_not_the_canonical_server():
    """A skill ZIP is device-local; it must never be sent to the shared server."""
    demo_source = _demo_source()

    assert 'make_api_multipart_request(\n        "/skills/uploads"' in demo_source
    assert "CHATBOT_SERVER_API" not in demo_source


def test_canonical_server_still_exposes_no_skill_upload_router():
    source = REPO_ROOT.joinpath("app", "main.py").read_text(encoding="utf-8")

    assert "skill_upload" not in source
    assert "skills_router" not in source


def test_streamlit_never_installs_skills_by_local_path():
    """The browser has no trustworthy filesystem path to offer.

    The path-based install route stays for local tooling, but a UI that posted a
    user-typed path would be asking the sidecar to read an arbitrary directory.
    """
    tree = ast.parse(_demo_source())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        arguments = [
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        ]
        assert "/skills/install" not in arguments
        assert "/skills/install/preview" not in arguments


def test_streamlit_skill_helpers_url_encode_every_identifier():
    """Identifiers reach these helpers from the sidecar, but still get encoded."""
    tree = ast.parse(_demo_source())
    helpers = {
        "start_skill_install",
        "get_skill_installation",
        "cancel_skill_upload",
        "cancel_skill_installation",
    }
    checked = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in helpers:
            continue
        checked.add(node.name)
        source = ast.unparse(node)
        assert "quote(" in source, f"{node.name} builds a URL without quoting its id"

    assert checked == helpers
