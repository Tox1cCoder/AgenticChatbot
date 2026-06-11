"""
Architecture guardrails for the production skills refactor.

These tests document the approved end state even where the runtime code has not
yet been refactored to match it.
"""

from pathlib import Path


def test_server_app_does_not_expose_public_skills_routes():
    main_source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(
        encoding="utf-8"
    )
    assert "app.include_router(skills_router)" not in main_source


def test_demo_manages_skills_through_the_sidecar_api():
    demo_source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")
    assert "skills_snapshot" not in demo_source
    assert 'make_api_request("GET", "/skills")' in demo_source
    assert 'make_api_request("POST", "/skills/reload")' in demo_source
