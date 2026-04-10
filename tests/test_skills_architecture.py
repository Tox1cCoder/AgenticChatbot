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


def test_demo_uses_repo_skills_snapshot_instead_of_deleted_skills_api():
    demo_source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    assert "get_repo_skill_detail_for_demo" in demo_source
    assert "list_repo_skills_for_demo" in demo_source
    assert "reload_repo_skills_for_demo" in demo_source
    assert 'make_api_request("GET", "/skills")' not in demo_source
    assert 'make_api_request("GET", f"/skills/{name}")' not in demo_source
    assert 'make_api_request("PATCH", f"/skills/{name}/toggle?enabled={str(enabled).lower()}")' not in demo_source
    assert 'make_api_request("POST", "/skills/reload")' not in demo_source
