from app.ai.hitl_config import requires_human_approval
from app.core.config import settings


def test_client_tools_follow_explicit_hitl_allowlist(monkeypatch):
    monkeypatch.setattr(settings, "enable_human_in_the_loop", True)
    monkeypatch.setattr(settings, "hitl_tools_require_approval", [])

    assert requires_human_approval(["client__desktop_commander__start_process"]) is False
    assert requires_human_approval(["server_only_tool"]) is False

    monkeypatch.setattr(
        settings,
        "hitl_tools_require_approval",
        ["client__desktop_commander__start_process"],
    )

    assert requires_human_approval(["client__desktop_commander__start_process"]) is True
