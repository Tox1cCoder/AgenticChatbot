"""Unit tests for the frozen-requirements resolver gate."""

from __future__ import annotations

from scripts import verify_frozen_requirements


def test_resolver_rejects_non_windows_platform_before_running_pip(monkeypatch, capsys):
    monkeypatch.setattr(verify_frozen_requirements.sys, "platform", "linux")
    monkeypatch.setattr(
        verify_frozen_requirements.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pip ran on an unsupported platform")
        ),
    )

    assert verify_frozen_requirements.main() == 2
    assert "Windows" in capsys.readouterr().err
