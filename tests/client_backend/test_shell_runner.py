import pytest

from client_backend.core.config import client_settings
from client_backend.services.shell_runner import ShellRunnerService


@pytest.mark.asyncio
async def test_shell_runner_executes_with_cmd_on_windows_without_workspace(monkeypatch):
    monkeypatch.setattr(client_settings, "workspace_roots", [])
    monkeypatch.setattr(client_settings, "allowed_shells", ["cmd", "powershell"])

    runner = ShellRunnerService()
    result = await runner.execute(
        command="echo hello-from-cmd",
        shell="cmd",
    )

    assert result.success is True
    assert "hello-from-cmd" in result.stdout.lower()
    assert result.working_dir is None
