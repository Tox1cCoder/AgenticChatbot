import json
from pathlib import Path

import client_backend.core.config as config_module
import client_backend.services.local_mcp_manager as mcp_module
from client_backend.cli import main


def test_doctor_reports_ok_with_valid_config(tmp_path, monkeypatch, capsys):
    profile_root = tmp_path / "profile"
    env_file = tmp_path / ".env.client"
    env_file.write_text(
        "\n".join(
            [
                "CLIENT_SERVER_API_BASE_URL=https://example.com",
                f"CLIENT_PROFILE_ROOT={profile_root}",
                "CLIENT_ENVIRONMENT=production",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("CLIENT_ENV_FILE", str(env_file))
    config_module.get_client_settings.cache_clear()
    settings = config_module.get_client_settings()
    monkeypatch.setattr(config_module, "client_settings", settings)
    monkeypatch.setattr(mcp_module, "client_settings", settings)

    exit_code = main(["doctor", "--config", str(env_file), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["config"]["profile_root"] == str(Path(profile_root))
    assert "workspace_roots_configured" not in payload["checks"]
    assert "workspace_roots" not in payload["config"]
