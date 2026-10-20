from __future__ import annotations

from pathlib import Path

import pytest
from uvicorn.config import Config
from uvicorn.supervisors.watchfilesreload import FileFilter

from shared.dev_reload import REPO_ROOT, reload_dirs


def _triggers_reload(path: Path) -> bool:
    """Whether uvicorn's reloader, configured as the servers are, restarts on ``path``."""
    config = Config(app="app.main:app", reload=True, reload_dirs=reload_dirs())
    watched = any(Path(directory) in path.parents for directory in config.reload_dirs)
    return watched and FileFilter(config)(path)


@pytest.mark.parametrize(
    "relative",
    [
        "create_late_apps_tmp.py",
        "skills/take100/take100_api.py",
        "skills/attendance/attendance_api.py",
        "tests/test_dev_reload.py",
        "scripts/anything.py",
    ],
)
def test_scratch_and_skill_scripts_do_not_restart_the_server(relative):
    assert not _triggers_reload(REPO_ROOT / relative)


@pytest.mark.parametrize(
    "relative",
    [
        "app/main.py",
        "client_backend/main.py",
        "shared/skills/hashing.py",
    ],
)
def test_server_source_changes_still_restart_the_server(relative):
    assert _triggers_reload(REPO_ROOT / relative)
