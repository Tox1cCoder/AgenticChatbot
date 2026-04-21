import subprocess
import sys
from pathlib import Path


def test_container_import_does_not_trigger_api_cycle():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-c", "import app.core.container"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
