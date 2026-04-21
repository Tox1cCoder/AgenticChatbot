import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_protocol_source_compiles_from_source():
    runtime_protocol = REPO_ROOT / "app" / "schemas" / "runtime_protocol.py"

    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(runtime_protocol)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="bundle script is PowerShell/Windows-specific")
def test_client_backend_bundle_can_import_local_skills_registry(tmp_path):
    output_root = tmp_path / "bundle-output"
    build_script = REPO_ROOT / "scripts" / "build-client-backend-bundle.ps1"

    build = subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(build_script),
            "-OutputRoot",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert build.returncode == 0, build.stderr or build.stdout

    bundle_root = output_root / "client-backend-bundle"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, r'{bundle_root}'); "
                "import client_backend.services.local_skills_registry; "
                "print('OK')"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=bundle_root,
    )

    assert probe.returncode == 0, probe.stderr or probe.stdout
