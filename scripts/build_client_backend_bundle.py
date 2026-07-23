"""
Build the client backend source bundle into dist/.

Python port of scripts/build-client-backend-bundle.ps1 — produces the same
artifacts (dist/client-backend-bundle/ and dist/client-backend-bundle.zip)
on any platform:

    python scripts/build_client_backend_bundle.py [--output-root PATH]
"""

# ruff: noqa: E501 — the embedded start-client-backend.ps1 content must match
# the shipped script verbatim; one PowerShell line exceeds 100 chars.

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
CACHE_FILE_SUFFIXES = {".pyc", ".pyo"}

REQUIREMENTS_CONTENT = """\
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
httpx>=0.25.0
PyJWT>=2.8.0
cryptography>=41.0.0
websockets>=12.0
langchain-mcp-adapters>=0.1.13
redis>=5.0.1
tavily-python>=0.3.0
tzdata>=2024.1
"""

START_PS1_CONTENT = """\
param(
    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"

function Resolve-BundleRoot {
    param([string]$ScriptRoot)

    $candidates = @(
        (Resolve-Path $ScriptRoot).Path,
        (Resolve-Path (Join-Path $ScriptRoot "..")).Path
    )

    foreach ($candidate in $candidates) {
        if (
            (Test-Path (Join-Path $candidate "client_backend")) -and
            (Test-Path (Join-Path $candidate "requirements-client.txt"))
        ) {
            return $candidate
        }
    }

    throw "Could not locate client backend bundle root."
}

$bundleRoot = Resolve-BundleRoot -ScriptRoot $PSScriptRoot
$requirementsPath = Join-Path $bundleRoot "requirements-client.txt"
$defaultConfigPath = Join-Path $bundleRoot ".env.client"
$exampleConfigPath = Join-Path $bundleRoot ".env.client.example"

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = $defaultConfigPath
}

if (!(Test-Path $ConfigPath) -and (Test-Path $exampleConfigPath)) {
    Copy-Item $exampleConfigPath $ConfigPath
    Write-Host "Created $ConfigPath from .env.client.example"
}

$venvPath = Join-Path $bundleRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\\python.exe"
$installMarker = Join-Path $venvPath ".client_requirements_installed"

if (!(Test-Path $venvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.10 -m venv $venvPath
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv $venvPath
    } else {
        throw "Python 3.10+ is required but was not found on PATH."
    }
}

$shouldInstall = $true
if ((Test-Path $installMarker) -and (Test-Path $requirementsPath)) {
    $shouldInstall = (Get-Item $requirementsPath).LastWriteTimeUtc -gt (Get-Item $installMarker).LastWriteTimeUtc
}

if ($shouldInstall) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $requirementsPath
    Set-Content -Path $installMarker -Value (Get-Date).ToString("o") -Encoding UTF8
}

$env:CLIENT_ENV_FILE = (Resolve-Path $ConfigPath).Path
Push-Location $bundleRoot
try {
    & $venvPython -m client_backend run --config $env:CLIENT_ENV_FILE
} finally {
    Pop-Location
}
"""

START_BAT_CONTENT = """\
@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start-client-backend.ps1" %*

endlocal
"""

README_CONTENT = """\
# Client Backend Bundle

This is the source-bundle distribution for the local client sidecar. It keeps the
same shipping model as the previous version: copy one folder to another machine,
run the startup script, and let the script create its own `.venv`.

## What Ships

- `client_backend/` source package
- `shared/` contract modules used by both server and client runtime
- minimal shared `app/` helpers required by the runtime bridge and local MCP manager
- bundled schema-v2 MCP registry and portable MCP server scripts
- `requirements-client.txt`
- `.env.client.example`
- `start-client-backend.ps1`
- `start-client-backend.bat`

## Why This Bundle Exists

The sidecar can also be packaged as a `.whl` file, which is a Python wheel
archive for `pip install`. That is useful for Python-native distribution, but it
is not the easiest handoff format for non-developer machines.

This bundle is the easier shipping format:

1. Copy or zip this folder.
2. Move it to the target machine.
3. Edit `.env.client`.
4. Run `start-client-backend.bat` or `start-client-backend.ps1`.

## Runtime Contract

- Client-side tools come only from MCP servers configured for the sidecar.
- Server-side tools come only from MCP servers configured on the canonical backend.
- The sidecar does not expose native shell or filesystem tools.

## Fast Start On Another PC

1. Install Python 3.10 or newer.
2. Copy this folder or unzip `client-backend-bundle.zip`.
3. Copy `.env.client.example` to `.env.client`.
4. Update at least:
   - `CLIENT_SERVER_API_BASE_URL`
5. Run `start-client-backend.bat`.

The startup script creates `.venv`, installs `requirements-client.txt`, and runs:

```powershell
python -m client_backend run --config .env.client
```

## CLI

Useful commands inside the bundle:

```powershell
python -m client_backend run --config .env.client
python -m client_backend doctor --config .env.client
python -m client_backend mcp migrate
python -m client_backend mcp doctor --servers widgets,tavily,time
```

## Rebuild From This Repo

```powershell
.\\scripts\\build-client-backend-bundle.ps1
```

or, cross-platform:

```bash
python scripts/build_client_backend_bundle.py
```

That regenerates `dist\\client-backend-bundle` and `dist\\client-backend-bundle.zip`.
"""


def _ignore_caches(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in CACHE_DIR_NAMES}


def _remove_cache_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in CACHE_FILE_SUFFIXES:
            path.unlink()


def build(output_root: Path) -> tuple[Path, Path]:
    bundle_root = output_root / "client-backend-bundle"
    bundle_zip = output_root / "client-backend-bundle.zip"

    # Empty the directory rather than removing it: on Windows the directory
    # node itself is often held open (shell cwd, IDE watcher) and rmdir fails.
    if bundle_root.exists():
        for child in bundle_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        bundle_root.mkdir(parents=True)

    shutil.copytree(
        REPO_ROOT / "client_backend",
        bundle_root / "client_backend",
        ignore=_ignore_caches,
    )
    shutil.copytree(
        REPO_ROOT / "shared",
        bundle_root / "shared",
        ignore=_ignore_caches,
    )

    app_dir = bundle_root / "app"
    (app_dir / "ai").mkdir(parents=True)
    (app_dir / "core").mkdir(parents=True)
    (app_dir / "schemas").mkdir(parents=True)
    (app_dir / "services").mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "app" / "ai" / "mcp_config.json",
        app_dir / "ai" / "mcp_config.json",
    )
    shutil.copytree(
        REPO_ROOT / "app" / "ai" / "mcp_servers",
        app_dir / "ai" / "mcp_servers",
        ignore=_ignore_caches,
    )
    shutil.copy2(
        REPO_ROOT / "app" / "core" / "config.py",
        app_dir / "core" / "config.py",
    )
    shutil.copy2(
        REPO_ROOT / "app" / "core" / "mcp_adapter_utils.py",
        app_dir / "core" / "mcp_adapter_utils.py",
    )
    shutil.copy2(
        REPO_ROOT / "app" / "schemas" / "runtime_protocol.py",
        app_dir / "schemas" / "runtime_protocol.py",
    )
    shutil.copy2(
        REPO_ROOT / "app" / "services" / "widget_contract.py",
        app_dir / "services" / "widget_contract.py",
    )
    shutil.copy2(
        REPO_ROOT / "app" / "services" / "widget_runtime.py",
        app_dir / "services" / "widget_runtime.py",
    )
    shutil.copy2(
        REPO_ROOT / ".env.client.example",
        bundle_root / ".env.client.example",
    )

    _remove_cache_files(bundle_root)

    (bundle_root / "requirements-client.txt").write_text(REQUIREMENTS_CONTENT, encoding="utf-8")
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "ai" / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "core" / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "schemas" / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "services" / "__init__.py").write_text("", encoding="utf-8")
    (bundle_root / "start-client-backend.ps1").write_text(START_PS1_CONTENT, encoding="utf-8")
    (bundle_root / "start-client-backend.bat").write_text(START_BAT_CONTENT, encoding="utf-8")
    (bundle_root / "README.client_backend.md").write_text(README_CONTENT, encoding="utf-8")

    if bundle_zip.exists():
        bundle_zip.unlink()
    with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_root))

    return bundle_root, bundle_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the client backend bundle.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "dist"),
        help="Directory that receives client-backend-bundle/ and the zip.",
    )
    args = parser.parse_args()

    bundle_root, bundle_zip = build(Path(args.output_root))
    print("Built client backend bundle:")
    print(f"  Folder: {bundle_root}")
    print(f"  Zip:    {bundle_zip}")


if __name__ == "__main__":
    main()
