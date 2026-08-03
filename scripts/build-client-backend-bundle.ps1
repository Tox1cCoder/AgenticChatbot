param(
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    param([string]$ScriptPath)
    return (Resolve-Path (Join-Path $ScriptPath "..")).Path
}

function Reset-Directory {
    param([string]$Path)

    if (Test-Path $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }

    New-Item -ItemType Directory -Path $Path | Out-Null
}

function Remove-CacheDirectories {
    param([string]$Root)

    Get-ChildItem -LiteralPath $Root -Recurse -Directory -Force |
        Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".mypy_cache") } |
        Remove-Item -Recurse -Force

    Get-ChildItem -LiteralPath $Root -Recurse -File -Force |
        Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
        Remove-Item -Force
}

function Write-BundleFile {
    param(
        [string]$Path,
        [string]$Content
    )

    $parent = Split-Path -Parent $Path
    if ($parent -and !(Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    Set-Content -LiteralPath $Path -Value $Content -Encoding UTF8
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Get-RepoRoot -ScriptPath $scriptRoot

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot "dist"
}

$bundleRoot = Join-Path $OutputRoot "client-backend-bundle"
$bundleZipPath = Join-Path $OutputRoot "client-backend-bundle.zip"

Reset-Directory -Path $bundleRoot

Copy-Item -LiteralPath (Join-Path $repoRoot "client_backend") -Destination $bundleRoot -Recurse -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "shared") -Destination $bundleRoot -Recurse -Force

New-Item -ItemType Directory -Path (Join-Path $bundleRoot "app") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundleRoot "app\\ai") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundleRoot "app\\core") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundleRoot "app\\schemas") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundleRoot "app\\services") -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $repoRoot "app\\ai\\mcp_config.json") -Destination (Join-Path $bundleRoot "app\\ai\\mcp_config.json") -Force
# Only the tracked *.py servers: developers keep unversioned server
# installations (OCR binaries, model weights) beside them, and a recursive copy
# would sweep those into the distributable and make the build depend on one
# machine's working tree.
$mcpServersDestination = Join-Path $bundleRoot "app\\ai\\mcp_servers"
New-Item -ItemType Directory -Force -Path $mcpServersDestination | Out-Null
Get-ChildItem -LiteralPath (Join-Path $repoRoot "app\\ai\\mcp_servers") -Filter *.py -File |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $mcpServersDestination -Force }
Copy-Item -LiteralPath (Join-Path $repoRoot "app\\core\\config.py") -Destination (Join-Path $bundleRoot "app\\core\\config.py") -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "app\\core\\build_info.py") -Destination (Join-Path $bundleRoot "app\\core\\build_info.py") -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "app\\core\\mcp_adapter_utils.py") -Destination (Join-Path $bundleRoot "app\\core\\mcp_adapter_utils.py") -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "app\\schemas\\runtime_protocol.py") -Destination (Join-Path $bundleRoot "app\\schemas\\runtime_protocol.py") -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "app\\services\\widget_contract.py") -Destination (Join-Path $bundleRoot "app\\services\\widget_contract.py") -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "app\\services\\widget_runtime.py") -Destination (Join-Path $bundleRoot "app\\services\\widget_runtime.py") -Force
Copy-Item -LiteralPath (Join-Path $repoRoot ".env.client.example") -Destination (Join-Path $bundleRoot ".env.client.example") -Force

Remove-CacheDirectories -Root $bundleRoot

$requirementsContent = @'
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
httpx>=0.25.0
PyJWT>=2.8.0
cryptography>=41.0.0
filelock>=3.20.0,<4.0.0
jsonschema>=4.20.0
tomli>=2.0.0; python_version < "3.11"
websockets>=12.0
langchain-mcp-adapters>=0.1.13
redis>=5.0.1
tavily-python>=0.3.0
tzdata>=2024.1
'@

$emptyInitContent = @'
'@

$startPs1Content = @'
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
$venvPython = Join-Path $venvPath "Scripts\python.exe"
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

# Compare requirement *contents*, not timestamps. Unzipping a bundle restores the
# mtime recorded in the archive, so a newly deployed requirements file routinely
# looks older than the marker written during the previous machine's last run, and
# a timestamp check then skips installing a dependency that was just added. The
# symptom is the sidecar dying at import on a module the bundle does declare.
$requirementsHash = (Get-FileHash -LiteralPath $requirementsPath -Algorithm SHA256).Hash
$installedHash = ""
if (Test-Path $installMarker) {
    $installedHash = (Get-Content -LiteralPath $installMarker -Raw).Trim()
}

if ($installedHash -ne $requirementsHash) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Installing $requirementsPath failed. The sidecar was not started; fix the error above and run this script again."
    }
    # Written only after pip succeeds. Recording it beforehand would mark a
    # failed install as done and skip every retry.
    Set-Content -Path $installMarker -Value $requirementsHash -Encoding UTF8
}

$env:CLIENT_ENV_FILE = (Resolve-Path $ConfigPath).Path
Push-Location $bundleRoot
try {
    & $venvPython -m client_backend run --config $env:CLIENT_ENV_FILE
} finally {
    Pop-Location
}
'@

$startBatContent = @'
@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start-client-backend.ps1" %*

endlocal
'@

$readmeContent = @'
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
.\scripts\build-client-backend-bundle.ps1
```

That regenerates `dist\client-backend-bundle` and `dist\client-backend-bundle.zip`.
'@

Write-BundleFile -Path (Join-Path $bundleRoot "requirements-client.txt") -Content $requirementsContent
Write-BundleFile -Path (Join-Path $bundleRoot "app\\__init__.py") -Content $emptyInitContent
Write-BundleFile -Path (Join-Path $bundleRoot "app\\ai\\__init__.py") -Content $emptyInitContent
Write-BundleFile -Path (Join-Path $bundleRoot "app\\core\\__init__.py") -Content $emptyInitContent
Write-BundleFile -Path (Join-Path $bundleRoot "app\\schemas\\__init__.py") -Content $emptyInitContent
Write-BundleFile -Path (Join-Path $bundleRoot "app\\services\\__init__.py") -Content $emptyInitContent
Write-BundleFile -Path (Join-Path $bundleRoot "start-client-backend.ps1") -Content $startPs1Content
Write-BundleFile -Path (Join-Path $bundleRoot "start-client-backend.bat") -Content $startBatContent
Write-BundleFile -Path (Join-Path $bundleRoot "README.client_backend.md") -Content $readmeContent

if (Test-Path $bundleZipPath) {
    Remove-Item -LiteralPath $bundleZipPath -Force
}

Compress-Archive -Path (Join-Path $bundleRoot "*") -DestinationPath $bundleZipPath -CompressionLevel Optimal

Write-Host "Built client backend bundle:"
Write-Host "  Folder: $bundleRoot"
Write-Host "  Zip:    $bundleZipPath"
