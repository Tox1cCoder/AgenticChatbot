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
$templateRoot = Join-Path $scriptRoot "client-backend-bundle"

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

$emptyInitContent = @'
'@

foreach ($name in @(
    "requirements-client.txt",
    "start-client-backend.ps1",
    "start-client-backend.bat",
    "README.client_backend.md"
)) {
    Copy-Item -LiteralPath (Join-Path $templateRoot $name) `
        -Destination (Join-Path $bundleRoot $name) -Force
}
Write-BundleFile -Path (Join-Path $bundleRoot "app\\__init__.py") -Content $emptyInitContent
Write-BundleFile -Path (Join-Path $bundleRoot "app\\ai\\__init__.py") -Content $emptyInitContent
Write-BundleFile -Path (Join-Path $bundleRoot "app\\core\\__init__.py") -Content $emptyInitContent
Write-BundleFile -Path (Join-Path $bundleRoot "app\\schemas\\__init__.py") -Content $emptyInitContent
Write-BundleFile -Path (Join-Path $bundleRoot "app\\services\\__init__.py") -Content $emptyInitContent

if (Test-Path $bundleZipPath) {
    Remove-Item -LiteralPath $bundleZipPath -Force
}

Compress-Archive -Path (Join-Path $bundleRoot "*") -DestinationPath $bundleZipPath -CompressionLevel Optimal

Write-Host "Built client backend bundle:"
Write-Host "  Folder: $bundleRoot"
Write-Host "  Zip:    $bundleZipPath"
