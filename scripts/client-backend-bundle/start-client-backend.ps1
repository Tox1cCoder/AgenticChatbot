param(
    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"

function Resolve-BundleRoot {
    param([string]$ScriptRoot)

    $candidates = @(
        (Resolve-Path -LiteralPath $ScriptRoot).Path,
        (Resolve-Path -LiteralPath (Join-Path $ScriptRoot "..")).Path
    )
    foreach ($candidate in $candidates) {
        if (
            (Test-Path -LiteralPath (Join-Path $candidate "client_backend")) -and
            (Test-Path -LiteralPath (Join-Path $candidate "requirements-client.txt"))
        ) {
            return $candidate
        }
    }
    throw "Could not locate client backend bundle root."
}

function Invoke-NativeChecked {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$Phase
    )

    & $FilePath @ArgumentList
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "$Phase failed with exit code $code while running '$FilePath'."
    }
}

function Get-Sha256 {
    param([string]$Path)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            $bytes = $sha.ComputeHash($stream)
        } finally {
            $stream.Dispose()
        }
    } finally {
        $sha.Dispose()
    }
    return ([System.BitConverter]::ToString($bytes)).Replace("-", "")
}

function Test-PythonCandidate {
    param(
        [string]$Command,
        [string[]]$PrefixArguments
    )

    if (!(Get-Command $Command -ErrorAction SilentlyContinue)) {
        return $false
    }
    $probeArguments = @($PrefixArguments) + @(
        "-c",
        "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
    )
    try {
        & $Command @probeArguments 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Find-CompatiblePython {
    $candidates = @(
        [pscustomobject]@{ Command = "py"; PrefixArguments = @("-3") },
        [pscustomobject]@{ Command = "python"; PrefixArguments = @() },
        [pscustomobject]@{ Command = "python3"; PrefixArguments = @() }
    )
    foreach ($candidate in $candidates) {
        if (Test-PythonCandidate `
            -Command $candidate.Command `
            -PrefixArguments $candidate.PrefixArguments) {
            return $candidate
        }
    }
    throw "Python 3.10 or newer is required but no compatible interpreter was found on PATH."
}

function Test-VenvPython {
    param([string]$PythonPath)

    if (!(Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        return $false
    }
    return Test-PythonCandidate -Command $PythonPath -PrefixArguments @()
}

function Test-Pip {
    param([string]$PythonPath)

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell converts a native process's stderr to error records.
        # A missing module is an expected probe result, not a terminating script error.
        $ErrorActionPreference = "Continue"
        & $PythonPath -m pip --version *> $null
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Reset-BundleVenv {
    param(
        [string]$BundleRoot,
        [string]$VenvPath
    )

    $expected = Join-Path $BundleRoot ".venv"
    if ([System.IO.Path]::GetFullPath($VenvPath) -ne [System.IO.Path]::GetFullPath($expected)) {
        throw "Refusing to reset a virtual environment outside the bundle."
    }
    if (Test-Path -LiteralPath $VenvPath) {
        Remove-Item -LiteralPath $VenvPath -Recurse -Force
    }
}

$bundleRoot = Resolve-BundleRoot -ScriptRoot $PSScriptRoot
$requirementsPath = Join-Path $bundleRoot "requirements-client.txt"
$defaultConfigPath = Join-Path $bundleRoot ".env.client"
$exampleConfigPath = Join-Path $bundleRoot ".env.client.example"

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = $defaultConfigPath
}
if (!(Test-Path -LiteralPath $ConfigPath) -and (Test-Path -LiteralPath $exampleConfigPath)) {
    Copy-Item -LiteralPath $exampleConfigPath -Destination $ConfigPath
    Write-Host "Created $ConfigPath from .env.client.example"
}
if (!(Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Client configuration was not found at '$ConfigPath'."
}

$venvPath = Join-Path $bundleRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$installMarker = Join-Path $venvPath ".client_requirements_installed"

if (!(Test-VenvPython -PythonPath $venvPython)) {
    Reset-BundleVenv -BundleRoot $bundleRoot -VenvPath $venvPath
    $hostPython = Find-CompatiblePython
    $createArguments = @($hostPython.PrefixArguments) + @("-m", "venv", $venvPath)
    Invoke-NativeChecked `
        -FilePath $hostPython.Command `
        -ArgumentList $createArguments `
        -Phase "Creating the sidecar virtual environment"
}
if (!(Test-VenvPython -PythonPath $venvPython)) {
    throw "The sidecar virtual environment was created but its Python is not usable."
}

if (!(Test-Pip -PythonPath $venvPython)) {
    Invoke-NativeChecked `
        -FilePath $venvPython `
        -ArgumentList @("-m", "ensurepip", "--upgrade") `
        -Phase "Repairing pip in the sidecar virtual environment"
}
if (!(Test-Pip -PythonPath $venvPython)) {
    throw "pip is unavailable after attempting repair with ensurepip."
}

$requirementsHash = Get-Sha256 -Path $requirementsPath
$pythonVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "Reading the sidecar Python version failed with exit code $LASTEXITCODE."
}
$fingerprint = "2|$($pythonVersion.Trim())|$requirementsHash"
$installedFingerprint = ""
if (Test-Path -LiteralPath $installMarker -PathType Leaf) {
    $installedFingerprint = (Get-Content -LiteralPath $installMarker -Raw).Trim()
}

if ($installedFingerprint -ne $fingerprint) {
    Invoke-NativeChecked `
        -FilePath $venvPython `
        -ArgumentList @("-m", "pip", "install", "-r", $requirementsPath) `
        -Phase "Installing sidecar requirements from '$requirementsPath'"
    $temporaryMarker = "$installMarker.tmp"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($temporaryMarker, $fingerprint, $utf8NoBom)
    Move-Item -LiteralPath $temporaryMarker -Destination $installMarker -Force
}

$env:CLIENT_ENV_FILE = (Resolve-Path -LiteralPath $ConfigPath).Path
$sidecarExitCode = 1
Push-Location $bundleRoot
try {
    & $venvPython -m client_backend run --config $env:CLIENT_ENV_FILE
    $sidecarExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $sidecarExitCode
