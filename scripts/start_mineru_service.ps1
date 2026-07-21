#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Launches the MinerU FastAPI service persistently.

.DESCRIPTION
    Starts the mineru-api service bound to 0.0.0.0:8765. This allows document parsing
    to use a persistent MinerU process instead of cold-starting a new one for each parse.
    GPU model weights load once on startup instead of on every document parse.

.PARAMETER Host
    The host to bind the service to. Default: 0.0.0.0

.PARAMETER Port
    The port to bind the service to. Default: 8765

.PARAMETER CudaDevices
    CUDA device(s) to use (e.g., "0" for GPU 0). Default: "0"
    Set to "" to let CUDA auto-select, or "CPU" to force CPU mode.

.EXAMPLE
    ./start_mineru_service.ps1
    # Starts mineru-api on 0.0.0.0:8765 with GPU 0

.EXAMPLE
    ./start_mineru_service.ps1 -Port 9000 -CudaDevices "0,1"
    # Starts mineru-api on 0.0.0.0:9000 with GPUs 0 and 1

.NOTES
    Prerequisites:
    - mineru-api command must be installed (part of magic-pdf/mineru package)
    - CUDA drivers and runtime installed for GPU support
    - After starting, set MINERU_API_URL in .env to http://localhost:8765 (or your host:port)
    - The service must be running before starting Celery workers or submitting document parsing tasks

    NSSM Registration (Windows Service Manager):
    To run this as a Windows service using NSSM (Non-Sucking Service Manager):
    1. Install NSSM: choco install nssm -y
    2. Register the service:
       nssm install MinerUService "C:\path\to\Scripts\mineru-api.exe" "--host 0.0.0.0 --port 8765"
    3. Set the working directory (optional):
       $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
       nssm set MinerUService AppDirectory $projectRoot
    4. Start the service:
       nssm start MinerUService
    5. Check status:
       nssm status MinerUService
    6. View logs:
       nssm get MinerUService AppStderr
       nssm get MinerUService AppStdout

    Manual Service Control (with NSSM):
    - Stop:  nssm stop MinerUService
    - Restart: nssm restart MinerUService
    - Remove: nssm remove MinerUService confirm
#>

param(
    [string]$Host = "0.0.0.0",
    [int]$Port = 8765,
    [string]$CudaDevices = "0"
)

# Set CUDA environment variables for GPU support
if ($CudaDevices -and $CudaDevices -ne "CPU") {
    $env:CUDA_VISIBLE_DEVICES = $CudaDevices
    Write-Host "Setting CUDA_VISIBLE_DEVICES=$CudaDevices"
} elseif ($CudaDevices -eq "CPU") {
    Write-Host "Running in CPU mode (no GPU)"
}

Write-Host "Starting MinerU service on $Host`:$Port ..."
Write-Host ""
Write-Host "Once started, configure your .env file with:"
Write-Host "  MINERU_API_URL=http://localhost:$Port"
Write-Host ""
Write-Host "The service must be running before starting Celery workers."
Write-Host "Press Ctrl+C to stop the service."
Write-Host ""

# Launch mineru-api
mineru-api --host $Host --port $Port
