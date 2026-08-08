$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

function Assert-CommandExists {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "$Name is not installed or not available in PATH. Install it and reopen PowerShell."
        exit 1
    }
}

Assert-CommandExists python
Assert-CommandExists node

Write-Host "Starting LifeTrack backend and proxy server..." -ForegroundColor Cyan

$pythonJob = Start-Job -Name "LifeTrackPython" -ScriptBlock {
    param($rootPath)
    Set-Location $rootPath
    python app.py
} -ArgumentList $root

Start-Sleep -Seconds 3

$nodeJob = Start-Job -Name "LifeTrackNode" -ScriptBlock {
    param($rootPath)
    Set-Location $rootPath
    node server.js
} -ArgumentList $root

Write-Host "LifeTrack services started." -ForegroundColor Green
Write-Host "  Python backend: http://localhost:8501"
Write-Host "  React frontend: http://localhost:3000"
Write-Host "Use Get-Job to inspect jobs and Stop-Job to stop them."
Write-Host "Example: Get-Job | Receive-Job -Keep"

Start-Process "http://localhost:8501"
Start-Process "http://localhost:3000"

Pop-Location
