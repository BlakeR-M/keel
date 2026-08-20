<#
.SYNOPSIS
Stops the processes deploy\onprem\run.ps1 started, using the PIDs it saved under data\.

.DESCRIPTION
Reads data\keel-web.pid and data\llama-server.pid (or KEEL_DATA_DIR when set), ends each process that is
still running and matches the expected executable, then removes the pid files. A llama-server that was
already running before run.ps1 has no pid file and is left alone.

.PARAMETER KeepLlama
Stop the web app only and leave llama-server running (handy while iterating on the app).

.EXAMPLE
.\deploy\onprem\stop.ps1
#>
[CmdletBinding()]
param(
    [switch]$KeepLlama
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$DataDir = [Environment]::GetEnvironmentVariable('KEEL_DATA_DIR')
if ([string]::IsNullOrWhiteSpace($DataDir)) { $DataDir = Join-Path $RepoRoot 'data' }

function Stop-FromPidFile([string]$Name, [string]$ExpectedProcessPattern) {
    $pidFile = Join-Path $DataDir "$Name.pid"
    if (-not (Test-Path $pidFile)) {
        Write-Host "${Name}: no pid file at $pidFile, nothing to stop."
        return
    }
    $processId = [int]((Get-Content $pidFile | Select-Object -First 1).Trim())
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Write-Host "${Name}: pid $processId already gone."
    } elseif ($process.ProcessName -notmatch $ExpectedProcessPattern) {
        Write-Host "${Name}: pid $processId is now '$($process.ProcessName)', a different program; leaving it alone."
    } else {
        Stop-Process -Id $processId -Force
        Write-Host "${Name}: pid $processId stopped."
    }
    Remove-Item $pidFile -Force
}

Stop-FromPidFile 'keel-web' '^python|^uvicorn'
if ($KeepLlama) {
    Write-Host "llama-server: kept running (-KeepLlama)."
} else {
    Stop-FromPidFile 'llama-server' '^llama-server'
}
