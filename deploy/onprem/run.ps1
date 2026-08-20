<#
.SYNOPSIS
Starts the Keel on-premise stack without Docker: llama-server on 127.0.0.1:8081, then the Keel web app on 127.0.0.1:8400.

.DESCRIPTION
Idempotent: a llama-server already answering on the port is left alone, and so is a web app already answering
on 8400. Processes this script starts run in the background with their PIDs saved under data\ so stop.ps1
can end them. Output goes to data\llama-server.out|err and data\keel-web.out|err.

Environment (every value has a default):
  KEEL_LLAMA_SERVER   path to llama-server.exe        default D:\llama.cpp\bin\llama-server.exe
  KEEL_MODEL_PATH     GGUF model file                 default D:\models\qwen2.5-3b-instruct-q4_k_m.gguf
  KEEL_NGL            GPU layers to offload           default 0 (CPU only)
  KEEL_LLAMA_CTX      context length                  default 8192
  KEEL_LLAMA_THREADS  CPU threads                     default logical processor count
  KEEL_LLAMA_ALIAS    model alias served by the API   default qwen2.5-3b-instruct
  KEEL_LLAMA_PORT     llama-server port               default 8081
  KEEL_LLAMA_EXTRA_ARGS  extra llama-server flags     default none
  KEEL_PYTHON         interpreter for the web app     default <repo>\.venv\Scripts\python.exe
  KEEL_WEB_APP        ASGI app for uvicorn            default keel.web.app:app
  KEEL_DATA_DIR       pid and log directory           default <repo>\data
  KEEL_AIRGAP         1 turns on the egress guard in the app (passed through unchanged)

.PARAMETER SkipWeb
Start llama-server only. demo.ps1 uses this to ingest the corpus before the web app comes up.

.PARAMETER SkipLlama
Start the web app only (the LLM endpoint is expected to be answering already).

.PARAMETER TimeoutSeconds
How long to wait for each service to answer. Default 300 (a large model on CPU takes a while to load).

.EXAMPLE
.\deploy\onprem\run.ps1
.EXAMPLE
$env:KEEL_MODEL_PATH = 'D:\models\Qwen3.5-9B-Q5_K_M.gguf'; $env:KEEL_NGL = '99'; .\deploy\onprem\run.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipWeb,
    [switch]$SkipLlama,
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Get-EnvOrDefault([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value
}

function Format-Arg([string]$Value) {
    if ($Value -match '\s') { return '"' + $Value + '"' }
    return $Value
}

$DataDir     = Get-EnvOrDefault 'KEEL_DATA_DIR' (Join-Path $RepoRoot 'data')
$LlamaServer = Get-EnvOrDefault 'KEEL_LLAMA_SERVER' 'D:\llama.cpp\bin\llama-server.exe'
$ModelPath   = Get-EnvOrDefault 'KEEL_MODEL_PATH' 'D:\models\qwen2.5-3b-instruct-q4_k_m.gguf'
$Ngl         = Get-EnvOrDefault 'KEEL_NGL' '0'
$Ctx         = Get-EnvOrDefault 'KEEL_LLAMA_CTX' '8192'
$Threads     = Get-EnvOrDefault 'KEEL_LLAMA_THREADS' ([string][Environment]::ProcessorCount)
$Alias       = Get-EnvOrDefault 'KEEL_LLAMA_ALIAS' 'qwen2.5-3b-instruct'
$LlamaPort   = [int](Get-EnvOrDefault 'KEEL_LLAMA_PORT' '8081')
$ExtraArgs   = Get-EnvOrDefault 'KEEL_LLAMA_EXTRA_ARGS' ''
$Python      = Get-EnvOrDefault 'KEEL_PYTHON' (Join-Path $RepoRoot '.venv\Scripts\python.exe')
$WebApp      = Get-EnvOrDefault 'KEEL_WEB_APP' 'keel.web.app:app'
$WebPort     = 8400

$LlamaModelsUrl = "http://127.0.0.1:$LlamaPort/v1/models"
$WebUrl         = "http://127.0.0.1:$WebPort/"

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
# An empty stdin file for the child processes, so a caller that pipes this script's output is never held open by them.
$NullIn = Join-Path $DataDir 'null.in'
if (-not (Test-Path $NullIn)) { New-Item -ItemType File -Path $NullIn -Force | Out-Null }

function Test-HttpOk([string]$Url) {
    # True for a 2xx answer. llama-server answers 503 while the model is still loading.
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Test-HttpAnswers([string]$Url) {
    # True for any HTTP answer, status code aside: the server is up.
    try {
        Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 | Out-Null
        return $true
    } catch {
        $response = $null
        try { $response = $_.Exception.Response } catch { }
        return ($null -ne $response)
    }
}

function Wait-ForService([string]$Name, [scriptblock]$Ready, [System.Diagnostics.Process]$Process, [int]$Seconds, [string]$ErrLog) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Ready) { return }
        if ($Process.HasExited) {
            $tail = ''
            if (Test-Path $ErrLog) { $tail = (Get-Content $ErrLog -Tail 15) -join "`n" }
            throw "$Name exited with code $($Process.ExitCode) before it was ready. Last lines of ${ErrLog}:`n$tail"
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name did not answer within $Seconds seconds. Check $ErrLog, or raise -TimeoutSeconds for a large model."
}

function Save-Pid([string]$Name, [System.Diagnostics.Process]$Process) {
    # Touching Handle now keeps ExitCode readable later, should the process die while we wait.
    $null = $Process.Handle
    Set-Content -Path (Join-Path $DataDir "$Name.pid") -Value $Process.Id -Encoding ascii
}

# --- llama-server -------------------------------------------------------------------------------
if ($SkipLlama) {
    Write-Host "llama-server: skipped (-SkipLlama). Expecting an OpenAI-compatible endpoint at $LlamaModelsUrl"
} elseif (Test-HttpOk $LlamaModelsUrl) {
    Write-Host "llama-server: already answering at $LlamaModelsUrl, leaving it running."
} else {
    if (-not (Test-Path $LlamaServer)) {
        throw "llama-server binary missing at $LlamaServer. Set KEEL_LLAMA_SERVER to your llama-server.exe (llama.cpp release binaries)."
    }
    if (-not (Test-Path $ModelPath)) {
        throw "Model file missing at $ModelPath. Set KEEL_MODEL_PATH to a GGUF file (see docs/onprem.md for the model swap)."
    }
    $llamaArgs = @(
        '-m', (Format-Arg $ModelPath),
        '--host', '127.0.0.1',
        '--port', "$LlamaPort",
        '--jinja',
        '--alias', $Alias,
        '-ngl', "$Ngl",
        '-c', "$Ctx",
        '-t', "$Threads"
    )
    if ($ExtraArgs) { $llamaArgs += ($ExtraArgs -split '\s+' | Where-Object { $_ }) }
    Write-Host "llama-server: starting $LlamaServer $($llamaArgs -join ' ')"
    $llamaProc = Start-Process -FilePath $LlamaServer -ArgumentList $llamaArgs -WorkingDirectory $RepoRoot `
        -RedirectStandardInput $NullIn `
        -RedirectStandardOutput (Join-Path $DataDir 'llama-server.out') `
        -RedirectStandardError (Join-Path $DataDir 'llama-server.err') `
        -WindowStyle Hidden -PassThru
    Save-Pid 'llama-server' $llamaProc
    Write-Host "llama-server: pid $($llamaProc.Id), waiting for $LlamaModelsUrl (model load can take a minute on CPU)"
    Wait-ForService 'llama-server' { Test-HttpOk $LlamaModelsUrl } $llamaProc $TimeoutSeconds (Join-Path $DataDir 'llama-server.err')
    Write-Host "llama-server: ready at http://127.0.0.1:$LlamaPort/v1 (alias $Alias, ngl $Ngl)"
}

# --- Keel web app -------------------------------------------------------------------------------
if ($SkipWeb) {
    Write-Host "keel web: skipped (-SkipWeb)."
    exit 0
}
if (Test-HttpAnswers $WebUrl) {
    Write-Host "keel web: already answering at $WebUrl, leaving it running."
    exit 0
}
if (-not (Test-Path $Python)) {
    throw "Python interpreter missing at $Python. Run demo.ps1 once (it creates .venv), or set KEEL_PYTHON."
}
if ([string]::IsNullOrWhiteSpace($env:KEEL_LOCAL_LLM_BASE_URL)) { $env:KEEL_LOCAL_LLM_BASE_URL = "http://127.0.0.1:$LlamaPort/v1" }
if ([string]::IsNullOrWhiteSpace($env:KEEL_LOCAL_LLM_MODEL)) { $env:KEEL_LOCAL_LLM_MODEL = $Alias }
if ([string]::IsNullOrWhiteSpace($env:HF_HUB_DISABLE_SYMLINKS_WARNING)) { $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1' }
if ([string]::IsNullOrWhiteSpace($env:KEEL_DATA_DIR)) { $env:KEEL_DATA_DIR = $DataDir }

$webArgs = @('-m', 'uvicorn', $WebApp, '--host', '127.0.0.1', '--port', "$WebPort")
Write-Host "keel web: starting $Python $($webArgs -join ' ')"
$webProc = Start-Process -FilePath $Python -ArgumentList $webArgs -WorkingDirectory $RepoRoot `
    -RedirectStandardInput $NullIn `
    -RedirectStandardOutput (Join-Path $DataDir 'keel-web.out') `
    -RedirectStandardError (Join-Path $DataDir 'keel-web.err') `
    -WindowStyle Hidden -PassThru
Save-Pid 'keel-web' $webProc
Wait-ForService 'keel web' { Test-HttpAnswers $WebUrl } $webProc $TimeoutSeconds (Join-Path $DataDir 'keel-web.err')
Write-Host "keel web: ready at $WebUrl (pid $($webProc.Id)). Stop everything with deploy\onprem\stop.ps1"
