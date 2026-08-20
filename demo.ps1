<#
.SYNOPSIS
Fresh-clone demo on Windows: creates .venv, installs Keel, starts llama-server, ingests the fixture corpus,
starts the web app and prints the URL plus a question to ask.

.DESCRIPTION
Every step is guarded with a plain message and a rerun hint. Rerunning is safe: services already answering
are left alone, ingest is idempotent (a second run adds zero chunks).

Prerequisites: Python 3.11+ on PATH (py launcher or python), llama.cpp binaries (KEEL_LLAMA_SERVER, default
D:\llama.cpp\bin\llama-server.exe) and a GGUF model (KEEL_MODEL_PATH, default
D:\models\qwen2.5-3b-instruct-q4_k_m.gguf). See docs\onprem.md for the model swap and GPU offload.

.PARAMETER Airgap
Run the app with KEEL_AIRGAP=1: every outbound connection other than loopback is refused in-process.

.PARAMETER SkipInstall
Skip `pip install -e .[dev]` (the environment is already prepared).

.PARAMETER TimeoutSeconds
How long to wait for llama-server and the web app to answer. Default 300.

.EXAMPLE
.\demo.ps1
.EXAMPLE
.\demo.ps1 -Airgap
#>
[CmdletBinding()]
param(
    [switch]$Airgap,
    [switch]$SkipInstall,
    [int]$TimeoutSeconds = 300
)

# Native commands print to stderr freely here; exit codes are checked by hand after each call.
$ErrorActionPreference = 'Continue'

$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot
$VenvPy = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$RunScript = Join-Path $RepoRoot 'deploy\onprem\run.ps1'

function Step([string]$Text) {
    Write-Host ''
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Fail([string]$Text) {
    Write-Host ''
    Write-Host "demo: $Text" -ForegroundColor Red
    exit 1
}

function Find-BasePython {
    # Returns @{ Exe; Args } for the first interpreter that is Python 3.11 or newer.
    $candidates = @(
        @{ Exe = 'py'; Args = @('-3.11') },
        @{ Exe = 'py'; Args = @('-3') },
        @{ Exe = 'python'; Args = @() },
        @{ Exe = 'python3'; Args = @() }
    )
    $probe = 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'
    foreach ($candidate in $candidates) {
        if ($null -eq (Get-Command $candidate.Exe -ErrorAction SilentlyContinue)) { continue }
        $probeArgs = @($candidate.Args) + @('-c', $probe)
        & $candidate.Exe @probeArgs | Out-Null
        if ($LASTEXITCODE -eq 0) { return $candidate }
    }
    return $null
}

# --- 1. virtual environment ---------------------------------------------------------------------
if (Test-Path $VenvPy) {
    Step "Using the existing .venv"
} else {
    Step "Creating .venv"
    $base = Find-BasePython
    if ($null -eq $base) {
        Fail "Python 3.11 or newer was not found on PATH. Install it from python.org (tick 'Add to PATH'), then rerun .\demo.ps1"
    }
    $venvArgs = @($base.Args) + @('-m', 'venv', '.venv')
    & $base.Exe @venvArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) {
        Fail "Creating .venv failed (exit $LASTEXITCODE). Remove the .venv folder and rerun .\demo.ps1"
    }
}

# --- 2. install ---------------------------------------------------------------------------------
if ($SkipInstall) {
    Step "Skipping pip install (-SkipInstall)"
} else {
    Step "Installing Keel into .venv (pip install -e .[dev])"
    & $VenvPy -m pip install --disable-pip-version-check -q -e ".[dev]"
    if ($LASTEXITCODE -ne 0) {
        Fail "pip install failed (exit $LASTEXITCODE). Check the message above (network access is needed for the first install), then rerun .\demo.ps1"
    }
}

# --- 3. air-gap flag ----------------------------------------------------------------------------
if ($Airgap) {
    $env:KEEL_AIRGAP = '1'
    Step "Air-gap mode on (KEEL_AIRGAP=1): the app refuses every outbound connection other than loopback"
}

# --- 4. llama-server ----------------------------------------------------------------------------
Step "Starting llama-server (deploy\onprem\run.ps1 -SkipWeb)"
try {
    & $RunScript -SkipWeb -TimeoutSeconds $TimeoutSeconds
} catch {
    Fail "llama-server did not start: $($_.Exception.Message)`nFix the cause and rerun .\demo.ps1 (KEEL_LLAMA_SERVER and KEEL_MODEL_PATH point at the binary and the model)."
}

# --- 5. ingest ----------------------------------------------------------------------------------
Step "Ingesting the fixture corpus (fixtures\corpus.yaml)"
& $VenvPy -m keel.cli ingest --manifest fixtures\corpus.yaml
if ($LASTEXITCODE -ne 0) {
    Fail "Ingest failed (exit $LASTEXITCODE). Read the message above, then rerun: .venv\Scripts\python.exe -m keel.cli ingest --manifest fixtures\corpus.yaml"
}

# --- 6. web app ---------------------------------------------------------------------------------
Step "Starting the Keel web app (deploy\onprem\run.ps1)"
try {
    & $RunScript -TimeoutSeconds $TimeoutSeconds
} catch {
    Fail "The web app did not start: $($_.Exception.Message)`nRead data\keel-web.err, fix the cause and rerun .\demo.ps1"
}

# --- 7. done ------------------------------------------------------------------------------------
Write-Host ''
Write-Host 'Keel is running: http://127.0.0.1:8400' -ForegroundColor Green
Write-Host 'Try asking: How many quotes does a $20,000 purchase need at Northbank Council?'
Write-Host 'Stop everything with: .\deploy\onprem\stop.ps1'
