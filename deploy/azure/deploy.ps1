<#
.SYNOPSIS
Deploys Keel to Azure with Bicep: managed identity, Container App, Azure OpenAI, AI Search, Key Vault.

.DESCRIPTION
Checks that the Azure CLI is installed and signed in, then either previews the deployment (-WhatIf,
`az deployment group what-if`) or creates the resource group, deploys deploy/azure/main.bicep with the
parameters file, prints the outputs and smoke-tests GET <appUrl>/health with five retries.

Auth for the running app is the user-assigned managed identity created by the template; the script
never writes a key anywhere.

.PARAMETER ResourceGroup
Resource group to create or reuse. Default keel-rg.

.PARAMETER Location
Azure region. Default australiaeast (carries gpt-4o-mini and text-embedding-3-small on Standard).

.PARAMETER ParametersFile
Bicep parameters file. Default deploy/azure/main.bicepparam next to this script. Edit `image` there first.

.PARAMETER Subscription
Optional subscription id or name to select before deploying.

.EXAMPLE
.\deploy.ps1 -WhatIf

.EXAMPLE
.\deploy.ps1 -ResourceGroup keel-rg -Location australiaeast
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ResourceGroup = 'keel-rg',
    [string]$Location = 'australiaeast',
    [string]$ParametersFile = (Join-Path $PSScriptRoot 'main.bicepparam'),
    [string]$TemplateFile = (Join-Path $PSScriptRoot 'main.bicep'),
    [string]$Subscription = ''
)

$ErrorActionPreference = 'Stop'

function Write-Step([string]$Text) {
    Write-Host ''
    Write-Host "==> $Text"
}

function Get-DeploymentArgs {
    # A .bicepparam file names its template through `using`, so only the parameters file is passed.
    if ($ParametersFile.ToLower().EndsWith('.bicepparam')) {
        return @('--parameters', $ParametersFile)
    }
    return @('--template-file', $TemplateFile, '--parameters', "@$ParametersFile")
}

# 1. Azure CLI present
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Host 'The Azure CLI is missing. Install it, open a new terminal, then run this script again:'
    Write-Host '  winget install -e --id Microsoft.AzureCLI'
    exit 1
}

# 2. Signed in
Write-Step 'Checking Azure CLI sign-in'
$accountJson = az account show -o json --only-show-errors
if ($LASTEXITCODE -ne 0 -or -not $accountJson) {
    Write-Host 'Sign in first, then run this script again:'
    Write-Host '  az login'
    exit 1
}
if ($Subscription) {
    az account set --subscription $Subscription --only-show-errors
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $accountJson = az account show -o json --only-show-errors
}
$account = ($accountJson | Out-String) | ConvertFrom-Json
Write-Host ("Subscription: {0} ({1})" -f $account.name, $account.id)

if (-not (Test-Path $ParametersFile)) {
    Write-Host "Parameters file $ParametersFile is missing. Copy main.bicepparam, set image, and rerun."
    exit 1
}

$deploymentArgs = Get-DeploymentArgs

# 3a. Preview only
if ($WhatIfPreference) {
    Write-Step "Previewing changes to resource group $ResourceGroup (what-if)"
    $exists = az group exists --name $ResourceGroup --only-show-errors
    if ("$exists".Trim() -ne 'true') {
        Write-Host "Resource group $ResourceGroup is absent. What-if at group scope needs it to exist; an empty group is free. Create it and rerun:"
        Write-Host "  az group create --name $ResourceGroup --location $Location"
        exit 1
    }
    az deployment group what-if --resource-group $ResourceGroup @deploymentArgs
    exit $LASTEXITCODE
}

# 3b. Deploy
Write-Step "Ensuring resource group $ResourceGroup in $Location"
az group create --name $ResourceGroup --location $Location -o none --only-show-errors
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$deploymentName = "keel-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
Write-Step "Deploying Keel as deployment $deploymentName (this takes 10 to 15 minutes; Azure OpenAI deployments are created one at a time)"
$outputsJson = az deployment group create --resource-group $ResourceGroup --name $deploymentName @deploymentArgs --query properties.outputs -o json --only-show-errors
if ($LASTEXITCODE -ne 0) {
    Write-Host "Deployment $deploymentName failed. Inspect it with:"
    Write-Host "  az deployment group show --resource-group $ResourceGroup --name $deploymentName --query properties.error"
    exit $LASTEXITCODE
}
$outputs = ($outputsJson | Out-String) | ConvertFrom-Json

# 4. Outputs
Write-Step 'Outputs'
foreach ($property in $outputs.PSObject.Properties) {
    Write-Host ("  {0} = {1}" -f $property.Name, $property.Value.value)
}

# 5. Smoke test
$appUrl = $outputs.appUrl.value
$appName = $outputs.appName.value
Write-Step "Smoke test: GET $appUrl/health (5 attempts)"
$healthy = $false
for ($attempt = 1; $attempt -le 5; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri "$appUrl/health" -UseBasicParsing -TimeoutSec 20
        if ($response.StatusCode -eq 200) {
            Write-Host "  attempt ${attempt}: 200 OK"
            $healthy = $true
            break
        }
        Write-Host "  attempt ${attempt}: HTTP $($response.StatusCode)"
    } catch {
        Write-Host "  attempt ${attempt}: $($_.Exception.Message)"
    }
    if ($attempt -lt 5) { Start-Sleep -Seconds 15 }
}

if ($healthy) {
    Write-Host ''
    Write-Host "Keel is up at $appUrl"
    Write-Host "Ingest a corpus and run the evals against it: see deploy/azure/README.md, section 'After deploy'."
    exit 0
}

Write-Host ''
Write-Host 'The app is still starting. Watch the container logs, then GET /health again:'
Write-Host "  az containerapp logs show --name $appName --resource-group $ResourceGroup --follow"
exit 2
