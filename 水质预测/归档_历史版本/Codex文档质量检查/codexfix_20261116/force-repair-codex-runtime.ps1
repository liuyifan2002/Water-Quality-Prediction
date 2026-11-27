param(
    [int]$TimeoutMinutes = 60
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logPath = Join-Path $scriptDir "force-repair-codex-runtime.log"
$donePath = Join-Path $scriptDir "force-repair-codex-runtime.done"

function Write-Step {
    param([string]$Message)
    $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$time] $Message"
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

function Invoke-Robocopy {
    param(
        [string]$Source,
        [string]$Destination,
        [string[]]$ExtraArgs = @()
    )

    Write-Step ("Running robocopy from '{0}' to '{1}'" -f $Source, $Destination)
    & robocopy.exe $Source $Destination @ExtraArgs | Tee-Object -FilePath $logPath -Append | Out-Host
    $code = $LASTEXITCODE
    Write-Step "Robocopy exit code: $code"

    if ($code -ge 8) {
        throw "Robocopy failed with exit code $code"
    }
}

$startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Set-Content -Path $logPath -Value "[$startTime] force-repair script started."
if (Test-Path $donePath) {
    Remove-Item $donePath -Force -ErrorAction SilentlyContinue
}

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$pkg = Get-AppxPackage -Name OpenAI.Codex

if (-not $pkg) {
    throw "OpenAI.Codex package was not found."
}

$packageRoot = $pkg.InstallLocation
$sourceRuntimeBin = Join-Path $packageRoot "app\resources\cua_node\bin"
$sourceBinFiles = @(
    "app\resources\codex.exe",
    "app\resources\codex",
    "app\resources\rg.exe",
    "app\resources\rg",
    "app\resources\codex-command-runner.exe"
)

$localRoot = Join-Path $env:LOCALAPPDATA "OpenAI\Codex"
$localBin = Join-Path $localRoot "bin"
$localRuntimeRoot = Join-Path $localRoot "runtimes\cua_node"
$localRuntimeBin = Join-Path $localRuntimeRoot "bin"
$backupRoot = Join-Path $env:LOCALAPPDATA ("OpenAI\Codex.forcebak-" + (Get-Date -Format "yyyyMMdd-HHmmss"))

Write-Step "Waiting for all Codex processes to exit."
while ((Get-Process -Name Codex -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
}

if (Get-Process -Name Codex -ErrorAction SilentlyContinue) {
    throw "Timeout: Codex is still running. Close every Codex window and try again."
}

Write-Step "Confirmed that all Codex processes have exited."

if (Test-Path $localRoot) {
    Write-Step "Backing up current local runtime directory to $backupRoot"
    Rename-Item -Path $localRoot -NewName ([System.IO.Path]::GetFileName($backupRoot))
}

Write-Step "Creating fresh local runtime directories."
New-Item -ItemType Directory -Force -Path $localBin | Out-Null
New-Item -ItemType Directory -Force -Path $localRuntimeBin | Out-Null

Write-Step "Copying packaged runtime files to local runtime directory."
Invoke-Robocopy -Source $sourceRuntimeBin -Destination $localRuntimeBin -ExtraArgs @("/E","/R:1","/W:1","/NFL","/NDL")

Write-Step "Copying packaged launcher files to local bin directory."
$launcherSourceDir = Join-Path $packageRoot "app\resources"
$launcherArgs = @(
    "codex.exe",
    "codex",
    "rg.exe",
    "rg",
    "codex-command-runner.exe",
    "/R:1",
    "/W:1",
    "/NFL",
    "/NDL"
)
Invoke-Robocopy -Source $launcherSourceDir -Destination $localBin -ExtraArgs $launcherArgs

Write-Step "Re-registering the Codex package."
Add-AppxPackage -DisableDevelopmentMode -Register (Join-Path $packageRoot "AppxManifest.xml")

Write-Step "Force repair complete. Restart Codex and verify that files now exist under %LOCALAPPDATA%\\OpenAI\\Codex."
Set-Content -Path $donePath -Value "completed $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
