# Build the quantem-server onedir distribution with PyInstaller.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File packaging\pyinstaller\build.ps1
#   powershell -ExecutionPolicy Bypass -File packaging\pyinstaller\build.ps1 -VerboseBuild
#
# Every cache/temp path is parameterised so the build can be kept off the
# system drive entirely, which some build machines require. Defaults resolve
# relative to the checkout; pass -Python/-Scratch to put them anywhere else.

param(
    [string]$Python = "",
    [string]$Scratch = "",
    [switch]$VerboseBuild
)

$ErrorActionPreference = "Stop"
$appRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # quantem_app/
$repoRoot = Split-Path -Parent $appRoot
if (-not $Scratch) { $Scratch = Join-Path $repoRoot ".scratch" }
if (-not $Python)  { $Python  = (Get-Command python).Source }

$env:TMP = "$Scratch\tmp"
$env:TEMP = "$Scratch\tmp"
$env:PYINSTALLER_CONFIG_DIR = "$Scratch\pyinstaller-config"
$env:PYTHONPATH = "$appRoot\src"
New-Item -ItemType Directory -Force "$Scratch\tmp" | Out-Null
New-Item -ItemType Directory -Force "$Scratch\pyi_build" | Out-Null

$logLevel = if ($VerboseBuild) { "INFO" } else { "WARN" }
# PyInstaller initializes logging while its package is imported, before it can
# parse the command-line flag below. Set both interfaces so even those startup
# messages use the requested level.
$env:PYI_LOG_LEVEL = $logLevel

# PyInstaller emits one misleading warning when a regular venv was created
# from a Conda-distributed base Python. The venv is intentionally not itself a
# Conda environment, so it has no conda-meta directory; the warning describes
# the expected setup and requires no action. Suppress only that exact message,
# while preserving every other warning and error. Native stderr is captured so
# PowerShell 5.1 does not promote warning lines to terminating ErrorRecords.
$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$pyinstallerOutput = & $Python -m PyInstaller `
    "$PSScriptRoot\quantem-server.spec" `
    --noconfirm `
    --log-level $logLevel `
    --workpath "$Scratch\pyi_build" `
    --distpath "$PSScriptRoot\dist" 2>&1
$pyinstallerExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference

$layeredVenvNotice = "Assuming this is not an Anaconda environment or an additional venv/pipenv/"
foreach ($line in $pyinstallerOutput) {
    $text = $line.ToString()
    if ($text.Contains($layeredVenvNotice)) { continue }
    Write-Host $text
}

if ($pyinstallerExitCode -ne 0) { throw "PyInstaller failed with exit code $pyinstallerExitCode" }

$dist = "$PSScriptRoot\dist\quantem-server"
$size = (Get-ChildItem -Recurse $dist | Measure-Object -Property Length -Sum).Sum
"Built $dist  ($([math]::Round($size / 1GB, 2)) GB)"
