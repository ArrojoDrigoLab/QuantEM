# Build the quantem-server onedir distribution with PyInstaller.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File packaging\pyinstaller\build.ps1
#
# Every cache/temp path is parameterised so the build can be kept off the
# system drive entirely, which some build machines require. Defaults resolve
# relative to the checkout; pass -Python/-Scratch to put them anywhere else.

param(
    [string]$Python = "",
    [string]$Scratch = ""
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

& $Python -m PyInstaller `
    "$PSScriptRoot\quantem-server.spec" `
    --noconfirm `
    --workpath "$Scratch\pyi_build" `
    --distpath "$PSScriptRoot\dist"

if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

$dist = "$PSScriptRoot\dist\quantem-server"
$size = (Get-ChildItem -Recurse $dist | Measure-Object -Property Length -Sum).Sum
"Built $dist  ($([math]::Round($size / 1GB, 2)) GB)"
