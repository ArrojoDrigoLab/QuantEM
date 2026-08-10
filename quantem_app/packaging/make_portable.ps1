# Assemble the portable QuantEM desktop distribution.
#
#   QuantEM-portable\
#     QuantEM.exe                the Tauri shell (double-click this)
#     quantem-server\            the PyInstaller onedir server sidecar
#       quantem-server.exe
#       _internal\...
#
# The shell looks for quantem-server\quantem-server.exe next to itself, so the
# zip needs no installer, no registry keys and no elevation. Run this after
# both halves are built:
#   1. packaging\pyinstaller\build.ps1           -> packaging\pyinstaller\dist\quantem-server
#   2. (in desktop\)  npm run build:exe          -> <cargo target>\release\QuantEM.exe
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File packaging\make_portable.ps1 `
#       [-CargoTargetDir <repo>\.scratch\cargo-target] [-NoZip]

param(
    [string]$CargoTargetDir = $env:CARGO_TARGET_DIR,
    [string]$OutDir = "",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
$appRoot = Split-Path -Parent $PSScriptRoot                      # quantem_app/
if (-not $CargoTargetDir) { $CargoTargetDir = "$appRoot\desktop\src-tauri\target" }
if (-not $OutDir) { $OutDir = "$appRoot\desktop\dist-portable" }

$shellExe = "$CargoTargetDir\release\QuantEM.exe"
$serverDir = "$appRoot\packaging\pyinstaller\dist\quantem-server"

if (-not (Test-Path $shellExe)) { throw "shell not built: $shellExe (run 'npm run build:exe' in desktop/)" }
if (-not (Test-Path "$serverDir\quantem-server.exe")) { throw "server not built: $serverDir (run packaging\pyinstaller\build.ps1)" }

$stage = "$OutDir\QuantEM-portable"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null

Copy-Item $shellExe "$stage\QuantEM.exe"
"Copying the server sidecar (large)..."
Copy-Item -Recurse $serverDir "$stage\quantem-server"

$size = (Get-ChildItem -Recurse $stage | Measure-Object -Property Length -Sum).Sum
"Staged $stage  ($([math]::Round($size / 1GB, 2)) GB)"

if (-not $NoZip) {
    $zip = "$OutDir\QuantEM-portable-win64.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    "Zipping (this takes a few minutes)..."
    Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
    $zsize = (Get-Item $zip).Length
    "Wrote $zip  ($([math]::Round($zsize / 1GB, 2)) GB)"
}
