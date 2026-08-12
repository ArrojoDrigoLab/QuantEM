[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Target,
    [string] $Config = "",
    [ValidateRange(1, 10)]
    [int] $Attempts = 3,
    [ValidateRange(0, 300)]
    [int] $DelaySeconds = 15
)

$ErrorActionPreference = "Stop"
$tauriArguments = @("tauri", "build", "--target", $Target, "--bundles", "nsis")
if ($Config) {
    $tauriArguments += @("--config", $Config)
}

for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
    # Tauri downloads its NSIS helper binaries during bundling. A transient
    # disconnect must not discard a completed Rust compilation.
    $ErrorActionPreference = "Continue"
    & npx @tauriArguments
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"

    if ($exitCode -eq 0) {
        return
    }
    if ($attempt -eq $Attempts) {
        throw "Tauri bundling failed after $Attempts attempts (last exit code $exitCode)."
    }

    $delay = $DelaySeconds * $attempt
    Write-Warning "Tauri bundling attempt $attempt failed with exit code $exitCode; retrying in $delay seconds."
    Start-Sleep -Seconds $delay
}
