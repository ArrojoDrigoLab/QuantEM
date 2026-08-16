param(
    [Parameter(Mandatory = $true)]
    [string] $Root,

    [Parameter(Mandatory = $true)]
    [string] $Manifest
)

$ErrorActionPreference = "Stop"

try {
    $document = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
    if ($document.schema -ne 1 -or -not $document.runtime_id -or -not $document.files) {
        exit 2
    }

    $rootPath = [IO.Path]::GetFullPath($Root).TrimEnd("\") + "\"
    foreach ($entry in $document.files) {
        $relative = [string]$entry.path
        if (-not $relative -or [IO.Path]::IsPathRooted($relative)) {
            exit 2
        }
        $path = [IO.Path]::GetFullPath((Join-Path $rootPath $relative))
        if (-not $path.StartsWith($rootPath, [StringComparison]::OrdinalIgnoreCase)) {
            exit 2
        }
        $item = Get-Item -LiteralPath $path -ErrorAction Stop
        if (-not $item.PSIsContainer -and $item.Length -eq [int64]$entry.size) {
            $stream = [IO.File]::OpenRead($path)
            try {
                $sha = [Security.Cryptography.SHA256]::Create()
                try {
                    $actual = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace("-", "").ToLowerInvariant()
                } finally {
                    $sha.Dispose()
                }
            } finally {
                $stream.Dispose()
            }
            if ($actual -eq ([string]$entry.sha256).ToLowerInvariant()) {
                continue
            }
        }
        exit 1
    }
    exit 0
} catch {
    exit 2
}
