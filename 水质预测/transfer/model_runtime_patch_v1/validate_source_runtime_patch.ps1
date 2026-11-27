param(
    [switch]$RebuildManifest
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $root 'sha256_manifest.csv'
$expectedTabICL = '0DB9CB538F114E79026BF08F45F41AD8DD7AD2DE2AACA9A5CA8CD3BD9748AE7A'

if ($RebuildManifest) {
    $rows = @(
        Get-ChildItem -LiteralPath $root -Recurse -File |
            Where-Object { $_.FullName -ne $manifestPath } |
            ForEach-Object {
                $relative = $_.FullName.Substring($root.Length + 1).Replace('\', '/')
                $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToUpperInvariant()
                [PSCustomObject]@{
                    relative_path = $relative
                    size_bytes = $_.Length
                    sha256 = $hash
                }
            }
    )
    $rows | Sort-Object relative_path | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8
    Write-Output ("MANIFEST_REBUILT rows=" + $rows.Count)
    exit 0
}

$failures = New-Object System.Collections.Generic.List[string]
$required = @(
    'historical_exact_versions.csv',
    'historical_pip_freeze.txt',
    'checkpoint_inventory.csv',
    'checkpoint_provenance.md',
    'unresolved_dependencies.md',
    'source_recovery_decision.md',
    'environment_probe.json',
    'checkpoints/tabicl-regressor-v2-20260212.ckpt',
    'source_code/run_tabular_pretrained_ab_original.py',
    'source_code/run_tabiclv2_chla_smoke_original.py',
    'source_code/tabiclv2_adapter_original.py'
)

foreach ($relative in $required) {
    $full = Join-Path $root ($relative.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        $failures.Add("MISSING|required=$relative")
    }
}

$checkpoint = Join-Path $root 'checkpoints\tabicl-regressor-v2-20260212.ckpt'
if (Test-Path -LiteralPath $checkpoint -PathType Leaf) {
    $actual = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actual -ne $expectedTabICL) {
        $failures.Add("HASH_MISMATCH|tabicl-regressor-v2-20260212.ckpt|$actual|$expectedTabICL")
    }
}

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    $failures.Add('MISSING|sha256_manifest.csv; run with -RebuildManifest first')
} else {
    foreach ($row in (Import-Csv -LiteralPath $manifestPath)) {
        $full = Join-Path $root ($row.relative_path.Replace('/', '\'))
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
            $failures.Add("MANIFEST_MISSING|$($row.relative_path)")
            continue
        }
        $file = Get-Item -LiteralPath $full
        $actual = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToUpperInvariant()
        if ([int64]$row.size_bytes -ne [int64]$file.Length -or $actual -ne $row.sha256.ToUpperInvariant()) {
            $failures.Add("MANIFEST_MISMATCH|$($row.relative_path)")
        }
    }
}

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Output 'SOURCE_RUNTIME_PATCH_VALIDATION_PASS'
Write-Output 'No model training or inference was started by this script.'
