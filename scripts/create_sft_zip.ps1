$baseDir = "C:\queryalign-rlvr-grpo\artifacts\phase_2"
$zipPath = "C:\queryalign-rlvr-grpo\artifacts\phase_2\sql_grpo_sft_artifacts.zip"

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Add-Type -AssemblyName System.IO.Compression.FileSystem

# Archive outputs folder (specifically best_adapter and metrics)
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)

$filesToInclude = Get-ChildItem -Path "$baseDir\outputs\sft\best_adapter", "$baseDir\outputs\sft\zero_shot_baseline_metrics.json", "$baseDir\outputs\sft\README.md" -Recurse | Where-Object { -not $_.PSIsContainer }

foreach ($f in $filesToInclude) {
    $relPath = $f.FullName.Substring($baseDir.Length + 1).Replace("\", "/")
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $relPath)
}
$zip.Dispose()

$sizeMb = (Get-Item $zipPath).Length / 1MB
Write-Host "Created $zipPath ($([Math]::Round($sizeMb, 2)) MB)"
