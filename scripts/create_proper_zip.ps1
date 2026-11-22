Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$baseDir = "C:\queryalign-rlvr-grpo\artifacts\phase_2"
$zipPath = "C:\queryalign-rlvr-grpo\artifacts\phase_2\sql_grpo_sft_artifacts.zip"

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)

$files = Get-ChildItem -Path "$baseDir\outputs\sft\best_adapter", "$baseDir\outputs\sft\zero_shot_baseline_metrics.json", "$baseDir\outputs\sft\README.md" -Recurse | Where-Object { -not $_.PSIsContainer }

foreach ($f in $files) {
    $entryName = $f.FullName.Substring($baseDir.Length + 1).Replace("\", "/")
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $entryName)
}
$zip.Dispose()

$z = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
Write-Host "SUCCESS: Created $zipPath with $($z.Entries.Count) entries:"
$z.Entries | ForEach-Object { Write-Host "  $($_.FullName)" }
$z.Dispose()
