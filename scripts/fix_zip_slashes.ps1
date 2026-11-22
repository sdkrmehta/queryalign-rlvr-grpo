$zipPath = 'C:\queryalign-rlvr-grpo\artifacts\phase_2\sql_grpo_sft_artifacts.zip'
Add-Type -AssemblyName System.IO.Compression.FileSystem

# Rebuild zip with strict forward-slash entries for Linux compatibility
$tempZip = 'C:\queryalign-rlvr-grpo\artifacts\phase_2\sql_grpo_sft_artifacts_fwd.zip'
if (Test-Path $tempZip) { Remove-Item $tempZip -Force }

$base = 'C:\queryalign-rlvr-grpo\artifacts\phase_2'
$files = Get-ChildItem -Path "$base\outputs\sft\best_adapter", "$base\outputs\sft\zero_shot_baseline_metrics.json", "$base\outputs\sft\README.md" -Recurse | Where-Object { -not $_.PSIsContainer }

$zip = [System.IO.Compression.ZipFile]::Open($tempZip, [System.IO.Compression.ZipArchiveMode]::Create)
foreach ($f in $files) {
    $entryName = $f.FullName.Substring($base.Length + 1).Replace("\", "/")
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $entryName)
}
$zip.Dispose()

Remove-Item $zipPath -Force
Move-Item $tempZip $zipPath -Force

$zipCheck = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
Write-Host "Re-verified zip with forward slashes ($($zipCheck.Entries.Count) entries):"
$zipCheck.Entries | Select-Object -First 5 | ForEach-Object { Write-Host ("  " + $_.FullName) }
$zipCheck.Dispose()
