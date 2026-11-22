$zipPath = 'C:\queryalign-rlvr-grpo\artifacts\phase_2\sql_grpo_sft_artifacts.zip'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
Write-Host "Zip Entries count: $($zip.Entries.Count)"
$zip.Entries | Select-Object -First 15 | ForEach-Object { Write-Host (" - " + $_.FullName + " (" + [Math]::Round($_.Length / 1KB, 1) + " KB)") }
$zip.Dispose()
