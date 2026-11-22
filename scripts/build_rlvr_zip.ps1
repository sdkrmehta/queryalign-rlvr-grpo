Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$baseDir = "C:\queryalign-rlvr-grpo\artifacts\phase_4\staging"
if (Test-Path $baseDir) { Remove-Item $baseDir -Recurse -Force }

$targetAdapter = "$baseDir\outputs\grpo\final_adapter"
New-Item -ItemType Directory -Path $targetAdapter -Force | Out-Null

# Copy final_adapter files
Copy-Item -Path "C:\queryalign-rlvr-grpo\artifacts\phase_4\grpo\final_adapter\*" -Destination $targetAdapter -Recurse -Force
# Copy metrics and README
Copy-Item -Path "C:\queryalign-rlvr-grpo\artifacts\phase_4\grpo\grpo_evaluation_metrics.json" -Destination "$baseDir\outputs\grpo" -Force
Copy-Item -Path "C:\queryalign-rlvr-grpo\artifacts\phase_4\grpo\README.md" -Destination "$baseDir\outputs\grpo" -Force

$zipPath = "C:\queryalign-rlvr-grpo\artifacts\phase_4\sql_grpo_rlvr_artifacts.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
$files = Get-ChildItem -Path "$baseDir\outputs" -Recurse | Where-Object { -not $_.PSIsContainer }

foreach ($f in $files) {
    $entryName = $f.FullName.Substring($baseDir.Length + 1).Replace("\", "/")
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $entryName)
}
$zip.Dispose()
Remove-Item $baseDir -Recurse -Force

$z = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
Write-Host "SUCCESS: Created $zipPath ($([Math]::Round((Get-Item $zipPath).Length / 1MB, 2)) MB) with $($z.Entries.Count) entries:"
$z.Entries | ForEach-Object { Write-Host "  $($_.FullName)" }
$z.Dispose()
