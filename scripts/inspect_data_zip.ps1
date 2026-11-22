Add-Type -AssemblyName System.IO.Compression.FileSystem

$z2 = "C:\queryalign-rlvr-grpo\sql_grpo_data.zip"
Write-Host "=== Inspecting: $z2 ==="
$zip2 = [System.IO.Compression.ZipFile]::OpenRead($z2)
Write-Host "Total entries: $($zip2.Entries.Count)"
$zip2.Entries | Select-Object -First 25 | ForEach-Object {
    Write-Host ("  " + $_.FullName + " (" + [Math]::Round($_.Length / 1KB, 1) + " KB)")
}
$zip2.Dispose()
