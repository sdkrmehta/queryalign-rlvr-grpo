Add-Type -AssemblyName System.IO.Compression.FileSystem

$z1 = "C:\queryalign-rlvr-grpo\artifacts\phase_2\New folder\sql_grpo_sft.zip"
Write-Host "=== Inspecting: $z1 ==="
$zip1 = [System.IO.Compression.ZipFile]::OpenRead($z1)
Write-Host "Total entries: $($zip1.Entries.Count)"
$zip1.Entries | ForEach-Object {
    Write-Host ("  " + $_.FullName + " (" + [Math]::Round($_.Length / 1KB, 1) + " KB)")
}
$zip1.Dispose()
