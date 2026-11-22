$tempDir = 'C:\queryalign-rlvr-grpo\artifacts\phase_2\staging\outputs\sft'
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
Copy-Item -Path 'C:\queryalign-rlvr-grpo\artifacts\phase_2\outputs\sft\best_adapter' -Destination $tempDir -Recurse -Force
Copy-Item -Path 'C:\queryalign-rlvr-grpo\artifacts\phase_2\outputs\sft\zero_shot_baseline_metrics.json' -Destination $tempDir -Force
Copy-Item -Path 'C:\queryalign-rlvr-grpo\artifacts\phase_2\outputs\sft\README.md' -Destination $tempDir -Force

$zipPath = 'C:\queryalign-rlvr-grpo\artifacts\phase_2\sql_grpo_sft_artifacts.zip'
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path 'C:\queryalign-rlvr-grpo\artifacts\phase_2\staging\outputs' -DestinationPath $zipPath -CompressionLevel Optimal
Remove-Item -Path 'C:\queryalign-rlvr-grpo\artifacts\phase_2\staging' -Recurse -Force

$item = Get-Item $zipPath
$mb = [Math]::Round($item.Length / 1MB, 2)
Write-Host "Created: $($item.FullName) ($mb MB)"
