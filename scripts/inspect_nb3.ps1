$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json
Write-Host "Notebook 3 total cells: $($nb3.cells.Count)"
for ($i=0; $i -lt $nb3.cells.Count; $i++) {
    $src = $nb3.cells[$i].source -join ""
    if ($src -match "GRPOConfig" -or $src -match "device_map" -or $src -match "CUDA_VISIBLE_DEVICES") {
        Write-Host "Cell $i contains match:"
        Write-Host $src.Substring(0, [Math]::Min(200, $src.Length))
        Write-Host "---"
    }
}
