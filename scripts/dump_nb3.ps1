$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json
Write-Host "Notebook 3 has $($nb3.cells.Count) cells."
for ($i=0; $i -lt $nb3.cells.Count; $i++) {
    Write-Host "`n==============================================="
    Write-Host "CELL $i ($($nb3.cells[$i].cell_type))"
    Write-Host "==============================================="
    $src = $nb3.cells[$i].source -join ""
    Write-Host $src
}
