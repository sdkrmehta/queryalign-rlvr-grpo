$nb4 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\04_kaggle_evaluation_majority8.ipynb' -Raw | ConvertFrom-Json
for ($i=2; $i -lt $nb4.cells.Count; $i++) {
    Write-Host "`n==============================================="
    Write-Host "CELL $i ($($nb4.cells[$i].cell_type))"
    Write-Host "==============================================="
    $src = $nb4.cells[$i].source -join ""
    Write-Host $src
}
