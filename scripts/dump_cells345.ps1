$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json
Write-Host "=== CELL 3 ==="
Write-Host ($nb3.cells[3].source -join "")
Write-Host "`n=== CELL 4 ==="
Write-Host ($nb3.cells[4].source -join "")
Write-Host "`n=== CELL 5 ==="
Write-Host ($nb3.cells[5].source -join "")
