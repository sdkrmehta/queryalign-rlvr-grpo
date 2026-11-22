$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json
Write-Host "--- Notebook 3 Step 2 (Paths & Auto Unpack) ---"
$src2 = $nb3.cells[2].source -join ""
Write-Host $src2

Write-Host "`n--- Notebook 3 Step 4 (Load Base Model & Resume SFT Adapter) ---"
$src4 = $nb3.cells[4].source -join ""
Write-Host $src4
