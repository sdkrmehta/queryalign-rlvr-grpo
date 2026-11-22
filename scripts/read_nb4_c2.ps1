$nb4 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\04_kaggle_evaluation_majority8.ipynb' -Raw | ConvertFrom-Json
Write-Host ($nb4.cells[2].source -join "")
