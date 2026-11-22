$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json
$src = $nb3.cells[6].source -join ""
Write-Host $src
