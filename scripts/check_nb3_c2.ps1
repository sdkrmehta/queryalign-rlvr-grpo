$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json
$c2 = $nb3.cells[2].source -join ""
Write-Host $c2
