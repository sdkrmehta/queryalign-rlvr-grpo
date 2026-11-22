$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json
$lines = $nb3.cells[3].source
for ($i=0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "sql") {
        Write-Host "Index $i : $($lines[$i])"
    }
}
