$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json
$src = $nb3.cells[3].source -join ""
$lines = $src -split "`n"
for ($i=0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "parse_sql" -or $lines[$i] -match "sql>") {
        Write-Host "Line $i : $($lines[$i])"
    }
}
