$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json

for ($c = 0; $c -lt $nb3.cells.Count; $c++) {
    $src = $nb3.cells[$c].source
    $newSrc = @()
    foreach ($line in $src) {
        $newLine = $line.Replace('<\/sql>', '</sql>')
        $newSrc += $newLine
    }
    $nb3.cells[$c].source = $newSrc
}

$nb3 | ConvertTo-Json -Depth 100 | Set-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Encoding UTF8
Write-Host "Replaced in Notebook 3."

$nb4 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\04_kaggle_evaluation_majority8.ipynb' -Raw | ConvertFrom-Json

for ($c = 0; $c -lt $nb4.cells.Count; $c++) {
    $src = $nb4.cells[$c].source
    $newSrc = @()
    foreach ($line in $src) {
        $newLine = $line.Replace('<\/sql>', '</sql>')
        $newSrc += $newLine
    }
    $nb4.cells[$c].source = $newSrc
}

$nb4 | ConvertTo-Json -Depth 100 | Set-Content 'c:\queryalign-rlvr-grpo\notebooks\04_kaggle_evaluation_majority8.ipynb' -Encoding UTF8
Write-Host "Replaced in Notebook 4."
