$nb = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\02_kaggle_sft_baseline.ipynb' -Raw | ConvertFrom-Json

# Add "import json\n" as the first line of Cell 6 if not already present
$src = $nb.cells[6].source
$firstLine = $src[0]
Write-Host "Current first line of Cell 6: $firstLine"

if ($firstLine -notmatch 'import json') {
    # Prepend import json line
    $newSrc = @("import json`n") + $src
    $nb.cells[6].source = $newSrc
    Write-Host "Added 'import json' to Cell 6."
} else {
    Write-Host "import json already present — no change needed."
}

$nb | ConvertTo-Json -Depth 100 | Set-Content 'c:\queryalign-rlvr-grpo\notebooks\02_kaggle_sft_baseline.ipynb' -Encoding UTF8
Write-Host "Saved."
