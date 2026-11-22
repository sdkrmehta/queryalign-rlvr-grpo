try {
    $content = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\02_kaggle_sft_baseline.ipynb' -Raw
    $null = $content | ConvertFrom-Json
    Write-Host "JSON VALID"
} catch {
    Write-Host "JSON ERROR"
    Write-Host $_.Exception.Message
}
