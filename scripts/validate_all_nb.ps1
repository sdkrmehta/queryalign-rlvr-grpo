$paths = @(
    'c:\queryalign-rlvr-grpo\notebooks\01_kaggle_data_engineering.ipynb',
    'c:\queryalign-rlvr-grpo\notebooks\02_kaggle_sft_baseline.ipynb',
    'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb',
    'c:\queryalign-rlvr-grpo\notebooks\04_kaggle_evaluation_majority8.ipynb'
)
foreach ($p in $paths) {
    try {
        $null = Get-Content $p -Raw | ConvertFrom-Json
        Write-Host "VALID: $p"
    } catch {
        Write-Host "ERROR in $p : $($_.Exception.Message)"
    }
}
