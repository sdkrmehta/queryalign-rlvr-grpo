$path3 = "c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb"
$raw3 = Get-Content $path3 -Raw
$fixed3 = $raw3.Replace('<\\/sql>', '</sql>').Replace('<\/sql>', '</sql>')
Set-Content -Path $path3 -Value $fixed3 -Encoding UTF8

$path4 = "c:\queryalign-rlvr-grpo\notebooks\04_kaggle_evaluation_majority8.ipynb"
$raw4 = Get-Content $path4 -Raw
$fixed4 = $raw4.Replace('<\\/sql>', '</sql>').Replace('<\/sql>', '</sql>')
Set-Content -Path $path4 -Value $fixed4 -Encoding UTF8

Write-Host "Replaced in Notebook 3 and Notebook 4."
