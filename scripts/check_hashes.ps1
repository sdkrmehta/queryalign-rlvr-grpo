$f1 = 'C:\queryalign-rlvr-grpo\artifacts\phase_2\outputs\sft\best_adapter\adapter_model.safetensors'
$f2 = 'C:\queryalign-rlvr-grpo\artifacts\phase_2\outputs\sft\checkpoint-1314\adapter_model.safetensors'
$h1 = (Get-FileHash $f1 -Algorithm SHA256).Hash
$h2 = (Get-FileHash $f2 -Algorithm SHA256).Hash
Write-Host "best_adapter hash:    $h1"
Write-Host "checkpoint-1314 hash: $h2"
Write-Host "Hashes match:         $($h1 -eq $h2)"
