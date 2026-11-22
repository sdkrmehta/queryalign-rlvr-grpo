$f1 = 'C:\queryalign-rlvr-grpo\artifacts\phase_4\grpo\final_adapter\adapter_model.safetensors'
$f2 = 'C:\queryalign-rlvr-grpo\artifacts\phase_4\grpo\checkpoint-100\adapter_model.safetensors'
$h1 = (Get-FileHash $f1 -Algorithm SHA256).Hash
$h2 = (Get-FileHash $f2 -Algorithm SHA256).Hash
Write-Host "final_adapter hash:   $h1"
Write-Host "checkpoint-100 hash:  $h2"
Write-Host "Hashes match:         $($h1 -eq $h2)"
