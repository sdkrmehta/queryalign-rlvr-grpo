$state = Get-Content 'C:\queryalign-rlvr-grpo\artifacts\phase_2\outputs\sft\checkpoint-1314\trainer_state.json' -Raw | ConvertFrom-Json
$state.log_history[-1] | ConvertTo-Json
