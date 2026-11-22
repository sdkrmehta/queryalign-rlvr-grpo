$state = Get-Content 'C:\queryalign-rlvr-grpo\artifacts\phase_4\grpo\checkpoint-100\trainer_state.json' -Raw | ConvertFrom-Json
$state.log_history[0] | ConvertTo-Json
