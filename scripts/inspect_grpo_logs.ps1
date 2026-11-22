$state = Get-Content 'C:\queryalign-rlvr-grpo\artifacts\phase_4\grpo\checkpoint-100\trainer_state.json' -Raw | ConvertFrom-Json
Write-Host "Global Step: $($state.global_step)"
Write-Host "Max Steps:    $($state.max_steps)"
Write-Host "Epoch:        $($state.epoch)"
Write-Host "Total FLOPs:  $($state.total_flos)"

Write-Host "`n--- Sample Log History (Every 10 steps) ---"
$logs = $state.log_history
foreach ($l in $logs) {
    if ($l.step) {
        $r_fmt = if ($l.PSObject.Properties['reward_format']) { [Math]::Round($l.reward_format, 3) } else { "-" }
        $r_ast = if ($l.PSObject.Properties['reward_ast_schema']) { [Math]::Round($l.reward_ast_schema, 3) } else { "-" }
        $r_exc = if ($l.PSObject.Properties['reward_exec_valid']) { [Math]::Round($l.reward_exec_valid, 3) } else { "-" }
        $r_acc = if ($l.PSObject.Properties['reward_accuracy']) { [Math]::Round($l.reward_accuracy, 3) } else { "-" }
        $r_tot = if ($l.PSObject.Properties['reward']) { [Math]::Round($l.reward, 3) } else { "-" }
        $loss  = if ($l.PSObject.Properties['loss']) { [Math]::Round($l.loss, 4) } else { "-" }
        $kl    = if ($l.PSObject.Properties['kl']) { [Math]::Round($l.kl, 4) } else { "-" }
        Write-Host "Step $($l.step) | Loss: $loss | Total Reward: $r_tot | Fmt: $r_fmt | AST: $r_ast | Exec: $r_exc | Acc: $r_acc | KL: $kl"
    }
}
