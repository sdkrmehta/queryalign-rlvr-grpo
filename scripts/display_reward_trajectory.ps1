$state = Get-Content 'C:\queryalign-rlvr-grpo\artifacts\phase_4\grpo\checkpoint-100\trainer_state.json' -Raw | ConvertFrom-Json
Write-Host "=== GRPO 4-STAGE REWARD TRAJECTORY (100 STEPS) ==="
Write-Host "Step | Total Reward | Accuracy Reward | Schema Reward | Exec Reward | Format Reward | KL Divergence"
Write-Host "---------------------------------------------------------------------------------------------------"
foreach ($l in $state.log_history) {
    if ($l.step) {
        $r_tot = [Math]::Round($l.reward, 3)
        $r_acc = [Math]::Round($l.'rewards/reward_accuracy/mean', 3)
        $r_ast = [Math]::Round($l.'rewards/reward_ast_schema/mean', 3)
        $r_exc = [Math]::Round($l.'rewards/reward_exec_valid/mean', 3)
        $r_fmt = [Math]::Round($l.'rewards/reward_format/mean', 3)
        $kl    = [Math]::Round($l.kl, 4)
        Write-Host ("{0,4} | {1,12} | {2,15} | {3,13} | {4,11} | {5,13} | {6,13}" -f $l.step, $r_tot, $r_acc, $r_ast, $r_exc, $r_fmt, $kl)
    }
}
