$state = Get-Content 'C:\queryalign-rlvr-grpo\artifacts\phase_2\outputs\sft\checkpoint-1314\trainer_state.json' -Raw | ConvertFrom-Json
Write-Host "Global Step: $($state.global_step)"
Write-Host "Max Steps: $($state.max_steps)"
Write-Host "Total FLOPs: $($state.total_flos)"
Write-Host "Epoch: $($state.epoch)"
Write-Host "Train Runtime (s): $($state.train_runtime)"

Write-Host "`n--- Sample Log History (First 5 & Last 5 steps) ---"
$logs = $state.log_history
Write-Host "Total log entries: $($logs.Count)"
for ($i=0; $i -lt [Math]::Min(5, $logs.Count); $i++) {
    Write-Host ("Step " + $logs[$i].step + ": loss = " + $logs[$i].loss + " (epoch " + $logs[$i].epoch + ")")
}
Write-Host "..."
for ($i=[Math]::Max(0, $logs.Count - 6); $i -lt $logs.Count; $i++) {
    if ($logs[$i].step) {
        Write-Host ("Step " + $logs[$i].step + ": loss = " + $logs[$i].loss + " (epoch " + $logs[$i].epoch + ")")
    } elseif ($logs[$i].train_loss) {
        Write-Host ("Final train_loss = " + $logs[$i].train_loss)
    }
}
