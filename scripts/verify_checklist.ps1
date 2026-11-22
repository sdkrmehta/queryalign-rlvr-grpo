$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json
$allText = ($nb3.cells | ForEach-Object { $_.source -join "" }) -join "`n"

$checks = @(
    @{ Name = "P0 - extract_completion_text for conversational completions"; Pattern = "def extract_completion_text" },
    @{ Name = "P0 - Defined uri & execute_sandbox_query in reward_accuracy"; Pattern = "execute_sandbox_query\(db_file, sql" },
    @{ Name = "P0 - Correct handling of empty ground-truth results"; Pattern = "is_empty = bool\(gold_is_empty" },
    @{ Name = "P0 - Genuine SQL execution timeout (set_progress_handler)"; Pattern = "conn\.set_progress_handler\(lambda:" },
    @{ Name = "P0 - Strict read-only SQL authorization (set_authorizer)"; Pattern = "conn\.set_authorizer\(create_authorizer\(\)\)" },
    @{ Name = "P0 - Mandatory SFT adapter loading assertion"; Pattern = "assert adapter_weight_file\.exists\(\)" },
    @{ Name = "P1 - Tokenizer padding to left"; Pattern = "tokenizer\.padding_side = `"left`"" },
    @{ Name = "P1 - AST / schema validation & table-whitelist"; Pattern = "cte_names = set\(\)" },
    @{ Name = "P1 - Pin compatible library versions"; Pattern = "transformers==4\.57\.3" },
    @{ Name = "P1 - Allocator & CUDA order before torch import"; Pattern = "(?s)CUDA_VISIBLE_DEVICES.*import torch" },
    @{ Name = "P1 - Dataset validation & reproducible sampling"; Pattern = "random\.seed\(42\)" },
    @{ Name = "P1 - Reward-function unit tests"; Pattern = "ALL REWARD FUNCTION UNIT TESTS PASSED" },
    @{ Name = "P1 - Held-out evaluation procedure on dev set"; Pattern = "Running Held-Out Evaluation on Spider Dev Set" }
)

Write-Host "================ CHECKLIST VERIFICATION ================"
$passCount = 0
foreach ($chk in $checks) {
    if ($allText -match $chk.Pattern) {
        Write-Host "✅ PASS: $($chk.Name)"
        $passCount++
    } else {
        Write-Host "❌ FAIL: $($chk.Name) (pattern: $($chk.Pattern))"
    }
}
Write-Host "========================================================"
Write-Host "Passed $passCount / $($checks.Count) checks."
