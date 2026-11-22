$nb3 = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Raw | ConvertFrom-Json

# Step 2: add CUDA_VISIBLE_DEVICES = "0"
for ($i=0; $i -lt $nb3.cells.Count; $i++) {
    $src = $nb3.cells[$i].source -join ""
    if ($src -match "Step 2: GPU Inspection") {
        Write-Host "Found Step 2 in NB3 at index $i"
        $lines = @(
            "# Step 2: GPU Inspection & Path Setup`n",
            "import os`n",
            "# Force single GPU 0 to prevent DataParallel wrapping on multi-GPU (T4 x2)`n",
            "os.environ[`"CUDA_VISIBLE_DEVICES`"] = `"0`"`n"
        )
        $oldLines = $nb3.cells[$i].source
        $nb3.cells[$i].source = $lines + ($oldLines | Select-Object -Skip 1)
        Write-Host "Cell $i updated in NB3."
    }
    if ($src -match "Step 6: Configure GRPOTrainer") {
        Write-Host "Found Step 6 in NB3 at index $i"
        $newCell6 = @(
            "# Step 6: Configure GRPOTrainer with Group Size G=8 and Batch Divisibility`n",
            "import os, inspect`n",
            "import torch`n",
            "from trl import GRPOTrainer, GRPOConfig`n",
            "`n",
            "# 1. Reduce CUDA memory allocator fragmentation`n",
            "os.environ[`"PYTORCH_CUDA_ALLOC_CONF`"] = `"expandable_segments:True`"`n",
            "`n",
            "# 2. Prevent Trainer from wrapping model in torch.nn.DataParallel`n",
            "model.is_parallelizable = True`n",
            "model.model_parallel = True`n",
            "if hasattr(model, `"base_model`"):`n",
            "    model.base_model.is_parallelizable = True`n",
            "    model.base_model.model_parallel = True`n",
            "`n",
            "# 3. Clean any rogue bfloat16 tensors`n",
            "for p in model.parameters():`n",
            "    if p.dtype == torch.bfloat16:`n",
            "        p.data = p.data.to(torch.float16)`n",
            "for b in model.buffers():`n",
            "    if b.dtype == torch.bfloat16:`n",
            "        b.data = b.data.to(torch.float16)`n",
            "`n",
            "NUM_GENERATIONS = 8 # G=8 candidate queries per prompt`n",
            "major, minor = torch.cuda.get_device_capability()`n",
            "use_bf16 = (major >= 8)`n",
            "use_fp16 = True # Native FP16 for T4`n",
            "print(f`"CUDA Capability: {major}.{minor} | Training precision: bf16={use_bf16}, fp16={use_fp16} | Device: {torch.cuda.get_device_name(0)}`")`n",
            "`n",
            "raw_grpo_args = {`n",
            "    `"output_dir`": str(GRPO_OUTPUTS_DIR),`n",
            "    `"num_generations`": NUM_GENERATIONS,`n",
            "    `"per_device_train_batch_size`": 1,`n",
            "    `"gradient_accumulation_steps`": 8, # 1 * 8 = 8, divisible by num_generations=8`n",
            "    `"temperature`": 0.8,`n",
            "    `"learning_rate`": 5.0e-6,`n",
            "    `"beta`": 0.04, # KL regularization`n",
            "    `"max_prompt_length`": 1024,`n",
            "    `"max_completion_length`": 256,`n",
            "    `"max_steps`": 100,`n",
            "    `"logging_steps`": 5,`n",
            "    `"save_steps`": 25,`n",
            "    `"fp16`": use_fp16,`n",
            "    `"bf16`": use_bf16,`n",
            "    `"remove_unused_columns`": False, # Essential: preserves custom metadata for reward callables`n",
            "    `"report_to`": `"none`",`n",
            "}`n",
            "`n",
            "grpo_sig = inspect.signature(GRPOConfig.__init__).parameters`n",
            "has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in grpo_sig.values())`n",
            "valid_grpo_args = raw_grpo_args if has_var_kw else {k: v for k, v in raw_grpo_args.items() if k in grpo_sig}`n",
            "grpo_args = GRPOConfig(**valid_grpo_args)`n",
            "`n",
            "# Force single GPU mode inside TrainingArguments so Trainer NEVER invokes DataParallel`n",
            "grpo_args._n_gpu = 1`n",
            "`n",
            "reward_funcs = [`n",
            "    reward_format,`n",
            "    reward_ast_schema,`n",
            "    reward_exec_valid,`n",
            "    reward_accuracy,`n",
            "]`n",
            "`n",
            "trainer_kwargs = {`n",
            "    `"model`": model,`n",
            "    `"reward_funcs`": reward_funcs,`n",
            "    `"args`": grpo_args,`n",
            "    `"train_dataset`": grpo_dataset,`n",
            "}`n",
            "trainer_sig = inspect.signature(GRPOTrainer.__init__).parameters`n",
            "if `"processing_class`" in trainer_sig:`n",
            "    trainer_kwargs[`"processing_class`"] = tokenizer`n",
            "elif `"tokenizer`" in trainer_sig:`n",
            "    trainer_kwargs[`"tokenizer`"] = tokenizer`n",
            "`n",
            "trainer = GRPOTrainer(**trainer_kwargs)`n",
            "`n",
            "print(`"GRPOTrainer initialized. Starting RLVR Fine-Tuning...`")`n",
            "trainer.train()`n",
            "`n",
            "# Save Final GRPO Policy Adapter`n",
            "final_adapter_dir = GRPO_OUTPUTS_DIR / `"final_adapter`"`n",
            "trainer.model.save_pretrained(str(final_adapter_dir))`n",
            "tokenizer.save_pretrained(str(final_adapter_dir))`n",
            "print(f`"✅ Successfully saved Final GRPO Adapter to: {final_adapter_dir}`")"
        )
        $nb3.cells[$i].source = $newCell6
        Write-Host "Cell $i (Step 6) updated in NB3."
    }
}

$nb3 | ConvertTo-Json -Depth 100 | Set-Content 'c:\queryalign-rlvr-grpo\notebooks\03_kaggle_grpo_training.ipynb' -Encoding UTF8
Write-Host "Notebook 3 saved successfully."
