$nb = Get-Content 'c:\queryalign-rlvr-grpo\notebooks\02_kaggle_sft_baseline.ipynb' -Raw | ConvertFrom-Json

# --- Update Cell 2 (index 1 in 0-indexed or find by Step 2) ---
for ($i=0; $i -lt $nb.cells.Count; $i++) {
    $src = $nb.cells[$i].source -join ""
    if ($src -match "Step 2: GPU Inspection") {
        Write-Host "Found Step 2 at index $i"
        # Prepend os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        $lines = @(
            "# Step 2: GPU Inspection & Path Setup (With Auto-Unpack for Kaggle Inputs)`n",
            "import os`n",
            "# Force single GPU 0 to prevent HuggingFace Trainer from wrapping bitsandbytes 4-bit in DataParallel`n",
            "os.environ[`"CUDA_VISIBLE_DEVICES`"] = `"0`"`n"
        )
        # Keep existing lines from index 1 onward
        $oldLines = $nb.cells[$i].source
        $nb.cells[$i].source = $lines + ($oldLines | Select-Object -Skip 1)
        Write-Host "Cell $i (Step 2) updated."
    }
}

# --- Update Cell 7 (find by Step 7) ---
for ($i=0; $i -lt $nb.cells.Count; $i++) {
    $src = $nb.cells[$i].source -join ""
    if ($src -match "Step 7: Train QLoRA SFT") {
        Write-Host "Found Step 7 at index $i"
        $newCell7 = @(
            "# Step 7: Train QLoRA SFT Baseline on Kaggle GPU (T4 Multi-GPU & Precision Fixed)`n",
            "import os, inspect`n",
            "import torch`n",
            "import transformers, trl, peft`n",
            "`n",
            "# 1. Reduce CUDA memory allocator fragmentation`n",
            "os.environ[`"PYTORCH_CUDA_ALLOC_CONF`"] = `"expandable_segments:True`"`n",
            "`n",
            "# 2. Prevent HuggingFace Trainer from wrapping 4-bit model in torch.nn.DataParallel (breaks bitsandbytes)`n",
            "model.is_parallelizable = True`n",
            "model.model_parallel = True`n",
            "if hasattr(model, `"base_model`"):`n",
            "    model.base_model.is_parallelizable = True`n",
            "    model.base_model.model_parallel = True`n",
            "`n",
            "# 3. Clean any rogue bfloat16 tensors to avoid GradScaler errors with fp16`n",
            "for p in model.parameters():`n",
            "    if p.dtype == torch.bfloat16:`n",
            "        p.data = p.data.to(torch.float16)`n",
            "for b in model.buffers():`n",
            "    if b.dtype == torch.bfloat16:`n",
            "        b.data = b.data.to(torch.float16)`n",
            "`n",
            "major, minor = torch.cuda.get_device_capability()`n",
            "# T4 = CC 7.5: Native FP16 Tensor Cores supported. BF16 NOT supported.`n",
            "# BitsAndBytes 4-bit matmul requires FP16 inputs (fp16=True).`n",
            "use_bf16 = (major >= 8)`n",
            "use_fp16 = True  # Native FP16 for T4 QLoRA`n",
            "print(f`"CUDA Capability: {major}.{minor} | Precision: bf16={use_bf16}, fp16={use_fp16} | Device: {torch.cuda.get_device_name(0)}`")`n",
            "`n",
            "raw_args = {`n",
            "    `"output_dir`": str(OUTPUTS_DIR),`n",
            "    `"num_train_epochs`": 3,`n",
            "    `"per_device_train_batch_size`": 1,       # T4 OOM fix: 1`n",
            "    `"gradient_accumulation_steps`": 16,      # compensates: 1x16 = 16 effective batch`n",
            "    `"learning_rate`": 2e-4,`n",
            "    `"lr_scheduler_type`": `"cosine`",`n",
            "    `"warmup_steps`": 20,`n",
            "    `"weight_decay`": 0.01,`n",
            "    `"logging_steps`": 10,`n",
            "    `"save_strategy`": `"steps`",`n",
            "    `"save_steps`": 100,`n",
            "    `"save_total_limit`": 2,`n",
            "    `"fp16`": use_fp16,                       # Mandatory for bitsandbytes 4-bit GEMM on T4`n",
            "    `"bf16`": use_bf16,`n",
            "    `"gradient_checkpointing`": True,         # T4 OOM fix: enabled`n",
            "    `"dataset_text_field`": `"text`",`n",
            "    `"packing`": False,`n",
            "    `"report_to`": `"none`",`n",
            "}`n",
            "`n",
            "sig_params = inspect.signature(SFTConfig.__init__).parameters`n",
            "if `"max_length`" in sig_params:`n",
            "    raw_args[`"max_length`"] = 1024           # T4 OOM fix: 1024`n",
            "elif `"max_seq_length`" in sig_params:`n",
            "    raw_args[`"max_seq_length`"] = 1024`n",
            "`n",
            "has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())`n",
            "valid_args = raw_args if has_var_keyword else {k: v for k, v in raw_args.items() if k in sig_params}`n",
            "training_args = SFTConfig(**valid_args)`n",
            "`n",
            "# Force single GPU mode inside TrainingArguments so Trainer NEVER invokes DataParallel`n",
            "training_args._n_gpu = 1`n",
            "`n",
            "# Resolve processing_class vs tokenizer across TRL versions`n",
            "trainer_kwargs = {`n",
            "    `"model`": model,`n",
            "    `"train_dataset`": train_dataset,`n",
            "    `"args`": training_args,`n",
            "}`n",
            "trainer_sig = inspect.signature(SFTTrainer.__init__).parameters`n",
            "if `"processing_class`" in trainer_sig:`n",
            "    trainer_kwargs[`"processing_class`"] = tokenizer`n",
            "elif `"tokenizer`" in trainer_sig:`n",
            "    trainer_kwargs[`"tokenizer`"] = tokenizer`n",
            "`n",
            "trainer = SFTTrainer(**trainer_kwargs)`n",
            "`n",
            "print(`"Starting QLoRA SFT Training...`")`n",
            "trainer.train()`n",
            "`n",
            "# Save trained SFT adapter`n",
            "sft_adapter_path = OUTPUTS_DIR / `"best_adapter`"`n",
            "trainer.model.save_pretrained(str(sft_adapter_path))`n",
            "tokenizer.save_pretrained(str(sft_adapter_path))`n",
            "print(f`"✅ Saved SFT Adapter to: {sft_adapter_path}`")"
        )
        $nb.cells[$i].source = $newCell7
        Write-Host "Cell $i (Step 7) updated."
    }
}

$nb | ConvertTo-Json -Depth 100 | Set-Content 'c:\queryalign-rlvr-grpo\notebooks\02_kaggle_sft_baseline.ipynb' -Encoding UTF8
Write-Host "Notebook 2 saved successfully."
