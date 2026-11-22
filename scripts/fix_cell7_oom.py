"""
Fix Notebook 2 Cell 7 OOM issues and Cell 6 gradient checkpointing.
Directly manipulates the notebook JSON cells.
"""
import json
from pathlib import Path

nb_path = Path(r"c:\queryalign-rlvr-grpo\notebooks\02_kaggle_sft_baseline.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

cells = nb["cells"]

# ── Find cells by their Step comment ──────────────────────────────────────────
def find_cell(cells, keyword):
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if keyword in src:
            return i
    return -1

idx6 = find_cell(cells, "Step 6: Configure QLoRA")
idx7 = find_cell(cells, "Step 7: Train QLoRA SFT")

print(f"Cell 6 index: {idx6}, Cell 7 index: {idx7}")

# ── Replace Cell 6 source ─────────────────────────────────────────────────────
NEW_CELL6 = [
    "# Step 6: Configure QLoRA Adapter & SFT Trainer\n",
    "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
    "from trl import SFTTrainer, SFTConfig\n",
    "from datasets import Dataset\n",
    "\n",
    "# This cell must run exactly once per freshly loaded base model.\n",
    "if hasattr(model, \"peft_config\"):\n",
    "    raise RuntimeError(\n",
    "        \"This model already has a PEFT adapter. Restart the Kaggle session, \"\n",
    "        \"rerun Cell 4, and execute Cell 6 only once.\"\n",
    "    )\n",
    "\n",
    "# Enable gradient checkpointing inside prepare_model_for_kbit_training.\n",
    "# This recomputes activations during the backward pass, reducing peak VRAM ~40%.\n",
    "model = prepare_model_for_kbit_training(\n",
    "    model,\n",
    "    use_gradient_checkpointing=True,\n",
    ")\n",
    "# Required when gradient_checkpointing=True so frozen base layers still receive grads\n",
    "model.enable_input_require_grads()\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=16,\n",
    "    lora_dropout=0.05,\n",
    "    bias=\"none\",\n",
    "    task_type=\"CAUSAL_LM\",\n",
    "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
    ")\n",
    "model = get_peft_model(model, lora_config)\n",
    "\n",
    "# Keep trainable LoRA parameters in their default FP32 dtype.\n",
    "# Do not force them to FP16: that can create dtype mismatches during gradient scaling.\n",
    "model.print_trainable_parameters()\n",
    "print(\"Trainable adapter dtypes:\", sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}))\n",
    "\n",
    "# Prepare Dataset\n",
    "train_manifest = DATA_DIR / \"processed\" / \"train_manifest.jsonl\"\n",
    "assert train_manifest.exists(), f\"train_manifest.jsonl not found at {train_manifest}! Please check Step 2.\"\n",
    "train_rows = []\n",
    "with open(train_manifest, \"r\", encoding=\"utf-8\") as f:\n",
    "    for line in f:\n",
    "        item = json.loads(line)\n",
    "        full_prompt = tokenizer.apply_chat_template(item[\"prompt\"], tokenize=False, add_generation_prompt=True)\n",
    "        full_text = full_prompt + item[\"completion\"] + tokenizer.eos_token\n",
    "        train_rows.append({\"text\": full_text})\n",
    "train_dataset = Dataset.from_list(train_rows)\n",
    "print(f\"Loaded {len(train_dataset)} training examples.\")\n",
    "assert len(train_dataset) >= 5000, f\"Expected ~7,000 training examples, but only found {len(train_dataset)}! Aborting to prevent incomplete training.\""
]

# ── Replace Cell 7 source ─────────────────────────────────────────────────────
NEW_CELL7 = [
    "# Step 7: Train QLoRA SFT Baseline on Kaggle GPU (OOM-safe settings for T4)\n",
    "import os, inspect\n",
    "import transformers, trl, peft\n",
    "\n",
    "# Reduce CUDA allocator fragmentation — prevents many OOM spikes on T4\n",
    "os.environ[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n",
    "\n",
    "print(\n",
    "    f\"Package versions | transformers={transformers.__version__} | \"\n",
    "    f\"trl={trl.__version__} | peft={peft.__version__}\"\n",
    ")\n",
    "\n",
    "major, minor = torch.cuda.get_device_capability()\n",
    "# T4 = CC 7.5 → no native BF16 GEMM; keep AMP off to avoid GradScaler/BF16 errors\n",
    "use_bf16 = False\n",
    "use_fp16 = False\n",
    "print(\n",
    "    f\"CUDA Capability: {major}.{minor} | Training precision: \"\n",
    "    f\"bf16={use_bf16}, fp16={use_fp16} | Mode: FP32 optimizer, 4-bit base\"\n",
    ")\n",
    "\n",
    "# ── Memory-safe training config for T4 (15 GB VRAM) ─────────────────────────\n",
    "# Effective batch size = per_device(1) * grad_accum(16) = 16  (same throughput)\n",
    "# gradient_checkpointing=True recomputes activations, saving ~40% VRAM at ~20% speed cost\n",
    "# max_length=1024 is sufficient for Spider queries; 1536 triggers OOM on T4\n",
    "raw_args = {\n",
    "    \"output_dir\": str(OUTPUTS_DIR),\n",
    "    \"num_train_epochs\": 3,\n",
    "    \"per_device_train_batch_size\": 1,       # T4 OOM fix: reduced from 4 → 1\n",
    "    \"gradient_accumulation_steps\": 16,      # compensates: 1×16 = same effective batch\n",
    "    \"learning_rate\": 2e-4,\n",
    "    \"lr_scheduler_type\": \"cosine\",\n",
    "    \"warmup_steps\": 20,\n",
    "    \"weight_decay\": 0.01,\n",
    "    \"logging_steps\": 10,\n",
    "    \"save_strategy\": \"steps\",\n",
    "    \"save_steps\": 100,\n",
    "    \"save_total_limit\": 2,\n",
    "    \"fp16\": False,\n",
    "    \"bf16\": False,\n",
    "    \"gradient_checkpointing\": True,         # T4 OOM fix: enabled\n",
    "    \"dataset_text_field\": \"text\",\n",
    "    \"packing\": False,\n",
    "    \"report_to\": \"none\",\n",
    "}\n",
    "\n",
    "# TRL 0.25.0 uses max_length; keep signature check for minor version resilience\n",
    "sig_params = inspect.signature(SFTConfig.__init__).parameters\n",
    "if \"max_length\" in sig_params:\n",
    "    raw_args[\"max_length\"] = 1024           # T4 OOM fix: reduced from 1536 → 1024\n",
    "elif \"max_seq_length\" in sig_params:\n",
    "    raw_args[\"max_seq_length\"] = 1024\n",
    "\n",
    "has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())\n",
    "valid_args = raw_args if has_var_keyword else {k: v for k, v in raw_args.items() if k in sig_params}\n",
    "training_args = SFTConfig(**valid_args)\n",
    "\n",
    "# Resolve processing_class vs tokenizer across TRL versions\n",
    "trainer_kwargs = {\n",
    "    \"model\": model,\n",
    "    \"train_dataset\": train_dataset,\n",
    "    \"args\": training_args,\n",
    "}\n",
    "trainer_sig = inspect.signature(SFTTrainer.__init__).parameters\n",
    "if \"processing_class\" in trainer_sig:\n",
    "    trainer_kwargs[\"processing_class\"] = tokenizer\n",
    "elif \"tokenizer\" in trainer_sig:\n",
    "    trainer_kwargs[\"tokenizer\"] = tokenizer\n",
    "\n",
    "trainer = SFTTrainer(**trainer_kwargs)\n",
    "\n",
    "print(\"Starting QLoRA SFT Training...\")\n",
    "trainer.train()\n",
    "\n",
    "# Save trained SFT adapter\n",
    "sft_adapter_path = OUTPUTS_DIR / \"best_adapter\"\n",
    "trainer.model.save_pretrained(str(sft_adapter_path))\n",
    "tokenizer.save_pretrained(str(sft_adapter_path))\n",
    "print(f\"✅ Saved SFT Adapter to: {sft_adapter_path}\")"
]

if idx6 >= 0:
    cells[idx6]["source"] = NEW_CELL6
    print(f"✅ Replaced Cell {idx6} (Step 6)")
else:
    print("⚠️  Cell 6 not found!")

if idx7 >= 0:
    cells[idx7]["source"] = NEW_CELL7
    print(f"✅ Replaced Cell {idx7} (Step 7)")
else:
    print("⚠️  Cell 7 not found!")

nb["cells"] = cells
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("✅ Notebook saved.")

# Verify JSON is valid
try:
    json.loads(nb_path.read_text(encoding="utf-8"))
    print("✅ Notebook JSON is valid.")
except json.JSONDecodeError as e:
    print(f"❌ JSON error: {e}")
