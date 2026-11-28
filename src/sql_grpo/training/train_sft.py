"""Standalone QLoRA SFT Baseline Trainer (Config-Driven & TRL Version-Resilient)."""

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from sql_grpo.data.validate_dataset import load_records
from sql_grpo.training.callbacks import RLVRMetricsCallback
from sql_grpo.training.checkpointing import find_latest_checkpoint, write_run_metadata


def load_sft_config(config_path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def format_sft_example(record: Dict[str, Any], tokenizer: Any = None) -> str:
    """Formats a verified train-split manifest record into a complete chat supervised target."""
    messages = list(record["prompt"]) + [{"role": "assistant", "content": record["completion"]}]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)


def filter_trainable_sft_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Retains only train records whose gold SQL did not error or timeout during data QA."""
    return [
        r
        for r in records
        if r.get("split", "train") == "train" and r.get("gold_status") not in ("error", "timeout", "ast_rejected", "truncated")
    ]


def build_sft_config_kwargs(cfg: Dict[str, Any], sft_config_cls: Any, use_bf16: bool = False, use_fp16: bool = True) -> Dict[str, Any]:
    """Builds SFTConfig kwargs filtered cleanly against the installed TRL SFTConfig signature."""
    model_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})
    sig_params = inspect.signature(sft_config_cls.__init__).parameters

    raw: Dict[str, Any] = {
        "output_dir": tr_cfg.get("output_dir", "./outputs/sft/run_001"),
        "num_train_epochs": int(tr_cfg.get("num_train_epochs", 3)),
        "per_device_train_batch_size": int(tr_cfg.get("per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(tr_cfg.get("per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": int(tr_cfg.get("gradient_accumulation_steps", 8)),
        "optim": tr_cfg.get("optim", "paged_adamw_8bit"),
        "learning_rate": float(tr_cfg.get("learning_rate", 2e-4)),
        "lr_scheduler_type": tr_cfg.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(tr_cfg.get("warmup_ratio", 0.05)),
        "weight_decay": float(tr_cfg.get("weight_decay", 0.01)),
        "logging_steps": int(tr_cfg.get("logging_steps", 10)),
        "save_strategy": tr_cfg.get("save_strategy", "epoch"),
        "eval_strategy": tr_cfg.get("eval_strategy", "no"),
        "evaluation_strategy": tr_cfg.get("eval_strategy", "no"),
        "save_total_limit": int(tr_cfg.get("save_total_limit", 2)),
        "seed": int(tr_cfg.get("seed", 3407)),
        "fp16": use_fp16,
        "bf16": use_bf16,
        "gradient_checkpointing": bool(tr_cfg.get("gradient_checkpointing", True)),
        "dataset_text_field": "text",
        "packing": False,
        "report_to": tr_cfg.get("report_to", "none"),
    }

    max_len = int(model_cfg.get("max_seq_length", 1024))
    if "max_length" in sig_params:
        raw["max_length"] = max_len
    elif "max_seq_length" in sig_params:
        raw["max_seq_length"] = max_len

    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
    return raw if has_var_kw else {k: v for k, v in raw.items() if k in sig_params}


def run_sft_training(
    config_path: str | Path,
    train_manifest: str | Path = "data/processed/train_manifest.jsonl",
    eval_manifest: Optional[str | Path] = None,
    resume: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Runs QLoRA SFT training on train split only (or validates setup cleanly under dry_run=True)."""
    cfg = load_sft_config(config_path)
    tr_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    out_dir = Path(tr_cfg.get("output_dir", "./outputs/sft/run_001"))

    raw_train = load_records(Path(train_manifest))
    train_rows = filter_trainable_sft_records(raw_train)
    eval_rows = load_records(Path(eval_manifest)) if eval_manifest and Path(eval_manifest).is_file() else []

    if dry_run:
        return {
            "mode": "dry_run",
            "model_id": model_cfg.get("model_id", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "output_dir": str(out_dir),
            "sample_text": format_sft_example(train_rows[0]) if train_rows else "",
        }

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    model_id = model_cfg.get("model_id", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(use_cuda and torch.cuda.get_device_capability(0)[0] >= 8)
    use_fp16 = bool(use_cuda and not use_bf16)

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if use_cuda and model_cfg.get("load_in_4bit", True):
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["torch_dtype"] = compute_dtype
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if use_cuda and model_cfg.get("load_in_4bit", True):
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    from peft import get_peft_model

    model = get_peft_model(model, peft_cfg)

    train_ds = Dataset.from_list([{"text": format_sft_example(r, tokenizer)} for r in train_rows])
    eval_ds = (
        Dataset.from_list([{"text": format_sft_example(r, tokenizer)} for r in eval_rows])
        if eval_rows
        else None
    )

    sft_args = SFTConfig(**build_sft_config_kwargs(cfg, SFTConfig, use_bf16=use_bf16, use_fp16=use_fp16))
    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "train_dataset": train_ds,
        "args": sft_args,
        "callbacks": [RLVRMetricsCallback(out_dir)],
    }
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    resume_ckpt = str(find_latest_checkpoint(out_dir)) if resume else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    best_dir = out_dir / "best_adapter"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    meta_path = write_run_metadata(out_dir, run_type="qlora_sft", config=cfg, adapter_dir=best_dir)
    return {
        "mode": "trained",
        "adapter_dir": str(best_dir),
        "metadata_path": str(meta_path),
        "train_rows": len(train_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QLoRA SFT baseline for QueryAlign.")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--train-manifest", default="data/processed/train_manifest.jsonl")
    parser.add_argument("--eval-manifest", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    res = run_sft_training(
        config_path=args.config,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
