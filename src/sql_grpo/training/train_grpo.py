"""Standalone GRPO Reinforcement Learning Trainer (Config-Driven with Deterministic RLVR Rewards)."""

import argparse
import inspect
import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional
import yaml

from sql_grpo.data.validate_dataset import load_records
from sql_grpo.rewards import (
    accuracy_reward_func,
    efficiency_reward_func,
    execution_reward_func,
    format_reward_func,
    schema_reward_func,
)
from sql_grpo.training.callbacks import RLVRMetricsCallback
from sql_grpo.training.checkpointing import find_latest_checkpoint, write_run_metadata


def load_grpo_config(config_path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def validate_batch_divisibility(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    world_size: int = 1,
) -> int:
    """Enforces TRL GRPO constraint that effective batch size is divisible by num_generations."""
    if num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {num_generations}")
    effective_batch = world_size * per_device_batch_size * gradient_accumulation_steps
    if effective_batch % num_generations != 0:
        raise ValueError(
            f"Effective batch size ({world_size}*{per_device_batch_size}*{gradient_accumulation_steps}={effective_batch}) "
            f"must be divisible by num_generations={num_generations}"
        )
    return effective_batch


def prepare_grpo_dataset_records(
    records: List[Dict[str, Any]],
    sample_size: Optional[int] = None,
    seed: int = 3407,
) -> List[Dict[str, Any]]:
    """Prepares prompt + metadata columns without leaking gold SQL inside the prompt contract."""
    eligible: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("split", "train") != "train":
            continue
        if rec.get("gold_status") in ("error", "timeout", "ast_rejected", "truncated"):
            continue
        eligible.append(
            {
                "prompt": rec["prompt"],
                "example_id": rec.get("example_id", ""),
                "db_id": rec["db_id"],
                "valid_tables": rec.get("valid_tables", []),
                "valid_columns_json": json.dumps(rec.get("valid_columns", {}))
                if rec.get("valid_columns") is not None
                else "",
                "gold_result_json": rec.get("gold_result_json", "[]"),
                "gold_is_empty": bool(rec.get("gold_is_empty", False)),
            }
        )

    if sample_size is not None and sample_size > 0 and len(eligible) > sample_size:
        rng = random.Random(seed)
        eligible = rng.sample(eligible, sample_size)
    return eligible


def get_registered_reward_funcs(include_efficiency: bool = False) -> List[Callable[..., List[float]]]:
    """Returns TRL-compatible deterministic reward callables in documented Blueprint order."""
    funcs: List[Callable[..., List[float]]] = [
        format_reward_func,
        schema_reward_func,
        execution_reward_func,
        accuracy_reward_func,
    ]
    if include_efficiency:
        funcs.append(efficiency_reward_func)
    return funcs


def build_grpo_config_kwargs(
    cfg: Dict[str, Any],
    grpo_config_cls: Any,
    use_bf16: bool = False,
    use_fp16: bool = True,
    world_size: int = 1,
) -> Dict[str, Any]:
    """Builds version-resilient GRPOConfig kwargs and validates batch divisibility."""
    model_cfg = cfg.get("model", {})
    g_cfg = cfg.get("grpo", {})

    num_generations = int(g_cfg.get("num_generations", 8))
    per_device = int(g_cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(g_cfg.get("gradient_accumulation_steps", 8))
    validate_batch_divisibility(per_device, grad_accum, num_generations, world_size=world_size)

    raw: Dict[str, Any] = {
        "output_dir": g_cfg.get("output_dir", "./outputs/grpo/run_001"),
        "num_generations": num_generations,
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": grad_accum,
        "temperature": float(g_cfg.get("temperature", 0.8)),
        "learning_rate": float(g_cfg.get("learning_rate", 5.0e-6)),
        "beta": float(g_cfg.get("beta", 0.04)),
        "max_prompt_length": int(model_cfg.get("max_prompt_length", 1024)),
        "max_completion_length": int(model_cfg.get("max_completion_length", 256)),
        "max_steps": int(g_cfg.get("max_steps", 100)),
        "warmup_ratio": float(g_cfg.get("warmup_ratio", 0.1)),
        "lr_scheduler_type": g_cfg.get("lr_scheduler_type", "cosine"),
        "logging_steps": int(g_cfg.get("logging_steps", 5)),
        "save_steps": int(g_cfg.get("save_steps", 50)),
        "seed": int(g_cfg.get("seed", 3407)),
        "fp16": use_fp16,
        "bf16": use_bf16,
        "remove_unused_columns": False,  # Required so db_id / gold_result_json reach reward_funcs
        "report_to": g_cfg.get("report_to", "none"),
    }

    sig_params = inspect.signature(grpo_config_cls.__init__).parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
    return raw if has_var_kw else {k: v for k, v in raw.items() if k in sig_params}


def load_policy_from_sft(
    base_model: Any,
    sft_adapter_path: Optional[str | Path],
    merge_sft_before_new_lora: bool = False,
    allow_base_ablation: bool = False,
    new_lora_config: Optional[Any] = None,
) -> Any:
    """Resumes SFT adapter as single trainable PEFT adapter (or merges to 16-bit first; never double-wraps)."""
    from peft import PeftModel, get_peft_model

    if not sft_adapter_path:
        if not allow_base_ablation:
            raise ValueError("sft_adapter_path is required unless allow_base_ablation=True (Variant D).")
        if new_lora_config is None:
            raise ValueError("new_lora_config must be supplied for Variant D base+GRPO ablation.")
        return get_peft_model(base_model, new_lora_config)

    adapter_dir = Path(sft_adapter_path)
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Mandatory SFT adapter weights missing at {adapter_dir}")

    if merge_sft_before_new_lora:
        sft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
        merged = sft_model.merge_and_unload()
        if new_lora_config is None:
            raise ValueError("new_lora_config required after merge_and_unload.")
        return get_peft_model(merged, new_lora_config)

    # Single active LoRA adapter resumed directly from SFT checkpoint (no double-wrapping)
    return PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)


def run_grpo_training(
    config_path: str | Path,
    train_manifest: str | Path = "data/processed/train_manifest.jsonl",
    sample_size: Optional[int] = None,
    include_efficiency_reward: bool = False,
    allow_base_ablation: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Runs RLVR-GRPO training (or validates dataset, divisibility, and reward batch under dry_run=True)."""
    cfg = load_grpo_config(config_path)
    model_cfg = cfg.get("model", {})
    g_cfg = cfg.get("grpo", {})
    out_dir = Path(g_cfg.get("output_dir", "./outputs/grpo/run_001"))

    validate_batch_divisibility(
        int(g_cfg.get("per_device_train_batch_size", 1)),
        int(g_cfg.get("gradient_accumulation_steps", 8)),
        int(g_cfg.get("num_generations", 8)),
    )

    raw_train = load_records(Path(train_manifest))
    grpo_records = prepare_grpo_dataset_records(
        raw_train,
        sample_size=sample_size,
        seed=int(g_cfg.get("seed", 3407)),
    )
    reward_funcs = get_registered_reward_funcs(include_efficiency=include_efficiency_reward)

    if dry_run:
        return {
            "mode": "dry_run",
            "model_id": model_cfg.get("model_id", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
            "sft_adapter_path": model_cfg.get("sft_adapter_path"),
            "prompt_count": len(grpo_records),
            "num_generations": int(g_cfg.get("num_generations", 8)),
            "reward_funcs": [fn.__name__ for fn in reward_funcs],
            "output_dir": str(out_dir),
        }

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    model_id = model_cfg.get("model_id", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    fresh_lora = LoraConfig(
        r=int(model_cfg.get("lora_rank", 16)),
        lora_alpha=int(model_cfg.get("lora_alpha", 16)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.0)),
        target_modules=model_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = load_policy_from_sft(
        base_model=base_model,
        sft_adapter_path=model_cfg.get("sft_adapter_path"),
        merge_sft_before_new_lora=bool(model_cfg.get("merge_sft_before_grpo", False)),
        allow_base_ablation=allow_base_ablation,
        new_lora_config=fresh_lora,
    )

    grpo_args = GRPOConfig(
        **build_grpo_config_kwargs(cfg, GRPOConfig, use_bf16=use_bf16, use_fp16=use_fp16)
    )
    train_ds = Dataset.from_list(grpo_records)
    trainer_sig = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs: Dict[str, Any] = {
        "model": policy_model,
        "reward_funcs": reward_funcs,
        "args": grpo_args,
        "train_dataset": train_ds,
        "callbacks": [RLVRMetricsCallback(out_dir)],
    }
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    resume_ckpt = str(find_latest_checkpoint(out_dir)) if resume else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = out_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    meta_path = write_run_metadata(out_dir, run_type="rlvr_grpo", config=cfg, adapter_dir=final_dir)
    return {
        "mode": "trained",
        "adapter_dir": str(final_dir),
        "metadata_path": str(meta_path),
        "prompt_count": len(grpo_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RLVR-GRPO policy for QueryAlign.")
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--train-manifest", default="data/processed/train_manifest.jsonl")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--include-efficiency", action="store_true")
    parser.add_argument("--allow-base-ablation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    res = run_grpo_training(
        config_path=args.config,
        train_manifest=args.train_manifest,
        sample_size=args.sample_size,
        include_efficiency_reward=args.include_efficiency,
        allow_base_ablation=args.allow_base_ablation,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
