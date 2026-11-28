"""Training Pipeline & Adapter Lifecycle Smoke Tests (Blueprint Sections 13.7, 13.8, 19.5)."""

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from sql_grpo.data.schema_serializer import build_chat_prompt
from sql_grpo.training.callbacks import RLVRMetricsCallback, summarize_reward_vector
from sql_grpo.training.checkpointing import (
    find_latest_checkpoint,
    verify_adapter_reload,
    write_run_metadata,
)
from sql_grpo.training.train_grpo import (
    build_grpo_config_kwargs,
    get_registered_reward_funcs,
    load_grpo_config,
    load_policy_from_sft,
    prepare_grpo_dataset_records,
    run_grpo_training,
    validate_batch_divisibility,
)
from sql_grpo.training.train_sft import (
    build_sft_config_kwargs,
    filter_trainable_sft_records,
    load_sft_config,
    run_sft_training,
)


class _DummySFTConfig:
    def __init__(self, output_dir="", max_length=1024, eval_strategy="no", **kwargs):
        pass


class _DummyGRPOConfig:
    def __init__(self, output_dir="", num_generations=8, remove_unused_columns=False, beta=0.04, **kwargs):
        pass


def test_sft_filtering_config_and_dry_run(tmp_path):
    cfg = load_sft_config("configs/sft.yaml")
    kwargs = build_sft_config_kwargs(cfg, _DummySFTConfig)
    assert kwargs["max_length"] == 2048

    manifest = tmp_path / "train.jsonl"
    prompt = build_chat_prompt("List depts", "DATABASE SCHEMA:\n  Table: department\n    - name (TEXT)")
    rows = [
        {"example_id": "t1", "split": "train", "db_id": "test_company", "gold_status": "success", "prompt": prompt, "completion": "<sql>SELECT name FROM department</sql>"},
        {"example_id": "t2", "split": "train", "db_id": "test_company", "gold_status": "empty", "prompt": prompt, "completion": "<sql>SELECT name FROM department WHERE 1=0</sql>"},
        {"example_id": "t3", "split": "train", "db_id": "test_company", "gold_status": "error", "prompt": prompt, "completion": "<sql>BAD</sql>"},
        {"example_id": "d1", "split": "dev", "db_id": "concert_singer", "gold_status": "success", "prompt": prompt, "completion": "<sql>SELECT 1</sql>"},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    filtered = filter_trainable_sft_records(rows)
    assert [r["example_id"] for r in filtered] == ["t1", "t2"]

    dry = run_sft_training("configs/sft.yaml", train_manifest=manifest, dry_run=True)
    assert dry["mode"] == "dry_run"
    assert dry["train_rows"] == 2


def test_grpo_batch_divisibility_rewards_and_dry_run(tmp_path):
    assert validate_batch_divisibility(1, 8, 8) == 8
    assert validate_batch_divisibility(1, 4, 4) == 4
    with pytest.raises(ValueError, match="divisible"):
        validate_batch_divisibility(1, 6, 8)

    grpo_cfg = load_grpo_config("configs/grpo.yaml")
    grpo_kwargs = build_grpo_config_kwargs(grpo_cfg, _DummyGRPOConfig)
    assert grpo_kwargs["remove_unused_columns"] is False
    assert grpo_kwargs["num_generations"] == 8

    pilot_cfg = load_grpo_config("configs/grpo_pilot.yaml")
    pilot_kwargs = build_grpo_config_kwargs(pilot_cfg, _DummyGRPOConfig)
    assert pilot_kwargs["num_generations"] == 4

    prompt = build_chat_prompt("List names", "DATABASE SCHEMA:\n  Table: employee\n    - name (TEXT)")
    manifest = tmp_path / "grpo_train.jsonl"
    raw = [
        {
            "example_id": "g1",
            "split": "train",
            "db_id": "test_company",
            "gold_status": "success",
            "gold_sql": "SELECT SECRET_GOLD_SQL FROM employee",
            "valid_tables": ["employee"],
            "valid_columns": {"employee": ["emp_id", "name"]},
            "gold_result_json": '[["Alice"]]',
            "gold_is_empty": False,
            "prompt": prompt,
        }
    ]
    manifest.write_text(json.dumps(raw[0]) + "\n", encoding="utf-8")

    prepared = prepare_grpo_dataset_records(raw)
    assert len(prepared) == 1
    assert "SECRET_GOLD_SQL" not in json.dumps(prepared[0]["prompt"])
    assert json.loads(prepared[0]["valid_columns_json"]) == {"employee": ["emp_id", "name"]}

    funcs = get_registered_reward_funcs(include_efficiency=False)
    assert [f.__name__ for f in funcs] == [
        "format_reward_func",
        "schema_reward_func",
        "execution_reward_func",
        "accuracy_reward_func",
    ]
    # Verify every registered reward callable returns one finite scalar per completion
    comps = [[{"role": "assistant", "content": "<sql>SELECT name FROM employee</sql>"}]]
    for fn in funcs:
        scores = fn(
            comps,
            db_id=["test_company"],
            valid_tables=[prepared[0]["valid_tables"]],
            valid_columns_json=[prepared[0]["valid_columns_json"]],
            gold_result_json=[prepared[0]["gold_result_json"]],
            gold_is_empty=[False],
        )
        assert len(scores) == 1 and isinstance(scores[0], float)

    dry = run_grpo_training("configs/grpo_pilot.yaml", train_manifest=manifest, dry_run=True)
    assert dry["mode"] == "dry_run"
    assert dry["prompt_count"] == 1


def test_callbacks_checkpointing_and_tiny_peft_step_reload(tmp_path):
    # Reward stats + zero-variance check
    stats = summarize_reward_vector([0.5, 0.5] * 4, group_size=8)
    assert stats["zero_variance_pct"] == 100.0

    # Callback + checkpoint metadata
    out_dir = tmp_path / "run_001"
    (out_dir / "checkpoint-10").mkdir(parents=True)
    (out_dir / "checkpoint-25").mkdir(parents=True)
    assert find_latest_checkpoint(out_dir).name == "checkpoint-25"

    cb = RLVRMetricsCallback(out_dir)
    cb.on_log(None, SimpleNamespace(global_step=5), None, logs={"loss": 0.42, "reward": 1.15})
    assert cb.log_jsonl_path.is_file() and "0.42" in cb.log_jsonl_path.read_text(encoding="utf-8")
    assert cb.resource_csv_path.is_file()

    # Tiny offline Qwen2 + PEFT LoRA step -> save -> resume SFT trainable (no double-wrap) -> save & reload
    real_tok_dir = Path("artifacts/phase_4/grpo/final_adapter")
    if not real_tok_dir.is_dir():
        pytest.skip("Tokenizer artifacts unavailable")

    tok = AutoTokenizer.from_pretrained(str(real_tok_dir), local_files_only=True)
    tiny_base_dir = tmp_path / "tiny_qwen2"
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(tiny_base_dir)
    tok.save_pretrained(tiny_base_dir)

    base = Qwen2ForCausalLM.from_pretrained(str(tiny_base_dir), local_files_only=True)
    lora_cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
    sft_model = get_peft_model(base, lora_cfg)
    sft_dir = tmp_path / "sft_ckpt"
    sft_model.save_pretrained(sft_dir)
    tok.save_pretrained(sft_dir)

    with pytest.raises(ValueError, match="sft_adapter_path is required"):
        load_policy_from_sft(base, None, allow_base_ablation=False)

    resumed_base = Qwen2ForCausalLM.from_pretrained(str(tiny_base_dir), local_files_only=True)
    grpo_model = load_policy_from_sft(resumed_base, sft_dir, merge_sft_before_new_lora=False)
    assert isinstance(grpo_model, PeftModel)

    # One optimizer step updates trainable LoRA parameters cleanly with finite loss
    grpo_model.train()
    opt = torch.optim.AdamW([p for p in grpo_model.parameters() if p.requires_grad], lr=1e-3)
    ids = torch.tensor([[1, 2, 3, 4]])
    loss = grpo_model(input_ids=ids, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()

    grpo_final = tmp_path / "grpo_final"
    grpo_model.save_pretrained(grpo_final)
    tok.save_pretrained(grpo_final)
    meta_path = write_run_metadata(out_dir, "smoke_test", {"lr": 1e-3}, adapter_dir=grpo_final)
    assert meta_path.is_file()

    reload_info = verify_adapter_reload(str(tiny_base_dir), grpo_final, local_files_only=True)
    assert reload_info["peft_loaded"] is True
    assert len(reload_info["adapter_sha256"]) == 64
