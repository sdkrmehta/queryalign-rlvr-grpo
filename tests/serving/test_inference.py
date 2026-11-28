"""Tests for LocalSQLInferenceBackend, adapter resolution, and serving configuration."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import pytest

from sql_grpo.serving.inference import (
    LocalSQLInferenceBackend,
    ModelUnavailableError,
    ServingModelConfig,
    _compute_sha256,
    _resolve_adapter_dir,
    load_serving_config,
)


EXPECTED_GRPO_SHA256 = "9EF9C5326FC68CE4D9D602392F3DE107877C2C8083BC77A74D3B68799409ECF3"
EXPECTED_SFT_SHA256 = "6F99666EC06BD2D01589A4AC9F9B92D1B870BB23CF2E11B382C773F7EDC6B137"


def test_load_serving_config_defaults():
    cfg = load_serving_config()
    assert cfg.base_model_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert "final_adapter" in cfg.default_adapter
    assert cfg.num_candidates == 8
    assert cfg.sampling_temperature == pytest.approx(0.8)


def test_real_trained_adapters_resolve_and_match_sha256():
    cfg = load_serving_config()
    grpo_dir = _resolve_adapter_dir(cfg.default_adapter, [])
    sft_dir = _resolve_adapter_dir(cfg.sft_adapter, [])
    assert grpo_dir is not None and (grpo_dir / "adapter_model.safetensors").is_file()
    assert sft_dir is not None and (sft_dir / "adapter_model.safetensors").is_file()
    assert _compute_sha256(grpo_dir / "adapter_model.safetensors") == EXPECTED_GRPO_SHA256
    assert _compute_sha256(sft_dir / "adapter_model.safetensors") == EXPECTED_SFT_SHA256


def test_missing_adapter_raises_model_unavailable(tmp_path):
    cfg = ServingModelConfig(
        default_adapter=str(tmp_path / "does_not_exist"),
        sft_adapter=str(tmp_path / "missing_sft"),
        lazy_load=True,
    )
    backend = LocalSQLInferenceBackend(config=cfg)
    backend._GRPO_FALLBACKS = []
    backend._SFT_FALLBACKS = []

    with patch.dict("sys.modules", {"torch": object(), "transformers": object(), "peft": object()}):
        with pytest.raises(ModelUnavailableError):
            backend._ensure_loaded()


def test_missing_ml_deps_raises_model_unavailable():
    cfg = ServingModelConfig(lazy_load=True)
    backend = LocalSQLInferenceBackend(config=cfg)
    with patch.dict("sys.modules", {"torch": None}):
        with pytest.raises(ModelUnavailableError):
            backend.generate(
                prompt_messages=[{"role": "user", "content": "hi"}],
                num_candidates=1,
            )


class _FakeTensor:
    def __init__(self, data):
        self._data = data

    @property
    def shape(self):
        if self._data and isinstance(self._data[0], list):
            return (len(self._data), len(self._data[0]))
        return (len(self._data),)

    def to(self, _device):
        return self

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]


class _FakeTokenizer:
    pad_token = "<|endoftext|>"
    eos_token = "<|im_end|>"
    pad_token_id = 151643
    eos_token_id = 151645
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=1536):
        return {"input_ids": _FakeTensor([[10, 11, 12]]), "attention_mask": _FakeTensor([[1, 1, 1]])}

    def decode(self, token_ids, skip_special_tokens=True):
        return "<sql>SELECT name FROM employee</sql>"


class _FakePeftModel:
    def __init__(self):
        self.active_adapter = "grpo"
        self.disabled_calls = 0
        self.set_adapter_calls = []
        self.generate_calls = []

    def set_adapter(self, name):
        self.active_adapter = name
        self.set_adapter_calls.append(name)

    def disable_adapter(self):
        self.disabled_calls += 1
        return nullcontext()

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        n = kwargs.get("num_return_sequences", 1)
        return [_FakeTensor([10, 11, 12, 200, 201, 151645]) for _ in range(n)]


def test_local_backend_generate_greedy_and_majority8_paths():
    cfg = ServingModelConfig(lazy_load=True, max_new_tokens=64, sampling_temperature=0.8, sampling_top_p=0.95)
    backend = LocalSQLInferenceBackend(config=cfg)
    fake_tok = _FakeTokenizer()
    fake_model = _FakePeftModel()
    backend._tokenizer = fake_tok
    backend._model = fake_model
    backend._device = "cpu"
    backend._loaded_adapters = ["grpo", "sft"]
    backend._grpo_sha256 = EXPECTED_GRPO_SHA256
    backend._sft_sha256 = EXPECTED_SFT_SHA256

    class _FakeTorch:
        @staticmethod
        def inference_mode():
            return nullcontext()

    with patch.dict("sys.modules", {"torch": _FakeTorch}):
        # 1. Greedy with GRPO
        out_g = backend.generate(
            prompt_messages=[{"role": "user", "content": "list employee"}],
            num_candidates=1,
            model_choice="SFT + GRPO (RLVR)",
        )
        assert len(out_g.completions) == 1
        assert out_g.completions[0] == "<sql>SELECT name FROM employee</sql>"
        assert out_g.adapter_sha256 == EXPECTED_GRPO_SHA256
        assert fake_model.generate_calls[-1]["do_sample"] is False

        # 2. Majority@8 with SFT
        out_m = backend.generate(
            prompt_messages=[{"role": "user", "content": "list employee"}],
            num_candidates=8,
            model_choice="QLoRA SFT Baseline",
        )
        assert len(out_m.completions) == 8
        assert out_m.adapter_sha256 == EXPECTED_SFT_SHA256
        assert fake_model.generate_calls[-1]["do_sample"] is True
        assert fake_model.generate_calls[-1]["num_return_sequences"] == 8
        assert fake_model.generate_calls[-1]["temperature"] == pytest.approx(0.8)

        # 3. Zero-shot disables adapter
        out_z = backend.generate(
            prompt_messages=[{"role": "user", "content": "list employee"}],
            num_candidates=1,
            model_choice="Qwen2.5-Coder-1.5B (Zero-Shot)",
        )
        assert out_z.adapter_sha256 is None
        assert fake_model.disabled_calls == 1


def test_real_peft_and_transformers_roundtrip_on_local_qwen2_fixture(tmp_path):
    """End-to-end load & generate via real AutoTokenizer, AutoModelForCausalLM, and PeftModel."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    grpo_real_dir = _resolve_adapter_dir("./artifacts/phase_4/grpo/final_adapter", [])
    assert grpo_real_dir is not None

    base_dir = tmp_path / "tiny_qwen2"
    grpo_dir = tmp_path / "grpo_adapter"
    sft_dir = tmp_path / "sft_adapter"

    tok = AutoTokenizer.from_pretrained(str(grpo_real_dir), trust_remote_code=False)
    tiny_cfg = Qwen2Config(
        vocab_size=151936,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    base_model = Qwen2ForCausalLM(tiny_cfg)
    base_model.save_pretrained(str(base_dir))
    tok.save_pretrained(str(base_dir))

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    peft_grpo = get_peft_model(Qwen2ForCausalLM.from_pretrained(str(base_dir)), lora_cfg)
    peft_grpo.save_pretrained(str(grpo_dir))
    tok.save_pretrained(str(grpo_dir))

    peft_sft = get_peft_model(Qwen2ForCausalLM.from_pretrained(str(base_dir)), lora_cfg)
    peft_sft.save_pretrained(str(sft_dir))

    cfg = ServingModelConfig(
        base_model_id=str(base_dir),
        default_adapter=str(grpo_dir),
        sft_adapter=str(sft_dir),
        device="cpu",
        load_in_4bit=False,
        lazy_load=True,
        max_new_tokens=6,
    )
    backend = LocalSQLInferenceBackend(config=cfg)
    assert backend.is_ready() is False

    msgs = [
        {"role": "system", "content": "Return SQL."},
        {"role": "user", "content": "List employee names."},
    ]
    out_grpo = backend.generate(msgs, num_candidates=1, model_choice="SFT + GRPO (RLVR)")
    assert backend.is_ready() is True
    assert len(out_grpo.completions) == 1
    assert out_grpo.adapter_sha256 == _compute_sha256(grpo_dir / "adapter_model.safetensors")

    out_sft = backend.generate(msgs, num_candidates=2, model_choice="QLoRA SFT Baseline")
    assert len(out_sft.completions) == 2
    assert out_sft.adapter_sha256 == _compute_sha256(sft_dir / "adapter_model.safetensors")

    out_zs = backend.generate(msgs, num_candidates=1, model_choice="Qwen2.5-Coder-1.5B (Zero-Shot)")
    assert len(out_zs.completions) == 1
    assert out_zs.adapter_sha256 is None
