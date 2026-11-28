"""Local LLM Inference Backend for QueryAlign Serving.

Loads Qwen2.5-Coder-1.5B-Instruct and trained PEFT LoRA adapters (GRPO / SFT)
strictly from local/Hugging Face open-weight checkpoints with zero external
SQL-generation API calls.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import inspect
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Protocol

import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "serving.yaml"


class ModelUnavailableError(RuntimeError):
    """Raised when local open-weight model dependencies or checkpoints cannot be loaded."""


@dataclass
class ServingModelConfig:
    """Configuration for local base model, LoRA adapters, and decoding parameters."""

    base_model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    revision: str = "main"
    default_adapter: str = "./artifacts/phase_4/grpo/final_adapter"
    sft_adapter: str = "./artifacts/phase_2&3/outputs/sft/best_adapter"
    device: str = "auto"
    load_in_4bit: bool = True
    lazy_load: bool = True
    max_prompt_tokens: int = 1536
    max_new_tokens: int = 256
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95
    num_candidates: int = 8
    min_plurality_threshold: int = 3


@dataclass
class InferenceOutput:
    """Raw completions and timing metadata produced by local model generation."""

    completions: List[str]
    generation_latency_ms: float
    model_choice: str
    base_model_id: str
    adapter_sha256: Optional[str] = None
    truncated_flags: List[bool] = field(default_factory=list)


class SQLInferenceBackend(Protocol):
    """Protocol implemented by local model inference backends (and test doubles)."""

    def is_ready(self) -> bool:
        """Return whether the model and adapters are currently loaded in memory."""

    def status(self) -> Dict[str, Any]:
        """Return non-sensitive diagnostic status for readiness checks."""

    def generate(
        self,
        prompt_messages: List[Dict[str, str]],
        num_candidates: int = 1,
        model_choice: str = "SFT + GRPO (RLVR)",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceOutput:
        """Generate one greedy or N stochastic completions locally."""


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _resolve_adapter_dir(configured: Optional[str], fallbacks: List[str]) -> Optional[Path]:
    """Resolve an adapter directory containing adapter_model.safetensors."""
    candidates: List[str] = []
    if configured:
        candidates.append(configured)
    candidates.extend(fallbacks)

    for raw in candidates:
        p = Path(raw)
        search_paths = [p] if p.is_absolute() else [Path.cwd() / p, _REPO_ROOT / p]
        for candidate in search_paths:
            resolved = candidate.resolve(strict=False)
            if (resolved / "adapter_model.safetensors").is_file():
                return resolved
    return None


def load_serving_config(config_path: Optional[str | Path] = None) -> ServingModelConfig:
    """Load serving configuration from YAML with environment variable overrides."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    model_cfg = raw.get("model", {})
    gen_cfg = raw.get("generation", {})

    return ServingModelConfig(
        base_model_id=os.getenv("MODEL_ID") or model_cfg.get("base_model_id", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        revision=os.getenv("MODEL_REVISION") or model_cfg.get("revision", "main"),
        default_adapter=os.getenv("GRPO_ADAPTER_PATH") or model_cfg.get(
            "default_adapter", "./artifacts/phase_4/grpo/final_adapter"
        ),
        sft_adapter=os.getenv("SFT_ADAPTER_PATH") or model_cfg.get(
            "sft_adapter", "./artifacts/phase_2&3/outputs/sft/best_adapter"
        ),
        device=os.getenv("SERVING_DEVICE") or model_cfg.get("device", "auto"),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        lazy_load=bool(model_cfg.get("lazy_load", True)),
        max_prompt_tokens=int(gen_cfg.get("max_prompt_tokens", 1536)),
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 256)),
        sampling_temperature=float(gen_cfg.get("sampling_temperature", 0.8)),
        sampling_top_p=float(gen_cfg.get("sampling_top_p", 0.95)),
        num_candidates=int(gen_cfg.get("num_candidates", 8)),
        min_plurality_threshold=int(gen_cfg.get("min_plurality_threshold", 3)),
    )


class LocalSQLInferenceBackend:
    """Thread-safe local Hugging Face Transformers + PEFT inference backend."""

    _GRPO_FALLBACKS = [
        "./artifacts/phase_4/grpo/final_adapter",
        "./artifacts/phase_4/outputs/grpo/final_adapter",
        "./outputs/grpo/run_001/final_adapter",
        "./outputs/grpo/final_adapter",
    ]
    _SFT_FALLBACKS = [
        "./artifacts/phase_2&3/outputs/sft/best_adapter",
        "./outputs/sft/run_001/best_adapter",
        "./outputs/sft/best_adapter",
    ]

    def __init__(self, config: Optional[ServingModelConfig] = None):
        self.config = config or load_serving_config()
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._grpo_adapter_dir: Optional[Path] = None
        self._sft_adapter_dir: Optional[Path] = None
        self._grpo_sha256: Optional[str] = None
        self._sft_sha256: Optional[str] = None
        self._loaded_adapters: List[str] = []
        self._load_error: Optional[str] = None

        if not self.config.lazy_load:
            self._ensure_loaded()

    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self.is_ready(),
            "base_model_id": self.config.base_model_id,
            "revision": self.config.revision,
            "device": str(self._device) if self._device is not None else self.config.device,
            "active_adapters": list(self._loaded_adapters),
            "grpo_adapter_sha256": self._grpo_sha256,
            "sft_adapter_sha256": self._sft_sha256,
            "error": self._load_error,
        }

    def _ensure_loaded(self) -> None:
        if self.is_ready():
            return

        with self._load_lock:
            if self.is_ready():
                return

            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                self._load_error = f"Missing local ML inference dependency: {exc}"
                raise ModelUnavailableError(
                    "Local ML dependencies (torch, transformers, peft, accelerate) are required "
                    f"for inference: {exc}"
                ) from exc

            grpo_dir = _resolve_adapter_dir(self.config.default_adapter, self._GRPO_FALLBACKS)
            if grpo_dir is None:
                self._load_error = f"GRPO adapter weights not found at {self.config.default_adapter}"
                raise ModelUnavailableError(
                    f"Trained GRPO adapter not found (checked {self.config.default_adapter})."
                )

            sft_dir = _resolve_adapter_dir(self.config.sft_adapter, self._SFT_FALLBACKS)
            self._grpo_adapter_dir = grpo_dir
            self._sft_adapter_dir = sft_dir
            self._grpo_sha256 = _compute_sha256(grpo_dir / "adapter_model.safetensors")
            if sft_dir is not None:
                self._sft_sha256 = _compute_sha256(sft_dir / "adapter_model.safetensors")

            req_device = (self.config.device or "auto").lower()
            use_cuda = req_device in ("auto", "cuda") and torch.cuda.is_available()
            if req_device == "cuda" and not use_cuda:
                logger.warning("CUDA requested in serving config but unavailable; falling back to CPU.")

            try:
                is_local_base = Path(self.config.base_model_id).exists()
                tok_source = str(grpo_dir) if (grpo_dir / "tokenizer_config.json").is_file() else self.config.base_model_id
                tok_kwargs: Dict[str, Any] = {"trust_remote_code": False}
                if tok_source == self.config.base_model_id and self.config.revision and not is_local_base:
                    tok_kwargs["revision"] = self.config.revision
                tokenizer = AutoTokenizer.from_pretrained(tok_source, **tok_kwargs)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"

                model_kwargs: Dict[str, Any] = {"trust_remote_code": False}
                if self.config.revision and not is_local_base:
                    model_kwargs["revision"] = self.config.revision

                if use_cuda:
                    major, _ = torch.cuda.get_device_capability(0)
                    compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
                    dtype_kw = "dtype" if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
                    model_kwargs[dtype_kw] = compute_dtype
                    model_kwargs["device_map"] = {"": 0}

                    if self.config.load_in_4bit:
                        try:
                            from transformers import BitsAndBytesConfig

                            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=compute_dtype,
                                bnb_4bit_use_double_quant=True,
                            )
                        except Exception as bnb_exc:
                            logger.warning("4-bit BitsAndBytes unavailable (%s); loading %s.", bnb_exc, compute_dtype)

                    target_device = torch.device("cuda:0")
                else:
                    dtype_kw = "dtype" if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
                    model_kwargs[dtype_kw] = torch.float32
                    target_device = torch.device("cpu")

                base_model = AutoModelForCausalLM.from_pretrained(self.config.base_model_id, **model_kwargs)
                if not use_cuda:
                    base_model = base_model.to(target_device)

                peft_model = PeftModel.from_pretrained(
                    base_model,
                    str(grpo_dir),
                    adapter_name="grpo",
                    is_trainable=False,
                )
                loaded_adapters = ["grpo"]
                if sft_dir is not None:
                    peft_model.load_adapter(str(sft_dir), adapter_name="sft", is_trainable=False)
                    loaded_adapters.append("sft")

                peft_model.set_adapter("grpo")
                peft_model.eval()

                self._tokenizer = tokenizer
                self._model = peft_model
                self._device = target_device
                self._loaded_adapters = loaded_adapters
                self._load_error = None
                logger.info(
                    "Loaded local model %s (%s) with adapters=%s grpo_sha256=%s",
                    self.config.base_model_id,
                    target_device,
                    loaded_adapters,
                    self._grpo_sha256,
                )
            except ModelUnavailableError:
                raise
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise ModelUnavailableError(
                    f"Failed to load local model '{self.config.base_model_id}': {exc}"
                ) from exc

    def _select_adapter_context(self, model_choice: str):
        choice_low = (model_choice or "").lower()
        if "zero-shot" in choice_low or "zero shot" in choice_low or choice_low == "base":
            return self._model.disable_adapter(), None
        if "sft" in choice_low and "grpo" not in choice_low and "rlvr" not in choice_low:
            if "sft" not in self._loaded_adapters:
                raise ModelUnavailableError("SFT baseline adapter is not loaded.")
            self._model.set_adapter("sft")
            return nullcontext(), self._sft_sha256
        self._model.set_adapter("grpo")
        return nullcontext(), self._grpo_sha256

    def generate(
        self,
        prompt_messages: List[Dict[str, str]],
        num_candidates: int = 1,
        model_choice: str = "SFT + GRPO (RLVR)",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceOutput:
        self._ensure_loaded()
        import torch

        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = self.config.sampling_temperature if temperature is None else temperature
        p_val = self.config.sampling_top_p if top_p is None else top_p

        prompt_text = self._tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_tokens,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[1])

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if num_candidates > 1:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": max(temp, 1e-4),
                    "top_p": p_val,
                    "num_return_sequences": num_candidates,
                }
            )
        else:
            gen_kwargs.update(
                {
                    "do_sample": False,
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                    "num_return_sequences": 1,
                }
            )

        t0 = time.perf_counter()
        with self._gen_lock:
            ctx, active_sha = self._select_adapter_context(model_choice)
            with ctx, torch.inference_mode():
                outputs = self._model.generate(**inputs, **gen_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        completions: List[str] = []
        truncated_flags: List[bool] = []
        eos_id = self._tokenizer.eos_token_id
        for seq in outputs:
            gen_tokens = seq[input_len:]
            is_trunc = len(gen_tokens) >= max_tokens and (eos_id is None or int(gen_tokens[-1]) != int(eos_id))
            truncated_flags.append(bool(is_trunc))
            completions.append(self._tokenizer.decode(gen_tokens, skip_special_tokens=True))

        return InferenceOutput(
            completions=completions,
            generation_latency_ms=latency_ms,
            model_choice=model_choice,
            base_model_id=self.config.base_model_id,
            adapter_sha256=active_sha,
            truncated_flags=truncated_flags,
        )
