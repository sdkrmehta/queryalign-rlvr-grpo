"""Checkpointing, Reproducibility Metadata, and Adapter Reload Verification."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
from typing import Any, Dict, Optional

from sql_grpo import __version__


def compute_file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def _safe_git_sha(cwd: Optional[Path] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        return out.strip()
    except Exception:
        return "unknown"


def write_run_metadata(
    output_dir: str | Path,
    run_type: str,
    config: Dict[str, Any],
    adapter_dir: Optional[str | Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Writes reproducible run metadata (versions, git SHA, hardware, config digest, adapter SHA-256)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cuda_info: Dict[str, Any] = {"torch_version": "not_installed", "cuda_available": False}
    try:
        import torch

        cuda_info["torch_version"] = torch.__version__
        cuda_info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
            cuda_info["device_count"] = int(torch.cuda.device_count())
    except ImportError:
        pass

    adapter_sha256: Optional[str] = None
    if adapter_dir:
        adapter_sha256 = compute_file_sha256(Path(adapter_dir) / "adapter_model.safetensors")

    cfg_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest().upper()

    metadata = {
        "sql_grpo_version": __version__,
        "run_type": run_type,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _safe_git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "sqlite_version": sqlite3.sqlite_version,
        "command_line": sys.argv,
        "hardware": cuda_info,
        "config_sha256": cfg_hash,
        "config": config,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "adapter_sha256": adapter_sha256,
        **(extra or {}),
    }

    meta_path = out / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return meta_path


def find_latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    """Finds the highest-step `checkpoint-*` subdirectory for resume recovery."""
    out = Path(output_dir)
    if not out.is_dir():
        return None
    ckpts = []
    for p in out.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-", 1)[1])
                ckpts.append((step, p))
            except ValueError:
                continue
    if not ckpts:
        return None
    ckpts.sort(key=lambda item: item[0], reverse=True)
    return ckpts[0][1]


def verify_adapter_reload(
    base_model_id: str,
    adapter_dir: str | Path,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    """Reloads a saved PEFT adapter + tokenizer cleanly and verifies integrity metadata."""
    adapter_path = Path(adapter_dir)
    weight_file = adapter_path / "adapter_model.safetensors"
    config_file = adapter_path / "adapter_config.json"
    if not weight_file.is_file() or not config_file.is_file():
        raise FileNotFoundError(f"Saved PEFT adapter incomplete under {adapter_path}")

    adapter_cfg = json.loads(config_file.read_text(encoding="utf-8"))
    sha = compute_file_sha256(weight_file)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=local_files_only)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, local_files_only=local_files_only)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    peft_model.eval()
    return {
        "adapter_dir": str(adapter_path),
        "adapter_sha256": sha,
        "base_model_name_or_path": adapter_cfg.get("base_model_name_or_path"),
        "vocab_size": len(tokenizer),
        "peft_loaded": True,
    }
