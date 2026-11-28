"""Training Callbacks & Telemetry: Logs RLVR reward breakdowns, zero-variance rate, and VRAM."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base class when transformers is unavailable in lightweight test envs."""


def capture_cuda_memory_mb() -> Dict[str, float]:
    """Safely captures allocated and reserved CUDA memory in MiB (zeros on CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2),
                "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2),
                "max_reserved_mb": round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2),
            }
    except Exception:
        pass
    return {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "max_reserved_mb": 0.0,
    }


def summarize_reward_vector(rewards: List[float], group_size: int = 8) -> Dict[str, float]:
    """Computes finite summary stats and zero-variance group fraction for a reward batch."""
    if not rewards:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "zero_variance_pct": 0.0}

    clean = [float(r) for r in rewards if math.isfinite(float(r))]
    if not clean:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "zero_variance_pct": 0.0}

    mean_v = sum(clean) / len(clean)
    var_v = sum((x - mean_v) ** 2 for x in clean) / len(clean)
    std_v = math.sqrt(var_v)

    zero_groups = 0
    total_groups = 0
    if group_size > 0 and len(clean) >= group_size:
        usable = len(clean) - (len(clean) % group_size)
        for start in range(0, usable, group_size):
            chunk = clean[start : start + group_size]
            total_groups += 1
            if max(chunk) - min(chunk) <= 1e-12:
                zero_groups += 1

    zero_pct = (zero_groups / total_groups * 100.0) if total_groups > 0 else 0.0
    return {
        "mean": round(mean_v, 4),
        "std": round(std_v, 4),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
        "zero_variance_pct": round(zero_pct, 2),
    }


class RLVRMetricsCallback(TrainerCallback):
    """Persists step-level training logs and resource usage locally as JSONL/CSV."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.log_jsonl_path = self.metrics_dir / "training_log.jsonl"
        self.resource_csv_path = self.metrics_dir / "resource_usage.csv"
        if not self.resource_csv_path.exists():
            with open(self.resource_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp_utc", "global_step", "allocated_mb", "reserved_mb", "max_allocated_mb", "max_reserved_mb"]
                )

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        if not logs:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        mem = capture_cuda_memory_mb()
        ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp_utc": ts,
            "global_step": step,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool))},
            **mem,
        }
        with open(self.log_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        with open(self.resource_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [ts, step, mem["allocated_mb"], mem["reserved_mb"], mem["max_allocated_mb"], mem["max_reserved_mb"]]
            )
