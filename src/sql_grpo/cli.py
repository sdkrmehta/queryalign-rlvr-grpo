"""Small environment checks used by the Phase 0 readiness gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from typing import Any


TRAINING_MODULES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "trl",
    "peft",
    "bitsandbytes",
    "unsloth",
    "sqlglot",
)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def collect_environment() -> dict[str, Any]:
    """Collect non-invasive environment details without installing packages."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "training_modules": {name: _module_available(name) for name in TRAINING_MODULES},
        "nvidia_smi": shutil.which("nvidia-smi"),
        "gpu": None,
        "cuda_available": None,
    }

    nvidia_smi = report["nvidia_smi"]
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            report["gpu"] = {
                "returncode": result.returncode,
                "output": result.stdout.strip(),
                "error": result.stderr.strip() or None,
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["gpu"] = {"error": str(exc)}

    if report["training_modules"]["torch"]:
        try:
            import torch  # type: ignore[import-not-found]

            report["torch_version"] = torch.__version__
            report["cuda_available"] = bool(torch.cuda.is_available())
            if report["cuda_available"]:
                report["cuda_device"] = torch.cuda.get_device_name(0)
                report["cuda_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        except Exception as exc:  # A diagnostic should report broken installs, not hide them.
            report["torch_error"] = f"{type(exc).__name__}: {exc}"

    return report


def main() -> int:
    parser = argparse.ArgumentParser(prog="sql-grpo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="report environment readiness details")
    args = parser.parse_args()

    if args.command == "doctor":
        print(json.dumps(collect_environment(), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
