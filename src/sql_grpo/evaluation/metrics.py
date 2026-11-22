"""Evaluation Metrics: Computes EX, Valid SQL Rate, AST Safety Rate, and Latency quantiles."""

from typing import Any, Dict, List, Optional
import numpy as np


def compute_benchmark_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes aggregate evaluation metrics across a list of per-example evaluation records."""
    total = len(records)
    if total == 0:
        return {}

    format_valid_count = sum(1 for r in records if r.get("format_valid", False))
    ast_safe_count = sum(1 for r in records if r.get("ast_safe", False))
    exec_success_count = sum(1 for r in records if r.get("execution_success", False))
    match_count = sum(1 for r in records if r.get("execution_match", False))

    latencies = [r.get("latency_total_ms", 0.0) for r in records if r.get("latency_total_ms", 0.0) > 0]

    p50_latency = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "total_examples": total,
        "format_valid_rate": round(format_valid_count / total * 100.0, 2),
        "ast_safe_rate": round(ast_safe_count / total * 100.0, 2),
        "valid_sql_rate": round(exec_success_count / total * 100.0, 2),
        "execution_accuracy": round(match_count / total * 100.0, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
    }
