"""Evaluation Metrics: Computes EX, validity, safety, consensus coverage, and latency quantiles."""

from collections import Counter
from typing import Any, Dict, List
import numpy as np


def _safe_percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(values, q, method="linear")), 3)


def compute_benchmark_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes aggregate evaluation metrics across per-example evaluation records."""
    total = len(records)
    if total == 0:
        return {}

    format_valid_count = sum(1 for r in records if r.get("format_valid", False))
    ast_safe_count = sum(1 for r in records if r.get("ast_safe", False))
    exec_success_count = sum(1 for r in records if r.get("execution_success", False))
    match_count = sum(
        1
        for r in records
        if r.get("execution_match", r.get("greedy_correct", False))
    )
    timeout_count = sum(1 for r in records if r.get("error_code") in ("TIMEOUT", "HARD_TIMEOUT_KILLED"))
    truncated_count = sum(1 for r in records if r.get("error_code") == "RESULT_TRUNCATED" or r.get("truncated", False))
    empty_gold_count = sum(1 for r in records if r.get("gold_is_empty", False))

    total_lats = [
        float(r.get("latency_total_ms", r.get("greedy_latency_ms", 0.0)))
        for r in records
        if float(r.get("latency_total_ms", r.get("greedy_latency_ms", 0.0))) > 0
    ]
    gen_lats = [float(r["latency_generation_ms"]) for r in records if float(r.get("latency_generation_ms", 0.0)) > 0]
    val_lats = [float(r["latency_validation_ms"]) for r in records if float(r.get("latency_validation_ms", 0.0)) > 0]
    exec_lats = [float(r["latency_execution_ms"]) for r in records if float(r.get("latency_execution_ms", 0.0)) > 0]

    result: Dict[str, Any] = {
        "total_examples": total,
        "percentile_method": "linear (NumPy default, Hyndman & Fan Type 7)",
        "format_valid_count": format_valid_count,
        "ast_safe_count": ast_safe_count,
        "exec_success_count": exec_success_count,
        "execution_match_count": match_count,
        "timeout_count": timeout_count,
        "truncated_count": truncated_count,
        "empty_gold_count": empty_gold_count,
        "format_valid_rate": round(format_valid_count / total * 100.0, 2),
        "ast_safe_rate": round(ast_safe_count / total * 100.0, 2),
        "valid_sql_rate": round(exec_success_count / total * 100.0, 2),
        "execution_accuracy": round(match_count / total * 100.0, 2),
        "p50_latency_ms": _safe_percentile(total_lats, 50),
        "p95_latency_ms": _safe_percentile(total_lats, 95),
        "p50_generation_ms": _safe_percentile(gen_lats, 50),
        "p95_generation_ms": _safe_percentile(gen_lats, 95),
        "p50_validation_ms": _safe_percentile(val_lats, 50),
        "p50_execution_ms": _safe_percentile(exec_lats, 50),
    }

    if any("majority8_status" in r for r in records):
        status_counts = Counter(r.get("majority8_status") for r in records)
        maj8_correct = sum(1 for r in records if r.get("majority8_correct", False))
        ce_total = status_counts.get("CONSENSUS_ELECTED", 0)
        ce_correct = sum(
            1
            for r in records
            if r.get("majority8_status") == "CONSENSUS_ELECTED" and r.get("majority8_correct", False)
        )
        abstained_count = sum(
            status_counts.get(s, 0)
            for s in ("LOW_CONFIDENCE", "NO_CONSENSUS_TIE", "ALL_FAILED")
        )
        maj8_lats = [float(r["majority8_latency_ms"]) for r in records if float(r.get("majority8_latency_ms", 0.0)) > 0]
        result["majority8"] = {
            "execution_accuracy": round(maj8_correct / total * 100.0, 2),
            "correct_count": maj8_correct,
            "consensus_coverage_pct": round(ce_total / total * 100.0, 2),
            "consensus_precision_pct": round((ce_correct / ce_total * 100.0) if ce_total else 0.0, 2),
            "abstention_rate_pct": round(abstained_count / total * 100.0, 2),
            "p50_latency_ms": _safe_percentile(maj8_lats, 50),
            "p95_latency_ms": _safe_percentile(maj8_lats, 95),
            "status_counts": dict(status_counts),
        }

    return result
