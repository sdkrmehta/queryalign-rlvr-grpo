"""Error Analysis: Classifies failure modes and Greedy -> Majority@8 flips."""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List


def classify_failure_mode(record: Dict[str, Any]) -> str:
    """Categorizes a single evaluation record into a clean diagnostic outcome bucket."""
    if record.get("execution_match", record.get("greedy_correct", False)):
        return "CORRECT"
    if not record.get("format_valid", record.get("greedy_sql_valid", True)):
        return "FORMAT_ERROR"
    if "ast_safe" in record and not record.get("ast_safe"):
        return f"AST_REJECTED:{record.get('error_code') or 'POLICY'}"
    err_code = record.get("error_code")
    if err_code in ("TIMEOUT", "HARD_TIMEOUT_KILLED"):
        return "TIMEOUT"
    if err_code == "RESULT_TRUNCATED" or record.get("truncated"):
        return "TRUNCATED"
    if not record.get("execution_success", record.get("greedy_exec_ok", False)):
        return "RUNTIME_ERROR"
    if record.get("gold_is_empty"):
        return "EMPTY_GOLD_MISMATCH"
    return "WRONG_RESULT"


def analyze_evaluation_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produces aggregate failure distributions and Majority@8 correction/regression IDs."""
    mode_counts: Counter[str] = Counter()
    db_failures: Counter[str] = Counter()
    corrections: List[str] = []
    regressions: List[str] = []

    for r in records:
        mode = classify_failure_mode(r)
        mode_counts[mode] += 1
        if mode != "CORRECT":
            db_failures[str(r.get("db_id", "unknown"))] += 1

        if "majority8_correct" in r:
            g_ok = bool(r.get("execution_match", r.get("greedy_correct", False)))
            m_ok = bool(r.get("majority8_correct", False))
            ex_id = str(r.get("example_id") or r.get("id") or "")
            if (not g_ok) and m_ok:
                corrections.append(ex_id)
            elif g_ok and (not m_ok):
                regressions.append(ex_id)

    return {
        "total": len(records),
        "failure_mode_counts": dict(mode_counts),
        "top_failing_dbs": dict(db_failures.most_common(10)),
        "self_corrections_count": len(corrections),
        "self_corrections": corrections,
        "regressions_count": len(regressions),
        "regressions": regressions,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze evaluation error modes.")
    parser.add_argument("--input", required=True, help="Path to JSON or JSONL evaluation predictions.")
    args = parser.parse_args()

    p = Path(args.input)
    txt = p.read_text(encoding="utf-8")
    rows = json.loads(txt) if p.suffix.lower() == ".json" else [json.loads(l) for l in txt.splitlines() if l.strip()]
    print(json.dumps(analyze_evaluation_records(rows), indent=2))
