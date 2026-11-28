"""Run Comparator: Compares saved evaluation runs cleanly and flags slice/denominator mismatches."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from sql_grpo.evaluation.metrics import compute_benchmark_metrics


def _load_eval_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return list(data) if isinstance(data, list) else list(data.get("results", []))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def compare_evaluation_runs(run_paths: Dict[str, str | Path]) -> Dict[str, Any]:
    """Compares multiple runs on accuracy, validity, latency, and exact ID alignment."""
    summaries: Dict[str, Any] = {}
    id_sets: Dict[str, List[str]] = {}

    for label, raw_path in run_paths.items():
        rows = _load_eval_rows(Path(raw_path))
        summaries[label] = compute_benchmark_metrics(rows)
        id_sets[label] = [str(r.get("example_id") or r.get("id") or i) for i, r in enumerate(rows)]

    labels = list(run_paths.keys())
    all_matched_slice = True
    if len(labels) > 1:
        ref = id_sets[labels[0]]
        for other in labels[1:]:
            if id_sets[other] != ref:
                all_matched_slice = False
                break

    return {
        "matched_example_slice": all_matched_slice,
        "runs": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark evaluation prediction files.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of label=path pairs, e.g. base=reports/base.jsonl sft=reports/sft.jsonl",
    )
    args = parser.parse_args()

    mapping: Dict[str, str] = {}
    for entry in args.runs:
        if "=" not in entry:
            raise ValueError(f"Expected label=path, got {entry!r}")
        k, v = entry.split("=", 1)
        mapping[k] = v
    print(json.dumps(compare_evaluation_runs(mapping), indent=2))


if __name__ == "__main__":
    main()
