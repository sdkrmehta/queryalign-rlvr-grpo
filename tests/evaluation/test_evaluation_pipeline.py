"""Evaluation Harness, Error Analysis, and Run Comparator Tests."""

import json
from pathlib import Path
import pytest

from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.evaluation.compare_runs import compare_evaluation_runs
from sql_grpo.evaluation.error_analysis import analyze_evaluation_records, classify_failure_mode
from sql_grpo.evaluation.metrics import compute_benchmark_metrics
from sql_grpo.evaluation.run_benchmark import evaluate_examples
from tests.serving.test_generator import StubInferenceBackend


class _PerPromptStubBackend(StubInferenceBackend):
    def __init__(self, responses):
        super().__init__()
        self._responses = responses
        self._idx = 0

    def generate(self, prompt_messages, num_candidates=1, model_choice="SFT + GRPO (RLVR)", **_kwargs):
        cands = self._responses[self._idx]
        self._idx += 1
        out = super().generate(prompt_messages, num_candidates=len(cands), model_choice=model_choice)
        out.completions = cands[:num_candidates]
        return out


def test_evaluate_examples_and_metrics_denominators():
    reg = DatabaseRegistry(data_root=Path("./data"))
    sb = SQLiteSandbox()
    examples = [
        {
            "example_id": "ex_ok",
            "db_id": "test_company",
            "question": "List department names",
            "valid_tables": ["department", "employee"],
            "gold_result_json": json.dumps([["Engineering"], ["Sales"], ["HR"]]),
            "gold_is_empty": False,
        },
        {
            "example_id": "ex_unsafe",
            "db_id": "test_company",
            "question": "Drop employee",
            "valid_tables": ["department", "employee"],
            "gold_result_json": json.dumps([]),
            "gold_is_empty": True,
        },
        {
            "example_id": "ex_tie",
            "db_id": "test_company",
            "question": "Ambiguous tie",
            "valid_tables": ["department", "employee"],
            "gold_result_json": json.dumps([["Engineering"], ["Sales"], ["HR"]]),
            "gold_is_empty": False,
        },
    ]

    backend = _PerPromptStubBackend(
        [
            ["<sql>SELECT name FROM department</sql>"] * 8,
            ["<sql>DROP TABLE employee;</sql>"] * 8,
            ["<sql>SELECT name FROM department</sql>"] * 4 + ["<sql>SELECT name FROM employee</sql>"] * 4,
        ]
    )

    records, summary = evaluate_examples(
        examples=examples,
        inference_backend=backend,
        model_variant="grpo",
        registry=reg,
        sandbox=sb,
        majority_k=8,
    )
    assert len(records) == 3
    assert records[0]["execution_match"] is True
    assert records[0]["majority8_status"] == "CONSENSUS_ELECTED"
    assert records[0]["majority8_correct"] is True

    # Failed / unsafe outputs stay in denominator and are never executed
    assert records[1]["format_valid"] is True
    assert records[1]["ast_safe"] is False
    assert records[1]["execution_success"] is False
    assert records[1]["majority8_status"] == "ALL_FAILED"

    # Tie abstains cleanly in evaluation harness
    assert records[2]["majority8_status"] == "NO_CONSENSUS_TIE"
    assert records[2]["majority8_sql"] is None

    metrics = summary["metrics"]
    assert metrics["total_examples"] == 3
    assert metrics["execution_match_count"] == 2
    assert metrics["majority8"]["consensus_coverage_pct"] == pytest.approx(33.33, abs=0.05)


def test_error_analysis_and_compare_runs(tmp_path):
    assert classify_failure_mode({"execution_match": False, "format_valid": False}) == "FORMAT_ERROR"
    assert classify_failure_mode({"execution_match": False, "format_valid": True, "ast_safe": False, "error_code": "UNAUTHORIZED_TABLE"}) == "AST_REJECTED:UNAUTHORIZED_TABLE"
    assert classify_failure_mode({"execution_match": False, "format_valid": True, "ast_safe": True, "error_code": "TIMEOUT"}) == "TIMEOUT"

    # Verify error analysis on real audited final_benchmark_results.json
    real_results_path = Path("reports/final_benchmark_results.json")
    if real_results_path.is_file():
        rows = json.loads(real_results_path.read_text(encoding="utf-8"))
        analysis = analyze_evaluation_records(rows)
        assert analysis["total"] == 50
        assert analysis["self_corrections_count"] == 8
        assert analysis["regressions_count"] == 1
        assert analysis["regressions"] == ["dev_000031"]

    # Compare mismatched slices vs identical slices
    run_a = tmp_path / "run_a.jsonl"
    run_b = tmp_path / "run_b.jsonl"
    run_c = tmp_path / "run_c.jsonl"
    run_a.write_text('{"example_id":"e1","execution_match":true}\n{"example_id":"e2","execution_match":false}\n', encoding="utf-8")
    run_b.write_text('{"example_id":"e1","execution_match":true}\n{"example_id":"e2","execution_match":true}\n', encoding="utf-8")
    run_c.write_text('{"example_id":"e1","execution_match":true}\n', encoding="utf-8")

    matched = compare_evaluation_runs({"a": run_a, "b": run_b})
    assert matched["matched_example_slice"] is True
    assert matched["runs"]["b"]["execution_accuracy"] == 100.0

    mismatched = compare_evaluation_runs({"a": run_a, "c": run_c})
    assert mismatched["matched_example_slice"] is False
