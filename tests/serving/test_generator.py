"""Tests for SQLGenerator, Local Inference Integration, AST Safety, and Majority@8 Voting."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.serving.generator import SQLGenerator
from sql_grpo.serving.inference import InferenceOutput


class StubInferenceBackend:
    """Deterministic local-inference test double returning preconfigured completions."""

    def __init__(self, completions_by_mode: Optional[Dict[int, List[str]]] = None):
        self.completions_by_mode = completions_by_mode or {
            1: ["<sql>SELECT name FROM employee</sql>"],
            8: ["<sql>SELECT name FROM employee</sql>"] * 8,
        }
        self.calls: List[Dict[str, Any]] = []

    def is_ready(self) -> bool:
        return True

    def status(self) -> Dict[str, Any]:
        return {"loaded": True, "base_model_id": "stub"}

    def generate(
        self,
        prompt_messages: List[Dict[str, str]],
        num_candidates: int = 1,
        model_choice: str = "SFT + GRPO (RLVR)",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceOutput:
        self.calls.append(
            {
                "prompt_messages": prompt_messages,
                "num_candidates": num_candidates,
                "model_choice": model_choice,
            }
        )
        comps = self.completions_by_mode.get(num_candidates) or self.completions_by_mode[1]
        return InferenceOutput(
            completions=list(comps[:num_candidates]),
            generation_latency_ms=12.5,
            model_choice=model_choice,
            base_model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            adapter_sha256="9EF9C5326FC68CE4D9D602392F3DE107877C2C8083BC77A74D3B68799409ECF3",
        )


def _make_generator(completions_by_mode: Optional[Dict[int, List[str]]] = None):
    data_dir = Path(__file__).resolve().parents[2] / "data"
    registry = DatabaseRegistry(data_root=data_dir)
    sandbox = SQLiteSandbox()
    stub = StubInferenceBackend(completions_by_mode=completions_by_mode)
    return SQLGenerator(registry=registry, sandbox=sandbox, inference_backend=stub), stub


def test_registry_list_databases():
    gen, _ = _make_generator()
    dbs = gen.registry.list_available_databases()
    assert "test_company" in dbs
    assert "concert_singer" in dbs
    assert len(dbs) > 10


def test_generator_uses_inference_backend_not_manifest_replay():
    # Ask a real Spider dev_manifest question, verify stub backend output is used (no gold_sql replay)
    gen, stub = _make_generator(
        {1: ["<sql>SELECT name FROM singer LIMIT 2</sql>"]}
    )
    res = gen.execute_and_evaluate(
        question="How many singers do we have?",
        database_id="concert_singer",
        model_choice="SFT + GRPO (RLVR)",
        inference_mode="Greedy (Pass@1)",
    )
    assert len(stub.calls) == 1
    user_msg = stub.calls[0]["prompt_messages"][1]["content"]
    assert "DATABASE SCHEMA:" in user_msg
    assert "Table: singer" in user_msg
    assert "How many singers do we have?" in user_msg
    assert res.selected_sql == "SELECT name FROM singer LIMIT 2"
    assert res.format_valid is True
    assert res.ast_safe is True
    assert res.execution_result is not None and res.execution_result.success is True
    assert len(res.rows) == 2
    assert res.rewards is None  # Live serving has no ground truth gold_rows


def test_ast_unsafe_model_output_is_never_executed():
    gen, _ = _make_generator(
        {1: ["<sql>DROP TABLE employee;</sql>"]}
    )
    res = gen.execute_and_evaluate(
        question="delete employees",
        database_id="test_company",
        inference_mode="Greedy (Pass@1)",
    )
    assert res.format_valid is True
    assert res.ast_safe is False
    assert res.ast_error_code == "DISALLOWED_OPERATION"
    assert res.execution_result is None
    assert res.rows == []


def test_majority8_clustering_and_single_execution():
    candidates = [
        "<sql>SELECT name FROM employee</sql>",
        "<sql>SELECT name FROM employee ORDER BY emp_id</sql>",
        "<sql>SELECT name FROM employee WHERE salary > 0</sql>",
        "<sql>SELECT name FROM employee WHERE emp_id >= 100</sql>",
        "<sql>SELECT name FROM employee LIMIT 10</sql>",
        "<sql>SELECT name FROM department</sql>",
        "<sql>SELECT name FROM department LIMIT 2</sql>",
        "<sql>DROP TABLE employee</sql>",
    ]
    gen, _ = _make_generator({8: candidates})
    res = gen.execute_and_evaluate(
        question="list employee name",
        database_id="test_company",
        inference_mode="Majority@8 (Self-Consistency)",
    )
    assert res.voting_outcome is not None
    assert res.voting_outcome.is_abstained is False
    assert res.voting_outcome.total_candidates == 8
    assert res.voting_outcome.num_safe == 7
    assert res.voting_outcome.num_executed == 7
    assert res.voting_outcome.cluster_size == 5
    assert res.selected_sql == "SELECT name FROM employee"
    assert len(res.candidates) == 8
    assert [c.cluster_id for c in res.candidates] == [1, 1, 1, 1, 1, 2, 3, None]


def test_majority8_tie_and_all_failed_abstention():
    # Tie: 3 vs 3
    tie_candidates = (
        ["<sql>SELECT name FROM employee</sql>"] * 3
        + ["<sql>SELECT name FROM department</sql>"] * 3
        + ["bad output"] * 2
    )
    gen_tie, _ = _make_generator({8: tie_candidates})
    res_tie = gen_tie.execute_and_evaluate(
        question="list names",
        database_id="test_company",
        inference_mode="Majority@8 (Self-Consistency)",
    )
    assert res_tie.voting_outcome is not None
    assert res_tie.voting_outcome.is_abstained is True
    assert res_tie.voting_outcome.abstention_reason == "NO_CONSENSUS_TIE"

    # All failed: no fallback SQL leaked
    bad_candidates = ["<sql>DROP TABLE employee</sql>"] * 4 + ["no tags"] * 4
    gen_bad, _ = _make_generator({8: bad_candidates})
    res_bad = gen_bad.execute_and_evaluate(
        question="list names",
        database_id="test_company",
        inference_mode="Majority@8 (Self-Consistency)",
    )
    assert res_bad.voting_outcome is not None
    assert res_bad.voting_outcome.is_abstained is True
    assert res_bad.voting_outcome.abstention_reason == "ALL_CANDIDATES_FAILED_OR_UNSAFE"
    assert res_bad.selected_sql is None


def test_execute_false_skips_sandbox_execution():
    gen, _ = _make_generator({1: ["<sql>SELECT name FROM employee</sql>"]})
    res = gen.execute_and_evaluate(
        question="list employee name",
        database_id="test_company",
        inference_mode="Greedy (Pass@1)",
        execute=False,
    )
    assert res.format_valid is True
    assert res.ast_safe is True
    assert res.selected_sql == "SELECT name FROM employee"
    assert res.execution_result is None
    assert res.execution_latency_ms == 0.0
