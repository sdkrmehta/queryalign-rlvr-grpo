"""Tests for SQLGenerator, Dynamic Introspection, and Serving Pipeline."""

import pytest
from pathlib import Path

from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.serving.generator import SQLGenerator


@pytest.fixture
def generator():
    data_dir = Path(__file__).resolve().parents[2] / "data"
    registry = DatabaseRegistry(data_root=data_dir)
    sandbox = SQLiteSandbox()
    return SQLGenerator(registry=registry, sandbox=sandbox)


def test_registry_list_databases(generator):
    dbs = generator.registry.list_available_databases()
    assert "test_company" in dbs
    assert "concert_singer" in dbs
    assert len(dbs) > 10


def test_question_change_changes_sql_and_results(generator):
    # Query 1: list employee name
    r1 = generator.execute_and_evaluate(
        question="list employee name",
        database_id="test_company",
        model_choice="QLoRA SFT Baseline",
        inference_mode="Greedy (Pass@1)",
    )
    assert r1.format_valid is True
    assert r1.ast_safe is True
    assert "employee" in r1.selected_sql.lower()
    assert r1.rows == [("Alice",), ("Bob",), ("Charlie",), ("David",), ("Eve",)]

    # Query 2: list departments with budget > 200000
    r2 = generator.execute_and_evaluate(
        question="List all departments with a budget greater than 200,000.",
        database_id="test_company",
        model_choice="QLoRA SFT Baseline",
        inference_mode="Greedy (Pass@1)",
    )
    assert r2.format_valid is True
    assert r2.ast_safe is True
    assert "department" in r2.selected_sql.lower()
    assert "budget" in r2.selected_sql.lower()
    # Results must differ from Query 1
    assert r1.selected_sql != r2.selected_sql
    assert r1.rows != r2.rows


def test_target_database_change(generator):
    # Query on test_company
    r_company = generator.execute_and_evaluate(
        question="list employee name",
        database_id="test_company",
        model_choice="QLoRA SFT Baseline",
        inference_mode="Greedy (Pass@1)",
    )
    # Query on concert_singer
    r_singer = generator.execute_and_evaluate(
        question="How many singers do we have?",
        database_id="concert_singer",
        model_choice="QLoRA SFT Baseline",
        inference_mode="Greedy (Pass@1)",
    )
    assert r_company.database_id == "test_company"
    assert r_singer.database_id == "concert_singer"
    assert r_singer.rows == [(6,)]
    assert r_company.selected_sql != r_singer.selected_sql


def test_inference_mode_majority8_vs_greedy(generator):
    r_greedy = generator.execute_and_evaluate(
        question="list employee name",
        database_id="test_company",
        model_choice="SFT + GRPO (RLVR)",
        inference_mode="Greedy (Pass@1)",
    )
    assert r_greedy.voting_outcome is None
    assert len(r_greedy.candidates) == 0

    r_maj8 = generator.execute_and_evaluate(
        question="list employee name",
        database_id="test_company",
        model_choice="SFT + GRPO (RLVR)",
        inference_mode="Majority@8 (Self-Consistency)",
    )
    assert r_maj8.voting_outcome is not None
    assert r_maj8.voting_outcome.total_candidates == 8
    assert r_maj8.voting_outcome.cluster_size >= 4
    assert len(r_maj8.candidates) == 8


def test_active_model_change_reward_telemetry(generator):
    # Zero-shot does not produce GRPO composite reward breakdown
    r_zs = generator.execute_and_evaluate(
        question="list employee name",
        database_id="test_company",
        model_choice="Qwen2.5-Coder-1.5B (Zero-Shot)",
        inference_mode="Greedy (Pass@1)",
    )
    assert r_zs.rewards is None

    # GRPO produces verifiable composite reward breakdown
    r_grpo = generator.execute_and_evaluate(
        question="list employee name",
        database_id="test_company",
        model_choice="SFT + GRPO (RLVR)",
        inference_mode="Greedy (Pass@1)",
    )
    assert r_grpo.rewards is not None
    assert r_grpo.rewards.r_fmt == 0.20
    assert r_grpo.rewards.r_ast == 0.20
    assert r_grpo.rewards.r_exec_valid == 0.30
    assert r_grpo.rewards.r_total >= 1.0


def test_table_extraction_correctness(generator):
    # car_makers in car_1 (must NOT extract continents)
    r_maker = generator.execute_and_evaluate(
        question="What are the names of all car makers?",
        database_id="car_1",
    )
    assert "car_makers" in r_maker.selected_sql.lower()
    assert "continents" not in r_maker.selected_sql.lower()
    assert len(r_maker.rows) == 22

    # publication in academic (must NOT extract author)
    r_paper = generator.execute_and_evaluate(
        question="How many papers are there?",
        database_id="academic",
    )
    assert "publication" in r_paper.selected_sql.lower()
    assert "author" not in r_paper.selected_sql.lower()


def test_top_5_budget_limit(generator):
    # top 5 budget must include LIMIT 5 and DESC ordering
    r_top = generator.execute_and_evaluate(
        question="list top 5 budget",
        database_id="test_company",
    )
    assert "limit 5" in r_top.selected_sql.lower()
    assert "desc" in r_top.selected_sql.lower()
    assert len(r_top.rows) == 3  # department has 3 records total
    assert r_top.rows[0][0] == 500000.0 or r_top.rows[0][-1] == 500000.0


def test_projection_precision(generator):
    # List all employee name with a salary greater than 2000.
    # Must only project name, not unrequested salary
    r = generator.execute_and_evaluate(
        question="List all employee name with a salary greater than 2000.",
        database_id="test_company",
    )
    assert "name" in r.columns
    assert "salary" not in r.columns
    assert len(r.rows) == 5
