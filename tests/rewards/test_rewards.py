"""Mandatory Verifiable Reward & Composite Bounds Tests (Blueprint Section 19.4)."""

import json
from pathlib import Path
import pytest

try:
    from data.fixtures.create_fixtures import create_test_db
except ImportError:
    from sql_grpo.data.fixtures.create_fixtures import create_test_db

from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.rewards.accuracy_reward import accuracy_reward_func, compute_accuracy_reward
from sql_grpo.rewards.composite import (
    compute_composite_reward,
    compute_composite_reward_batch,
)
from sql_grpo.rewards.efficiency_reward import compute_efficiency_reward, efficiency_reward_func
from sql_grpo.rewards.execution_reward import compute_execution_reward, execution_reward_func
from sql_grpo.rewards.format_reward import compute_format_reward, format_reward_func
from sql_grpo.rewards.schema_reward import compute_schema_reward, schema_reward_func
from sql_grpo.safety.sql_parser import SQLCompletionParser

FIXTURE_TABLES = {"department", "employee"}
FIXTURE_COLUMNS = {
    "department": {"dept_id", "name", "budget"},
    "employee": {"emp_id", "name", "dept_id", "salary"},
}


@pytest.fixture(scope="module")
def temp_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("reward_data")
    db_file = root / "fixtures" / "company.sqlite"
    create_test_db(db_file)
    reg = DatabaseRegistry(data_root=root)
    reg.register("company", db_file)
    sb = SQLiteSandbox(row_limit=5000, timeout_seconds=2.0)
    return reg, sb, db_file


def test_format_parser_and_rewards():
    assert compute_format_reward("<sql>SELECT name FROM department</sql>") == 0.20
    assert compute_format_reward("SELECT name FROM department") == -0.50
    assert compute_format_reward("<sql>SELECT name FROM department") == -0.50  # truncated
    assert compute_format_reward("<sql>SELECT 1</sql><sql>SELECT 2</sql>") == -0.50  # extra tags
    assert compute_format_reward("<sql>SELECT name FROM department</sql>\nTrailing note.") == -0.50

    parsed_md = SQLCompletionParser.parse("<sql>```sql\nSELECT 1\n```</sql>", strip_markdown=True)
    assert parsed_md.is_valid_format is True
    assert parsed_md.cleaned_markdown is True
    assert parsed_md.sql == "SELECT 1"

    parsed_abs = SQLCompletionParser.parse("<abstain>Schema lacks table</abstain>")
    assert parsed_abs.is_valid_format is False
    assert parsed_abs.is_abstention is True
    assert parsed_abs.error_code == "MODEL_ABSTENTION"


def test_schema_and_execution_rejections_and_column_checks(temp_env):
    reg, sb, _ = temp_env
    # Unknown column must be blocked at AST stage and receive R_ast=0.00, R_exec_valid=-0.20
    assert compute_schema_reward("<sql>SELECT name FROM department</sql>", FIXTURE_TABLES, FIXTURE_COLUMNS) == 0.20
    assert compute_schema_reward("<sql>SELECT secret FROM department</sql>", FIXTURE_TABLES, FIXTURE_COLUMNS) == 0.00
    assert compute_schema_reward("<sql>SELECT name FROM ghost_tbl</sql>", FIXTURE_TABLES, FIXTURE_COLUMNS) == 0.00
    assert compute_schema_reward("<sql>SELECT FROM department</sql>", FIXTURE_TABLES, FIXTURE_COLUMNS) == 0.00
    assert compute_schema_reward("<sql>SELECT ??? FROM department</sql>", FIXTURE_TABLES, FIXTURE_COLUMNS) == 0.00

    for bad_comp in (
        "no tags at all",
        "<sql>DROP TABLE department;</sql>",
        "<sql>SELECT 1; SELECT 2;</sql>",
        "<sql>SELECT nonexistent FROM department</sql>",
        "<sql>SELECT * FROM nonexistent_tbl</sql>",
    ):
        assert (
            compute_execution_reward(
                bad_comp,
                db_id="company",
                valid_tables=FIXTURE_TABLES,
                valid_columns=FIXTURE_COLUMNS,
                registry=reg,
                sandbox=sb,
            )
            == -0.20
        )


def test_composite_flow_extrema_and_gating(temp_env):
    reg, sb, _ = temp_env
    gold_depts = [(1, "Engineering"), (2, "Sales"), (3, "HR")]

    # 1. Malformed format -> [-0.50, 0, -0.20, 0, 0] = -0.70 (minimum bound)
    worst = compute_composite_reward(
        completion="gibberish without tags",
        db_id="company",
        gold_rows=gold_depts,
        valid_tables=FIXTURE_TABLES,
        valid_columns=FIXTURE_COLUMNS,
        registry=reg,
        sandbox=sb,
    )
    assert (worst.r_fmt, worst.r_ast, worst.r_exec_valid, worst.r_result, worst.r_eff, worst.r_total) == (
        -0.50,
        0.00,
        -0.20,
        0.00,
        0.00,
        -0.70,
    )

    # 2. Well-formed but AST rejected -> [+0.20, 0, -0.20, 0, 0] = 0.00
    ast_bad = compute_composite_reward(
        completion="<sql>DELETE FROM employee</sql>",
        db_id="company",
        gold_rows=gold_depts,
        valid_tables=FIXTURE_TABLES,
        valid_columns=FIXTURE_COLUMNS,
        registry=reg,
        sandbox=sb,
    )
    assert (ast_bad.r_fmt, ast_bad.r_ast, ast_bad.r_exec_valid, ast_bad.r_total) == (0.20, 0.00, -0.20, 0.00)

    # 3. Valid AST but runtime error / unknown function runtime failure or nonexistent DB -> +0.20
    missing_db = compute_composite_reward(
        completion="<sql>SELECT dept_id, name FROM department</sql>",
        db_id="missing_db",
        gold_rows=gold_depts,
        valid_tables=FIXTURE_TABLES,
        valid_columns=FIXTURE_COLUMNS,
        registry=reg,
        sandbox=sb,
    )
    assert (missing_db.r_fmt, missing_db.r_ast, missing_db.r_exec_valid, missing_db.r_total) == (
        0.20,
        0.20,
        -0.20,
        0.20,
    )

    # 4. Valid execution, wrong result -> efficiency MUST stay gated at 0.00 even when fast vs gold
    wrong_res = compute_composite_reward(
        completion="<sql>SELECT dept_id, name FROM department WHERE dept_id = 1</sql>",
        db_id="company",
        gold_rows=gold_depts,
        valid_tables=FIXTURE_TABLES,
        valid_columns=FIXTURE_COLUMNS,
        candidate_latency_ms=1.0,
        gold_latency_ms=10.0,
        registry=reg,
        sandbox=sb,
    )
    assert (
        wrong_res.r_fmt,
        wrong_res.r_ast,
        wrong_res.r_exec_valid,
        wrong_res.r_result,
        wrong_res.r_eff,
        wrong_res.r_total,
    ) == (0.20, 0.20, 0.30, 0.00, 0.00, 0.70)

    # 5. Exact match with clean SQL + fast paired latency -> [+0.20, +0.20, +0.30, +1.00, +0.30] = +2.00 (max bound)
    best = compute_composite_reward(
        completion="<sql>SELECT dept_id, name FROM department</sql>",
        db_id="company",
        gold_rows=gold_depts,
        valid_tables=FIXTURE_TABLES,
        valid_columns=FIXTURE_COLUMNS,
        candidate_latency_ms=5.0,
        gold_latency_ms=10.0,
        registry=reg,
        sandbox=sb,
    )
    assert (best.r_fmt, best.r_ast, best.r_exec_valid, best.r_result, best.r_eff, best.r_total) == (
        0.20,
        0.20,
        0.30,
        1.00,
        0.30,
        2.00,
    )

    # 6. Omitting gold_latency_ms does NOT grant free +0.15 latency bonus (only +0.15 structural = 1.85 total)
    no_paired_lat = compute_composite_reward(
        completion="<sql>SELECT dept_id, name FROM department</sql>",
        db_id="company",
        gold_rows=gold_depts,
        valid_tables=FIXTURE_TABLES,
        valid_columns=FIXTURE_COLUMNS,
        registry=reg,
        sandbox=sb,
    )
    assert no_paired_lat.r_eff == 0.15
    assert no_paired_lat.r_total == 1.85

    # Unmeasured latency (0.0) in compute_efficiency_reward directly must not award latency bonus
    assert compute_efficiency_reward(is_correct=True, candidate_latency_ms=0.0, gold_latency_ms=10.0) == 0.00


def test_timeout_truncation_order_duplicates_and_empty_gold_policy(temp_env):
    reg, _, _ = temp_env

    # Truncated result never receives +0.30 exec or +1.00 accuracy credit
    trunc_sb = SQLiteSandbox(row_limit=1)
    trunc_bd = compute_composite_reward(
        completion="<sql>SELECT name FROM department</sql>",
        db_id="company",
        gold_rows=[("Engineering",), ("Sales",), ("HR",)],
        valid_tables=FIXTURE_TABLES,
        valid_columns=FIXTURE_COLUMNS,
        registry=reg,
        sandbox=trunc_sb,
    )
    assert trunc_bd.r_exec_valid == -0.20
    assert trunc_bd.r_result == 0.00
    assert trunc_bd.r_total == 0.20

    # Timeout receives r_exec_valid=-0.20 and r_result=0.00
    aliases = [f"e{i}" for i in range(11)]
    cross_clause = " CROSS JOIN ".join(f"employee {a}" for a in aliases)
    sum_expr = " + ".join(f"{a}.emp_id" for a in aliases)
    slow_sql = f"<sql>SELECT COUNT(*) FROM {cross_clause} WHERE {sum_expr} > -1</sql>"
    timeout_sb = SQLiteSandbox(timeout_seconds=0.002, progress_opcodes=100)
    to_bd = compute_composite_reward(
        completion=slow_sql,
        db_id="company",
        gold_rows=[(1,)],
        valid_tables=FIXTURE_TABLES,
        valid_columns=FIXTURE_COLUMNS,
        registry=reg,
        sandbox=timeout_sb,
    )
    assert to_bd.r_exec_valid == -0.20
    assert to_bd.r_result == 0.00

    # Ordered vs unordered & duplicate count checks
    sb = SQLiteSandbox()
    reversed_sql = "<sql>SELECT name FROM department ORDER BY dept_id DESC</sql>"
    gold_asc = [("Engineering",), ("Sales",), ("HR",)]
    assert (
        compute_accuracy_reward(
            reversed_sql, "company", gold_rows=gold_asc, ordered=True,
            valid_tables=FIXTURE_TABLES, valid_columns=FIXTURE_COLUMNS, registry=reg, sandbox=sb,
        )
        == 0.00
    )
    assert (
        compute_accuracy_reward(
            reversed_sql, "company", gold_rows=gold_asc, ordered=False,
            valid_tables=FIXTURE_TABLES, valid_columns=FIXTURE_COLUMNS, registry=reg, sandbox=sb,
        )
        == 1.00
    )

    # Empty-gold anti-hacking: unverified WHERE 1=0 shortcut receives 0.00
    trivial_empty = "<sql>SELECT name FROM department WHERE 1 = 0</sql>"
    assert (
        compute_accuracy_reward(
            trivial_empty, "company", gold_rows=[], gold_is_empty=True, empty_gold_verified=False,
            valid_tables=FIXTURE_TABLES, valid_columns=FIXTURE_COLUMNS, registry=reg, sandbox=sb,
        )
        == 0.00
    )
    # Verified empty-gold succeeds only if result is also empty
    assert (
        compute_accuracy_reward(
            "<sql>SELECT name FROM department WHERE budget > 9999999</sql>",
            "company", gold_rows=[], gold_is_empty=True, empty_gold_verified=True,
            valid_tables=FIXTURE_TABLES, valid_columns=FIXTURE_COLUMNS, registry=reg, sandbox=sb,
        )
        == 1.00
    )
    assert (
        compute_accuracy_reward(
            "<sql>SELECT name FROM department</sql>",
            "company", gold_rows=[], gold_is_empty=True, empty_gold_verified=True,
            valid_tables=FIXTURE_TABLES, valid_columns=FIXTURE_COLUMNS, registry=reg, sandbox=sb,
        )
        == 0.00
    )


def test_trl_conversational_adapter_and_zero_variance_batch():
    # Exercises all 5 TRL *_reward_func callables with mixed string, dict, and list[dict] completions
    convs = [
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<sql>SELECT name FROM department</sql>"}],
        {"role": "assistant", "content": "<sql>SELECT nonexistent_col FROM department</sql>"},
        "<sql>SELECT name FROM employee</sql>",
    ]
    tbls = [["department", "employee"]] * 3
    cols_json = [json.dumps({"department": ["dept_id", "name", "budget"], "employee": ["emp_id", "name", "dept_id", "salary"]})] * 3
    dbs = ["test_company"] * 3
    gold_json = [json.dumps([["Engineering"], ["Sales"], ["HR"]])] * 3

    assert format_reward_func(convs) == [0.20, 0.20, 0.20]
    assert schema_reward_func(convs, valid_tables=tbls, valid_columns_json=cols_json) == [0.20, 0.00, 0.20]
    assert execution_reward_func(convs, db_id=dbs, valid_tables=tbls, valid_columns_json=cols_json) == [0.30, -0.20, 0.30]
    assert accuracy_reward_func(
        convs, db_id=dbs, gold_result_json=gold_json, gold_is_empty=[False] * 3,
        valid_tables=tbls, valid_columns_json=cols_json,
    ) == [1.00, 0.00, 0.00]
    assert efficiency_reward_func(convs) == [0.00, 0.00, 0.00]
    assert efficiency_reward_func(
        convs, is_correct=[True, False, False], candidate_latency_ms=[2.0, 2.0, 2.0], gold_latency_ms=[5.0, 5.0, 5.0]
    ) == [0.30, 0.00, 0.00]

    # Batch zero-variance check for G=8 uniform group
    same = compute_composite_reward("bad sql", db_id="test_company")
    summary = compute_composite_reward_batch([same] * 8, group_size=8)
    assert summary.total_groups == 1
    assert summary.zero_variance_groups == 1
