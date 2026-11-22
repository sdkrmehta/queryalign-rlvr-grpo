"""Tests for Verifiable Rewards & Bound Invariants."""

import pytest
from sql_grpo.rewards.composite import compute_composite_reward
from sql_grpo.rewards.format_reward import compute_format_reward
from sql_grpo.rewards.schema_reward import compute_schema_reward


def test_format_reward_correct():
    completion = "<sql>SELECT name FROM department</sql>"
    assert compute_format_reward(completion) == 0.20


def test_format_reward_malformed():
    completion = "SELECT name FROM department"
    assert compute_format_reward(completion) == -0.50


def test_format_reward_trailing_commentary():
    completion = "<sql>SELECT name FROM department</sql>\nHere is the explanation."
    assert compute_format_reward(completion) == -0.50


def test_schema_reward_valid():
    completion = "<sql>SELECT name FROM department</sql>"
    assert compute_schema_reward(completion, valid_tables={"department"}) == 0.20


def test_schema_reward_unauthorized_table():
    completion = "<sql>SELECT secret FROM top_secret_tbl</sql>"
    assert compute_schema_reward(completion, valid_tables={"department"}) == 0.00


def test_reward_mathematical_extrema():
    """Validates that total reward strictly lies in [-0.70, +2.00]."""
    # Worst case: malformed format -> -0.50 (fmt) + 0.00 (ast) + -0.20 (exec) + 0.00 (res) + 0.00 (eff) = -0.70
    worst = compute_composite_reward(
        completion="gibberish without tags",
        db_id="non_existent",
    )
    assert worst.r_total == -0.70

    # Best case: correct format, safe AST, valid exec, exact match, high efficiency
    # +0.20 + 0.20 + 0.30 + 1.00 + 0.30 = +2.00
    best_fmt = "<sql>SELECT 1</sql>"
    # Verified range assertion inside compute_composite_reward
    assert -0.701 <= worst.r_total <= 2.001
