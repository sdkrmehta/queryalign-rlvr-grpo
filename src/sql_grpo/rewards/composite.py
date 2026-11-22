"""Composite Reward: Aggregates verifiable reward components with strict range assertions."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from sql_grpo.rewards.accuracy_reward import compute_accuracy_reward
from sql_grpo.rewards.efficiency_reward import compute_efficiency_reward
from sql_grpo.rewards.execution_reward import compute_execution_reward
from sql_grpo.rewards.format_reward import compute_format_reward
from sql_grpo.rewards.schema_reward import compute_schema_reward


@dataclass
class RewardBreakdown:
    r_fmt: float
    r_ast: float
    r_exec_valid: float
    r_result: float
    r_eff: float
    r_total: float


def compute_composite_reward(
    completion: str,
    db_id: str,
    gold_rows: Optional[List[Tuple[Any, ...]]] = None,
    gold_is_empty: bool = False,
    ordered: bool = False,
    valid_tables: Optional[Set[str]] = None,
    candidate_latency_ms: float = 0.0,
    gold_latency_ms: Optional[float] = None,
) -> RewardBreakdown:
    """Computes all 5 reward components and total reward with strict gating.

    Theoretical range: [-0.70, +2.00]
    """
    # 1. Format reward: +0.20 or -0.50
    r_fmt = compute_format_reward(completion)

    # 2. AST/Schema reward: +0.20 or 0.00
    r_ast = compute_schema_reward(completion, valid_tables=valid_tables)

    # 3. Execution validity reward: +0.30 or -0.20
    r_exec_valid = compute_execution_reward(completion, db_id=db_id, valid_tables=valid_tables)

    # 4. Result accuracy reward: +1.00 or 0.00
    r_result = compute_accuracy_reward(
        completion=completion,
        db_id=db_id,
        gold_rows=gold_rows,
        gold_is_empty=gold_is_empty,
        ordered=ordered,
        valid_tables=valid_tables,
    )

    # 5. Efficiency reward: 0.00 to +0.30 (gated on correctness)
    is_correct = (r_result == 1.00)
    r_eff = compute_efficiency_reward(
        is_correct=is_correct,
        candidate_latency_ms=candidate_latency_ms,
        gold_latency_ms=gold_latency_ms,
    )

    r_total = r_fmt + r_ast + r_exec_valid + r_result + (r_eff if is_correct else 0.0)

    # Assert mathematical bounds [-0.70, +2.00]
    assert -0.701 <= r_total <= 2.001, f"Reward {r_total} out of bounds [-0.70, +2.00]"

    return RewardBreakdown(
        r_fmt=round(r_fmt, 3),
        r_ast=round(r_ast, 3),
        r_exec_valid=round(r_exec_valid, 3),
        r_result=round(r_result, 3),
        r_eff=round(r_eff, 3),
        r_total=round(r_total, 3),
    )
