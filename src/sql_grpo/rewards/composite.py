"""Composite Reward: Aggregates verifiable reward components with single-pass execution and strict bounds."""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.rewards.accuracy_reward import score_accuracy_from_rows
from sql_grpo.rewards.common import normalize_and_validate_candidate
from sql_grpo.rewards.efficiency_reward import compute_efficiency_reward

_SHARED_REGISTRY = DatabaseRegistry()
_SHARED_SANDBOX = SQLiteSandbox(row_limit=5000, timeout_seconds=2.0)


@dataclass
class RewardBreakdown:
    r_fmt: float
    r_ast: float
    r_exec_valid: float
    r_result: float
    r_eff: float
    r_total: float


@dataclass
class BatchRewardSummary:
    breakdowns: List[RewardBreakdown]
    r_fmt: List[float]
    r_ast: List[float]
    r_exec_valid: List[float]
    r_result: List[float]
    r_eff: List[float]
    r_total: List[float]
    zero_variance_groups: int
    total_groups: int


def compute_composite_reward(
    completion: Any,
    db_id: str,
    gold_rows: Optional[List[Tuple[Any, ...]]] = None,
    gold_is_empty: bool = False,
    empty_gold_verified: bool = False,
    ordered: bool = False,
    float_tolerance: float = 0.0,
    valid_tables: Optional[Set[str]] = None,
    valid_columns: Optional[Union[Set[str], Mapping[str, Set[str]]]] = None,
    candidate_latency_ms: Optional[float] = None,
    gold_latency_ms: Optional[float] = None,
    registry: Optional[DatabaseRegistry] = None,
    sandbox: Optional[SQLiteSandbox] = None,
) -> RewardBreakdown:
    """Computes all 5 reward components using a single sandbox execution and enforces [-0.70, +2.00]."""
    cand = normalize_and_validate_candidate(
        completion=completion,
        valid_tables=valid_tables,
        valid_columns=valid_columns,
    )

    r_fmt = 0.20 if cand.is_format_valid else -0.50
    r_ast = 0.20 if (cand.is_format_valid and cand.is_ast_safe) else 0.00

    # Never execute malformed or AST-rejected SQL; R_exec_valid is always -0.20 on rejection.
    if not cand.is_format_valid or not cand.is_ast_safe or not cand.sql:
        r_exec_valid = -0.20
        r_result = 0.00
        r_eff = 0.00
    else:
        reg = registry or _SHARED_REGISTRY
        sb = sandbox or _SHARED_SANDBOX
        try:
            db_path = reg.get_path(db_id)
        except Exception:
            db_path = None

        if db_path is None:
            r_exec_valid = -0.20
            r_result = 0.00
            r_eff = 0.00
        else:
            cand_res = sb.execute(
                db_path=db_path,
                database_id=db_id,
                sql=cand.sql,
                allowed_tables=valid_tables,
                allowed_columns=valid_columns,
            )
            exec_ok = cand_res.success and cand_res.complete and not cand_res.truncated
            r_exec_valid = 0.30 if exec_ok else -0.20

            r_result = score_accuracy_from_rows(
                cand_rows=cand_res.rows,
                cand_success=cand_res.success,
                cand_complete=cand_res.complete,
                cand_truncated=cand_res.truncated,
                gold_rows=gold_rows,
                gold_is_empty=gold_is_empty,
                empty_gold_verified=empty_gold_verified,
                ordered=ordered,
                float_tolerance=float_tolerance,
            )

            is_correct = r_result == 1.00
            effective_cand_latency = (
                candidate_latency_ms
                if (candidate_latency_ms is not None and candidate_latency_ms > 0.0)
                else cand_res.latency_ms
            )
            r_eff = compute_efficiency_reward(
                is_correct=is_correct,
                candidate_latency_ms=effective_cand_latency,
                gold_latency_ms=gold_latency_ms,
                sql=cand.sql,
                ast=cand.ast_result.ast if cand.ast_result else None,
            )

    gated_eff = r_eff if r_result == 1.00 else 0.00
    r_total = r_fmt + r_ast + r_exec_valid + r_result + gated_eff

    assert math.isfinite(r_total), f"Reward {r_total} must be finite"
    assert -0.701 <= r_total <= 2.001, f"Reward {r_total} out of bounds [-0.70, +2.00]"

    return RewardBreakdown(
        r_fmt=round(r_fmt, 3),
        r_ast=round(r_ast, 3),
        r_exec_valid=round(r_exec_valid, 3),
        r_result=round(r_result, 3),
        r_eff=round(gated_eff, 3),
        r_total=round(r_total, 3),
    )


def compute_composite_reward_batch(
    breakdowns: List[RewardBreakdown],
    group_size: int = 8,
) -> BatchRewardSummary:
    """Aggregates per-component reward arrays and tracks zero-variance GRPO groups."""
    totals = [b.r_total for b in breakdowns]
    zero_var = 0
    total_groups = 0
    if group_size > 0 and len(totals) >= group_size:
        for start in range(0, len(totals) - (len(totals) % group_size), group_size):
            chunk = totals[start : start + group_size]
            total_groups += 1
            if max(chunk) - min(chunk) <= 1e-12:
                zero_var += 1

    return BatchRewardSummary(
        breakdowns=breakdowns,
        r_fmt=[b.r_fmt for b in breakdowns],
        r_ast=[b.r_ast for b in breakdowns],
        r_exec_valid=[b.r_exec_valid for b in breakdowns],
        r_result=[b.r_result for b in breakdowns],
        r_eff=[b.r_eff for b in breakdowns],
        r_total=totals,
        zero_variance_groups=zero_var,
        total_groups=total_groups,
    )
