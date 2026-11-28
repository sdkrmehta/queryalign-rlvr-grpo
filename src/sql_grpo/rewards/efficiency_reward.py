"""Efficiency Reward (R_eff): Deterministic efficiency bonus strictly gated on exact correctness."""

from typing import Any, List, Optional
import sqlglot
from sqlglot import exp

from sql_grpo.rewards.common import extract_completion_text
from sql_grpo.safety.sql_parser import SQLCompletionParser


def _has_inefficient_ast_patterns(ast: exp.Expression) -> bool:
    """Detects unconstrained Cartesian cross-joins in a parsed SELECT AST."""
    for join_node in ast.find_all(exp.Join):
        kind = (join_node.args.get("kind") or "").upper()
        has_condition = join_node.args.get("on") is not None or join_node.args.get("using") is not None
        if kind == "CROSS" or (not has_condition and kind not in ("NATURAL",)):
            return True
    return False


def compute_efficiency_reward(
    is_correct: bool,
    candidate_latency_ms: Optional[float] = None,
    gold_latency_ms: Optional[float] = None,
    sql: Optional[str] = None,
    ast: Optional[exp.Expression] = None,
) -> float:
    """Computes efficiency reward in [0.00, +0.30], strictly gated on correctness.

    Components (max +0.30):
        1. Paired measured latency ratio vs gold baseline (up to +0.15):
           requires both `candidate_latency_ms > 0` and `gold_latency_ms > 0`.
           Never grants free latency bonus when timings are omitted (0.0 or None).
        2. Deterministic structural query-efficiency check (up to +0.15):
           awards +0.15 when the parsed query contains no unconstrained Cartesian cross joins.
           No reasoning-word-count bonus is ever computed.
    """
    if not is_correct:
        return 0.00

    bonus = 0.0

    # 1. Paired hardware-matched latency ratio (up to +0.15)
    if (
        candidate_latency_ms is not None
        and candidate_latency_ms > 0.0
        and gold_latency_ms is not None
        and gold_latency_ms > 0.0
    ):
        if candidate_latency_ms <= 1.2 * gold_latency_ms:
            bonus += 0.15
        elif candidate_latency_ms <= 2.0 * gold_latency_ms:
            bonus += 0.075

    # 2. Deterministic query-cost structural criterion (up to +0.15)
    parsed_ast = ast
    if parsed_ast is None and sql:
        try:
            stmts = sqlglot.parse(sql, read="sqlite")
            if len(stmts) == 1 and stmts[0] is not None:
                parsed_ast = stmts[0]
        except Exception:
            parsed_ast = None

    if parsed_ast is not None and not _has_inefficient_ast_patterns(parsed_ast):
        bonus += 0.15

    return round(min(bonus, 0.30), 4)


def efficiency_reward_func(
    completions: List[Any],
    is_correct: Optional[List[bool]] = None,
    candidate_latency_ms: Optional[List[float]] = None,
    gold_latency_ms: Optional[List[float]] = None,
    **kwargs,
) -> List[float]:
    """TRL-compatible reward callable for correctness-gated query efficiency."""
    if is_correct is None:
        return [0.00 for _ in completions]

    rewards: List[float] = []
    for idx, comp in enumerate(completions):
        correct_flag = bool(is_correct[idx]) if idx < len(is_correct) else False
        if not correct_flag:
            rewards.append(0.00)
            continue

        text = extract_completion_text(comp)
        parsed = SQLCompletionParser.parse(text)
        cand_lat = (
            candidate_latency_ms[idx]
            if (candidate_latency_ms and idx < len(candidate_latency_ms))
            else None
        )
        gold_lat = (
            gold_latency_ms[idx]
            if (gold_latency_ms and idx < len(gold_latency_ms))
            else None
        )
        rewards.append(
            compute_efficiency_reward(
                is_correct=True,
                candidate_latency_ms=cand_lat,
                gold_latency_ms=gold_lat,
                sql=parsed.sql if parsed.is_valid_format else None,
            )
        )
    return rewards
