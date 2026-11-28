"""Accuracy / Result Equivalence Reward (R_result): Compares candidate rows against gold rows."""

import json
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.result_normalizer import compare_results
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.rewards.common import (
    NormalizedCandidate,
    normalize_and_validate_candidate,
    resolve_columns_arg,
    resolve_tables_arg,
)

_SHARED_REGISTRY = DatabaseRegistry()
_SHARED_SANDBOX = SQLiteSandbox(row_limit=5000, timeout_seconds=2.0)


def score_accuracy_from_rows(
    cand_rows: Optional[List[Tuple[Any, ...]]],
    cand_success: bool,
    cand_complete: bool,
    cand_truncated: bool,
    gold_rows: Optional[List[Tuple[Any, ...]]] = None,
    gold_is_empty: bool = False,
    empty_gold_verified: bool = False,
    ordered: bool = False,
    float_tolerance: float = 0.0,
) -> float:
    """Computes R_result (+1.00 or 0.00) from an already-executed candidate result."""
    if not cand_success or not cand_complete or cand_truncated or cand_rows is None:
        return 0.00

    if gold_rows is None:
        return 0.00

    effective_empty = gold_is_empty or (len(gold_rows) == 0)
    if effective_empty:
        # Empty-result anti-hacking policy (Blueprint V2.0 Sections 9.4 & 10.2):
        # Never award +1.00 solely because both outputs are empty; require independent
        # non-triviality/perturbation verification via empty_gold_verified=True.
        if empty_gold_verified and len(cand_rows) == 0:
            return 1.00
        return 0.00

    is_match = compare_results(
        candidate_rows=cand_rows,
        gold_rows=gold_rows,
        ordered=ordered,
        candidate_truncated=cand_truncated,
        float_tolerance=float_tolerance,
    )
    return 1.00 if is_match else 0.00


def compute_accuracy_reward(
    completion: Any,
    db_id: str,
    gold_rows: Optional[List[Tuple[Any, ...]]] = None,
    gold_is_empty: bool = False,
    empty_gold_verified: bool = False,
    ordered: bool = False,
    float_tolerance: float = 0.0,
    valid_tables: Optional[Set[str]] = None,
    valid_columns: Optional[Union[Set[str], Mapping[str, Set[str]]]] = None,
    registry: Optional[DatabaseRegistry] = None,
    sandbox: Optional[SQLiteSandbox] = None,
) -> float:
    """Computes result equivalence reward.

    Returns:
        +1.00 if candidate completes safely and matches gold result under ordered/multiset policy.
        0.00 if unsafe, errored, truncated, incorrect, or unverified empty-gold match.
    """
    cand = normalize_and_validate_candidate(
        completion=completion,
        valid_tables=valid_tables,
        valid_columns=valid_columns,
    )
    if not cand.is_format_valid or not cand.is_ast_safe or not cand.sql:
        return 0.00

    reg = registry or _SHARED_REGISTRY
    sb = sandbox or _SHARED_SANDBOX

    try:
        db_path = reg.get_path(db_id)
    except Exception:
        return 0.00

    cand_res = sb.execute(
        db_path=db_path,
        database_id=db_id,
        sql=cand.sql,
        allowed_tables=valid_tables,
        allowed_columns=valid_columns,
    )

    return score_accuracy_from_rows(
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


def accuracy_reward_func(
    completions: List[Any],
    db_id: Optional[List[str]] = None,
    gold_result_json: Optional[List[str]] = None,
    gold_is_empty: Optional[List[bool]] = None,
    empty_gold_verified: Optional[List[bool]] = None,
    ordered: Optional[List[bool]] = None,
    valid_tables: Optional[List[Any]] = None,
    valid_columns: Optional[List[Any]] = None,
    valid_columns_json: Optional[List[Any]] = None,
    **kwargs,
) -> List[float]:
    """TRL-compatible reward callable for result equivalence."""
    cache: Dict[Tuple[str, Any, Any], NormalizedCandidate] = {}
    rewards: List[float] = []
    for idx, comp in enumerate(completions):
        target_db = db_id[idx] if (db_id and idx < len(db_id)) else ""
        is_empty = bool(gold_is_empty[idx]) if (gold_is_empty and idx < len(gold_is_empty)) else False
        is_verified = (
            bool(empty_gold_verified[idx])
            if (empty_gold_verified and idx < len(empty_gold_verified))
            else False
        )
        is_ordered = bool(ordered[idx]) if (ordered and idx < len(ordered)) else False
        tables = resolve_tables_arg(valid_tables, idx)
        cols = resolve_columns_arg(valid_columns, valid_columns_json, idx)

        cand = normalize_and_validate_candidate(
            completion=comp,
            valid_tables=tables,
            valid_columns=cols,
            cache=cache,
        )
        if not cand.is_format_valid or not cand.is_ast_safe or not cand.sql or not target_db:
            rewards.append(0.00)
            continue

        gold_rows: Optional[List[Tuple[Any, ...]]] = None
        if gold_result_json and idx < len(gold_result_json) and gold_result_json[idx] is not None:
            try:
                raw_data = json.loads(gold_result_json[idx])
                gold_rows = [tuple(r) if isinstance(r, list) else (r,) for r in raw_data]
            except Exception:
                gold_rows = None

        rewards.append(
            compute_accuracy_reward(
                completion=cand.raw_text,
                db_id=target_db,
                gold_rows=gold_rows,
                gold_is_empty=is_empty,
                empty_gold_verified=is_verified,
                ordered=is_ordered,
                valid_tables=tables,
                valid_columns=cols,
            )
        )
    return rewards
