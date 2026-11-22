"""Accuracy / Result Equivalence Reward (R_result): Compares candidate rows against gold rows."""

import json
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple, Union

from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.result_normalizer import compare_results
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.safety.ast_policy import ASTPolicyChecker
from sql_grpo.safety.sql_parser import SQLCompletionParser

_SHARED_REGISTRY = DatabaseRegistry()
_SHARED_SANDBOX = SQLiteSandbox(row_limit=5000, timeout_seconds=2.0)


def compute_accuracy_reward(
    completion: str,
    db_id: str,
    gold_rows: Optional[List[Tuple[Any, ...]]] = None,
    gold_is_empty: bool = False,
    ordered: bool = False,
    valid_tables: Optional[Set[str]] = None,
    registry: Optional[DatabaseRegistry] = None,
    sandbox: Optional[SQLiteSandbox] = None,
) -> float:
    """Computes result equivalence reward.

    Returns:
        +1.00 if candidate result matches gold result under order/multiset policy.
        0.00 if incorrect, truncated, or empty-result shortcut detected.
    """
    parsed = SQLCompletionParser.parse(completion)
    if not parsed.is_valid_format or not parsed.sql:
        return 0.00

    checker = ASTPolicyChecker(allowed_tables=valid_tables)
    ast_res = checker.validate(parsed.sql)
    if not ast_res.is_safe:
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
        sql=parsed.sql,
        allowed_tables=valid_tables,
    )

    if not cand_res.success or not cand_res.complete or cand_res.truncated:
        return 0.00

    cand_rows = cand_res.rows or []

    # Anti-hacking: If gold output is empty, both returning empty is not rewarded
    # to prevent rewarding trivial contradictory SQL (e.g. WHERE 1=0).
    if gold_is_empty:
        if len(cand_rows) == 0:
            # Special policy: Cap/zero denotation reward for empty gold
            return 0.00
        return 0.00

    is_match = compare_results(
        candidate_rows=cand_rows,
        gold_rows=gold_rows,
        ordered=ordered,
        candidate_truncated=cand_res.truncated,
    )

    return 1.00 if is_match else 0.00


def accuracy_reward_func(
    completions: List[Union[str, dict]],
    db_id: Optional[List[str]] = None,
    gold_result_json: Optional[List[str]] = None,
    gold_is_empty: Optional[List[bool]] = None,
    valid_tables: Optional[List[Any]] = None,
    **kwargs,
) -> List[float]:
    """TRL-compatible reward callable for result equivalence."""
    rewards: List[float] = []
    for idx, comp in enumerate(completions):
        text = comp["content"] if isinstance(comp, dict) else str(comp)
        target_db = db_id[idx] if (db_id and idx < len(db_id)) else ""
        is_empty = bool(gold_is_empty[idx]) if (gold_is_empty and idx < len(gold_is_empty)) else False
        tables = set(valid_tables[idx]) if (valid_tables and idx < len(valid_tables) and valid_tables[idx]) else None

        gold_rows: Optional[List[Tuple[Any, ...]]] = None
        if gold_result_json and idx < len(gold_result_json) and gold_result_json[idx]:
            try:
                raw_data = json.loads(gold_result_json[idx])
                gold_rows = [tuple(r) for r in raw_data]
            except Exception:
                gold_rows = None

        rewards.append(
            compute_accuracy_reward(
                completion=text,
                db_id=target_db,
                gold_rows=gold_rows,
                gold_is_empty=is_empty,
                valid_tables=tables,
            )
        )
    return rewards
