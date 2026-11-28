"""Execution Validity Reward (R_exec_valid): Scores safe sandbox query completion."""

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.rewards.common import (
    NormalizedCandidate,
    normalize_and_validate_candidate,
    resolve_columns_arg,
    resolve_tables_arg,
)

_SHARED_REGISTRY = DatabaseRegistry()
_SHARED_SANDBOX = SQLiteSandbox(row_limit=5000, timeout_seconds=2.0)


def compute_execution_reward(
    completion: Any,
    db_id: str,
    valid_tables: Optional[Set[str]] = None,
    valid_columns: Optional[Union[Set[str], Mapping[str, Set[str]]]] = None,
    registry: Optional[DatabaseRegistry] = None,
    sandbox: Optional[SQLiteSandbox] = None,
) -> float:
    """Computes execution validity reward for a completion.

    Returns:
        +0.30 if safe execution completes without errors, timeouts, or row/byte truncation.
        -0.20 on format failure, AST rejection, runtime error, timeout, truncation, or missing DB.
    """
    cand = normalize_and_validate_candidate(
        completion=completion,
        valid_tables=valid_tables,
        valid_columns=valid_columns,
    )
    if not cand.is_format_valid or not cand.is_ast_safe or not cand.sql:
        return -0.20  # Never execute malformed or AST-rejected SQL

    reg = registry or _SHARED_REGISTRY
    sb = sandbox or _SHARED_SANDBOX

    try:
        db_path = reg.get_path(db_id)
    except Exception:
        return -0.20

    res = sb.execute(
        db_path=db_path,
        database_id=db_id,
        sql=cand.sql,
        allowed_tables=valid_tables,
        allowed_columns=valid_columns,
    )

    if res.success and res.complete and not res.truncated:
        return 0.30
    return -0.20


def execution_reward_func(
    completions: List[Any],
    db_id: Optional[List[str]] = None,
    valid_tables: Optional[List[Any]] = None,
    valid_columns: Optional[List[Any]] = None,
    valid_columns_json: Optional[List[Any]] = None,
    **kwargs,
) -> List[float]:
    """TRL-compatible reward callable for query execution validity."""
    cache: Dict[Tuple[str, Any, Any], NormalizedCandidate] = {}
    rewards: List[float] = []
    for idx, comp in enumerate(completions):
        target_db = db_id[idx] if (db_id and idx < len(db_id)) else ""
        tables = resolve_tables_arg(valid_tables, idx)
        cols = resolve_columns_arg(valid_columns, valid_columns_json, idx)
        cand = normalize_and_validate_candidate(
            completion=comp,
            valid_tables=tables,
            valid_columns=cols,
            cache=cache,
        )
        if not cand.is_format_valid or not cand.is_ast_safe or not cand.sql or not target_db:
            rewards.append(-0.20)
            continue
        rewards.append(
            compute_execution_reward(
                completion=cand.raw_text,
                db_id=target_db,
                valid_tables=tables,
                valid_columns=cols,
            )
        )
    return rewards
