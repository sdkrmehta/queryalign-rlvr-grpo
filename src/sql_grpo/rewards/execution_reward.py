"""Execution Validity Reward (R_exec_valid): Scores successful query execution."""

from pathlib import Path
from typing import Any, List, Optional, Set, Union

from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.safety.ast_policy import ASTPolicyChecker
from sql_grpo.safety.sql_parser import SQLCompletionParser

_SHARED_REGISTRY = DatabaseRegistry()
_SHARED_SANDBOX = SQLiteSandbox(row_limit=5000, timeout_seconds=2.0)


def compute_execution_reward(
    completion: str,
    db_id: str,
    valid_tables: Optional[Set[str]] = None,
    registry: Optional[DatabaseRegistry] = None,
    sandbox: Optional[SQLiteSandbox] = None,
) -> float:
    """Computes execution validity reward for a completion.

    Returns:
        +0.30 if safe execution completes without errors, timeouts or truncation.
        -0.20 if AST rejected, runtime error, timeout, truncated, or DB missing.
    """
    parsed = SQLCompletionParser.parse(completion)
    if not parsed.is_valid_format or not parsed.sql:
        return -0.20

    checker = ASTPolicyChecker(allowed_tables=valid_tables)
    ast_res = checker.validate(parsed.sql)
    if not ast_res.is_safe:
        return -0.20  # Never execute AST-rejected SQL

    reg = registry or _SHARED_REGISTRY
    sb = sandbox or _SHARED_SANDBOX

    try:
        db_path = reg.get_path(db_id)
    except Exception:
        return -0.20

    res = sb.execute(
        db_path=db_path,
        database_id=db_id,
        sql=parsed.sql,
        allowed_tables=valid_tables,
    )

    if res.success and res.complete and not res.truncated:
        return 0.30
    return -0.20


def execution_reward_func(
    completions: List[Union[str, dict]],
    db_id: Optional[List[str]] = None,
    valid_tables: Optional[List[Any]] = None,
    **kwargs,
) -> List[float]:
    """TRL-compatible reward callable for query execution validity."""
    rewards: List[float] = []
    for idx, comp in enumerate(completions):
        text = comp["content"] if isinstance(comp, dict) else str(comp)
        target_db = db_id[idx] if (db_id and idx < len(db_id)) else ""
        tables = set(valid_tables[idx]) if (valid_tables and idx < len(valid_tables) and valid_tables[idx]) else None
        rewards.append(compute_execution_reward(text, target_db, valid_tables=tables))
    return rewards
