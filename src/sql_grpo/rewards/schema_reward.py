"""Schema & AST Reward (R_ast): Scores syntactic validity and schema compliance."""

from typing import Any, List, Optional, Set, Union
from sql_grpo.safety.ast_policy import ASTPolicyChecker
from sql_grpo.safety.sql_parser import SQLCompletionParser


def compute_schema_reward(completion: str, valid_tables: Optional[Set[str]] = None) -> float:
    """Computes AST / Schema reward for a single completion.

    Returns:
        +0.20 if valid SELECT-family AST with authorized tables.
        0.00 otherwise.
    """
    parsed = SQLCompletionParser.parse(completion)
    if not parsed.is_valid_format or not parsed.sql:
        return 0.00

    checker = ASTPolicyChecker(allowed_tables=valid_tables)
    ast_res = checker.validate(parsed.sql)
    if ast_res.is_safe:
        return 0.20
    return 0.00


def schema_reward_func(completions: List[Union[str, dict]], valid_tables: Optional[List[Any]] = None, **kwargs) -> List[float]:
    """TRL-compatible reward callable for AST & schema compliance."""
    rewards: List[float] = []
    for idx, comp in enumerate(completions):
        text = comp["content"] if isinstance(comp, dict) else str(comp)
        tables = set(valid_tables[idx]) if (valid_tables and idx < len(valid_tables) and valid_tables[idx]) else None
        rewards.append(compute_schema_reward(text, valid_tables=tables))
    return rewards
