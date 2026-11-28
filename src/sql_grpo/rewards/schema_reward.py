"""Schema & AST Reward (R_ast): Scores syntactic validity and schema allowlist compliance."""

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from sql_grpo.rewards.common import (
    NormalizedCandidate,
    normalize_and_validate_candidate,
    resolve_columns_arg,
    resolve_tables_arg,
)


def compute_schema_reward(
    completion: Any,
    valid_tables: Optional[Set[str]] = None,
    valid_columns: Optional[Union[Set[str], Mapping[str, Set[str]]]] = None,
) -> float:
    """Computes AST / Schema reward for a single completion.

    Returns:
        +0.20 if format valid and AST conforms to SELECT-only table/column/function allowlists.
        0.00 otherwise.
    """
    cand = normalize_and_validate_candidate(
        completion=completion,
        valid_tables=valid_tables,
        valid_columns=valid_columns,
    )
    if cand.is_format_valid and cand.is_ast_safe:
        return 0.20
    return 0.00


def schema_reward_func(
    completions: List[Any],
    valid_tables: Optional[List[Any]] = None,
    valid_columns: Optional[List[Any]] = None,
    valid_columns_json: Optional[List[Any]] = None,
    **kwargs,
) -> List[float]:
    """TRL-compatible reward callable for AST & schema compliance."""
    cache: Dict[Tuple[str, Any, Any], NormalizedCandidate] = {}
    rewards: List[float] = []
    for idx, comp in enumerate(completions):
        tables = resolve_tables_arg(valid_tables, idx)
        cols = resolve_columns_arg(valid_columns, valid_columns_json, idx)
        cand = normalize_and_validate_candidate(
            completion=comp,
            valid_tables=tables,
            valid_columns=cols,
            cache=cache,
        )
        rewards.append(0.20 if (cand.is_format_valid and cand.is_ast_safe) else 0.00)
    return rewards
