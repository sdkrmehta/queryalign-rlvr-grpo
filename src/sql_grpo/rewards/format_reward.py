"""Format Reward (R_fmt): Scores tag structure and absence of disallowed text."""

from typing import List, Union
from sql_grpo.safety.sql_parser import SQLCompletionParser


def compute_format_reward(completion: str) -> float:
    """Computes format reward for a single completion.

    Returns:
        +0.20 if valid <sql>...</sql> block without trailing text.
        -0.50 if missing, malformed, truncated, or contains disallowed text.
    """
    parsed = SQLCompletionParser.parse(completion)
    if parsed.is_valid_format:
        return 0.20
    return -0.50


def format_reward_func(completions: List[Union[str, dict]], **kwargs) -> List[float]:
    """TRL-compatible reward callable for format validation."""
    rewards: List[float] = []
    for comp in completions:
        text = comp["content"] if isinstance(comp, dict) else str(comp)
        rewards.append(compute_format_reward(text))
    return rewards
