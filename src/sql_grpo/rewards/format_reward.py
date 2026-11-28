"""Format Reward (R_fmt): Scores tag structure and absence of disallowed text."""

from typing import Any, List
from sql_grpo.rewards.common import extract_completion_text
from sql_grpo.safety.sql_parser import SQLCompletionParser


def compute_format_reward(completion: Any) -> float:
    """Computes format reward for a single completion.

    Returns:
        +0.20 if valid <sql>...</sql> block without trailing text.
        -0.50 if missing, malformed, abstention, truncated, or contains disallowed trailing text.
    """
    text = extract_completion_text(completion)
    parsed = SQLCompletionParser.parse(text)
    if parsed.is_valid_format and parsed.sql:
        return 0.20
    return -0.50


def format_reward_func(completions: List[Any], **kwargs) -> List[float]:
    """TRL-compatible reward callable for format validation."""
    return [compute_format_reward(comp) for comp in completions]
