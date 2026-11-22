"""Efficiency Reward (R_eff): Rewards query execution efficiency strictly gated on correctness."""

from typing import List, Optional, Union


def compute_efficiency_reward(
    is_correct: bool,
    candidate_latency_ms: float,
    gold_latency_ms: Optional[float] = None,
) -> float:
    """Computes efficiency reward.

    Returns:
        0.00 if candidate is not correct.
        Up to +0.30 if correct and fast compared to gold latency baseline.
    """
    if not is_correct:
        return 0.00

    bonus = 0.0

    # 1. Latency comparison bonus (up to +0.15)
    if gold_latency_ms and gold_latency_ms > 0:
        if candidate_latency_ms <= 1.2 * gold_latency_ms:
            bonus += 0.15
        elif candidate_latency_ms <= 2.0 * gold_latency_ms:
            bonus += 0.075
    else:
        # Fallback absolute threshold: <= 50ms
        if candidate_latency_ms <= 50.0:
            bonus += 0.15

    # 2. Structural efficiency bonus (up to +0.15)
    # Additional proxy bonus for queries completing rapidly
    if candidate_latency_ms <= 30.0:
        bonus += 0.15

    return min(bonus, 0.30)


def efficiency_reward_func(
    completions: List[Union[str, dict]],
    **kwargs,
) -> List[float]:
    """TRL-compatible reward callable for efficiency."""
    # During parallel rollout without paired gold execution times, return 0.00 or base efficiency
    return [0.0 for _ in completions]
