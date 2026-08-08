"""Pure-Python mask construction for locally reduced metric domains."""

import math
from typing import Iterable, Sequence

from .aggregators import SufficientStats


def _same_length(*values: Sequence[object]) -> None:
    if len({len(value) for value in values}) != 1:
        raise ValueError("mask inputs must have equal outer length")


def valid_latent_position_mask(
    topk_token_ids: Sequence[Sequence[int]],
    response_mask: Sequence[bool],
    attention_mask: Sequence[bool],
    loss_mask: Sequence[bool],
    sentinel: int = -100,
) -> list[bool]:
    """Return positions that are response, attended, loss-eligible latent states."""
    _same_length(topk_token_ids, response_mask, attention_mask, loss_mask)
    return [
        bool(response and attention and loss and any(token != sentinel for token in token_ids))
        for token_ids, response, attention, loss in zip(topk_token_ids, response_mask, attention_mask, loss_mask)
    ]


def valid_latent_component_mask(
    topk_token_ids: Sequence[Sequence[int]],
    position_mask: Sequence[bool],
    sentinel: int = -100,
) -> list[list[bool]]:
    _same_length(topk_token_ids, position_mask)
    return [
        [bool(position_valid and token != sentinel) for token in token_ids]
        for token_ids, position_valid in zip(topk_token_ids, position_mask)
    ]


def zero_advantage_stats(
    advantages: Iterable[float], eligible_latent_mask: Iterable[bool], zero_threshold: float = 0.0,
) -> SufficientStats:
    """Return a rate package whose denominator is exactly the eligible domain."""
    values = list(advantages)
    eligible = list(eligible_latent_mask)
    if len(values) != len(eligible):
        raise ValueError("advantages and eligible_latent_mask must have equal length")
    zeros = [abs(float(value)) <= zero_threshold if math.isfinite(float(value)) else False for value in values]
    # Use the actual advantages so non-finite eligible values are counted as NaN
    # rather than silently entering the zero-rate denominator.
    return SufficientStats.from_values(values, eligible, zeros)
