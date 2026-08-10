"""Stage 4 checkpoint probe metric reducers."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import random
from typing import Any, Callable, Iterable, Sequence, TypeVar

from .aggregators import SufficientStats


PROBE_DEFINITION_VERSION = "checkpoint_probe_v1"
ONE_SIDED_DEFINITION_VERSION = "onesided_delta_v1"
CREDIT_DEFINITION_VERSION = "credit_autograd_v1"
SURROGATE_ALIGNMENT_DEFINITION_VERSION = "surrogate_alignment_grad_sign_v1"
DEFAULT_NEAR_ZERO_THRESHOLD = 1e-6

T = TypeVar("T")


@dataclass(frozen=True)
class CreditMetrics:
    top1_share: float | None
    effective_k: float | None
    weight_credit_spearman: float | None
    surrogate_alignment_rate: float | None
    concentration_count: int
    spearman_count: int
    alignment_count: int
    concentration_available: bool
    concentration_unavailable_reason: str | None
    spearman_available: bool
    spearman_unavailable_reason: str | None
    alignment_available: bool
    alignment_unavailable_reason: str | None
    credit_autograd_executed: bool = True


@dataclass(frozen=True)
class PreservedProbeResult:
    value: T
    state_restored: bool
    cuda_rng_checked: bool


def build_probe_metric_row(
    *,
    profile_name: str,
    seed: int,
    global_step: int,
    optimizer_step: int,
    checkpoint_step: int,
    probe_batch_id: str,
    deltas: Iterable[float],
    valid_delta_mask: Iterable[bool],
    flipgrad_trigger_mask: Iterable[bool],
    credit: CreditMetrics | None,
    near_zero_threshold: float = DEFAULT_NEAR_ZERO_THRESHOLD,
    trajectory_group: str = "all",
    latent_position_group: str = "all",
    probe_rng_restore_succeeded: bool = True,
) -> dict[str, object]:
    delta_values = [float(value) for value in deltas]
    valid = [bool(value) for value in valid_delta_mask]
    flips = [bool(value) for value in flipgrad_trigger_mask]
    if len(delta_values) != len(valid) or len(delta_values) != len(flips):
        raise ValueError("probe delta values and masks must have equal length")
    effective = [include and math.isfinite(value) for value, include in zip(delta_values, valid)]
    stats = SufficientStats.from_values(delta_values, effective)
    negative = SufficientStats.from_values(delta_values, effective, [value < 0 for value in delta_values])
    near_zero = SufficientStats.from_values(
        delta_values, effective, [abs(value) <= near_zero_threshold for value in delta_values]
    )
    flip_count = sum(bool(trigger) for trigger, include in zip(flips, effective) if include)
    onesided_available = stats.available
    credit = credit or disabled_credit_metrics()
    return {
        "profile_name": profile_name,
        "seed": seed,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "checkpoint_step": checkpoint_step,
        "observation_phase": "checkpoint_probe",
        "probe_batch_id": probe_batch_id,
        "probe_definition_version": PROBE_DEFINITION_VERSION,
        "trajectory_group": trajectory_group,
        "latent_position_group": latent_position_group,
        "onesided/delta_mean": stats.mean() if stats.available else None,
        "onesided/delta_std": stats.std() if stats.available else None,
        "onesided/delta_p05": _quantile([value for value, include in zip(delta_values, effective) if include], 0.05)
        if stats.available else None,
        "onesided/delta_min": stats.min if stats.available else None,
        "onesided/delta_negative_rate": negative.rate() if stats.available else None,
        "onesided/delta_near_zero_rate": near_zero.rate() if stats.available else None,
        "onesided/flipgrad_rate": flip_count / stats.count if stats.available else None,
        "onesided/delta_count": stats.count,
        "onesided/flipgrad_count": flip_count,
        "valid_flipgrad_denominator": stats.count,
        "onesided_near_zero_threshold": near_zero_threshold,
        "onesided_near_zero_definition_version": "abs_delta_lte_threshold_v1",
        "onesided_definition_version": ONE_SIDED_DEFINITION_VERSION,
        "credit/top1_share": credit.top1_share,
        "credit/effective_k": credit.effective_k,
        "credit/weight_credit_spearman": credit.weight_credit_spearman,
        "credit/surrogate_alignment_rate": credit.surrogate_alignment_rate,
        "credit/concentration_count": credit.concentration_count,
        "credit/spearman_count": credit.spearman_count,
        "credit/alignment_count": credit.alignment_count,
        "credit_definition_version": CREDIT_DEFINITION_VERSION,
        "surrogate_alignment_definition_version": SURROGATE_ALIGNMENT_DEFINITION_VERSION,
        "record_available": onesided_available,
        "record_unavailable_reason": None if onesided_available else "empty_effective_mask",
        "onesided_available": onesided_available,
        "onesided_unavailable_reason": None if onesided_available else "empty_effective_mask",
        "credit_concentration_available": credit.concentration_available,
        "credit_concentration_unavailable_reason": credit.concentration_unavailable_reason,
        "credit_spearman_available": credit.spearman_available,
        "credit_spearman_unavailable_reason": credit.spearman_unavailable_reason,
        "credit_alignment_available": credit.alignment_available,
        "credit_alignment_unavailable_reason": credit.alignment_unavailable_reason,
        "credit/weight_credit_spearman__available": credit.spearman_available,
        "credit/weight_credit_spearman__unavailable_reason": credit.spearman_unavailable_reason,
        "credit/surrogate_alignment_rate__available": credit.alignment_available,
        "credit/surrogate_alignment_rate__unavailable_reason": credit.alignment_unavailable_reason,
        "probe_rng_restore_succeeded": probe_rng_restore_succeeded,
        "record_version": "probe_metrics_v1",
    }


def disabled_credit_metrics() -> CreditMetrics:
    return CreditMetrics(
        top1_share=None,
        effective_k=None,
        weight_credit_spearman=None,
        surrogate_alignment_rate=None,
        concentration_count=0,
        spearman_count=0,
        alignment_count=0,
        concentration_available=False,
        concentration_unavailable_reason="disabled_by_config",
        spearman_available=False,
        spearman_unavailable_reason="disabled_by_config",
        alignment_available=False,
        alignment_unavailable_reason="disabled_by_config",
        credit_autograd_executed=False,
    )


def build_probe_benchmark_row(
    *,
    profile_name: str,
    seed: int,
    global_step: int,
    checkpoint_step: int,
    probe_batch_id: str,
    probe_trajectory_count: int,
    probe_latent_position_count: int,
    credit_autograd_executed: bool,
    probe_rng_restore_succeeded: bool,
    probe_extra_time_seconds: float | None = None,
    probe_peak_memory_bytes: int = 0,
    record_available: bool = True,
    record_unavailable_reason: str | None = None,
) -> dict[str, object]:
    return {
        "profile_name": profile_name,
        "seed": seed,
        "global_step": global_step,
        "checkpoint_step": checkpoint_step,
        "probe_batch_id": probe_batch_id,
        "probe_extra_time_seconds": probe_extra_time_seconds,
        "probe_peak_memory_bytes": probe_peak_memory_bytes,
        "probe_trajectory_count": probe_trajectory_count,
        "probe_latent_position_count": probe_latent_position_count,
        "credit_autograd_executed": credit_autograd_executed,
        "probe_rng_restore_succeeded": probe_rng_restore_succeeded,
        "record_available": record_available,
        "record_unavailable_reason": record_unavailable_reason,
    }


def collect_credit_from_autograd(
    *,
    policy_loss: Any,
    topk_log_probs: Any,
    mixture_weights: Any,
    valid_component_mask: Any,
    advantages: Any,
) -> CreditMetrics:
    import torch

    if not getattr(topk_log_probs, "requires_grad", False):
        raise ValueError("topk_log_probs must require grad")
    gradients = torch.autograd.grad(policy_loss, topk_log_probs, retain_graph=False, create_graph=False)[0]
    credit = -gradients.detach().float()
    weights = mixture_weights.detach().float()
    valid = valid_component_mask.detach().bool()
    advantage_values = advantages.detach().float()
    if credit.shape != weights.shape or credit.shape != valid.shape or credit.shape != advantage_values.shape:
        raise ValueError("credit tensors must have identical shape")

    finite = torch.isfinite(credit) & torch.isfinite(weights) & torch.isfinite(advantage_values) & valid
    credit_values = credit[finite].flatten()
    weight_values = weights[finite].flatten()
    advantage_flat = advantage_values[finite].flatten()
    abs_credit = credit_values.abs()
    credit_mass = float(abs_credit.double().sum().item())
    if credit_values.numel() == 0 or credit_mass <= 0.0:
        concentration_available = False
        top1_share = None
        effective_k = None
        concentration_reason = "empty_credit_mass"
        concentration_count = int(credit_values.numel())
    else:
        q = (abs_credit.double() / credit_mass).tolist()
        concentration_available = True
        top1_share = max(q)
        effective_k = math.exp(-sum(value * math.log(value) for value in q if value > 0))
        concentration_reason = None
        concentration_count = len(q)

    spearman_value, spearman_available, spearman_reason, spearman_count = _spearman(
        weight_values.tolist(), abs_credit.tolist()
    )
    directional = (credit_values != 0) & (advantage_flat != 0)
    alignment_count = int(directional.sum().item())
    if alignment_count:
        aligned = ((credit_values[directional] * advantage_flat[directional]) > 0).sum().item()
        alignment_available = True
        alignment_rate = float(aligned) / alignment_count
        alignment_reason = None
    else:
        alignment_available = False
        alignment_rate = None
        alignment_reason = "zero_gradient_direction"
    return CreditMetrics(
        top1_share=top1_share,
        effective_k=effective_k,
        weight_credit_spearman=spearman_value,
        surrogate_alignment_rate=alignment_rate,
        concentration_count=concentration_count,
        spearman_count=spearman_count,
        alignment_count=alignment_count,
        concentration_available=concentration_available,
        concentration_unavailable_reason=concentration_reason,
        spearman_available=spearman_available,
        spearman_unavailable_reason=spearman_reason,
        alignment_available=alignment_available,
        alignment_unavailable_reason=alignment_reason,
    )


def run_preserving_training_state(model: Any, optimizer: Any, probe: Callable[[], T]) -> PreservedProbeResult[T]:
    import numpy as np
    import torch

    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    grads_before = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in model.parameters()]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    training_before = bool(model.training)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        value = probe()
    finally:
        model.train(training_before)
        for parameter, saved in zip(model.parameters(), grads_before):
            parameter.grad = None if saved is None else saved.clone()
        optimizer.load_state_dict(optimizer_before)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    state_restored = _states_equal(model, optimizer, parameters_before, grads_before, optimizer_before, training_before)
    return PreservedProbeResult(value=value, state_restored=state_restored, cuda_rng_checked=cuda_rng is not None)


def _states_equal(
    model: Any,
    optimizer: Any,
    parameters_before: Sequence[Any],
    grads_before: Sequence[Any],
    optimizer_before: dict[str, Any],
    training_before: bool,
) -> bool:
    import torch

    if bool(model.training) != training_before:
        return False
    for parameter, before, grad_before in zip(model.parameters(), parameters_before, grads_before):
        if not torch.equal(parameter.detach(), before):
            return False
        if grad_before is None:
            if parameter.grad is not None:
                return False
        elif parameter.grad is None or not torch.equal(parameter.grad, grad_before):
            return False
    return _nested_equal(optimizer.state_dict(), optimizer_before)


def _nested_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    if hasattr(left, "detach") and hasattr(right, "detach"):
        return bool(torch.equal(left, right))
    return left == right


def _quantile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _spearman(left: Sequence[float], right: Sequence[float]) -> tuple[float | None, bool, str | None, int]:
    if len(left) != len(right):
        raise ValueError("spearman vectors must have equal length")
    finite_pairs = [(float(a), float(b)) for a, b in zip(left, right) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(finite_pairs) < 2:
        return None, False, "effective_k_lt_2", len(finite_pairs)
    left_values = [item[0] for item in finite_pairs]
    right_values = [item[1] for item in finite_pairs]
    if len(set(left_values)) < 2 or len(set(right_values)) < 2:
        return None, False, "constant_rank", len(finite_pairs)
    left_ranks = _average_ranks(left_values)
    right_ranks = _average_ranks(right_values)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks))
    left_den = math.sqrt(sum((a - left_mean) ** 2 for a in left_ranks))
    right_den = math.sqrt(sum((b - right_mean) ** 2 for b in right_ranks))
    if left_den == 0.0 or right_den == 0.0:
        return None, False, "constant_rank", len(finite_pairs)
    return numerator / (left_den * right_den), True, None, len(finite_pairs)


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average = (i + 1 + j) / 2.0
        for index, _ in indexed[i:j]:
            ranks[index] = average
        i = j
    return ranks
