"""Stage 4 checkpoint probe metric reducers."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import random
import time
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

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


def restore_packed_probe_tensors(
    packed: Mapping[str, Any],
    *,
    indices: Any,
    batch: int,
    seqlen: int,
    response_length: int,
) -> dict[str, Any]:
    """Restore packed next-token probe tensors without detaching their graph."""
    if batch < 1 or seqlen < 2 or not 0 < response_length < seqlen:
        raise ValueError("invalid packed probe output dimensions")
    required = {"topk_log_probs", "raw_diff", "flipgrad_trigger_mask"}
    if set(packed) != required:
        raise ValueError("packed probe tensor fields are incomplete")
    restored: dict[str, Any] = {}
    for name in sorted(required):
        values = packed[name]
        if values.dim() < 2 or values.size(0) != indices.numel():
            raise ValueError("packed probe tensor does not align with unpadding indices")
        flat_shape = (batch * seqlen, *values.shape[1:])
        full = values.new_zeros(flat_shape).index_copy(0, indices, values)
        restored[name] = full.view(batch, seqlen, *values.shape[1:])[
            :, -response_length - 1 : -1
        ].contiguous()
    return restored


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
    state_preservation: dict[str, bool] | None = None,
) -> dict[str, object]:
    state = state_preservation or {
        "parameters_unchanged": False,
        "optimizer_state_unchanged": False,
        "training_grads_unchanged": False,
        "cpu_rng_restored": False,
        "cuda_rng_restored": False,
        "python_rng_restored": False,
        "numpy_rng_restored": False,
        "module_mode_restored": False,
    }
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
        **state,
        "extra_loss_backward_executed": False,
        "extra_optimizer_step_executed": False,
    }


def collect_credit_from_autograd(
    *,
    policy_loss: Any,
    topk_log_probs: Any,
    mixture_weights: Any,
    valid_component_mask: Any,
    advantages: Any,
    retain_graph: bool = False,
) -> CreditMetrics:
    import torch

    if not getattr(topk_log_probs, "requires_grad", False):
        raise ValueError("topk_log_probs must require grad")
    gradients = torch.autograd.grad(
        policy_loss,
        topk_log_probs,
        retain_graph=retain_graph,
        create_graph=False,
    )[0]
    return collect_credit_from_values(
        credit=-gradients.detach(),
        mixture_weights=mixture_weights,
        valid_component_mask=valid_component_mask,
        advantages=advantages,
    )


def collect_credit_from_values(
    *,
    credit: Any,
    mixture_weights: Any,
    valid_component_mask: Any,
    advantages: Any,
) -> CreditMetrics:
    import torch

    credit = torch.as_tensor(credit).detach().float()
    weights = torch.as_tensor(mixture_weights).detach().float()
    valid = torch.as_tensor(valid_component_mask).detach().bool()
    advantage_values = torch.as_tensor(advantages).detach().float()
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


_STATE_PRESERVATION_FIELDS = (
    "parameters_unchanged",
    "optimizer_state_unchanged",
    "training_grads_unchanged",
    "cpu_rng_restored",
    "cuda_rng_restored",
    "python_rng_restored",
    "numpy_rng_restored",
    "module_mode_restored",
)


def collect_checkpoint_probe_packet(
    *,
    policy_loss: Any,
    topk_log_probs: Any,
    deltas: Any,
    mixture_weights: Any,
    valid_component_mask: Any,
    advantages: Any,
    trajectory_masks: dict[str, Any],
    position_masks: dict[str, Any],
    model: Any,
    optimizer: Any,
    flipgrad_trigger_mask: Any | None = None,
    retain_graph: bool = True,
) -> dict[str, object]:
    """Run one bounded credit autograd and return a Ray-safe worker packet."""
    import numpy as np
    import torch

    tensors = {
        "topk_log_probs": topk_log_probs,
        "deltas": deltas,
        "mixture_weights": mixture_weights,
        "valid_component_mask": valid_component_mask,
        "advantages": advantages,
    }
    expected_shape = tuple(topk_log_probs.shape)
    if not getattr(topk_log_probs, "requires_grad", False):
        raise ValueError("topk_log_probs must require grad")
    if any(tuple(value.shape) != expected_shape for value in tensors.values()):
        raise ValueError("checkpoint probe tensors must have identical shape")
    for family, masks in (("trajectory", trajectory_masks), ("position", position_masks)):
        if "all" not in masks:
            raise ValueError(f"{family} masks require an all group")
        if any(tuple(mask.shape) != expected_shape for mask in masks.values()):
            raise ValueError(f"{family} masks must match checkpoint probe tensors")

    parameters = tuple(model.parameters())
    parameter_before = _parameter_token(parameters)
    optimizer_before = _optimizer_token(optimizer)
    grads_before = _grad_token(parameters)
    module_mode_before = bool(model.training)
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    peak_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    started = time.perf_counter()
    try:
        gradients = torch.autograd.grad(
            policy_loss,
            topk_log_probs,
            retain_graph=retain_graph,
            create_graph=False,
        )[0]
    finally:
        model.train(module_mode_before)
        random.setstate(python_rng_before)
        np.random.set_state(numpy_rng_before)
        torch.random.set_rng_state(cpu_rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state_all(cuda_rng_before)

    state = {
        "parameters_unchanged": _parameter_token(parameters) == parameter_before,
        "optimizer_state_unchanged": _optimizer_token(optimizer) == optimizer_before,
        "training_grads_unchanged": _grad_token(parameters) == grads_before,
        "cpu_rng_restored": torch.equal(torch.random.get_rng_state(), cpu_rng_before),
        "cuda_rng_restored": (
            True
            if cuda_rng_before is None
            else all(
                torch.equal(left, right)
                for left, right in zip(torch.cuda.get_rng_state_all(), cuda_rng_before)
            )
        ),
        "python_rng_restored": random.getstate() == python_rng_before,
        "numpy_rng_restored": _numpy_rng_equal(np.random.get_state(), numpy_rng_before),
        "module_mode_restored": bool(model.training) == module_mode_before,
    }
    if gradients is None or tuple(gradients.shape) != expected_shape:
        raise RuntimeError("credit autograd did not return an aligned gradient")

    detached_delta = deltas.detach().float()
    detached_advantages = advantages.detach().float()
    detached_valid = valid_component_mask.detach().bool()
    detached_flips = (
        flipgrad_trigger_mask.detach().bool()
        if flipgrad_trigger_mask is not None
        else (detached_advantages <= 0) & (detached_delta < 0)
    )
    if tuple(detached_flips.shape) != expected_shape:
        raise ValueError("flipgrad trigger mask must match checkpoint probe tensors")
    peak_after = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {
        "available": True,
        "deltas": detached_delta.flatten().cpu().tolist(),
        "credit": (-gradients.detach().float()).flatten().cpu().tolist(),
        "mixture_weights": mixture_weights.detach().float().flatten().cpu().tolist(),
        "advantages": detached_advantages.flatten().cpu().tolist(),
        "valid_component_mask": detached_valid.flatten().cpu().tolist(),
        "flipgrad_trigger_mask": detached_flips.flatten().cpu().tolist(),
        "trajectory_masks": {
            name: mask.detach().bool().flatten().cpu().tolist()
            for name, mask in trajectory_masks.items()
        },
        "position_masks": {
            name: mask.detach().bool().flatten().cpu().tolist()
            for name, mask in position_masks.items()
        },
        "trajectory_count": int(expected_shape[0]) if expected_shape else 0,
        "latent_position_count": int(
            detached_valid.any(dim=-1).sum().item()
        ),
        "credit_autograd_executed": True,
        "state_preservation": state,
        "probe_extra_time_seconds": time.perf_counter() - started,
        "probe_peak_memory_bytes": max(0, int(peak_after - peak_before)),
    }


def build_checkpoint_probe_event(
    packets: Iterable[dict[str, object]],
    *,
    expected_worker_count: int,
    profile_name: str,
    seed: int,
    global_step: int,
    optimizer_step: int,
    checkpoint_step: int,
    probe_batch_id: str,
) -> dict[str, object]:
    """Merge bounded rank packets and derive every group from one autograd result."""
    import torch

    rows = [dict(packet) for packet in packets]
    ranks = [row.get("worker_rank") for row in rows]
    if len(rows) != expected_worker_count or sorted(ranks) != list(range(expected_worker_count)):
        raise ValueError("checkpoint probe requires one uniquely ranked packet per worker")
    if any(row.get("available") is not True for row in rows):
        raise ValueError("checkpoint probe worker packet is unavailable")

    vector_fields = (
        "deltas",
        "credit",
        "mixture_weights",
        "advantages",
        "valid_component_mask",
        "flipgrad_trigger_mask",
    )
    merged = {
        field: [value for packet in rows for value in packet[field]]
        for field in vector_fields
    }
    size = len(merged["deltas"])
    if any(len(merged[field]) != size for field in vector_fields):
        raise ValueError("checkpoint probe worker vectors are misaligned")

    trajectory_names = _common_mask_names(rows, "trajectory_masks")
    position_names = _common_mask_names(rows, "position_masks")
    output_rows = []
    for trajectory_name in trajectory_names:
        trajectory_mask = [
            value
            for packet in rows
            for value in packet["trajectory_masks"][trajectory_name]
        ]
        for position_name in position_names:
            position_mask = [
                value
                for packet in rows
                for value in packet["position_masks"][position_name]
            ]
            selected = [
                bool(valid and trajectory and position)
                for valid, trajectory, position in zip(
                    merged["valid_component_mask"], trajectory_mask, position_mask
                )
            ]
            credit = collect_credit_from_values(
                credit=torch.tensor(merged["credit"]),
                mixture_weights=torch.tensor(merged["mixture_weights"]),
                valid_component_mask=torch.tensor(selected),
                advantages=torch.tensor(merged["advantages"]),
            )
            output_rows.append(
                build_probe_metric_row(
                    profile_name=profile_name,
                    seed=seed,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    checkpoint_step=checkpoint_step,
                    probe_batch_id=probe_batch_id,
                    deltas=merged["deltas"],
                    valid_delta_mask=selected,
                    flipgrad_trigger_mask=merged["flipgrad_trigger_mask"],
                    credit=credit,
                    trajectory_group=trajectory_name,
                    latent_position_group=position_name,
                    probe_rng_restore_succeeded=all(
                        packet["state_preservation"][field]
                        for packet in rows
                        for field in (
                            "cpu_rng_restored",
                            "cuda_rng_restored",
                            "python_rng_restored",
                            "numpy_rng_restored",
                        )
                    ),
                )
            )

    state = {
        field: all(packet["state_preservation"][field] for packet in rows)
        for field in _STATE_PRESERVATION_FIELDS
    }
    benchmark = build_probe_benchmark_row(
        profile_name=profile_name,
        seed=seed,
        global_step=global_step,
        checkpoint_step=checkpoint_step,
        probe_batch_id=probe_batch_id,
        probe_trajectory_count=sum(int(packet["trajectory_count"]) for packet in rows),
        probe_latent_position_count=sum(int(packet["latent_position_count"]) for packet in rows),
        credit_autograd_executed=all(
            packet.get("credit_autograd_executed") is True for packet in rows
        ),
        probe_rng_restore_succeeded=all(
            state[field]
            for field in (
                "cpu_rng_restored",
                "cuda_rng_restored",
                "python_rng_restored",
                "numpy_rng_restored",
            )
        ),
        probe_extra_time_seconds=sum(float(packet["probe_extra_time_seconds"]) for packet in rows),
        probe_peak_memory_bytes=max(int(packet["probe_peak_memory_bytes"]) for packet in rows),
        record_available=all(state.values()),
        record_unavailable_reason=None if all(state.values()) else "state_preservation_failed",
        state_preservation=state,
    )
    worker_runtime = [
        {
            "worker_rank": int(packet["worker_rank"]),
            "probe_extra_time_seconds": float(packet["probe_extra_time_seconds"]),
            "probe_peak_memory_bytes": int(packet["probe_peak_memory_bytes"]),
        }
        for packet in rows
    ]
    return {"rows": output_rows, "benchmark": benchmark, "worker_runtime": worker_runtime}


def _common_mask_names(packets: Sequence[dict[str, object]], field: str) -> list[str]:
    names = set(packets[0][field])
    if any(set(packet[field]) != names for packet in packets[1:]):
        raise ValueError(f"checkpoint probe {field} differ across workers")
    return ["all", *sorted(names - {"all"})]


def _parameter_token(parameters: Sequence[Any]) -> tuple[object, ...]:
    return tuple(
        (id(parameter), int(parameter._version), tuple(parameter.shape), str(parameter.dtype))
        for parameter in parameters
    )


def _grad_token(parameters: Sequence[Any]) -> tuple[object, ...]:
    return tuple(
        None
        if parameter.grad is None
        else (
            id(parameter.grad),
            int(parameter.grad._version),
            tuple(parameter.grad.shape),
            str(parameter.grad.dtype),
        )
        for parameter in parameters
    )


def _optimizer_token(optimizer: Any) -> tuple[object, ...]:
    state = tuple(
        sorted(
            (id(parameter), _nested_version_token(values))
            for parameter, values in optimizer.state.items()
        )
    )
    groups = tuple(
        tuple(
            sorted(
                (key, _nested_version_token(value))
                for key, value in group.items()
                if key != "params"
            )
        )
        for group in optimizer.param_groups
    )
    return state, groups


def _nested_version_token(value: Any) -> object:
    if hasattr(value, "_version") and hasattr(value, "shape"):
        return (id(value), int(value._version), tuple(value.shape), str(value.dtype))
    if isinstance(value, dict):
        return tuple(sorted((str(key), _nested_version_token(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_nested_version_token(item) for item in value)
    return value


def _numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    import numpy as np

    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
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
