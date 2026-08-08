"""Driver-side Stage 1 train-step records from detached sufficient statistics."""

import math
from typing import Iterable, Mapping, Optional, TYPE_CHECKING

from .aggregators import SufficientStats
from .events import StepContext

if TYPE_CHECKING:
    from .stage2 import Stage2SufficientStats


_CORE = (
    ("train/policy_loss", "mean", "train/policy_loss", "train/policy_loss_count"),
    ("train/entropy", "mean", "train/entropy", "train/entropy_count"),
    ("train/kl", "mean", "train/kl", "train/kl_count"),
    ("train/clip_fraction", "rate", "train/clip_fraction", "train/clip_fraction_count"),
    ("train/importance_ratio_mean", "mean", "train/importance_ratio", "train/importance_ratio_count"),
    ("train/importance_ratio_std", "std", "train/importance_ratio", "train/importance_ratio_count"),
    ("train/response_length", "mean", "train/response_length", "train/response_length_count"),
    ("train/latent_length", "mean", "train/latent_length", "train/latent_length_count"),
)


def _metric_value(stat: SufficientStats, method: str) -> float:
    return {"mean": stat.mean, "std": stat.std, "rate": stat.rate}[method]()


def build_train_step_metrics(
    context: StepContext,
    statistics: Mapping[str, SufficientStats],
    final_training_trajectory_lengths: Iterable[int],
    *,
    driver_step_time_seconds: Optional[float] = None,
    stage2_statistics: Optional["Stage2SufficientStats"] = None,
    aggregation_worker_count: Optional[int] = None,
    record_version: str = "metrics_record_v1",
    metrics_compute_time: Optional[float] = None,
    metrics_write_time: Optional[float] = None,
) -> dict:
    """Build one authoritative train/probe metrics row on the driver.

    ``driver_step_time_seconds`` is intentionally not accepted as a worker
    statistic.  A pre-backward monitor probe must leave it unavailable because
    no actor update has completed.
    """
    if context.observation_phase not in {"post_update", "pre_backward_probe"}:
        raise ValueError("train step metrics require post_update or pre_backward_probe")
    is_pre_backward_probe = context.observation_phase == "pre_backward_probe"
    lengths = list(final_training_trajectory_lengths)
    if any(not isinstance(length, int) or length < 0 for length in lengths):
        raise ValueError("trajectory lengths must be non-negative integers")
    record = context.to_record()
    record.update({
        "metric_scope": "train_step", "aggregation_worker_count": aggregation_worker_count,
        "record_version": record_version, "metrics_compute_time": metrics_compute_time,
        "metrics_write_time": metrics_write_time,
        "generated_token_count_definition_version": "paper_mixed_trajectory_sum_v1",
        "generated_token_count_scope": "final_training_rollout_trajectories",
        "entropy_source": "runtime_policy_entropy", "entropy_probability_space": "runtime_policy_distribution",
        "entropy_mask_definition": "runtime_policy_entropy_mask", "entropy_definition_version": "runtime_policy_entropy_v1",
        "response_length_definition_version": "runtime_response_length_v1",
        "latent_length_definition_version": "runtime_latent_position_v1",
        "length_counting_rule_version": "runtime_eos_stop_pending_probe_v1",
    })
    all_available = True
    for output_name, method, source_name, count_name in _CORE:
        stat = statistics.get(source_name)
        available = stat is not None and stat.available
        record[output_name] = _metric_value(stat, method) if available else None
        record[f"{output_name}__available"] = available
        record[f"{output_name}__unavailable_reason"] = None if available else (stat.unavailable_reason if stat else "missing_runtime_interface")
        record[count_name] = stat.count if stat is not None else 0
        all_available = all_available and available
    generated_available = bool(lengths)
    record["train/generated_token_count"] = sum(lengths) if generated_available else None
    record["train/generated_token_count__available"] = generated_available
    record["train/generated_token_count__unavailable_reason"] = None if generated_available else "empty_effective_mask"
    record["final_training_trajectory_count"] = len(lengths)
    time_available = driver_step_time_seconds is not None and math.isfinite(float(driver_step_time_seconds))
    if is_pre_backward_probe and time_available:
        raise ValueError("pre_backward_probe must not report post-update step_time")
    record["train/step_time"] = float(driver_step_time_seconds) if time_available else None
    record["train/step_time__available"] = time_available
    record["train/step_time__unavailable_reason"] = (
        None
        if time_available
        else (
            "pre_backward_probe_no_actor_update"
            if is_pre_backward_probe
            else "missing_runtime_interface"
        )
    )
    all_available = all_available and generated_available and time_available
    from .stage2 import Stage2SufficientStats, build_stage2_metrics
    if stage2_statistics is None:
        stage2_statistics = Stage2SufficientStats(
            SufficientStats(), SufficientStats(), SufficientStats(), SufficientStats(), SufficientStats(),
        )
        stage2_record = build_stage2_metrics(stage2_statistics)
        stage2_record["stage2_unavailable_reason"] = "missing_runtime_interface"
        record.update(stage2_record)
    else:
        record.update(build_stage2_metrics(stage2_statistics))
    record["train_core_available"] = all_available
    record["train_core_unavailable_reason"] = None if all_available else "metric_unavailable"
    record["record_available"] = True
    record["record_unavailable_reason"] = None
    return record
