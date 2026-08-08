"""P1 Stage 1/2 aggregation contracts for Latent-GRPO.

This module is intentionally torch-free.  Workers and the driver reduce runtime
Tensors before crossing the observer boundary; only bounded scalar sufficient
statistics reach this layer.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .aggregators import SufficientStats
from .events import StepContext
from .stage1 import build_train_step_metrics
from .stage2 import Stage2SufficientStats


WORKER_P1_METRICS = (
    "train/policy_loss",
    "train/kl",
    "train/clip_fraction",
    "train/importance_ratio",
)

DRIVER_P1_METRICS = (
    "train/entropy",
    "train/response_length",
    "train/latent_length",
    "mixture/effective_k_noisy",
    "mixture/top1_weight_noisy",
    "mask/zero_advantage_rate",
    "signal/reward",
    "signal/advantage",
)

_STAT_COUNT_FIELDS = ("count", "nan_count", "masked_count", "numerator_count")
_STAT_FLOAT_FIELDS = ("sum", "sum_sq")


class P1AggregationError(RuntimeError):
    """Fail-closed error for malformed or incomplete P1 observer packets."""


def sufficient_stats_to_record(
    stats: SufficientStats, *, definition_version: str
) -> dict[str, int | float | str | None]:
    if not definition_version:
        raise ValueError("definition_version must be non-empty")
    return {
        "sum": float(stats.sum),
        "sum_sq": float(stats.sum_sq),
        "count": int(stats.count),
        "nan_count": int(stats.nan_count),
        "masked_count": int(stats.masked_count),
        "min": None if stats.min is None else float(stats.min),
        "max": None if stats.max is None else float(stats.max),
        "numerator_count": int(stats.numerator_count),
        "definition_version": definition_version,
    }


def sufficient_stats_from_record(record: Mapping[str, Any]) -> SufficientStats:
    """Validate and deserialize one bounded sufficient-statistic record."""
    if not isinstance(record, Mapping):
        raise P1AggregationError("sufficient statistics must be a mapping")
    version = record.get("definition_version")
    if not isinstance(version, str) or not version:
        raise P1AggregationError("sufficient statistics require definition_version")
    for field in _STAT_COUNT_FIELDS:
        value = record.get(field)
        if type(value) is not int or value < 0:
            raise P1AggregationError(f"invalid sufficient-stat integer field: {field}")
    for field in _STAT_FLOAT_FIELDS:
        value = record.get(field)
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise P1AggregationError(f"invalid sufficient-stat numeric field: {field}")
    extrema: dict[str, float | None] = {}
    for field in ("min", "max"):
        value = record.get(field)
        if value is None:
            extrema[field] = None
        elif type(value) in {int, float} and math.isfinite(float(value)):
            extrema[field] = float(value)
        else:
            raise P1AggregationError(f"invalid sufficient-stat extrema field: {field}")
    if record["numerator_count"] > record["count"]:
        raise P1AggregationError("numerator_count cannot exceed count")
    return SufficientStats(
        sum=float(record["sum"]),
        sum_sq=float(record["sum_sq"]),
        count=int(record["count"]),
        nan_count=int(record["nan_count"]),
        masked_count=int(record["masked_count"]),
        min=extrema["min"],
        max=extrema["max"],
        numerator_count=int(record["numerator_count"]),
    )


def merge_serialized_sufficient_stats(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, int | float | str | None]:
    rows = [dict(record) for record in records]
    if not rows:
        raise P1AggregationError("cannot merge an empty sufficient-statistic set")
    versions = {row.get("definition_version") for row in rows}
    if len(versions) != 1 or not isinstance(next(iter(versions)), str):
        raise P1AggregationError("sufficient-statistic definition versions disagree")
    merged = SufficientStats.merge_all([sufficient_stats_from_record(row) for row in rows])
    return sufficient_stats_to_record(merged, definition_version=str(next(iter(versions))))


def merge_worker_p1_packets(
    packets: Iterable[Mapping[str, Any]], *, expected_worker_count: int
) -> dict[str, Any]:
    """Merge disjoint DP worker shards without ever averaging worker means."""
    rows = [dict(packet) for packet in packets]
    base = {
        "p1_worker_metrics_available": False,
        "p1_worker_metrics_unavailable_reason": None,
        "p1_worker_sufficient_stats": None,
    }
    ranks = [row.get("worker_rank") for row in rows]
    if (
        type(expected_worker_count) is not int
        or expected_worker_count < 1
        or len(rows) != expected_worker_count
        or any(type(rank) is not int for rank in ranks)
        or sorted(ranks) != list(range(expected_worker_count))
    ):
        base["p1_worker_metrics_unavailable_reason"] = "worker_packet_set_incomplete_or_duplicate"
        return base

    merged: dict[str, dict[str, int | float | str | None]] = {}
    try:
        for metric_name in WORKER_P1_METRICS:
            local_records = []
            for row in rows:
                mapping = row.get("p1_sufficient_stats")
                if not isinstance(mapping, Mapping) or metric_name not in mapping:
                    raise P1AggregationError(f"missing worker P1 statistic: {metric_name}")
                local_record = mapping[metric_name]
                if not isinstance(local_record, Mapping):
                    raise P1AggregationError(f"invalid worker P1 statistic: {metric_name}")
                local_records.append(local_record)
            merged[metric_name] = merge_serialized_sufficient_stats(local_records)
    except P1AggregationError as error:
        base["p1_worker_metrics_unavailable_reason"] = str(error)
        return base

    base.update(
        {
            "p1_worker_metrics_available": True,
            "p1_worker_sufficient_stats": merged,
        }
    )
    return base


def _required_stat(
    mapping: Mapping[str, Any], name: str, *, source: str
) -> SufficientStats:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        raise P1AggregationError(f"missing {source} P1 statistic: {name}")
    return sufficient_stats_from_record(value)


def build_p1_train_step_metrics(
    *,
    context: StepContext,
    worker_statistics: Mapping[str, Any],
    driver_statistics: Mapping[str, Any],
    final_training_trajectory_lengths: Sequence[int],
    driver_step_time_seconds: float | None,
    aggregation_worker_count: int,
    metrics_compute_time: float | None = None,
) -> dict[str, Any]:
    """Build the one authoritative Stage 1/2 row from global sufficient stats."""
    if type(aggregation_worker_count) is not int or aggregation_worker_count < 1:
        raise P1AggregationError("aggregation_worker_count must be positive")
    if driver_step_time_seconds is None:
        if context.observation_phase != "pre_backward_probe":
            raise P1AggregationError("driver_step_time_seconds is required post-update")
    elif not math.isfinite(float(driver_step_time_seconds)) or driver_step_time_seconds < 0:
        raise P1AggregationError("driver_step_time_seconds must be finite and non-negative")

    stage1_statistics = {
        "train/policy_loss": _required_stat(worker_statistics, "train/policy_loss", source="worker"),
        "train/entropy": _required_stat(driver_statistics, "train/entropy", source="driver"),
        "train/kl": _required_stat(worker_statistics, "train/kl", source="worker"),
        "train/clip_fraction": _required_stat(worker_statistics, "train/clip_fraction", source="worker"),
        "train/importance_ratio": _required_stat(worker_statistics, "train/importance_ratio", source="worker"),
        "train/response_length": _required_stat(driver_statistics, "train/response_length", source="driver"),
        "train/latent_length": _required_stat(driver_statistics, "train/latent_length", source="driver"),
    }
    stage2_statistics = Stage2SufficientStats(
        mixture_effective_k=_required_stat(
            driver_statistics, "mixture/effective_k_noisy", source="driver"
        ),
        mixture_top1_weight=_required_stat(
            driver_statistics, "mixture/top1_weight_noisy", source="driver"
        ),
        zero_advantage=_required_stat(
            driver_statistics, "mask/zero_advantage_rate", source="driver"
        ),
        reward=_required_stat(driver_statistics, "signal/reward", source="driver"),
        advantage=_required_stat(driver_statistics, "signal/advantage", source="driver"),
    )
    return build_train_step_metrics(
        context,
        stage1_statistics,
        final_training_trajectory_lengths,
        driver_step_time_seconds=(
            None if driver_step_time_seconds is None else float(driver_step_time_seconds)
        ),
        stage2_statistics=stage2_statistics,
        aggregation_worker_count=aggregation_worker_count,
        record_version=(
            "metrics_record_p1_pre_backward_probe_v1"
            if context.observation_phase == "pre_backward_probe"
            else "metrics_record_p1_v1"
        ),
        metrics_compute_time=metrics_compute_time,
    )
