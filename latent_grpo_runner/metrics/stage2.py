"""Stage 2 worker reductions, driver aggregation, and group raw facts."""

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional, Sequence

from .aggregators import SufficientStats
from .events import StepContext
from .masks import zero_advantage_stats


def _mixture_sufficient_stats(weights_by_position: Iterable[Sequence[float]]) -> tuple[SufficientStats, SufficientStats]:
    effective, top1 = [], []
    for weights in weights_by_position:
        row = [float(weight) for weight in weights]
        if not row or any(not math.isfinite(weight) or weight < 0 for weight in row):
            continue
        total = sum(row)
        if total <= 0:
            continue
        normalized = [weight / total for weight in row]
        effective.append(math.exp(-sum(weight * math.log(weight) for weight in normalized if weight)))
        top1.append(max(normalized))
    return SufficientStats.from_values(effective), SufficientStats.from_values(top1)


@dataclass(frozen=True)
class Stage2SufficientStats:
    """The only cross-worker Stage 2 payload; it contains no worker means."""

    mixture_effective_k: SufficientStats
    mixture_top1_weight: SufficientStats
    zero_advantage: SufficientStats
    reward: SufficientStats
    advantage: SufficientStats

    @classmethod
    def from_local(
        cls, noisy_mixture_weights: Iterable[Sequence[float]], advantages_for_zero_mask: Iterable[float],
        eligible_latent_mask: Iterable[bool], rewards: Iterable[float], final_advantages: Iterable[float],
        *, zero_threshold: float = 0.0,
    ) -> "Stage2SufficientStats":
        effective, top1 = _mixture_sufficient_stats(noisy_mixture_weights)
        return cls(effective, top1, zero_advantage_stats(advantages_for_zero_mask, eligible_latent_mask, zero_threshold),
                   SufficientStats.from_values(rewards), SufficientStats.from_values(final_advantages))

    def merge(self, other: "Stage2SufficientStats") -> "Stage2SufficientStats":
        return Stage2SufficientStats(
            self.mixture_effective_k.merge(other.mixture_effective_k),
            self.mixture_top1_weight.merge(other.mixture_top1_weight),
            self.zero_advantage.merge(other.zero_advantage), self.reward.merge(other.reward),
            self.advantage.merge(other.advantage),
        )


def build_stage2_metrics(stats: Stage2SufficientStats) -> dict:
    mixture_available = stats.mixture_effective_k.available and stats.mixture_top1_weight.available
    mask_available = stats.zero_advantage.available
    signal_available = stats.reward.available and stats.advantage.available
    record = {
        "mixture/effective_k_noisy": stats.mixture_effective_k.mean() if mixture_available else None,
        "mixture/top1_weight_noisy": stats.mixture_top1_weight.mean() if mixture_available else None,
        "mixture/noisy_count": stats.mixture_effective_k.count,
        "mixture_available": mixture_available,
        "mixture_unavailable_reason": None if mixture_available else "empty_effective_mask",
        "mask/zero_advantage_rate": stats.zero_advantage.rate() if mask_available else None,
        "mask/eligible_latent_token_count": stats.zero_advantage.count,
        "mask_available": mask_available,
        "mask_unavailable_reason": None if mask_available else "empty_effective_mask",
        "signal/reward_mean": stats.reward.mean() if stats.reward.available else None,
        "signal/reward_std": stats.reward.std() if stats.reward.available else None,
        "signal/advantage_std": stats.advantage.std() if stats.advantage.available else None,
        "signal/reward_count": stats.reward.count, "signal/advantage_count": stats.advantage.count,
        "signal_available": signal_available,
        "signal_unavailable_reason": None if signal_available else "empty_effective_mask",
        "stage2_available": mixture_available and mask_available and signal_available,
        "stage2_unavailable_reason": None if mixture_available and mask_available and signal_available else "metric_unavailable",
    }
    for name, available, reason in [
        ("mixture/effective_k_noisy", mixture_available, record["mixture_unavailable_reason"]),
        ("mixture/top1_weight_noisy", mixture_available, record["mixture_unavailable_reason"]),
        ("mask/zero_advantage_rate", mask_available, record["mask_unavailable_reason"]),
        ("signal/reward_mean", stats.reward.available, None if stats.reward.available else "empty_effective_mask"),
        ("signal/reward_std", stats.reward.available, None if stats.reward.available else "empty_effective_mask"),
        ("signal/advantage_std", stats.advantage.available, None if stats.advantage.available else "empty_effective_mask"),
    ]:
        record[f"{name}__available"] = available
        record[f"{name}__unavailable_reason"] = None if available else reason
    return record


def noisy_mixture_stats(weights_by_position: Iterable[Sequence[float]]) -> dict:
    effective, top1 = _mixture_sufficient_stats(weights_by_position)
    return build_stage2_metrics(Stage2SufficientStats(effective, top1, SufficientStats(), SufficientStats(), SufficientStats()))


def signal_stats(rewards: Iterable[float], advantages: Iterable[float]) -> dict:
    return build_stage2_metrics(Stage2SufficientStats(SufficientStats(), SufficientStats(), SufficientStats(), SufficientStats.from_values(rewards), SufficientStats.from_values(advantages)))


def build_group_metrics(group_id: str, trajectories: Iterable[Mapping[str, object]]) -> dict:
    """Count-only reducer retained for callers that do not yet have StepContext."""
    rows = list(trajectories)
    classes = [row.get("trajectory_class") for row in rows]
    if any(value not in {"correct", "non_correct"} for value in classes):
        raise ValueError("trajectory_class must be correct or non_correct")
    overlong = [row for row in rows if row.get("is_overlong_or_truncated_by_length")]
    return {"group_id": group_id, "group/trajectory_count": len(rows), "group/correct_trajectory_count": classes.count("correct"),
            "group/non_correct_trajectory_count": classes.count("non_correct"), "group/overlong_trajectory_count": len(overlong),
            "group/overlong_generated_token_count": sum(int(row.get("generated_token_count", 0)) for row in overlong),
            "group/overlong_response_length_max": max((int(row.get("response_length", 0)) for row in overlong), default=None)}


@dataclass(frozen=True)
class OptimalCorrectPath:
    trajectory_id: int
    mean_old_log_prob: float


def select_optimal_correct_path(trajectories: Iterable[Mapping[str, object]]) -> Optional[OptimalCorrectPath]:
    """Select from correct, positive-first-step-advantage trajectories in memory."""
    candidates = []
    for row in trajectories:
        if row.get("trajectory_class") != "correct" or float(row.get("first_step_advantage", 0.0)) <= 0:
            continue
        score = float(row.get("trajectory_mean_old_log_prob", math.nan))
        trajectory_id = row.get("trajectory_id")
        if isinstance(trajectory_id, int) and math.isfinite(score):
            candidates.append((score, trajectory_id))
    if not candidates:
        return None
    score, trajectory_id = max(candidates)
    return OptimalCorrectPath(trajectory_id, score)


def build_train_group_metrics(
    context: StepContext, group_id: str, prompt_id_or_hash: str, trajectories: Iterable[Mapping[str, object]],
    winner: Optional[OptimalCorrectPath], *, aggregation_worker_count: Optional[int] = None,
    record_version: str = "metrics_record_v1", metrics_compute_time: Optional[float] = None,
    metrics_write_time: Optional[float] = None,
) -> dict:
    if context.observation_phase != "post_advantage_pre_update":
        raise ValueError("group metrics must be created post_advantage_pre_update")
    if not isinstance(prompt_id_or_hash, str) or not prompt_id_or_hash:
        raise ValueError("prompt_id_or_hash must be a stable non-empty string")
    rows = list(trajectories)
    record = context.to_record()
    record.update(build_group_metrics(group_id, rows))
    rewards = [float(row["reward"]) for row in rows if "reward" in row and math.isfinite(float(row["reward"]))]
    record.update({
        "metric_scope": "train_group", "prompt_id_or_hash": prompt_id_or_hash,
        "aggregation_worker_count": aggregation_worker_count, "record_version": record_version,
        "metrics_compute_time": metrics_compute_time, "metrics_write_time": metrics_write_time,
        "group/zero_variance_reward": bool(rewards) and min(rewards) == max(rewards),
        "optimal_correct_trajectory_id": winner.trajectory_id if winner else None,
        "optimal_correct_mean_old_log_prob": winner.mean_old_log_prob if winner else None,
        "group_definition_version": "prompt_rollout_group_v1",
        "trajectory_classification_version": "correct_non_correct_v1",
        "overlong_definition_version": "length_threshold_v1",
        "group_available": bool(rows), "group_unavailable_reason": None if rows else "empty_effective_mask",
        "record_available": True, "record_unavailable_reason": None,
    })
    record["optimal_correct_path_available"] = winner is not None
    record["optimal_correct_path_unavailable_reason"] = None if winner else "empty_effective_mask"
    return record


def gumbel_diagnostic(raw_values: Iterable[float], one_sided_values: Iterable[float], *, enabled: bool,
                      lower_clip: float = 0.0, upper_clip: float = 0.0) -> dict:
    if not enabled:
        return {"gumbel_diagnostics_mode": "disabled", "gumbel_available": False, "gumbel_unavailable_reason": "disabled_by_config"}
    raw, one_sided = list(raw_values), list(one_sided_values)
    raw_stats, one_stats = SufficientStats.from_values(raw), SufficientStats.from_values(one_sided)
    lower = SufficientStats.from_values(raw, numerator_mask=[value <= lower_clip for value in raw])
    upper = SufficientStats.from_values(raw, numerator_mask=[value >= upper_clip for value in raw])
    zero = SufficientStats.from_values(one_sided, numerator_mask=[value == 0.0 for value in one_sided])
    available = raw_stats.available and one_stats.available
    return {"gumbel_diagnostics_mode": "diagnostic", "gumbel/raw_mean": raw_stats.mean() if raw_stats.available else None,
            "gumbel/raw_std": raw_stats.std() if raw_stats.available else None,
            "gumbel/lower_clip_rate": lower.rate() if raw_stats.available else None,
            "gumbel/upper_clip_rate": upper.rate() if raw_stats.available else None,
            "gumbel/zero_rate": zero.rate() if one_stats.available else None,
            "gumbel/raw_count": raw_stats.count, "gumbel/one_sided_count": one_stats.count,
            "gumbel_available": available, "gumbel_unavailable_reason": None if available else "empty_effective_mask"}


def mechanism_stats(surrogate_margins: Iterable[float], valid_component_mask: Iterable[bool],
                    flipgrad_trigger_mask: Iterable[bool], *, near_zero_threshold: float) -> dict:
    margins, valid, triggers = list(surrogate_margins), list(valid_component_mask), list(flipgrad_trigger_mask)
    if len(margins) != len(valid) or len(margins) != len(triggers):
        raise ValueError("mechanism values and masks must have equal length")
    effective = [include and math.isfinite(float(value)) for value, include in zip(margins, valid)]
    stats = SufficientStats.from_values(margins, valid)
    return {"sum": stats.sum, "sum_sq": stats.sum_sq, "count": stats.count, "nan_count": stats.nan_count,
            "masked_count": stats.masked_count, "min": stats.min,
            "negative_count": sum(value < 0 for value, include in zip(margins, effective) if include),
            "near_zero_count": sum(abs(value) <= near_zero_threshold for value, include in zip(margins, effective) if include),
            "flipgrad_trigger_count": sum(bool(trigger) for trigger, include in zip(triggers, effective) if include)}
