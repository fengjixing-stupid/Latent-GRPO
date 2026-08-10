"""Stage 3 Support metrics from already-computed Top-K tensors."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Iterable, Sequence


OBSERVATION_PHASE = "pre_update_old_log_prob"
SUPPORT_RECORD_VERSION = "support_metrics_v1"
SUPPORT_SELECTION_RULE_VERSION = "support_selection_v1"


def collect_support_metrics(
    *,
    profile_name: str,
    seed: int,
    global_step: int,
    optimizer_step_at_observation: int,
    group_ids: Sequence[str],
    trajectory_ids: Sequence[int],
    trajectory_classes: Sequence[str],
    trajectory_mean_old_log_probs: Sequence[float],
    response_mask: Any,
    rollout_topk_ids: Any,
    old_topk_indices: Any,
    is_overlong_or_truncated_by_length: Sequence[bool] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return per-selected-trajectory Support rows plus one benchmark row.

    This collector only consumes detached values produced by rollout generation
    and the pre-update old-log-prob forward. Any alignment uncertainty returns
    an unavailable benchmark and no metric rows.
    """
    started = perf_counter()
    try:
        rollout = _rank3_ints(rollout_topk_ids, "rollout_topk_ids")
        clean = _rank3_ints(old_topk_indices, "old_topk_indices")
        mask = _rank2_bools(response_mask, "response_mask")
        _validate_id_vectors(
            group_ids, trajectory_ids, trajectory_classes, trajectory_mean_old_log_probs, len(mask)
        )
        if len(rollout) != len(clean) or len(rollout) != len(mask):
            raise ValueError("batch_size_mismatch")
        if any(len(rollout_row) != len(clean_row) or len(rollout_row) != len(mask_row)
               for rollout_row, clean_row, mask_row in zip(rollout, clean, mask)):
            raise ValueError("response_width_mismatch")
        k_values = {len(position) for row in rollout for position in row}
        k_values.update(len(position) for row in clean for position in row)
        if len(k_values) != 1:
            raise ValueError("topk_k_mismatch")
        k = next(iter(k_values))
        if k < 1:
            raise ValueError("topk_k_empty")
        if any(token == -100 for row in clean for position in row for token in position):
            raise ValueError("old_topk_contains_padding_sentinel")
        overlong = (
            [False] * len(group_ids)
            if is_overlong_or_truncated_by_length is None
            else [bool(value) for value in is_overlong_or_truncated_by_length]
        )
        if len(overlong) != len(group_ids):
            raise ValueError("overlong_vector_length_mismatch")
    except (TypeError, ValueError) as error:
        reason = str(error) or error.__class__.__name__
        return [], _support_benchmark_record(
            profile_name=profile_name,
            seed=seed,
            global_step=global_step,
            optimizer_step_at_observation=optimizer_step_at_observation,
            elapsed=perf_counter() - started,
            selected_count=0,
            candidate_count=0,
            effective_position_count=0,
            available=False,
            reason=reason,
        )

    selected_indices = _select_trajectories(
        group_ids=group_ids,
        trajectory_ids=trajectory_ids,
        trajectory_classes=trajectory_classes,
        scores=trajectory_mean_old_log_probs,
        overlong=overlong,
    )
    rows: list[dict[str, object]] = []
    total_effective = 0
    for index in selected_indices:
        retention_sum = 0.0
        top1_count = 0
        effective_count = 0
        for rollout_position, clean_position, include in zip(rollout[index], clean[index], mask[index]):
            if not include or _is_hard_token(rollout_position):
                continue
            if any(token == -100 for token in rollout_position):
                continue
            rollout_set = set(rollout_position)
            clean_set = set(clean_position)
            retention_sum += len(rollout_set & clean_set) / k
            top1_count += int(rollout_position[0] in clean_set)
            effective_count += 1
        total_effective += effective_count
        available = effective_count > 0
        rows.append({
            "profile_name": profile_name,
            "seed": seed,
            "global_step": global_step,
            "optimizer_step_at_observation": optimizer_step_at_observation,
            "observation_phase": OBSERVATION_PHASE,
            "group_id": group_ids[index],
            "trajectory_id": int(trajectory_ids[index]),
            "trajectory_class": trajectory_classes[index],
            "trajectory_mean_old_log_prob": float(trajectory_mean_old_log_probs[index]),
            "trajectory_selection_rule_version": SUPPORT_SELECTION_RULE_VERSION,
            "candidate_trajectory_count": _candidate_count(group_ids[index], group_ids, overlong),
            "support/retention_rate": retention_sum / effective_count if available else None,
            "support/top1_retention_rate": top1_count / effective_count if available else None,
            "support/effective_position_count": effective_count,
            "record_available": available,
            "record_unavailable_reason": None if available else "empty_effective_mask",
            "support_available": available,
            "support_unavailable_reason": None if available else "empty_effective_mask",
            "record_version": SUPPORT_RECORD_VERSION,
        })
    benchmark = _support_benchmark_record(
        profile_name=profile_name,
        seed=seed,
        global_step=global_step,
        optimizer_step_at_observation=optimizer_step_at_observation,
        elapsed=perf_counter() - started,
        selected_count=len(rows),
        candidate_count=sum(not flag for flag in overlong),
        effective_position_count=total_effective,
        available=True,
        reason=None,
    )
    return rows, benchmark


def _select_trajectories(
    *,
    group_ids: Sequence[str],
    trajectory_ids: Sequence[int],
    trajectory_classes: Sequence[str],
    scores: Sequence[float],
    overlong: Sequence[bool],
) -> list[int]:
    selected: list[int] = []
    for group_id in dict.fromkeys(group_ids):
        indices = [i for i, value in enumerate(group_ids) if value == group_id and not overlong[i]]
        for klass in ("correct", "non_correct"):
            candidates = [
                (float(scores[i]), -int(trajectory_ids[i]), i)
                for i in indices
                if trajectory_classes[i] == klass and math.isfinite(float(scores[i]))
            ]
            if candidates:
                selected.append(max(candidates)[2])
    return selected


def _candidate_count(group_id: str, group_ids: Sequence[str], overlong: Sequence[bool]) -> int:
    return sum(value == group_id and not flag for value, flag in zip(group_ids, overlong))


def _support_benchmark_record(
    *,
    profile_name: str,
    seed: int,
    global_step: int,
    optimizer_step_at_observation: int,
    elapsed: float,
    selected_count: int,
    candidate_count: int,
    effective_position_count: int,
    available: bool,
    reason: str | None,
) -> dict[str, object]:
    return {
        "profile_name": profile_name,
        "seed": seed,
        "global_step": global_step,
        "optimizer_step_at_observation": optimizer_step_at_observation,
        "observation_phase": OBSERVATION_PHASE,
        "support_extra_time_seconds": elapsed,
        "support_cache_peak_bytes": 0,
        "support_selected_trajectory_count": selected_count,
        "support_candidate_trajectory_count": candidate_count,
        "support_benchmark/total_effective_position_count": effective_position_count,
        "record_available": available,
        "record_unavailable_reason": reason,
        "support_available": available,
        "support_unavailable_reason": reason,
    }


def _validate_id_vectors(
    group_ids: Sequence[str],
    trajectory_ids: Sequence[int],
    trajectory_classes: Sequence[str],
    trajectory_mean_old_log_probs: Sequence[float],
    batch_size: int,
) -> None:
    lengths = {len(group_ids), len(trajectory_ids), len(trajectory_classes), len(trajectory_mean_old_log_probs), batch_size}
    if len(lengths) != 1:
        raise ValueError("identity_vector_length_mismatch")
    if any(not isinstance(value, str) or not value for value in group_ids):
        raise ValueError("group_id_unavailable")
    if any(type(value) is not int or value < 0 for value in trajectory_ids):
        raise ValueError("trajectory_id_unavailable")
    if any(value not in {"correct", "non_correct"} for value in trajectory_classes):
        raise ValueError("trajectory_class_unavailable")


def _is_hard_token(topk_ids: Sequence[int]) -> bool:
    return len(topk_ids) > 1 and all(token == -100 for token in topk_ids[1:])


def _rank3_ints(values: Any, label: str) -> list[list[list[int]]]:
    nested = _tolist(values)
    if not isinstance(nested, list) or not all(isinstance(row, list) for row in nested):
        raise ValueError(f"{label}_rank_mismatch")
    result: list[list[list[int]]] = []
    for row in nested:
        converted_row: list[list[int]] = []
        if not all(isinstance(position, list) for position in row):
            raise ValueError(f"{label}_rank_mismatch")
        for position in row:
            if not position or any(type(token) is not int for token in position):
                raise ValueError(f"{label}_value_unavailable")
            converted_row.append(list(position))
        result.append(converted_row)
    return result


def _rank2_bools(values: Any, label: str) -> list[list[bool]]:
    nested = _tolist(values)
    if not isinstance(nested, list) or not all(isinstance(row, list) for row in nested):
        raise ValueError(f"{label}_rank_mismatch")
    return [[bool(value) for value in row] for row in nested]


def _tolist(values: Any) -> Any:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        return values.tolist()
    return values
