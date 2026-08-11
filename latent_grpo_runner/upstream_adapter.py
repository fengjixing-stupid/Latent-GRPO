"""Mac-safe adapter for optional upstream training instrumentation.

This module deliberately depends only on the Python standard library and the
runner's identity helpers.  Upstream code may import it without importing
torch, Ray, or SGLang.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numbers
import os
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from latent_grpo_runner.metrics.identity import stable_group_id


class NoOpObserver:
    """Observer used when instrumentation is absent or explicitly disabled."""

    enabled = False

    def emit(self, event_type: str, facts: Mapping[str, object]) -> None:
        del event_type, facts
        return None


class BufferedObserver:
    """Small in-memory observer used by synthetic tests and local adapters."""

    enabled = True

    def __init__(self, max_events: int = 1024) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self._max_events = max_events
        self._events: list[dict[str, object]] = []

    def emit(self, event_type: str, facts: Mapping[str, object]) -> None:
        event = {"event_type": event_type, **dict(facts)}
        if len(self._events) == self._max_events:
            self._events.pop(0)
        self._events.append(event)

    def drain(self) -> list[dict[str, object]]:
        events, self._events = self._events, []
        return events


@dataclass(frozen=True)
class OCPSelectionFacts:
    """Detached facts captured at the exact OCP winner-selection point."""

    group_id: object
    winner_local_index: int
    trajectory_id: object
    mean_old_log_prob: float | None


@dataclass(frozen=True)
class EvalQuestionFacts:
    """One checkpoint-evaluation generation before any aggregation."""

    data_source: str
    question_id: str
    generation_id: int
    predicted_answer: str
    reference_answer: str
    reward: float
    is_correct: bool | None
    correctness_unavailable_reason: str | None


def ocp_selection_event(facts: OCPSelectionFacts) -> dict[str, object]:
    """Convert OCP facts to a storage-friendly observer event."""
    return {"event_type": "ocp_selection", **asdict(facts)}


def eval_question_event(facts: EvalQuestionFacts) -> dict[str, object]:
    """Convert one raw evaluation generation to a storage-friendly event."""
    return {"event_type": "eval_question", **asdict(facts)}


def emit_eval_question_facts(
    observer: Any,
    *,
    data_sources: Sequence[Any],
    extra_infos: Sequence[Mapping[str, Any]],
    reward_models: Sequence[Mapping[str, Any]],
    outputs: Sequence[str],
    scores: Sequence[float],
    correctness: Sequence[bool | None] | None = None,
    generation_ordinals: MutableMapping[str, int] | None = None,
) -> int:
    """Emit complete per-generation eval facts without deriving aggregates."""
    lengths = {len(data_sources), len(extra_infos), len(reward_models), len(outputs), len(scores)}
    if correctness is not None:
        lengths.add(len(correctness))
    if len(lengths) != 1:
        raise ValueError("eval raw-fact inputs must have equal lengths")

    generation_ordinals = {} if generation_ordinals is None else generation_ordinals
    for position, (source, extra_info, reward_model, output, score) in enumerate(
        zip(data_sources, extra_infos, reward_models, outputs, scores)
    ):
        if "index" not in extra_info:
            raise ValueError("eval raw facts require extra_info.index")
        data_source = str(source)
        question_id = f"{data_source}:{extra_info['index']}"
        generation_id = generation_ordinals.get(question_id, 0)
        generation_ordinals[question_id] = generation_id + 1
        raw_correctness = None if correctness is None else correctness[position]
        if raw_correctness is None:
            is_correct = None
        elif isinstance(raw_correctness, bool):
            is_correct = raw_correctness
        elif isinstance(raw_correctness, numbers.Real) and raw_correctness in (0, 1):
            is_correct = bool(raw_correctness)
        else:
            raise ValueError("correctness must be bool, 0, 1, or null")
        event = eval_question_event(
            EvalQuestionFacts(
                data_source=data_source,
                question_id=question_id,
                generation_id=generation_id,
                predicted_answer=str(output),
                reference_answer=str(reward_model.get("ground_truth", extra_info.get("answer", ""))),
                reward=float(score),
                is_correct=is_correct,
                correctness_unavailable_reason=(
                    "reward_extra_info.acc_missing" if is_correct is None else None
                ),
            )
        )
        observer.emit("eval_question", {key: value for key, value in event.items() if key != "event_type"})
    return len(outputs)


def load_observer_from_env(environ: Mapping[str, str] | None = None, *, sink: Any = None) -> Any:
    """Enable observation only with an explicit durable coordinator sink.

    ``BufferedObserver`` intentionally remains a synthetic-test utility.  It
    drops old events at its configured bound and must never be selected by the
    production environment switch.
    """
    values = os.environ if environ is None else environ
    enabled = values.get("LATENT_GRPO_OBSERVER_ENABLED", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return NoOpObserver()
    if sink is None:
        raise RuntimeError(
            "LATENT_GRPO_OBSERVER_ENABLED requires an authoritative observer sink; "
            "eval raw-fact persistence is interface-only"
        )
    if isinstance(sink, BufferedObserver):
        raise TypeError("BufferedObserver is synthetic-only and cannot be a production observer sink")
    if not getattr(sink, "durable", False) or not callable(getattr(sink, "emit", None)):
        raise TypeError("authoritative observer sink must declare durable=true and implement emit")
    if getattr(sink, "enabled", None) is not True:
        raise TypeError("authoritative observer sink must declare enabled=true")
    return sink


def attach_stable_ids(
    global_step: int,
    prompt_identities: Iterable[str],
) -> tuple[list[str], list[int]]:
    """Build deterministic post-repeat group IDs and per-group ordinals."""
    ordinals: dict[str, int] = {}
    group_ids: list[str] = []
    trajectory_ids: list[int] = []
    for prompt_identity in prompt_identities:
        group_id = stable_group_id(global_step, prompt_identity)
        trajectory_id = ordinals.get(group_id, 0)
        ordinals[group_id] = trajectory_id + 1
        group_ids.append(group_id)
        trajectory_ids.append(trajectory_id)
    return group_ids, trajectory_ids


def _prompt_identities(non_tensor_batch: Mapping[str, Sequence[Any]]) -> list[str]:
    explicit = non_tensor_batch.get("prompt_identity")
    if explicit is not None:
        return [str(value) for value in explicit]

    indices = non_tensor_batch.get("index")
    if indices is None:
        raise ValueError("post-repeat instrumentation requires prompt_identity or index")
    data_sources = non_tensor_batch.get("data_source")
    if data_sources is None:
        data_sources = ["unknown"] * len(indices)
    if len(data_sources) != len(indices):
        raise ValueError("data_source and index lengths must match")
    return [f"{source}:{index}" for source, index in zip(data_sources, indices)]


def attach_stable_ids_to_batch(
    batch: Any,
    global_step: int,
    observer: Any = None,
) -> Any:
    """Attach IDs to a DataProto-like object only when observation is enabled."""
    observer = observer or NoOpObserver()
    if not observer.enabled:
        return batch

    try:
        import numpy as np
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("NumPy is required for enabled DataProto instrumentation") from error

    non_tensor_batch: MutableMapping[str, Sequence[Any]] = batch.non_tensor_batch
    group_ids, trajectory_ids = attach_stable_ids(global_step, _prompt_identities(non_tensor_batch))
    non_tensor_batch["group_id"] = np.asarray(group_ids, dtype=object)
    non_tensor_batch["trajectory_id"] = np.asarray(trajectory_ids, dtype=np.int64)
    observer.emit(
        "post_repeat_ids",
        {
            "global_step": int(global_step),
            "trajectory_count": len(trajectory_ids),
            "group_count": len(set(group_ids)),
        },
    )
    return batch


def build_component_sufficient_stats(
    margins: Iterable[float],
    valid_mask: Iterable[bool],
    flip_mask: Iterable[bool],
    near_zero_threshold: float,
) -> dict[str, int | float | None]:
    """Reduce synthetic/CPU component facts with the same Stage-2 masks.

    This standard-library reference implementation is intentionally separate
    from the upstream torch hook.  It lets Mac tests verify the contract
    without importing verl, torch, CUDA, or retaining tensor objects.
    """
    if near_zero_threshold < 0:
        raise ValueError("near_zero_threshold must be non-negative")
    margin_values = [float(value) for value in margins]
    validity = [bool(value) for value in valid_mask]
    flips = [bool(value) for value in flip_mask]
    if not (len(margin_values) == len(validity) == len(flips)):
        raise ValueError("margins, valid_mask, and flip_mask must have equal lengths")

    effective = [valid and math.isfinite(value) for value, valid in zip(margin_values, validity)]
    values = [value for value, include in zip(margin_values, effective) if include]
    return {
        "sum": float(sum(values)),
        "sum_sq": float(sum(value * value for value in values)),
        "count": len(values),
        "nan_count": sum(valid and not math.isfinite(value) for value, valid in zip(margin_values, validity)),
        "masked_count": sum(not valid for valid in validity),
        "min": min(values) if values else None,
        "negative_count": sum(value < 0 for value in values),
        "near_zero_count": sum(abs(value) <= near_zero_threshold for value in values),
        "flipgrad_trigger_count": sum(flip and include for flip, include in zip(flips, effective)),
    }


def merge_worker_observer_packets(
    packets: Iterable[Mapping[str, Any]], *, expected_worker_count: int
) -> dict[str, Any]:
    """Fail-closed coordinator merge of rank-local scalar observer packets.

    Optimizer attempts are replicated FSDP facts and therefore require exact
    rank consensus. Component statistics cover disjoint DP shards and are
    merged by sufficient-statistic addition/minimum, never worker means.
    """
    rows = [dict(packet) for packet in packets]
    base = {
        "aggregation_worker_count": len(rows),
        "optimizer_update_available": False,
        "optimizer_update_unavailable_reason": None,
        "did_update": None,
        "update_count": None,
        "component_available": False,
        "component_unavailable_reason": None,
        "component_sufficient_stats": None,
        "gpu_memory_available": False,
        "gpu_memory_unavailable_reason": None,
        "gpu_memory_by_worker": None,
    }
    ranks = [row.get("worker_rank") for row in rows]
    if (
        expected_worker_count < 1
        or len(rows) != expected_worker_count
        or any(type(rank) is not int for rank in ranks)
        or sorted(ranks) != list(range(expected_worker_count))
    ):
        reason = "worker_packet_set_incomplete_or_duplicate"
        return {
            **base,
            "optimizer_update_unavailable_reason": reason,
            "component_unavailable_reason": reason,
            "gpu_memory_unavailable_reason": reason,
        }

    memory_fields = (
        "device_index",
        "current_allocated_bytes",
        "current_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    memory_rows: list[dict[str, int]] = []
    for row in rows:
        memory = row.get("gpu_memory")
        if not isinstance(memory, Mapping) or any(
            type(memory.get(field)) is not int or int(memory[field]) < 0
            for field in memory_fields
        ):
            memory_rows = []
            break
        memory_rows.append(
            {"worker_rank": int(row["worker_rank"]), **{field: int(memory[field]) for field in memory_fields}}
        )
    if len(memory_rows) == expected_worker_count:
        base.update({"gpu_memory_available": True, "gpu_memory_by_worker": memory_rows})
    else:
        base["gpu_memory_unavailable_reason"] = "gpu_memory_missing_or_invalid"

    outcome_signatures = []
    for row in rows:
        attempts = row.get("optimizer_steps")
        update_count = row.get("update_count")
        did_update = row.get("did_update")
        if not isinstance(attempts, list) or type(update_count) is not int or type(did_update) is not bool:
            outcome_signatures.append(None)
            continue
        attempt_flags = tuple(attempt.get("did_step") for attempt in attempts if isinstance(attempt, Mapping))
        if len(attempt_flags) != len(attempts) or any(type(flag) is not bool for flag in attempt_flags):
            outcome_signatures.append(None)
            continue
        if update_count != sum(attempt_flags) or did_update != (update_count > 0):
            outcome_signatures.append(None)
            continue
        outcome_signatures.append((update_count, did_update, attempt_flags))
    if outcome_signatures and outcome_signatures[0] is not None and all(
        signature == outcome_signatures[0] for signature in outcome_signatures
    ):
        update_count, did_update, _ = outcome_signatures[0]
        base.update({"optimizer_update_available": True, "did_update": did_update, "update_count": update_count})
    else:
        base["optimizer_update_unavailable_reason"] = "optimizer_outcome_rank_disagreement"

    stats_rows: list[Mapping[str, Any]] = []
    unavailable_reasons: list[str] = []
    for row in rows:
        local_stats = row.get("component_sufficient_stats")
        if not isinstance(local_stats, list) or not local_stats:
            unavailable_reasons.append("component_stats_missing")
            continue
        for stats in local_stats:
            if not isinstance(stats, Mapping):
                unavailable_reasons.append("component_stats_invalid")
            elif stats.get("available") is False:
                unavailable_reasons.append(str(stats.get("unavailable_reason") or "component_stats_unavailable"))
            else:
                stats_rows.append(stats)
    if unavailable_reasons:
        base["component_unavailable_reason"] = sorted(set(unavailable_reasons))[0]
        return base

    count_fields = (
        "count", "nan_count", "masked_count", "negative_count", "near_zero_count", "flipgrad_trigger_count"
    )
    versions = {stats.get("definition_version") for stats in stats_rows}
    thresholds = {stats.get("near_zero_threshold") for stats in stats_rows}
    valid_stats = bool(stats_rows) and len(versions) == 1 and len(thresholds) == 1
    for stats in stats_rows:
        valid_stats = valid_stats and all(type(stats.get(name)) is int and stats[name] >= 0 for name in count_fields)
        valid_stats = valid_stats and all(
            type(stats.get(name)) in {int, float} and math.isfinite(float(stats[name])) for name in ("sum", "sum_sq")
        )
        minimum = stats.get("min")
        valid_stats = valid_stats and (minimum is None or (type(minimum) in {int, float} and math.isfinite(float(minimum))))
    if not valid_stats:
        base["component_unavailable_reason"] = "component_stats_schema_or_definition_mismatch"
        return base

    minima = [float(stats["min"]) for stats in stats_rows if stats.get("min") is not None]
    merged = {
        "sum": sum(float(stats["sum"]) for stats in stats_rows),
        "sum_sq": sum(float(stats["sum_sq"]) for stats in stats_rows),
        **{name: sum(int(stats[name]) for stats in stats_rows) for name in count_fields},
        "min": min(minima) if minima else None,
        "near_zero_threshold": thresholds.pop(),
        "definition_version": versions.pop(),
    }
    base.update({"component_available": True, "component_sufficient_stats": merged})
    return base
