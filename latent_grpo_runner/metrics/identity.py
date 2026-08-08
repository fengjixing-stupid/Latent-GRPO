"""Stable post-repeat identities and binary trajectory classification."""

import hashlib
from typing import Iterable, Mapping


def stable_group_id(global_step: int, prompt_identity: str) -> str:
    """Deterministic, resumable group identity based on step and prompt identity."""
    if not isinstance(prompt_identity, str) or not prompt_identity:
        raise ValueError("prompt_identity must be a non-empty stable string or hash")
    payload = f"v1:{int(global_step)}:{prompt_identity}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def attach_stable_trajectory_ids(records: Iterable[Mapping[str, object]], global_step: int) -> list[dict]:
    """Assign per-group ordinal IDs in the already-repeated, pre-reorder order."""
    ordinals: dict[str, int] = {}
    attached = []
    for record in records:
        if "prompt_identity" not in record:
            raise ValueError("prompt_identity is required before stable IDs can be attached")
        copied = dict(record)
        group_id = copied.get("group_id") or stable_group_id(global_step, copied["prompt_identity"])
        ordinal = ordinals.get(group_id, 0)
        ordinals[group_id] = ordinal + 1
        copied["group_id"] = group_id
        copied["trajectory_id"] = ordinal
        attached.append(copied)
    return attached


def classify_trajectory(is_correct: bool, response_length: int, max_response_length: int) -> dict:
    if max_response_length < 0 or response_length < 0:
        raise ValueError("lengths must be non-negative")
    return {
        "trajectory_class": "correct" if is_correct else "non_correct",
        "is_overlong_or_truncated_by_length": response_length >= max_response_length,
    }
