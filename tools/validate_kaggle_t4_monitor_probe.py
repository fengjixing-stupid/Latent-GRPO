#!/usr/bin/env python3
"""Validate one real-model Kaggle pre-backward monitoring run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REAL_PRE_BACKWARD_METRICS = (
    "train/policy_loss",
    "train/entropy",
    "train/kl",
    "train/clip_fraction",
    "train/importance_ratio_mean",
    "train/importance_ratio_std",
    "train/response_length",
    "train/latent_length",
    "mixture/effective_k_noisy",
    "mixture/top1_weight_noisy",
    "mask/zero_advantage_rate",
    "signal/reward_mean",
    "signal/reward_std",
    "signal/advantage_std",
)

DEFERRED_TO_THREE_GPU = (
    "train/step_time",
    "support_retention",
    "top1_retention",
    "delta_mean",
    "delta_std",
    "delta_p05",
    "delta_min",
    "delta_negative_rate",
    "delta_near_zero_rate",
    "flipgrad_rate",
    "top1_share",
    "effective_k",
    "weight_credit_spearman",
    "surrogate_alignment_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()
    table_dir = root / "train_step_metrics"
    parts = sorted(table_dir.glob("part-*.parquet"))
    if not parts:
        raise SystemExit("MONITOR_PROBE: FAIL: no train_step_metrics parquet part")

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise SystemExit("MONITOR_PROBE: FAIL: pyarrow is required") from error

    rows = []
    for part in parts:
        rows.extend(pq.read_table(part).to_pylist())
    candidates = [
        row for row in rows
        if row.get("profile_name") == "kaggle-t4-monitor"
        and row.get("observation_phase") == "pre_backward_probe"
    ]
    if not candidates:
        raise SystemExit("MONITOR_PROBE: FAIL: no pre_backward_probe row")
    row = max(candidates, key=lambda item: int(item["global_step"]))

    blockers = []
    if row.get("optimizer_step") != 0:
        blockers.append("optimizer_step_must_remain_zero")
    if row.get("train/step_time") is not None or row.get("train/step_time__available") is not False:
        blockers.append("step_time_must_be_deferred_without_actor_update")
    if row.get("train/step_time__unavailable_reason") != "pre_backward_probe_no_actor_update":
        blockers.append("step_time_deferred_reason_mismatch")
    for name in REAL_PRE_BACKWARD_METRICS:
        if row.get(f"{name}__available") is not True:
            blockers.append(f"real_metric_unavailable:{name}")

    generated_status = (
        "REAL_VALUE_PROVISIONAL_DEFINITION"
        if row.get("train/generated_token_count__available") is True
        else "UNAVAILABLE"
    )
    report = {
        "status": "PASS" if not blockers else "BLOCKED",
        "gate": "kaggle_t4_real_pre_backward_monitor",
        "profile_name": row.get("profile_name"),
        "global_step": row.get("global_step"),
        "optimizer_step": row.get("optimizer_step"),
        "optimizer_step_performed": False,
        "real_pre_backward_validated_metrics": list(REAL_PRE_BACKWARD_METRICS),
        "generated_token_count": {
            "status": generated_status,
            "reason": "current P1 scope is final_training_rollout_trajectories; all-attempt scope fix is still required",
        },
        "deferred_metrics": {
            "train/step_time": "DEFERRED_NO_ACTOR_UPDATE",
            **{name: "DEFERRED_TO_THREE_GPU_OR_LATER_STAGE" for name in DEFERRED_TO_THREE_GPU if name != "train/step_time"},
        },
        "blockers": blockers,
    }
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else root / "kaggle_t4_monitor_probe_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    print("REAL_PRE_BACKWARD_MONITOR_GATE:", report["status"])
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
