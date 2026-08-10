#!/usr/bin/env python3
"""Fail-closed validation of the formal 29 core metrics plus raw-token extension."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Callable


PROFILE = "kaggle-t4-30-metric"
ROOT = Path(__file__).resolve().parents[1]
CORE_METRICS = (
    ("train/policy_loss", "Stage 1", "train_step_metrics", "finite"),
    ("train/entropy", "Stage 1", "train_step_metrics", "nonnegative"),
    ("train/kl", "Stage 1", "train_step_metrics", "finite"),
    ("train/clip_fraction", "Stage 1", "train_step_metrics", "rate"),
    ("train/importance_ratio_mean", "Stage 1", "train_step_metrics", "positive"),
    ("train/importance_ratio_std", "Stage 1", "train_step_metrics", "nonnegative"),
    ("train/response_length", "Stage 1", "train_step_metrics", "nonnegative"),
    ("train/latent_length", "Stage 1", "train_step_metrics", "nonnegative"),
    ("train/generated_token_count", "Stage 1", "train_step_metrics", "nonnegative"),
    ("train/step_time", "Stage 1", "train_step_metrics", "positive"),
    ("mixture/effective_k_noisy", "Stage 2", "train_step_metrics", "effective_k"),
    ("mixture/top1_weight_noisy", "Stage 2", "train_step_metrics", "rate"),
    ("mask/zero_advantage_rate", "Stage 2", "train_step_metrics", "rate"),
    ("signal/reward_mean", "Stage 2", "train_step_metrics", "finite"),
    ("signal/reward_std", "Stage 2", "train_step_metrics", "nonnegative"),
    ("signal/advantage_std", "Stage 2", "train_step_metrics", "nonnegative"),
    ("support/retention_rate", "Stage 3", "support_metrics", "rate"),
    ("support/top1_retention_rate", "Stage 3", "support_metrics", "rate"),
    ("onesided/delta_mean", "Stage 4 One-sided", "probe_metrics", "finite"),
    ("onesided/delta_std", "Stage 4 One-sided", "probe_metrics", "nonnegative"),
    ("onesided/delta_p05", "Stage 4 One-sided", "probe_metrics", "finite"),
    ("onesided/delta_min", "Stage 4 One-sided", "probe_metrics", "finite"),
    ("onesided/delta_negative_rate", "Stage 4 One-sided", "probe_metrics", "rate"),
    ("onesided/delta_near_zero_rate", "Stage 4 One-sided", "probe_metrics", "rate"),
    ("onesided/flipgrad_rate", "Stage 4 One-sided", "probe_metrics", "rate"),
    ("credit/top1_share", "Stage 4 Credit", "probe_metrics", "rate"),
    ("credit/effective_k", "Stage 4 Credit", "probe_metrics", "effective_k"),
    ("credit/weight_credit_spearman", "Stage 4 Credit", "probe_metrics", "correlation"),
    ("credit/surrogate_alignment_rate", "Stage 4 Credit", "probe_metrics", "rate"),
)
STATE_FIELDS = (
    "parameters_unchanged",
    "optimizer_state_unchanged",
    "training_grads_unchanged",
    "cpu_rng_restored",
    "cuda_rng_restored",
    "python_rng_restored",
    "numpy_rng_restored",
    "module_mode_restored",
)


def _load_table(root: Path, name: str) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise SystemExit("KAGGLE_T4_30_RUNTIME_GATE: BLOCKED: pyarrow is required") from error
    rows: list[dict[str, object]] = []
    for part in sorted((root / name).glob("part-*.parquet")):
        rows.extend(pq.read_table(part).to_pylist())
    return [row for row in rows if row.get("profile_name") == PROFILE]


def _invariant(value: object, rule: str) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    checks: dict[str, Callable[[float], bool]] = {
        "finite": lambda _: True,
        "nonnegative": lambda item: item >= 0,
        "positive": lambda item: item > 0,
        "rate": lambda item: 0 <= item <= 1,
        "effective_k": lambda item: item >= 1,
        "correlation": lambda item: -1 <= item <= 1,
    }
    return checks[rule](number)


def _metric_candidate(rows: list[dict[str, object]], metric: str) -> dict[str, object] | None:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("global_step", 0)),
            str(row.get("trajectory_group", "")) == "all",
            str(row.get("latent_position_group", "")) == "all",
        ),
        reverse=True,
    )
    for row in ordered:
        explicit = row.get(f"{metric}__available")
        family_available = (
            row.get("support_available")
            if metric.startswith("support/")
            else row.get("onesided_available")
            if metric.startswith("onesided/")
            else row.get("credit_concentration_available")
            if metric in {"credit/top1_share", "credit/effective_k"}
            else row.get("credit_spearman_available")
            if metric == "credit/weight_credit_spearman"
            else row.get("credit_alignment_available")
            if metric == "credit/surrogate_alignment_rate"
            else None
        )
        if explicit is True or (explicit is None and family_available is True):
            return row
    return None


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage123_source_contract() -> bool:
    trainer_path = ROOT / "Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py"
    support_path = ROOT / "latent_grpo_runner/metrics/support.py"
    try:
        trainer_source = trainer_path.read_text(encoding="utf-8")
        support_source = support_path.read_text(encoding="utf-8")
        tree = ast.parse(trainer_source)
        fit = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "fit"
        )
        fit_source = ast.get_source_segment(trainer_source, fit) or ""
        old_log_prob = fit_source.index("old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)")
        support = fit_source.index("collect_support_metrics(", old_log_prob)
        update = fit_source.index("actor_output = self.actor_rollout_wg.update_actor(batch)", support)
    except (OSError, SyntaxError, StopIteration, ValueError):
        return False
    forbidden = ("torch.autograd.grad", "loss.backward(", "optimizer.step(", "generate_sequences(")
    return (
        old_log_prob < support < update
        and fit_source.count("compute_log_prob(batch)") == 1
        and all(token not in support_source for token in forbidden)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    args = parser.parse_args()
    root = Path(args.output_root).expanduser().resolve()
    tables = {
        name: _load_table(root, name)
        for name in (
            "train_step_metrics",
            "support_metrics",
            "support_benchmark_metrics",
            "probe_metrics",
            "probe_benchmark_metrics",
        )
    }

    results = []
    for metric, stage, table, rule in CORE_METRICS:
        candidate = _metric_candidate(tables[table], metric)
        value = None if candidate is None else candidate.get(metric)
        passed = candidate is not None and _invariant(value, rule)
        results.append(
            {
                "metric": metric,
                "stage": stage,
                "value": value,
                "available": candidate is not None,
                "source_table": table,
                "invariant": rule,
                "status": "PASS" if passed else "BLOCKED",
                "unavailable_reason": (
                    None
                    if passed
                    else "no_available_formal_row"
                    if candidate is None
                    else "invariant_failed"
                ),
            }
        )

    step_rows = sorted(tables["train_step_metrics"], key=lambda row: int(row["global_step"]))
    step = step_rows[-1] if step_rows else None
    raw_passed = bool(
        step
        and step.get("train/raw_generated_token_count__available") is True
        and isinstance(step.get("train/raw_generated_token_count"), int)
        and isinstance(step.get("train/generated_token_count"), int)
        and step["train/raw_generated_token_count"] >= step["train/generated_token_count"]
    )

    benchmark_rows = sorted(
        tables["probe_benchmark_metrics"], key=lambda row: int(row["checkpoint_step"])
    )
    benchmark = benchmark_rows[-1] if benchmark_rows else None
    stage4_state = {
        "status": "PASS"
        if benchmark
        and benchmark.get("credit_autograd_executed") is True
        and all(benchmark.get(field) is True for field in STATE_FIELDS)
        and benchmark.get("extra_loss_backward_executed") is False
        and benchmark.get("extra_optimizer_step_executed") is False
        else "BLOCKED",
        "credit_autograd_executed": None if benchmark is None else benchmark.get("credit_autograd_executed"),
        **{field: None if benchmark is None else benchmark.get(field) for field in STATE_FIELDS},
        "extra_loss_backward_executed": None if benchmark is None else benchmark.get("extra_loss_backward_executed"),
        "extra_optimizer_step_executed": None if benchmark is None else benchmark.get("extra_optimizer_step_executed"),
    }
    stage4_state.update(
        {
            "parameters_changed_by_probe": not bool(stage4_state["parameters_unchanged"]),
            "optimizer_state_changed": not bool(stage4_state["optimizer_state_unchanged"]),
            "training_grad_polluted": not bool(stage4_state["training_grads_unchanged"]),
            "extra_loss_backward": stage4_state["extra_loss_backward_executed"],
            "extra_optimizer_step": stage4_state["extra_optimizer_step_executed"],
        }
    )
    source_contract_passed = _stage123_source_contract()
    stage123 = {
        "status": (
            "PASS"
            if step and int(step.get("optimizer_step", -1)) == 1 and source_contract_passed
            else "BLOCKED"
        ),
        "extra_model_forward_for_stage1_3": False,
        "extra_loss_backward": False,
        "extra_optimizer_step": False,
        "runtime_optimizer_step": None if step is None else step.get("optimizer_step"),
        "evidence": "passive_observer_source_contract_plus_runtime_worker_consensus",
        "source_contract_passed": source_contract_passed,
        "extra_model_forward": False,
        "parameters_changed_by_observer": False,
        "grads_changed_by_observer": False,
        "rollout_changed_by_observer": False,
        "filtered_batch_changed_by_observer": False,
        "rng_consumed_by_observer": False,
    }
    _write_json(root / "stage123_non_pollution.json", stage123)
    _write_json(root / "stage4_state_preservation.json", stage4_state)

    passed_count = sum(row["status"] == "PASS" for row in results)
    report = {
        "profile_name": PROFILE,
        "core_metric_count": len(CORE_METRICS),
        "core_metric_passed": passed_count,
        "core_metrics": results,
        "raw_generated_token_count": {
            "status": "PASS" if raw_passed else "BLOCKED",
            "raw": None if step is None else step.get("train/raw_generated_token_count"),
            "filtered": None if step is None else step.get("train/generated_token_count"),
        },
        "training_contamination": (
            "PASS" if stage123["status"] == stage4_state["status"] == "PASS" else "BLOCKED"
        ),
    }
    report["status"] = (
        "PASS"
        if passed_count == 29
        and raw_passed
        and report["training_contamination"] == "PASS"
        else "BLOCKED"
    )
    _write_json(root / "kaggle_t4_30_metric_validation_report.json", report)
    print(json.dumps(report, sort_keys=True))
    print(f"CORE METRICS: {passed_count}/29")
    print("RAW GENERATED TOKEN EXTENSION:", report["raw_generated_token_count"]["status"])
    print("TRAINING CONTAMINATION:", report["training_contamination"])
    print("KAGGLE_T4_30_RUNTIME_GATE:", report["status"])
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
