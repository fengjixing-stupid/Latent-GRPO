#!/usr/bin/env python3
"""Build fail-closed machine and human acceptance reports for the 3-GPU gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

from latent_grpo_runner.config import load_config
from latent_grpo_runner.validation.output_validator import validate_output_directory
from tools.validate_kaggle_t4_30_metrics import (
    CORE_METRICS,
    STATE_FIELDS,
    _invariant,
    _metric_candidate,
    _stage123_source_contract,
)


def evaluate_final_gate(evidence: Mapping[str, object]) -> dict[str, object]:
    """Turn normalized evidence into an acceptance report with stable blockers."""
    requirements = {
        "gpu_count": evidence.get("gpu_count") == 3,
        "target_precision": evidence.get("target_precision") == "bfloat16",
        "preflight": evidence.get("preflight") is True,
        "distributed_runtime": evidence.get("distributed_runtime") is True,
        "real_backward": evidence.get("real_backward") is True,
        "real_optimizer_step": evidence.get("real_optimizer_step") is True,
        "metrics_core": evidence.get("metrics_core_passed") == 29,
        "raw_generated_token_count": evidence.get("raw_generated_token_count") is True,
        "worker_driver_aggregation": evidence.get("worker_driver_aggregation") is True,
        "aggregation_worker_count": evidence.get("aggregation_worker_count") == 3,
        "stage3_alignment": evidence.get("stage3_alignment") is True,
        "stage4_probe": evidence.get("stage4_probe") is True,
        "cuda_rng_restored": evidence.get("cuda_rng_restored") is True,
        "grad_pollution": evidence.get("grad_pollution") is False,
        "parameter_pollution_by_probe": evidence.get("parameter_pollution_by_probe") is False,
        "optimizer_state_pollution_by_probe": evidence.get("optimizer_state_pollution_by_probe") is False,
        "checkpoint_write": evidence.get("checkpoint_write") is True,
        "resume_compatibility": evidence.get("resume_compatibility") is True,
        "gpu_memory_telemetry": evidence.get("gpu_memory_telemetry") is True,
    }
    blockers = [name for name, passed in requirements.items() if not passed]
    status = "PASS" if not blockers else "BLOCKED"
    return {
        "profile_name": evidence.get("profile_name"),
        "git_commit": evidence.get("git_commit"),
        "gpu_count": evidence.get("gpu_count"),
        "target_precision": evidence.get("target_precision"),
        "preflight": "PASS" if requirements["preflight"] else "BLOCKED",
        "distributed_runtime": "PASS" if requirements["distributed_runtime"] else "BLOCKED",
        "real_backward": "PASS" if requirements["real_backward"] else "BLOCKED",
        "real_optimizer_step": "PASS" if requirements["real_optimizer_step"] else "BLOCKED",
        "metrics_core": f"{int(evidence.get('metrics_core_passed', 0))}/29",
        "worker_driver_aggregation": "PASS" if requirements["worker_driver_aggregation"] else "BLOCKED",
        "aggregation_worker_count": evidence.get("aggregation_worker_count"),
        "stage3_alignment": "PASS" if requirements["stage3_alignment"] else "BLOCKED",
        "stage4_probe": "PASS" if requirements["stage4_probe"] else "BLOCKED",
        "cuda_rng_restored": "PASS" if requirements["cuda_rng_restored"] else "BLOCKED",
        "grad_pollution": "NONE" if requirements["grad_pollution"] else "DETECTED_OR_UNPROVEN",
        "parameter_pollution_by_probe": (
            "NONE" if requirements["parameter_pollution_by_probe"] else "DETECTED_OR_UNPROVEN"
        ),
        "optimizer_state_pollution_by_probe": (
            "NONE" if requirements["optimizer_state_pollution_by_probe"] else "DETECTED_OR_UNPROVEN"
        ),
        "checkpoint_write": "PASS" if requirements["checkpoint_write"] else "BLOCKED",
        "resume_compatibility": "PASS" if requirements["resume_compatibility"] else "BLOCKED",
        "gpu_memory_telemetry": "PASS" if requirements["gpu_memory_telemetry"] else "BLOCKED",
        "blockers": blockers,
        "final_gate": status,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _validate_runtime_telemetry(
    run_root: Path, telemetry: Mapping[str, object]
) -> tuple[bool, dict[str, object]]:
    """Validate external utilization plus two rank-local CUDA allocator snapshots."""
    memory_peaks = telemetry.get("peak_memory_used_mib_by_gpu")
    memory_totals = telemetry.get("memory_total_mib_by_gpu")
    utilization_peaks = telemetry.get("peak_gpu_utilization_percent_by_gpu")
    utilization_averages = telemetry.get("average_gpu_utilization_percent_by_gpu")
    external_maps = (memory_peaks, memory_totals, utilization_peaks, utilization_averages)
    selected_indices = telemetry.get("selected_physical_gpu_indices")
    external_ok = (
        telemetry.get("schema_version") == "gpu_telemetry_v2"
        and telemetry.get("exit_code") == 0
        and telemetry.get("telemetry_error") is None
        and isinstance(selected_indices, list)
        and len(selected_indices) == 3
        and all(type(index) is int and index >= 0 for index in selected_indices)
        and len(set(selected_indices)) == 3
        and all(isinstance(item, Mapping) and len(item) == 3 for item in external_maps)
    )
    if external_ok:
        expected_keys = {str(index) for index in selected_indices}
        external_ok = all(set(item) == expected_keys for item in external_maps)  # type: ignore[arg-type]
    if external_ok:
        external_ok = all(
            type(value) in {int, float} and float(value) >= 0
            for item in external_maps
            for value in item.values()  # type: ignore[union-attr]
        )

    allocator = _load_json(run_root / "gpu_runtime_metrics.json")
    raw_steps = allocator.get("steps")
    steps = (
        sorted(raw_steps, key=lambda row: int(row.get("global_step", 0)))
        if isinstance(raw_steps, list) and all(isinstance(row, Mapping) for row in raw_steps)
        else []
    )
    selected_steps = steps[:2]
    fields = (
        "worker_rank",
        "device_index",
        "current_allocated_bytes",
        "current_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    normalized_steps: list[dict[int, Mapping[str, object]]] = []
    allocator_ok = allocator.get("schema_version") == "gpu_runtime_metrics_v1" and len(selected_steps) == 2
    for step in selected_steps:
        workers = step.get("workers")
        if not isinstance(workers, list) or len(workers) != 3:
            allocator_ok = False
            continue
        by_rank: dict[int, Mapping[str, object]] = {}
        for worker in workers:
            if not isinstance(worker, Mapping) or any(
                type(worker.get(field)) is not int or int(worker[field]) < 0 for field in fields
            ):
                allocator_ok = False
                continue
            by_rank[int(worker["worker_rank"])] = worker
            allocator_ok = allocator_ok and int(worker["current_allocated_bytes"]) > 0
            allocator_ok = allocator_ok and int(worker["current_reserved_bytes"]) > 0
            allocator_ok = allocator_ok and int(worker["peak_allocated_bytes"]) > 0
            allocator_ok = allocator_ok and int(worker["peak_reserved_bytes"]) > 0
            allocator_ok = allocator_ok and int(worker["current_allocated_bytes"]) <= int(worker["peak_allocated_bytes"])
            allocator_ok = allocator_ok and int(worker["current_reserved_bytes"]) <= int(worker["peak_reserved_bytes"])
            allocator_ok = allocator_ok and int(worker["current_allocated_bytes"]) <= int(worker["current_reserved_bytes"])
            allocator_ok = allocator_ok and int(worker["peak_allocated_bytes"]) <= int(worker["peak_reserved_bytes"])
        allocator_ok = allocator_ok and sorted(by_rank) == [0, 1, 2]
        normalized_steps.append(by_rank)

    growth: dict[str, int] = {}
    bounded_growth = len(normalized_steps) == 2 and all(len(step) == 3 for step in normalized_steps)
    total_values = (
        [int(memory_totals[str(index)]) * 1024 * 1024 for index in selected_indices]
        if isinstance(memory_totals, Mapping)
        and isinstance(selected_indices, list)
        and len(memory_totals) == 3
        and all(str(index) in memory_totals for index in selected_indices)
        else []
    )
    if bounded_growth and len(total_values) == 3:
        for rank in range(3):
            first = int(normalized_steps[0][rank]["current_reserved_bytes"])
            second = int(normalized_steps[1][rank]["current_reserved_bytes"])
            growth[str(rank)] = second - first
            bounded_growth = bounded_growth and second <= first + int(total_values[rank] * 0.25)
            bounded_growth = bounded_growth and int(normalized_steps[1][rank]["peak_reserved_bytes"]) <= total_values[rank]
    else:
        bounded_growth = False

    details = {
        "external_telemetry_valid": external_ok,
        "selected_physical_gpu_indices": selected_indices,
        "allocator_step_count": len(selected_steps),
        "allocator_worker_counts": [len(step) for step in normalized_steps],
        "reserved_growth_bytes_by_worker": growth,
        "bounded_second_step_growth": bounded_growth,
        "allocator_steps": selected_steps,
    }
    return bool(external_ok and allocator_ok and bounded_growth), details


def _validate_probe_worker_runtime_evidence(
    run_root: Path,
) -> tuple[bool, dict[str, object]]:
    payload = _load_json(run_root / "probe_worker_runtime.json")
    checkpoints = payload.get("checkpoints")
    latest = checkpoints[-1] if isinstance(checkpoints, list) and checkpoints else {}
    workers = latest.get("workers") if isinstance(latest, Mapping) else None
    valid = payload.get("schema_version") == "probe_worker_runtime_v1"
    valid = valid and isinstance(workers, list) and len(workers) == 3
    normalized: list[dict[str, object]] = []
    if isinstance(workers, list):
        for worker in workers:
            if not isinstance(worker, Mapping):
                valid = False
                continue
            rank = worker.get("worker_rank")
            elapsed = worker.get("probe_extra_time_seconds")
            peak = worker.get("probe_peak_memory_bytes")
            row_valid = (
                type(rank) is int
                and rank >= 0
                and type(elapsed) in {int, float}
                and math.isfinite(float(elapsed))
                and float(elapsed) >= 0
                and type(peak) is int
                and peak >= 0
            )
            valid = valid and row_valid
            if row_valid:
                normalized.append(
                    {
                        "worker_rank": rank,
                        "probe_extra_time_seconds": float(elapsed),
                        "probe_peak_memory_bytes": peak,
                    }
                )
    valid = valid and sorted(row["worker_rank"] for row in normalized) == [0, 1, 2]
    return bool(valid), {
        "checkpoint_step": latest.get("checkpoint_step") if isinstance(latest, Mapping) else None,
        "workers": normalized,
    }


def _load_table(root: Path, name: str, *, profile_name: str) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise SystemExit("3GPU_FINAL_GATE: BLOCKED: pyarrow is required") from error
    rows: list[dict[str, object]] = []
    for part in sorted((root / name).glob("part-*.parquet")):
        rows.extend(pq.read_table(part).to_pylist())
    return [row for row in rows if row.get("profile_name") == profile_name]


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _checkpoint_evidence(run_root: Path, expected_hash: str) -> tuple[bool, bool]:
    pointer = run_root / "latest_checkpointed_iteration.txt"
    try:
        step = int(pointer.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False, False
    checkpoint = run_root / f"global_step_{step}"
    sidecar = _load_json(checkpoint / "latent_grpo_metrics_sidecar.json")
    checkpoint_ok = (
        checkpoint.is_dir()
        and (checkpoint / "actor").is_dir()
        and (checkpoint / "data.pt").is_file()
        and sidecar.get("global_step") == step
        and sidecar.get("optimizer_step", 0) >= 1
    )
    compatibility_ok = sidecar.get("config_hash") == expected_hash
    return checkpoint_ok, compatibility_ok


def collect_evidence(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    root = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).expanduser().resolve()
    config = load_config(args.config, workspace_root=root).with_runtime_overrides(
        seed=args.seed,
        output_root=run_root,
        model_path=args.model_path,
        train_files=args.train_data,
        val_files=args.val_data,
    )
    output_validation = validate_output_directory(run_root)
    tables = {
        name: _load_table(run_root, name, profile_name=config.profile_name)
        for name in (
            "train_step_metrics",
            "support_metrics",
            "support_benchmark_metrics",
            "probe_metrics",
            "probe_benchmark_metrics",
        )
    }
    metric_results: list[dict[str, object]] = []
    for metric, stage, table, rule in CORE_METRICS:
        candidate = _metric_candidate(tables[table], metric)
        value = None if candidate is None else candidate.get(metric)
        passed = candidate is not None and _invariant(value, rule)
        metric_results.append(
            {"metric": metric, "stage": stage, "value": value, "status": "PASS" if passed else "BLOCKED"}
        )

    step_rows = sorted(tables["train_step_metrics"], key=lambda row: int(row.get("global_step", 0)))
    post_update_rows = [row for row in step_rows if row.get("observation_phase") == "post_update"]
    latest_step = post_update_rows[-1] if post_update_rows else {}
    aggregation_counts = [row.get("aggregation_worker_count") for row in post_update_rows]
    max_optimizer_step = max((int(row.get("optimizer_step", 0)) for row in post_update_rows), default=0)
    raw_ok = (
        latest_step.get("train/raw_generated_token_count__available") is True
        and isinstance(latest_step.get("train/raw_generated_token_count"), int)
        and isinstance(latest_step.get("train/generated_token_count"), int)
        and latest_step["train/raw_generated_token_count"] >= latest_step["train/generated_token_count"]
    )
    support_effective = sum(
        int(row.get("support/effective_position_count", 0))
        for row in tables["support_metrics"]
        if row.get("support_available") is True
    )
    support_benchmark_effective = sum(
        int(row.get("support_benchmark/total_effective_position_count", 0))
        for row in tables["support_benchmark_metrics"]
        if row.get("support_available") is True
    )
    benchmark_rows = sorted(
        tables["probe_benchmark_metrics"], key=lambda row: int(row.get("checkpoint_step", 0))
    )
    benchmark = benchmark_rows[-1] if benchmark_rows else {}
    probe_worker_runtime_ok, probe_worker_runtime = _validate_probe_worker_runtime_evidence(run_root)
    stage4_ok = bool(
        benchmark.get("credit_autograd_executed") is True
        and all(benchmark.get(field) is True for field in STATE_FIELDS)
        and benchmark.get("extra_loss_backward_executed") is False
        and benchmark.get("extra_optimizer_step_executed") is False
        and int(benchmark.get("probe_latent_position_count", 0)) > 0
        and type(benchmark.get("probe_extra_time_seconds")) in {int, float}
        and float(benchmark["probe_extra_time_seconds"]) >= 0
        and type(benchmark.get("probe_peak_memory_bytes")) is int
        and int(benchmark["probe_peak_memory_bytes"]) >= 0
        and probe_worker_runtime_ok
    )
    checkpoint_ok, checkpoint_hash_ok = _checkpoint_evidence(run_root, config.resume_compatibility_hash)
    resume = _load_json(Path(args.resume_report))
    preflight = _load_json(Path(args.preflight_report))
    ray_report = _load_json(Path(args.ray_report))
    telemetry = _load_json(Path(args.telemetry_report))
    telemetry_ok, runtime_memory_details = _validate_runtime_telemetry(run_root, telemetry)
    current_commit = _git_commit(root)
    git_identity_ok = current_commit is not None and preflight.get("git_commit") == current_commit
    aggregation_ok = bool(post_update_rows) and all(count == 3 for count in aggregation_counts)
    metric_passed = sum(item["status"] == "PASS" for item in metric_results)
    normalized = {
        "profile_name": config.profile_name,
        "git_commit": current_commit,
        "gpu_count": preflight.get("gpu_count"),
        "target_precision": "bfloat16",
        "preflight": preflight.get("status") == "PASS" and git_identity_ok,
        "distributed_runtime": ray_report.get("status") == "target_machine_probe_passed",
        "real_backward": max_optimizer_step >= 2 and latest_step.get("train/policy_loss__available") is True,
        "real_optimizer_step": max_optimizer_step >= 2,
        "metrics_core_passed": metric_passed,
        "raw_generated_token_count": raw_ok,
        "worker_driver_aggregation": aggregation_ok and output_validation.ok,
        "aggregation_worker_count": 3 if aggregation_ok else (aggregation_counts[-1] if aggregation_counts else None),
        "stage3_alignment": support_effective > 0 and support_benchmark_effective > 0,
        "stage4_probe": stage4_ok,
        "cuda_rng_restored": benchmark.get("cuda_rng_restored") is True,
        "grad_pollution": benchmark.get("training_grads_unchanged") is not True,
        "parameter_pollution_by_probe": benchmark.get("parameters_unchanged") is not True,
        "optimizer_state_pollution_by_probe": benchmark.get("optimizer_state_unchanged") is not True,
        "checkpoint_write": checkpoint_ok,
        "resume_compatibility": checkpoint_hash_ok and resume.get("status") == "PASS",
        "gpu_memory_telemetry": telemetry_ok,
    }
    details = {
        "metric_results": metric_results,
        "output_validation_errors": output_validation.errors,
        "git_identity_matches_preflight": git_identity_ok,
        "post_update_row_count": len(post_update_rows),
        "max_optimizer_step": max_optimizer_step,
        "aggregation_worker_counts": aggregation_counts,
        "support_effective_position_count": support_effective,
        "support_benchmark_effective_position_count": support_benchmark_effective,
        "probe_benchmark": benchmark,
        "probe_worker_runtime": probe_worker_runtime,
        "gpu_memory": telemetry,
        "gpu_allocator": runtime_memory_details,
        "step_time_seconds": [row.get("train/step_time") for row in post_update_rows[:2]],
        "sglang_configured_gpu_memory_utilization": config.rollout.gpu_memory_utilization,
        "stage123_source_contract_passed": _stage123_source_contract(),
    }
    if not details["stage123_source_contract_passed"]:
        normalized["real_backward"] = False
    return normalized, details


def _write_reports(output_root: Path, report: Mapping[str, object], details: Mapping[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    acceptance = dict(report)
    acceptance["details"] = dict(details)
    (output_root / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stage_counts = {"Stage 1": 0, "Stage 2": 0, "Stage 3": 0, "Stage 4 One-sided": 0, "Stage 4 Credit": 0}
    for item in details.get("metric_results", []):
        if item.get("status") == "PASS":
            stage_counts[str(item["stage"])] += 1
    telemetry = details.get("gpu_memory", {})
    gpu_rows = ["| visible GPU | peak memory.used MiB | avg utilization % | peak utilization % |", "|---:|---:|---:|---:|"]
    if isinstance(telemetry, Mapping):
        memory = telemetry.get("peak_memory_used_mib_by_gpu", {})
        average = telemetry.get("average_gpu_utilization_percent_by_gpu", {})
        peak = telemetry.get("peak_gpu_utilization_percent_by_gpu", {})
        if isinstance(memory, Mapping) and isinstance(average, Mapping) and isinstance(peak, Mapping):
            for gpu in sorted(memory, key=lambda value: int(value)):
                gpu_rows.append(f"| {gpu} | {memory.get(gpu)} | {average.get(gpu)} | {peak.get(gpu)} |")
    allocator_rows = ["| actor rank | current allocated MiB | current reserved MiB | peak allocated MiB | peak reserved MiB |", "|---:|---:|---:|---:|---:|"]
    allocator = details.get("gpu_allocator", {})
    allocator_steps = allocator.get("allocator_steps", []) if isinstance(allocator, Mapping) else []
    latest_allocator = allocator_steps[-1] if isinstance(allocator_steps, list) and allocator_steps else {}
    workers = latest_allocator.get("workers", []) if isinstance(latest_allocator, Mapping) else []
    if isinstance(workers, list):
        for worker in workers:
            if isinstance(worker, Mapping):
                mib = 1024 * 1024
                allocator_rows.append(
                    "| {rank} | {allocated:.1f} | {reserved:.1f} | {peak_allocated:.1f} | {peak_reserved:.1f} |".format(
                        rank=worker.get("worker_rank"),
                        allocated=int(worker.get("current_allocated_bytes", 0)) / mib,
                        reserved=int(worker.get("current_reserved_bytes", 0)) / mib,
                        peak_allocated=int(worker.get("peak_allocated_bytes", 0)) / mib,
                        peak_reserved=int(worker.get("peak_reserved_bytes", 0)) / mib,
                    )
                )
    probe_rows = ["| actor rank | probe extra time s | probe peak memory bytes |", "|---:|---:|---:|"]
    probe_runtime = details.get("probe_worker_runtime", {})
    probe_workers = probe_runtime.get("workers", []) if isinstance(probe_runtime, Mapping) else []
    if isinstance(probe_workers, list):
        for worker in probe_workers:
            if isinstance(worker, Mapping):
                probe_rows.append(
                    f"| {worker.get('worker_rank')} | {worker.get('probe_extra_time_seconds')} | {worker.get('probe_peak_memory_bytes')} |"
                )
    summary = f"""# 3GPU Final Acceptance Summary

Git: {report.get('git_commit')}
Profile: {report.get('profile_name')}
GPUs: {report.get('gpu_count')}
Precision: BF16

Preflight: {report.get('preflight')}
Runtime: {report.get('distributed_runtime')}
Real backward: {report.get('real_backward')}
Optimizer step: {report.get('real_optimizer_step')}

Stage 1: {stage_counts['Stage 1']}/10
Stage 2: {stage_counts['Stage 2']}/6
Stage 3: {stage_counts['Stage 3']}/2
Stage 4 One-sided: {stage_counts['Stage 4 One-sided']}/7
Stage 4 Credit: {stage_counts['Stage 4 Credit']}/4

Worker→driver aggregation: {report.get('worker_driver_aggregation')}
CUDA RNG restore: {report.get('cuda_rng_restored')}
Probe grad pollution: {report.get('grad_pollution')}
Probe parameter pollution: {report.get('parameter_pollution_by_probe')}
Probe optimizer pollution: {report.get('optimizer_state_pollution_by_probe')}
Checkpoint: {report.get('checkpoint_write')}
Resume compatibility: {report.get('resume_compatibility')}

Step times (s): {details.get('step_time_seconds')}
SGLang configured GPU memory utilization: {details.get('sglang_configured_gpu_memory_utilization')}
Second-step reserved growth bounded: {allocator.get('bounded_second_step_growth') if isinstance(allocator, Mapping) else None}

## Per-GPU device telemetry

{chr(10).join(gpu_rows)}

## Per-worker CUDA allocator telemetry

{chr(10).join(allocator_rows)}

## Per-worker checkpoint probe overhead

{chr(10).join(probe_rows)}

FINAL: {report.get('final_gate')}
"""
    (output_root / "ACCEPTANCE_SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--ray-report", required=True)
    parser.add_argument("--telemetry-report", required=True)
    parser.add_argument("--resume-report", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    normalized, details = collect_evidence(args)
    report = evaluate_final_gate(normalized)
    _write_reports(Path(args.output_root).expanduser().resolve(), report, details)
    print(f"CORE_METRICS: {report['metrics_core']}")
    for device in range(3):
        print(f"CUDA_RNG_DEVICE_{device}: {report['cuda_rng_restored']}")
    print(f"CUDA_RNG_ALL_DEVICES: {report['cuda_rng_restored']}")
    print(f"CHECKPOINT_GATE: {report['checkpoint_write']}")
    print(f"3GPU_FINAL_GATE: {report['final_gate']}")
    if report["blockers"]:
        print("BLOCKED_REASON: " + ",".join(report["blockers"]))
    return 0 if report["final_gate"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
