"""Target-only Ray GPU placement and error-propagation probe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_grpo_runner.environment import build_report_envelope


def collect_ray_placement_evidence(ray: Any, *, num_gpus: int) -> dict[str, Any]:
    """Submit one-GPU workers, then prove placement and driver error handling."""
    driver_gpu_ids = list(ray.get_gpu_ids())

    def worker_binding(worker_index: int) -> dict[str, Any]:
        return {
            "worker_index": worker_index,
            "ray_gpu_ids": list(ray.get_gpu_ids()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    one_gpu_worker = ray.remote(num_gpus=1)(worker_binding)
    worker_bindings = ray.get([one_gpu_worker.remote(index) for index in range(num_gpus)])
    assigned_ids = [binding["ray_gpu_ids"] for binding in worker_bindings]
    flattened_ids = [str(ids[0]) for ids in assigned_ids if len(ids) == 1]
    binding_validation_passed = (
        not driver_gpu_ids
        and len(worker_bindings) == num_gpus
        and len(flattened_ids) == num_gpus
        and len(set(flattened_ids)) == num_gpus
    )

    def intentional_worker_failure() -> None:
        raise RuntimeError("ray_probe_intentional_worker_failure")

    failing_worker = ray.remote(num_gpus=1)(intentional_worker_failure)
    worker_exception_propagated = False
    worker_exception_type: str | None = None
    try:
        ray.get(failing_worker.remote())
    except Exception as error:
        worker_exception_propagated = True
        worker_exception_type = type(error).__name__
    return {
        "driver_ray_gpu_ids": driver_gpu_ids,
        "driver_gpu_usage_empty": not driver_gpu_ids,
        "worker_bindings": worker_bindings,
        "unique_worker_gpu_ids": sorted(set(flattened_ids)),
        "binding_validation_passed": binding_validation_passed,
        "worker_exception_propagated": worker_exception_propagated,
        "worker_exception_type": worker_exception_type,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--output", default="artifacts/target_machine/ray_gpu_probe.json")
    args = parser.parse_args(argv)
    started_at = _timestamp()
    command = [sys.executable, "scripts/probe_ray_distributed.py", "--num-gpus", str(args.num_gpus)]
    output = Path(args.output)
    evidence: dict[str, Any] = {"requested_gpus": args.num_gpus}
    failure_reason: str | None = None
    exit_code = 1
    ray = None
    try:
        import ray as imported_ray

        ray = imported_ray
        ray.init(ignore_reinit_error=False, include_dashboard=False)
        available = int(ray.cluster_resources().get("GPU", 0))
        evidence["ray_gpu_resources"] = available
        if available < args.num_gpus:
            failure_reason = "ray_gpu_count_below_requirement"
        else:
            evidence.update(collect_ray_placement_evidence(ray, num_gpus=args.num_gpus))
            if not evidence["binding_validation_passed"]:
                failure_reason = "ray_gpu_binding_validation_failed"
            elif not evidence["worker_exception_propagated"]:
                failure_reason = "ray_worker_exception_not_propagated"
            else:
                exit_code = 0
    except ModuleNotFoundError:
        failure_reason = "ray_missing"
    except Exception as error:
        failure_reason = type(error).__name__
    finally:
        if ray is not None and ray.is_initialized():
            ray.shutdown()
    status = "target_machine_probe_passed" if exit_code == 0 else "blocked"
    report = build_report_envelope(
        command=command,
        started_at=started_at,
        finished_at=_timestamp(),
        exit_code=exit_code,
        status=status,
        environment_summary=evidence,
        stdout_log_path=None,
        stderr_log_path=None,
        artifacts=[str(output)],
        failure_reason=failure_reason,
    )
    _write(output, report)
    print(json.dumps(report, sort_keys=True))
    return exit_code


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
