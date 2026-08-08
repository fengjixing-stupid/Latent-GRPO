#!/usr/bin/env python3
"""Probe whether the current Git checkout can enter a dual-T4 P1 smoke.

The probe is read-only with respect to model/training state and training data.
It intentionally accepts no train/validation dataset arguments.  If the current
implementation is incompatible with T4, it writes a BLOCKED report and exits
non-zero before any data is requested or any training entrypoint is launched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_grpo_runner.environment import collect_environment
from latent_grpo_runner.validation.kaggle_t4_compatibility import assess_kaggle_t4_compatibility


ACTOR_PATH = ROOT / "Latent-GRPO" / "verl-0.4.x" / "verl" / "workers" / "actor" / "dp_actor.py"
FSDP_WORKER_PATH = ROOT / "Latent-GRPO" / "verl-0.4.x" / "verl" / "workers" / "fsdp_workers.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="/kaggle/working/latent-grpo-p1-t4-compatibility.json",
        help="Machine-readable compatibility report path",
    )
    return parser.parse_args(argv)


def _read_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"required source file is unreadable: {path}: {error}") from error


def _git_identity() -> dict[str, object]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return {
        "commit": completed.stdout.strip() if completed.returncode == 0 else None,
        "git_identity_available": completed.returncode == 0,
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = Path(args.report).expanduser().resolve()
    try:
        actor_source = _read_source(ACTOR_PATH)
        fsdp_source = _read_source(FSDP_WORKER_PATH)
        environment = collect_environment(
            mode="target",
            require_gpus=2,
            min_vram_gb=14,
            workspace_root=ROOT,
        )
        report = assess_kaggle_t4_compatibility(
            environment,
            actor_hardcodes_bfloat16=(
                "dtype=torch.bfloat16" in actor_source or ".to(torch.bfloat16)" in actor_source
            ),
            model_forces_flash_attention_2=(
                'attn_implementation="flash_attention_2"' in fsdp_source
            ),
        )
        report.update(
            {
                "environment_status": environment.get("status"),
                "environment_failure_reasons": environment.get("failure_reasons", []),
                "source_checks": {
                    "actor_path": str(ACTOR_PATH.relative_to(ROOT)),
                    "fsdp_worker_path": str(FSDP_WORKER_PATH.relative_to(ROOT)),
                    "actor_hardcodes_bfloat16": (
                        "dtype=torch.bfloat16" in actor_source or ".to(torch.bfloat16)" in actor_source
                    ),
                    "model_forces_flash_attention_2": (
                        'attn_implementation="flash_attention_2"' in fsdp_source
                    ),
                },
                **_git_identity(),
            }
        )
    except Exception as error:
        report = {
            "status": "BLOCKED",
            "gate": "kaggle_dual_t4_p1_compatibility",
            "training_started": False,
            "training_data_required_now": False,
            "training_data_inspected": False,
            "training_data_generated": False,
            "blockers": [f"compatibility_probe_error:{error}"],
            "next_gate": "fix_probe_or_runtime_before_requesting_data",
            **_git_identity(),
        }

    _write_report(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "READY_FOR_DATA" else 3


if __name__ == "__main__":
    raise SystemExit(main())
