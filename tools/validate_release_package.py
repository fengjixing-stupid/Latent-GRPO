#!/usr/bin/env python3
"""Validate the distributable package without claiming target-GPU execution.

This gate covers repository integrity, source syntax, unit tests, dependency-pin
consistency, and final-profile dry-runs. CUDA/NCCL/SGLang/FSDP execution remains
an explicit target-machine gate handled by tools/run_3gpu_final_validation.sh.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import warnings
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIGS = (
    "configs/3gpu-final-validation.yaml",
    "configs/3gpu-final-low.yaml",
    "configs/3gpu-final-high-validation.yaml",
    "configs/3gpu-final-high.yaml",
)
REQUIRED_FILES = (
    "FINAL_EXPERIMENT_PACKAGE.md",
    "PACKAGE_VERSION",
    "docs/3GPU_RUNBOOK.md",
    "docs/FINAL_PACKAGE_FIXES.md",
    "docs/3GPU_ACCEPTANCE_CHECKLIST.md",
    "docs/AUTHOR_HYPERPARAMETER_AUDIT.md",
    "docs/3GPU_HYPERPARAMETER_DEVIATIONS.md",
    "docs/PROJECT_TECHNICAL_HANDOFF.md",
    "tools/run_3gpu_preflight.sh",
    "tools/run_3gpu_final_validation.sh",
    "tools/run_3gpu_training.sh",
    "tools/validate_3gpu_final.py",
    "scripts/target_machine/run_with_gpu_telemetry.py",
    "scripts/target_machine/install_runtime.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(name: str, command: Sequence[str], *, cwd: Path = ROOT) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "name": name,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "command": list(command),
        "exit_code": completed.returncode,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def source_syntax_check() -> dict[str, object]:
    started = time.perf_counter()
    failures: list[str] = []
    count = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        for path in sorted(ROOT.rglob("*.py")):
            if ".git" in path.parts or ".venv-target" in path.parts:
                continue
            count += 1
            try:
                compile(path.read_bytes(), str(path), "exec")
            except (OSError, SyntaxError, UnicodeError) as error:
                failures.append(f"{path.relative_to(ROOT)}:{type(error).__name__}:{error}")
    return {
        "name": "python_source_syntax",
        "status": "PASS" if not failures else "FAIL",
        "files_checked": count,
        "failures": failures,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def shell_syntax_check() -> dict[str, object]:
    started = time.perf_counter()
    failures: list[dict[str, object]] = []
    count = 0
    for path in sorted(ROOT.rglob("*.sh")):
        if ".git" in path.parts:
            continue
        count += 1
        completed = subprocess.run(
            ["bash", "-n", str(path)], text=True, capture_output=True, check=False
        )
        if completed.returncode:
            failures.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "exit_code": completed.returncode,
                    "stderr": completed.stderr[-4000:],
                }
            )
    return {
        "name": "shell_source_syntax",
        "status": "PASS" if not failures else "FAIL",
        "files_checked": count,
        "failures": failures,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def required_files_check() -> dict[str, object]:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    return {
        "name": "required_release_files",
        "status": "PASS" if not missing else "FAIL",
        "required_count": len(REQUIRED_FILES),
        "missing": missing,
    }


def tracked_cruft_check() -> dict[str, object]:
    completed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    tracked = completed.stdout.splitlines() if completed.returncode == 0 else []
    cruft = [
        name
        for name in tracked
        if name.endswith(".DS_Store")
        or "/._" in name
        or name.startswith("._")
        or "__MACOSX/" in name
        or "/__pycache__/" in name
        or name.endswith(".pyc")
    ]
    return {
        "name": "tracked_packaging_cruft",
        "status": "PASS" if completed.returncode == 0 and not cruft else "FAIL",
        "git_exit_code": completed.returncode,
        "tracked_cruft": cruft,
        "stderr": completed.stderr[-4000:],
    }


def dependency_consistency_check() -> dict[str, object]:
    paths = {
        "constraint": ROOT / "constraints/linux-cu124-py311.txt",
        "runtime": ROOT / "requirements/runtime-sglang.txt",
        "metadata": ROOT / "Latent-GRPO/sglang_latent_reasoning_pkg/python/pyproject.toml",
        "installer": ROOT / "scripts/target_machine/install_runtime.py",
    }
    text = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    assertions = {
        "constraint_flashinfer_0_2_5": "flashinfer-python==0.2.5" in text["constraint"],
        "runtime_flashinfer_0_2_5": "flashinfer-python==0.2.5" in text["runtime"],
        "metadata_flashinfer_0_2_5": "flashinfer_python==0.2.5" in text["metadata"],
        "runtime_sgl_kernel_0_1_1": "sgl-kernel==0.1.1" in text["runtime"],
        "metadata_sgl_kernel_0_1_1": "sgl-kernel==0.1.1" in text["metadata"],
        "installer_expected_sgl_kernel": 'EXPECTED_SGL_KERNEL = "0.1.1"' in text["installer"],
        "installer_expected_flashinfer": 'EXPECTED_FLASHINFER = "0.2.5"' in text["installer"],
        "installer_pip_check": 'run("check")' in text["installer"],
        "no_active_flashinfer_0_2_3": all("0.2.3" not in value for value in text.values()),
    }
    return {
        "name": "active_dependency_pin_consistency",
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "assertions": assertions,
    }


def git_identity_check() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    clean = commit.returncode == 0 and status.returncode == 0 and not status.stdout.strip()
    return {
        "name": "git_identity_and_clean_tree",
        "status": "PASS" if clean else "FAIL",
        "git_commit": commit.stdout.strip() or None,
        "working_tree_entries": status.stdout.splitlines(),
        "stderr": (commit.stderr + status.stderr)[-4000:],
    }


def config_dry_runs() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="latent-grpo-release-") as temporary:
        temp_root = Path(temporary)
        for config in FINAL_CONFIGS:
            output = temp_root / Path(config).stem
            results.append(
                run_command(
                    f"config_dry_run:{config}",
                    [
                        sys.executable,
                        "train_latent_grpo.py",
                        "--config",
                        config,
                        "--output-root",
                        str(output),
                        "--dry-run",
                        "--validate-config",
                    ],
                )
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "release_validation" / "LOCAL_RELEASE_ACCEPTANCE.json"),
    )
    args = parser.parse_args()
    started_at = utc_now()

    checks: list[dict[str, object]] = [
        required_files_check(),
        tracked_cruft_check(),
        dependency_consistency_check(),
        source_syntax_check(),
        shell_syntax_check(),
        run_command("unit_tests", [sys.executable, "-m", "pytest", "-q"]),
        *config_dry_runs(),
        # Run this after tests/dry-runs so ignored outputs cannot hide a source-tree mutation.
        git_identity_check(),
    ]
    passed = all(check.get("status") == "PASS" for check in checks)
    git_check = next(check for check in checks if check.get("name") == "git_identity_and_clean_tree")
    report = {
        "schema_version": "latent_grpo_local_release_acceptance_v1",
        "scope": "local_static_unit_config_package_acceptance",
        "status": "PASS" if passed else "FAIL",
        "started_at": started_at,
        "finished_at": utc_now(),
        "python_version": sys.version.split()[0],
        "git_commit": git_check.get("git_commit"),
        "checks": checks,
        "target_gpu_runtime_status": "DEFERRED_TO_L20_SERVER",
        "target_gpu_runtime_command": "bash tools/run_3gpu_final_validation.sh ... --gpus 4,5,6",
        "truthfulness_note": (
            "PASS here does not assert CUDA/NCCL/SGLang/FSDP execution. "
            "Target runtime is accepted only when acceptance.json reports 3GPU_FINAL_GATE: PASS."
        ),
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"LOCAL_RELEASE_GATE: {report['status']}")
    print(f"REPORT: {destination}")
    print("TARGET_GPU_RUNTIME: DEFERRED_TO_L20_SERVER")
    if not passed:
        failed = [str(check.get("name")) for check in checks if check.get("status") != "PASS"]
        print("FAILED_CHECKS: " + ",".join(failed))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
