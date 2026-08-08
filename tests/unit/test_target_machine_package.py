from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "target_machine"
REQUIRED_FIELDS = {
    "command",
    "started_at",
    "finished_at",
    "exit_code",
    "status",
    "environment_summary",
    "stdout_log_path",
    "stderr_log_path",
    "artifacts",
    "failure_reason",
}
SCRIPT_NAMES = [
    "00_probe_environment.sh",
    "01_create_venv.sh",
    "02_install_pytorch.sh",
    "03_install_runtime.sh",
    "04_import_check.sh",
    "05_probe_ray_gpus.sh",
    "06_single_gpu_smoke.sh",
    "07_validate_single_gpu_output.sh",
    "08_three_gpu_smoke.sh",
    "09_validate_three_gpu_output.sh",
    "10_resume_smoke.sh",
    "11_collect_reports.sh",
]
REPORT_NAMES = [
    "runtime_probe.json",
    "import_check.json",
    "ray_gpu_probe.json",
    "single_gpu_smoke.json",
    "single_gpu_output_validation.json",
    "three_gpu_smoke.json",
    "three_gpu_output_validation.json",
    "resume_smoke.json",
    "requirements_validation.json",
    "report_manifest.json",
]


def test_all_target_scripts_exist_and_pass_bash_syntax_check() -> None:
    assert [path.name for path in sorted(SCRIPT_DIR.glob("[0-9][0-9]_*.sh"))] == SCRIPT_NAMES
    for name in SCRIPT_NAMES:
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPT_DIR / name)],
            capture_output=True,
            check=False,
            text=True,
        )
        assert completed.returncode == 0, f"{name}: {completed.stderr}"


def test_scripts_use_report_wrapper_and_safe_training_commands() -> None:
    contents = {name: (SCRIPT_DIR / name).read_text(encoding="utf-8") for name in SCRIPT_NAMES}
    for name, content in contents.items():
        assert "set -u" in content
        assert "TARGET_REPORT_DIR" in content or "build_report_manifest.py" in content

    single = contents["06_single_gpu_smoke.sh"]
    assert "CUDA_VISIBLE_DEVICES=0" in single
    assert "configs/smoke.yaml" in single
    assert "--max-steps 2" in single
    assert "single_gpu_memory.json" in single

    three = contents["08_three_gpu_smoke.sh"]
    assert "CUDA_VISIBLE_DEVICES=0,1,2" in three
    assert "configs/3gpu-low.yaml" in three
    assert "--max-steps 2" in three
    assert "torchrun" not in three
    assert "three_gpu_memory.json" in three

    assert "--resume-from" in contents["10_resume_smoke.sh"]


def test_unexecuted_report_templates_are_complete_and_deferred() -> None:
    report_dir = ROOT / "artifacts" / "target_machine"
    for name in REPORT_NAMES:
        payload = json.loads((report_dir / name).read_text(encoding="utf-8"))
        assert REQUIRED_FIELDS <= set(payload)
        assert payload["status"] == "target_machine_test_deferred"
        assert payload["exit_code"] is None
        assert payload["failure_reason"] == "not_executed_on_target_machine"


def test_report_runner_records_success_and_failure(tmp_path: Path) -> None:
    runner = SCRIPT_DIR / "run_reported.py"
    for expected_exit in (0, 7):
        report = tmp_path / f"report-{expected_exit}.json"
        stdout_log = tmp_path / f"stdout-{expected_exit}.log"
        stderr_log = tmp_path / f"stderr-{expected_exit}.log"
        command = [
            sys.executable,
            str(runner),
            "--report",
            str(report),
            "--stdout-log",
            str(stdout_log),
            "--stderr-log",
            str(stderr_log),
            "--success-status",
            "static_check_passed",
            "--",
            sys.executable,
            "-c",
            f"import sys; print('out'); print('err', file=sys.stderr); sys.exit({expected_exit})",
        ]
        completed = subprocess.run(command, capture_output=True, check=False, text=True)
        assert completed.returncode == expected_exit
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert REQUIRED_FIELDS == set(payload)
        assert payload["exit_code"] == expected_exit
        assert payload["status"] == ("static_check_passed" if expected_exit == 0 else "blocked")
        assert payload["failure_reason"] == (None if expected_exit == 0 else "command_exit_7")
        assert stdout_log.read_text(encoding="utf-8") == "out\n"
        assert stderr_log.read_text(encoding="utf-8") == "err\n"


def test_runbooks_document_install_order_abi_gates_and_return_bundle() -> None:
    teammate = (ROOT / "docs" / "teammate_target_machine_runbook.md").read_text(encoding="utf-8")
    operator = (ROOT / "docs" / "operator_runbook.md").read_text(encoding="utf-8")
    plan = (ROOT / "docs" / "target_machine_validation_plan.md").read_text(encoding="utf-8")
    combined = "\n".join((teammate, operator, plan))
    for required in (
        "download.pytorch.org/whl/cu124",
        "--no-deps",
        "sgl-kernel",
        "FlashAttention",
        "FlashInfer",
        "ABI",
        "artifacts/target_machine/",
        "ray_direct",
        "target_machine_test_deferred",
    ):
        assert required in combined
    assert "01_create_venv.sh" in teammate
    assert "11_collect_reports.sh" in teammate
