#!/usr/bin/env python3
"""Build a machine-readable inventory without promoting unverified reports."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "artifacts" / "target_machine"
EXPECTED = (
    "runtime_probe.json",
    "import_check.json",
    "ray_gpu_probe.json",
    "single_gpu_smoke.json",
    "single_gpu_output_validation.json",
    "three_gpu_smoke.json",
    "three_gpu_output_validation.json",
    "resume_smoke.json",
    "requirements_validation.json",
)


def main() -> int:
    reports: list[dict[str, object]] = []
    missing: list[str] = []
    blocked: list[str] = []
    for name in EXPECTED:
        path = REPORT_DIR / name
        if not path.is_file():
            missing.append(name)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            status = payload.get("status")
            exit_code = payload.get("exit_code")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            status = "blocked"
            exit_code = None
        reports.append({"path": str(path.relative_to(ROOT)), "status": status, "exit_code": exit_code})
        if status in {"blocked", "target_machine_test_deferred", "unavailable_with_reason"}:
            blocked.append(name)

    now = datetime.now(timezone.utc).isoformat()
    complete = not missing and not blocked
    payload = {
        "command": ["python", "scripts/target_machine/build_report_manifest.py"],
        "started_at": now,
        "finished_at": now,
        "exit_code": 0 if complete else 1,
        "status": "target_machine_probe_passed" if complete else "blocked",
        "environment_summary": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "expected_report_count": len(EXPECTED),
            "present_report_count": len(reports),
        },
        "stdout_log_path": "artifacts/target_machine/logs/11_collect_reports.stdout.log",
        "stderr_log_path": "artifacts/target_machine/logs/11_collect_reports.stderr.log",
        "artifacts": [str(REPORT_DIR.relative_to(ROOT)), *[item["path"] for item in reports]],
        "failure_reason": None if complete else "incomplete_or_blocked_reports",
        "reports": reports,
        "missing_reports": missing,
        "blocked_or_deferred_reports": blocked,
    }
    destination = REPORT_DIR / "report_manifest.json"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(payload, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

