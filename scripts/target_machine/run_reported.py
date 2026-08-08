#!/usr/bin/env python3
"""Run one target-machine command and atomically persist its report envelope.

This helper is intentionally standard-library-only so it can wrap the earliest
environment and virtual-environment steps.  It does not interpret a successful
process as proof beyond the explicit ``--success-status`` selected by the
calling script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


REQUIRED_FIELDS = (
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
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment_summary() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _read_environment_summary(path: Path | None) -> dict[str, Any]:
    summary = _environment_summary()
    if path is None or not path.is_file():
        return summary
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        summary["detail_report_read_error"] = str(path)
        return summary
    detail = payload.get("environment_summary", payload)
    if isinstance(detail, dict):
        summary["detail_report"] = detail
    return summary


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--success-status", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--environment-summary-json")
    args, command = parser.parse_known_args(argv)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    return args, command


def main(argv: list[str] | None = None) -> int:
    args, command = parse_args(argv)
    report_path = Path(args.report)
    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _timestamp()
    exit_code = 1
    failure_reason: str | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            completed = subprocess.run(command, stdout=stdout_file, stderr=stderr_file, check=False)
        exit_code = completed.returncode
        if exit_code != 0:
            failure_reason = f"command_exit_{exit_code}"
    except FileNotFoundError as error:
        exit_code = 127
        failure_reason = f"command_not_found:{error.filename}"
        stderr_path.write_text(f"{error}\n", encoding="utf-8")
        stdout_path.touch(exist_ok=True)
    except Exception as error:  # preserve an envelope for unexpected wrapper failures
        exit_code = 1
        failure_reason = f"wrapper_exception:{type(error).__name__}"
        with stderr_path.open("a", encoding="utf-8") as stderr_file:
            stderr_file.write(f"{type(error).__name__}: {error}\n")
        stdout_path.touch(exist_ok=True)

    report = {
        "command": command,
        "started_at": started_at,
        "finished_at": _timestamp(),
        "exit_code": exit_code,
        "status": args.success_status if exit_code == 0 else "blocked",
        "environment_summary": _read_environment_summary(
            Path(args.environment_summary_json) if args.environment_summary_json else None
        ),
        "stdout_log_path": str(stdout_path),
        "stderr_log_path": str(stderr_path),
        "artifacts": list(dict.fromkeys([str(report_path), *args.artifact])),
        "failure_reason": failure_reason,
    }
    assert tuple(report) == REQUIRED_FIELDS
    _atomic_write_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

