"""Write a read-only development or target-machine environment report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_grpo_runner.environment import build_report_envelope, collect_environment, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "target"), required=True)
    parser.add_argument("--require-gpus", type=int, default=3)
    parser.add_argument("--min-vram-gb", type=int, default=40)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    started_at = datetime.now(timezone.utc).isoformat()
    default_output = (
        ROOT / "artifacts" / "mac_development_environment.json"
        if args.mode == "development"
        else ROOT / "artifacts" / "target_machine" / "runtime_probe.json"
    )
    report = collect_environment(
        mode=args.mode,
        require_gpus=args.require_gpus,
        min_vram_gb=args.min_vram_gb,
        workspace_root=ROOT,
    )
    exit_code = 0 if args.mode == "development" or not report["failure_reasons"] else 1
    if args.mode == "target":
        envelope = build_report_envelope(
            command=[sys.executable, "scripts/check_environment.py", "--mode", "target"],
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            exit_code=exit_code,
            status=report["status"],
            environment_summary=report,
            stdout_log_path=None,
            stderr_log_path=None,
            artifacts=[str(args.output or default_output)],
            failure_reason=report["failure_reasons"][0] if report["failure_reasons"] else None,
        )
        output = write_report(args.output or default_output, envelope)
        print(json.dumps({"report": str(output), **envelope}, sort_keys=True))
    else:
        output = write_report(args.output or default_output, report)
        print(json.dumps({"report": str(output), **report}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
