#!/usr/bin/env python3
"""Run a bounded smoke command while sampling NVIDIA memory usage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample() -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"nvidia-smi_exit_{completed.returncode}")
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_used_mib": int(fields[3]),
                "memory_total_mib": int(fields[4]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args, command = parser.parse_known_args()
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("command required after --")
    started_at = timestamp()
    telemetry_error: str | None = None
    samples: list[dict[str, object]] = []
    process = subprocess.Popen(command)
    while process.poll() is None:
        try:
            samples.append({"sampled_at": timestamp(), "gpus": sample()})
        except (OSError, RuntimeError, ValueError) as error:
            telemetry_error = f"{type(error).__name__}:{error}"
        time.sleep(max(args.interval_seconds, 0.1))
    try:
        samples.append({"sampled_at": timestamp(), "gpus": sample()})
    except (OSError, RuntimeError, ValueError) as error:
        telemetry_error = f"{type(error).__name__}:{error}"

    peaks: dict[str, int] = {}
    totals: dict[str, int] = {}
    for item in samples:
        for gpu in item["gpus"]:  # type: ignore[index]
            index = str(gpu["index"])
            peaks[index] = max(peaks.get(index, 0), int(gpu["memory_used_mib"]))
            totals[index] = int(gpu["memory_total_mib"])
    payload = {
        "command": command,
        "started_at": started_at,
        "finished_at": timestamp(),
        "exit_code": process.returncode,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "samples": samples,
        "peak_memory_used_mib_by_gpu": peaks,
        "memory_total_mib_by_gpu": totals,
        "telemetry_error": telemetry_error,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if process.returncode:
        return int(process.returncode)
    return 5 if telemetry_error or not samples else 0


if __name__ == "__main__":
    raise SystemExit(main())

