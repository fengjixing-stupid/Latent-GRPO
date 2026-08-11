#!/usr/bin/env python3
"""Run a command while sampling only the GPUs selected for that command."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def visible_device_selectors(raw: str | None = None) -> tuple[str, ...] | None:
    """Return CUDA-visible physical-index/UUID selectors, or ``None`` for all GPUs.

    The three-GPU wrappers use numeric physical indices such as ``4,5,6``. UUID
    selectors are accepted as well. Empty CUDA_VISIBLE_DEVICES means that the
    sampler follows normal nvidia-smi behaviour and records every visible GPU.
    """

    value = os.environ.get("CUDA_VISIBLE_DEVICES", "") if raw is None else raw
    value = value.strip()
    if not value:
        return None
    selectors = tuple(part.strip() for part in value.split(",") if part.strip())
    if not selectors:
        return None
    if len(set(selectors)) != len(selectors):
        raise ValueError("duplicate_cuda_visible_devices_selector")
    for selector in selectors:
        if not (selector.isdigit() or selector.startswith("GPU-")):
            raise ValueError(f"unsupported_cuda_visible_devices_selector:{selector}")
    return selectors


def _resolve_rows(
    rows: list[dict[str, object]], selectors: tuple[str, ...] | None
) -> list[dict[str, object]]:
    """Return rows in CUDA_VISIBLE_DEVICES order, rejecting ambiguity."""

    if selectors is None:
        return rows
    ordered: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    for selector in selectors:
        matches = [
            row
            for row in rows
            if (selector.isdigit() and int(row["index"]) == int(selector))
            or (selector.startswith("GPU-") and str(row["uuid"]).startswith(selector))
        ]
        if len(matches) != 1:
            resolved = [str(row["index"]) for row in rows]
            raise RuntimeError(
                "cuda_visible_devices_resolution_failed:"
                f"selector={selector}:available={','.join(resolved)}"
            )
        index = int(matches[0]["index"])
        if index in seen_indices:
            raise RuntimeError(f"cuda_visible_devices_duplicate_physical_gpu:{index}")
        seen_indices.add(index)
        ordered.append(matches[0])
    return ordered


def sample() -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
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
        fields = [field.strip() for field in line.split(",", 5)]
        if len(fields) != 6:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_used_mib": int(fields[3]),
                    "memory_total_mib": int(fields[4]),
                    "gpu_utilization_percent": int(fields[5]),
                }
            )
        except ValueError:
            continue
    return _resolve_rows(rows, visible_device_selectors())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args, command = parser.parse_known_args()
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("command required after --")

    selectors = visible_device_selectors()
    started_at = timestamp()
    telemetry_error: str | None = None
    samples: list[dict[str, object]] = []
    process = subprocess.Popen(command, start_new_session=True)

    def forward_signal(signum: int, _frame: object) -> None:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    previous_handlers: dict[int, object] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, forward_signal)

    try:
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
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if process.poll() is None:
            process.terminate()
        process.wait()

    peaks: dict[str, int] = {}
    totals: dict[str, int] = {}
    utilization: dict[str, list[int]] = {}
    for item in samples:
        for gpu in item["gpus"]:  # type: ignore[index]
            index = str(gpu["index"])
            peaks[index] = max(peaks.get(index, 0), int(gpu["memory_used_mib"]))
            totals[index] = int(gpu["memory_total_mib"])
            utilization.setdefault(index, []).append(int(gpu["gpu_utilization_percent"]))

    sample_orders = [
        [int(gpu["index"]) for gpu in item["gpus"]]  # type: ignore[index]
        for item in samples
        if item.get("gpus")
    ]
    selected_physical_gpu_indices = sample_orders[0] if sample_orders else []
    if any(order != selected_physical_gpu_indices for order in sample_orders[1:]):
        telemetry_error = "selected_gpu_order_changed_during_sampling"
        selected_physical_gpu_indices = []

    payload = {
        "schema_version": "gpu_telemetry_v2",
        "command": command,
        "started_at": started_at,
        "finished_at": timestamp(),
        "exit_code": process.returncode,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", ""),
        "selected_device_selectors": list(selectors or ()),
        "selected_physical_gpu_indices": selected_physical_gpu_indices,
        "samples": samples,
        "peak_memory_used_mib_by_gpu": peaks,
        "memory_total_mib_by_gpu": totals,
        "peak_gpu_utilization_percent_by_gpu": {
            index: max(values) for index, values in utilization.items()
        },
        "average_gpu_utilization_percent_by_gpu": {
            index: sum(values) / len(values) for index, values in utilization.items()
        },
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
