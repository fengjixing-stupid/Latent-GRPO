#!/usr/bin/env python3
"""Run the formal one-step dual-T4 validation and validate all 30 metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/kaggle-t4-30-metric.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def _require_dual_t4() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if result.returncode != 0 or len(rows) != 2:
        raise SystemExit("KAGGLE_T4_30_RUNTIME_GATE: BLOCKED: expected exactly two visible GPUs")
    if any("T4" not in row or "7.5" not in row for row in rows):
        raise SystemExit(f"KAGGLE_T4_30_RUNTIME_GATE: BLOCKED: expected dual T4 CC 7.5, got {rows}")


def main() -> int:
    args = parse_args()
    train_path = Path(args.train_path).expanduser().resolve()
    val_path = Path(args.val_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    for label, path in (("train", train_path), ("validation", val_path)):
        if not path.is_file():
            raise SystemExit(f"KAGGLE_T4_30_RUNTIME_GATE: BLOCKED: {label} parquet missing: {path}")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"KAGGLE_T4_30_RUNTIME_GATE: BLOCKED: output root is not empty: {output_root}")
    _require_dual_t4()

    training = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_latent_grpo.py"),
            "--config",
            str(PROFILE),
            "--profile-name",
            "kaggle-t4-30-metric",
            "--model-path",
            args.model_path,
            "--train-files",
            str(train_path),
            "--val-files",
            str(val_path),
            "--output-root",
            str(output_root),
        ],
        cwd=ROOT,
        check=False,
    )
    if training.returncode != 0:
        print(f"KAGGLE_T4_30_RUNTIME_GATE: BLOCKED: training_exit={training.returncode}")
        return training.returncode

    validation = subprocess.run(
        [sys.executable, str(ROOT / "tools/validate_kaggle_t4_30_metrics.py"), str(output_root)],
        cwd=ROOT,
        check=False,
    )
    return validation.returncode


if __name__ == "__main__":
    raise SystemExit(main())
