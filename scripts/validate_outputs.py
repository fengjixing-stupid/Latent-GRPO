#!/usr/bin/env python3
"""Validate a Latent-GRPO output directory and return nonzero on violations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_grpo_runner.validation.output_validator import validate_output_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="run output directory")
    args = parser.parse_args(argv)
    result = validate_output_directory(args.input)
    for error in result.errors:
        print(error, file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
