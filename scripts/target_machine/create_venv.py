#!/usr/bin/env python3
"""Create the target CPython 3.11 venv without installing dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import venv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        print(f"CPython 3.11 required; found {sys.version.split()[0]}", file=sys.stderr)
        return 2
    path = Path(args.path)
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=True).create(path)
    print(f"created target venv: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

