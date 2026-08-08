"""Target-only alias for the strict runtime environment probe."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_environment import main


if __name__ == "__main__":
    raise SystemExit(main(["--mode", "target", *sys.argv[1:]]))
