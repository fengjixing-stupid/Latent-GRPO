#!/usr/bin/env python3
"""Install audited target groups in order using the active target venv."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def run(*args: str) -> None:
    completed = subprocess.run([sys.executable, "-m", "pip", *args], cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> int:
    kernel_version = os.environ.get("SGL_KERNEL_VERSION", "").strip()
    if kernel_version not in {"0.1.0", "0.1.1"}:
        print(
            "Set SGL_KERNEL_VERSION=0.1.0 or 0.1.1 only after resolving the vendored-metadata/README conflict.",
            file=sys.stderr,
        )
        return 2
    try:
        installed_kernel = importlib.metadata.version("sgl-kernel")
    except importlib.metadata.PackageNotFoundError:
        installed_kernel = None
    if installed_kernel is not None and installed_kernel != kernel_version:
        print(
            f"Refusing sgl-kernel conflict: installed={installed_kernel}, requested={kernel_version}",
            file=sys.stderr,
        )
        return 3

    constraint = str(ROOT / "constraints" / "linux-cu124-py311.txt")
    for requirement in (
        "requirements/runtime-core.txt",
        "requirements/metrics.txt",
        "requirements/reward-math.txt",
        "requirements/runtime-sglang.txt",
    ):
        run("install", "-c", constraint, "-r", str(ROOT / requirement))
    run("install", f"sgl-kernel=={kernel_version}")
    run("install", "--no-build-isolation", "flash-attn==2.7.3")
    run("install", "--no-deps", "-e", str(ROOT / "Latent-GRPO" / "verl-0.4.x"))
    run("install", "--no-deps", "-e", str(ROOT / "Latent-GRPO" / "sglang_latent_reasoning_pkg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

