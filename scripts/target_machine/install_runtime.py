#!/usr/bin/env python3
"""Install the audited CUDA 12.4 / Python 3.11 runtime in a fixed order."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SGL_KERNEL = "0.1.1"
EXPECTED_FLASHINFER = "0.2.5"


def run(*args: str) -> None:
    completed = subprocess.run([sys.executable, "-m", "pip", *args], cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    kernel_version = os.environ.get("SGL_KERNEL_VERSION", EXPECTED_SGL_KERNEL).strip()
    if kernel_version != EXPECTED_SGL_KERNEL:
        print(
            f"This package is internally aligned to sgl-kernel=={EXPECTED_SGL_KERNEL}; "
            f"refusing requested version {kernel_version!r}.",
            file=sys.stderr,
        )
        return 2

    installed_kernel = installed_version("sgl-kernel")
    if installed_kernel is not None and installed_kernel != EXPECTED_SGL_KERNEL:
        print(
            f"Refusing sgl-kernel conflict: installed={installed_kernel}, "
            f"required={EXPECTED_SGL_KERNEL}",
            file=sys.stderr,
        )
        return 3

    constraint = str(ROOT / "constraints" / "linux-cu124-py311.txt")
    for requirement in (
        "requirements/runtime-core.txt",
        "requirements/metrics.txt",
        "requirements/reward-math.txt",
        "requirements/runtime-sglang.txt",
        "requirements/tracking-optional.txt",
    ):
        run("install", "-c", constraint, "-r", str(ROOT / requirement))

    if installed_version("sgl-kernel") != EXPECTED_SGL_KERNEL:
        print("sgl-kernel version verification failed after installation", file=sys.stderr)
        return 4
    if installed_version("flashinfer-python") != EXPECTED_FLASHINFER:
        print("flashinfer-python version verification failed after installation", file=sys.stderr)
        return 5

    installed_flash_attn = installed_version("flash-attn")
    if installed_flash_attn != "2.7.3" and shutil.which("nvcc") is None:
        print(
            "flash-attn==2.7.3 is not installed and nvcc is unavailable. "
            "Use a CUDA-devel image/toolkit or preinstall the matching wheel.",
            file=sys.stderr,
        )
        return 6
    run("install", "--no-build-isolation", "flash-attn==2.7.3")
    run("install", "--no-deps", "-e", str(ROOT / "Latent-GRPO" / "verl-0.4.x"))
    run(
        "install",
        "--no-deps",
        "-e",
        str(ROOT / "Latent-GRPO" / "sglang_latent_reasoning_pkg" / "python"),
    )
    # Fail in the installer rather than several minutes into a target smoke.
    run("check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
