"""Read-only development and target-machine environment probes."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Sequence
from typing import Any, Iterable


_TARGET_PACKAGES = (
    "torch",
    "ray",
    "transformers",
    "pyarrow",
    "numpy",
    "tensordict",
    "torchdata",
    "sglang",
    "compressed-tensors",
    "sgl-kernel",
    "flash-attn",
    "flashinfer-python",
    "cuda-python",
    "cuda-bindings",
    "cachetools",
    "openai",
    "tiktoken",
    "torch-memory-saver",
)


def collect_environment(
    *,
    mode: str,
    require_gpus: int = 3,
    min_vram_gb: int = 40,
    required_precision: str = "bf16",
    workspace_root: str | Path | None = None,
    platform_name: str | None = None,
    machine: str | None = None,
    hostname: str | None = None,
    username: str | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Collect a JSON-safe report; development mode never imports torch."""
    if mode not in {"development", "target"}:
        raise ValueError("mode must be development or target")
    platform_value = platform_name or platform.system()
    machine_value = machine or platform.machine()
    python_value = python_executable or sys.executable
    root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    report: dict[str, Any] = {
        "probe_version": 1,
        "mode": mode,
        "host_platform": _platform_label(platform_value, machine_value),
        "hostname_redacted": _redact(hostname or socket.gethostname()),
        "username_redacted": _redact(username or _safe_username()),
        "python_version": platform.python_version(),
        "python_executable_fingerprint": _redact(python_value),
        "workspace_fingerprint": _redact(str(root)),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "disk_free_bytes": shutil.disk_usage(root).free,
        "dependency_check_status": _package_statuses(_TARGET_PACKAGES),
        "target_gpu_environment_available": False,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_total_memory_bytes": [],
        "gpu_compute_capabilities": [],
        "cuda_runtime_version": None,
        "cuda_driver_version": None,
        "cudnn_version": None,
        "nccl_version": None,
        "bf16_supported": False,
        "fp16_supported": False,
        "required_precision": _normalize_precision(required_precision),
        "nccl_available": False,
        "distributed_backend": None,
        "world_size": _environment_int("WORLD_SIZE", 1),
        "local_world_size": _environment_int("LOCAL_WORLD_SIZE", 1),
        "rank": _environment_int("RANK", 0),
        "local_rank": _environment_int("LOCAL_RANK", 0),
    }
    if mode == "development":
        report.update(
            {
                "status": "mac_development_check_passed",
                "training_runtime_validation": "deferred_to_target_machine",
                "deferred_reasons": ["target_machine_test_deferred"],
            }
        )
        return report
    report.update(_target_gpu_probe())
    reasons = validate_target_environment(
        report,
        require_gpus=require_gpus,
        min_vram_gb=min_vram_gb,
        required_precision=required_precision,
    )
    report["failure_reasons"] = reasons
    report["status"] = "target_machine_probe_passed" if not reasons else "blocked"
    report["training_runtime_validation"] = "target_machine_probe_passed" if not reasons else "target_machine_test_deferred"
    return report


def validate_target_environment(
    report: dict[str, Any],
    *,
    require_gpus: int,
    min_vram_gb: int,
    required_precision: str = "bf16",
) -> list[str]:
    """Return stable target gate reasons in deterministic priority order."""
    reasons: list[str] = []
    if report.get("host_platform") not in {"linux_x86_64", "linux_aarch64"}:
        reasons.append("target_platform_not_linux")
    if int(report.get("gpu_count", 0)) < require_gpus:
        reasons.append("gpu_count_below_requirement")
    threshold = min_vram_gb * 1024**3
    memory = report.get("gpu_total_memory_bytes", [])
    if len(memory) >= require_gpus and any(int(value) < threshold for value in memory[:require_gpus]):
        reasons.append("gpu_vram_below_requirement")
    if not report.get("cuda_available", False):
        reasons.append("cuda_unavailable")
    if report.get("cuda_driver_wheel_compatible") is False:
        reasons.append("cuda_driver_wheel_incompatible")
    precision = _normalize_precision(required_precision)
    if precision == "bf16":
        if not report.get("bf16_supported", False):
            reasons.append("bf16_unsupported")
    elif precision == "fp16":
        if not report.get("fp16_supported", False):
            reasons.append("fp16_unsupported")
    else:  # defensive; _normalize_precision currently rejects this branch
        reasons.append("unsupported_precision_requirement")
    if not report.get("nccl_available", False):
        reasons.append("nccl_unavailable")
    return reasons


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def build_report_envelope(
    *,
    command: Sequence[str],
    started_at: str,
    finished_at: str,
    exit_code: int,
    status: str,
    environment_summary: dict[str, Any],
    artifacts: Sequence[str],
    failure_reason: str | None,
    stdout_log_path: str | None = None,
    stderr_log_path: str | None = None,
) -> dict[str, Any]:
    """Create the uniform target-machine artifact envelope."""
    return {
        "command": list(command),
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": exit_code,
        "status": status,
        "environment_summary": environment_summary,
        "stdout_log_path": stdout_log_path,
        "stderr_log_path": stderr_log_path,
        "artifacts": list(artifacts),
        "failure_reason": failure_reason,
    }


def _target_gpu_probe() -> dict[str, Any]:
    nvidia = _nvidia_smi()
    torch_info = _torch_target_info()
    gpu_count = len(nvidia["names"])
    return {
        "target_gpu_environment_available": bool(torch_info["cuda_available"] and gpu_count),
        "cuda_available": torch_info["cuda_available"],
        "gpu_count": gpu_count,
        "gpu_names": nvidia["names"],
        "gpu_total_memory_bytes": nvidia["memory_bytes"],
        "gpu_compute_capabilities": nvidia["compute_capabilities"],
        "cuda_runtime_version": torch_info["cuda_runtime_version"],
        "cuda_driver_version": nvidia["driver_version"],
        "cuda_driver_wheel_compatible": _driver_wheel_compatible(
            torch_info["cuda_runtime_version"], nvidia["driver_version"]
        ),
        "cudnn_version": torch_info["cudnn_version"],
        "nccl_version": torch_info["nccl_version"],
        "bf16_supported": torch_info["bf16_supported"],
        "fp16_supported": torch_info["fp16_supported"],
        "nccl_available": torch_info["nccl_available"],
        "distributed_backend": "nccl" if torch_info["nccl_available"] else None,
        "torch_version": torch_info["torch_version"],
    }


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except OSError:
        return {"names": [], "memory_bytes": [], "compute_capabilities": [], "driver_version": None}
    if completed.returncode:
        return {"names": [], "memory_bytes": [], "compute_capabilities": [], "driver_version": None}
    rows = list(csv.reader(line for line in completed.stdout.splitlines() if line.strip()))
    names, memory, capabilities, drivers = [], [], [], []
    for row in rows:
        if len(row) != 4:
            continue
        names.append(row[0].strip())
        try:
            memory.append(int(float(row[1])) * 1024**2)
        except ValueError:
            memory.append(0)
        capabilities.append(row[3].strip())
        drivers.append(row[2].strip())
    return {
        "names": names,
        "memory_bytes": memory,
        "compute_capabilities": capabilities,
        "driver_version": drivers[0] if drivers else None,
    }


def _torch_target_info() -> dict[str, Any]:
    """Import torch only for an explicitly requested target-machine probe."""
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "torch_version": None,
            "cuda_available": False,
            "cuda_runtime_version": None,
            "cudnn_version": None,
            "nccl_version": None,
            "bf16_supported": False,
            "fp16_supported": False,
            "nccl_available": False,
        }
    cuda_available = bool(torch.cuda.is_available())
    nccl_version = None
    if cuda_available:
        try:
            nccl_version = ".".join(str(part) for part in torch.cuda.nccl.version())
        except (AttributeError, RuntimeError):
            nccl_version = None
    try:
        bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        bf16_supported = False
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nccl_version": nccl_version,
        "bf16_supported": bf16_supported,
        "fp16_supported": bool(cuda_available),
        "nccl_available": nccl_version is not None,
    }


def _driver_wheel_compatible(cuda_runtime_version: str | None, driver_version: str | None) -> bool | None:
    """Conservatively apply the documented CUDA 12 minimum driver family."""
    if not cuda_runtime_version or not driver_version:
        return None
    try:
        driver_major = int(driver_version.split(".", 1)[0])
    except ValueError:
        return None
    if cuda_runtime_version.startswith("12."):
        return driver_major >= 525
    return None


def _package_statuses(packages: Iterable[str]) -> dict[str, dict[str, str | None]]:
    status: dict[str, dict[str, str | None]] = {}
    for package in packages:
        try:
            status[package] = {"status": "present", "version": importlib.metadata.version(package)}
        except importlib.metadata.PackageNotFoundError:
            status[package] = {"status": "missing", "version": None}
    return status


def _platform_label(system_name: str, machine: str) -> str:
    system_key = system_name.lower()
    machine_key = machine.lower().replace("arm64", "arm64").replace("x86_64", "x86_64")
    if system_key == "darwin":
        return f"macos_{machine_key}"
    return f"{system_key}_{machine_key}"


def _redact(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_username() -> str:
    try:
        return os.getlogin()
    except OSError:
        return os.environ.get("USER", "unknown")


def _normalize_precision(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    if normalized in {"fp16", "float16", "half", "16"}:
        return "fp16"
    raise ValueError(f"unsupported target precision: {value}")


def _environment_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default
