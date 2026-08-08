#!/usr/bin/env python3
"""Strict target import/CUDA/extension ABI smoke; never import this on Mac."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import sys


IMPORTS = {
    "torch": "torch",
    "ray": "ray",
    "transformers": "transformers",
    "pyarrow": "pyarrow",
    "numpy": "numpy",
    "tensordict": "tensordict",
    "torchdata": "torchdata",
    "sglang": "sglang",
    "sgl_kernel": "sgl-kernel",
    "flash_attn": "flash-attn",
    "flashinfer": "flashinfer-python",
    "verl": "verl",
}


def main() -> int:
    evidence: dict[str, object] = {
        "status": "blocked",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "imports": {},
        "failures": [],
    }
    failures: list[str] = evidence["failures"]  # type: ignore[assignment]
    imports: dict[str, object] = evidence["imports"]  # type: ignore[assignment]
    modules: dict[str, object] = {}
    for module_name, distribution_name in IMPORTS.items():
        try:
            modules[module_name] = importlib.import_module(module_name)
            imports[module_name] = importlib.metadata.version(distribution_name)
        except Exception as error:
            imports[module_name] = {"error": type(error).__name__}
            failures.append(f"import_failed:{module_name}:{type(error).__name__}")

    torch = modules.get("torch")
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())  # type: ignore[attr-defined]
            evidence["torch_cuda_available"] = cuda_available
            evidence["torch_cuda_version"] = torch.version.cuda  # type: ignore[attr-defined]
            evidence["torch_cxx11_abi"] = bool(torch._C._GLIBCXX_USE_CXX11_ABI)  # type: ignore[attr-defined]
            evidence["torch_gpu_count"] = int(torch.cuda.device_count())  # type: ignore[attr-defined]
            evidence["bf16_supported"] = bool(torch.cuda.is_bf16_supported()) if cuda_available else False  # type: ignore[attr-defined]
            if not cuda_available:
                failures.append("torch_cuda_unavailable")
            elif int(evidence["torch_gpu_count"]) < 3:
                failures.append("torch_visible_gpu_count_below_3")
            if str(evidence["torch_cuda_version"]) != "12.4":
                failures.append("torch_cuda_version_not_12_4")
        except Exception as error:
            failures.append(f"torch_cuda_probe_failed:{type(error).__name__}")

    # Successful imports execute the extension loaders and therefore form the
    # minimum ABI smoke.  A real kernel launch remains a later training probe.
    evidence["extension_abi_imports_passed"] = all(
        name in modules for name in ("sgl_kernel", "flash_attn", "flashinfer")
    )
    evidence["status"] = "cuda_runtime_verified" if not failures else "blocked"
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

