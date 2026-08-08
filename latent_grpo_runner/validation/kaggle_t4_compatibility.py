"""Fail-closed compatibility gate for a Kaggle dual-T4 P1 runtime smoke.

This gate is intentionally data-free. READY_FOR_DATA means the code/runtime
stack can preserve Latent-GRPO semantics on T4 using FP16 + padded actor SDPA +
SGLang Triton attention while keeping the author's sampling backend unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

EXPECTED_T4_COMPUTE_CAPABILITY = "7.5"
_REQUIRED_PACKAGES = (
    "torch", "ray", "sglang", "sgl-kernel", "flashinfer-python",
    "cuda-python", "cuda-bindings", "pyarrow",
)
_EXPECTED_VERSION_PREFIXES = {
    "torch": "2.6.0",
    "sglang": "0.4.6.post1",
    "sgl-kernel": "0.1.1",
    "flashinfer-python": "0.2.5",
    "cuda-python": "11.8.6",
    "cuda-bindings": "11.8.6",
}


def assess_kaggle_t4_compatibility(
    environment: Mapping[str, Any],
    *,
    runner_attention_backend_exposed: bool,
    actor_hardcodes_bfloat16: bool,
    model_forces_flash_attention_2: bool,
    padded_latent_path_present: bool,
    gumbel_logprob_requires_flash_attention: bool,
    sglang_triton_attention_forwarded: bool,
    flashinfer_sampling_preserved: bool,
    runtime_imports_ok: bool,
) -> dict[str, Any]:
    names = [str(name) for name in environment.get("gpu_names", [])]
    capabilities = [str(value) for value in environment.get("gpu_compute_capabilities", [])]
    blockers: list[str] = []

    if len(names) != 2 or any("T4" not in name.upper() for name in names):
        blockers.append("expected_exactly_two_t4_gpus")
    if len(capabilities) != 2 or any(value != EXPECTED_T4_COMPUTE_CAPABILITY for value in capabilities):
        blockers.append("expected_t4_compute_capability_7_5")
    if not environment.get("cuda_available", False):
        blockers.append("cuda_unavailable")
    if not environment.get("nccl_available", False):
        blockers.append("nccl_unavailable")
    if environment.get("fp16_supported") is not True:
        blockers.append("fp16_unavailable_on_t4_runtime")
    cuda_runtime = str(environment.get("cuda_runtime_version") or "")
    if not cuda_runtime.startswith("11.8"):
        blockers.append("expected_t4_cuda_runtime_11_8")

    package_status = environment.get("dependency_check_status", {})
    if isinstance(package_status, Mapping):
        for package in _REQUIRED_PACKAGES:
            item = package_status.get(package, {})
            if not isinstance(item, Mapping) or item.get("status") != "present":
                blockers.append(f"missing_runtime_package:{package}")
        for package, expected_prefix in _EXPECTED_VERSION_PREFIXES.items():
            item = package_status.get(package, {})
            if isinstance(item, Mapping) and item.get("status") == "present":
                version = str(item.get("version") or "")
                if not version.startswith(expected_prefix):
                    blockers.append(f"runtime_version_mismatch:{package}:{version}")

    if not runner_attention_backend_exposed:
        blockers.append("runner_sglang_attention_backend_not_exposed")
    if actor_hardcodes_bfloat16:
        blockers.append("actor_forward_hardcodes_bfloat16")
    if model_forces_flash_attention_2:
        blockers.append("actor_model_forces_flash_attention_2_on_turing")
    if not padded_latent_path_present:
        blockers.append("padded_latent_actor_path_missing")
    if gumbel_logprob_requires_flash_attention:
        blockers.append("latent_gumbel_logprob_still_requires_flash_attention")
    if not sglang_triton_attention_forwarded:
        blockers.append("sglang_triton_attention_not_forwarded")
    if not flashinfer_sampling_preserved:
        blockers.append("author_sampling_backend_not_preserved")
    if not runtime_imports_ok:
        blockers.append("t4_runtime_import_check_failed")

    blockers = sorted(set(blockers))
    status = "BLOCKED" if blockers else "READY_FOR_DATA"
    return {
        "status": status,
        "gate": "kaggle_dual_t4_p1_runtime_compatibility",
        "training_started": False,
        "training_data_required_now": status == "READY_FOR_DATA",
        "training_data_inspected": False,
        "training_data_generated": False,
        "gpu_names": names,
        "gpu_compute_capabilities": capabilities,
        "bf16_supported": environment.get("bf16_supported"),
        "fp16_supported": environment.get("fp16_supported"),
        "runtime_precision": "fp16",
        "actor_attention_path": "padded_sdpa",
        "sglang_attention_backend": "triton",
        "sampling_backend": "flashinfer_preserved",
        "semantic_guards": [
            "latent_trajectory_generation_unchanged",
            "topk_gumbel_one_sided_sampling_unchanged",
            "latent_end_transition_unchanged",
            "old_current_gumbel_logprob_formula_unchanged",
            "advantage_and_dynamic_filtering_unchanged",
            "flipgrad_straight_through_formula_unchanged",
            "p1_metric_definitions_and_sampling_points_unchanged",
        ],
        "blockers": blockers,
        "next_gate": (
            "request_user_real_train_and_val_parquet"
            if status == "READY_FOR_DATA"
            else "resolve_t4_runtime_compatibility_before_requesting_data"
        ),
    }
