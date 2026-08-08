"""Fail-closed compatibility gate for a Kaggle dual-T4 P1 runtime smoke.

This module does not import torch and never touches training data.  It classifies
read-only environment/source facts so an unsupported T4 runtime cannot be
mistaken for a valid Latent-GRPO training smoke.
"""

from __future__ import annotations

from typing import Any, Mapping


EXPECTED_T4_COMPUTE_CAPABILITY = "7.5"


def assess_kaggle_t4_compatibility(
    environment: Mapping[str, Any],
    *,
    actor_hardcodes_bfloat16: bool,
    model_forces_flash_attention_2: bool,
) -> dict[str, Any]:
    """Return READY_FOR_DATA only when platform blockers are absent.

    READY_FOR_DATA is intentionally not a training PASS.  It means only that it
    is worth asking the user for real Parquet inputs for the next gate.
    """
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

    # T4/SM75 does not provide the BF16 capability required by the repository's
    # current generic target gate and actor runtime path.
    if environment.get("bf16_supported") is not True:
        blockers.append("current_target_path_requires_bf16_but_t4_is_not_bf16_capable")
    if actor_hardcodes_bfloat16:
        blockers.append("actor_forward_hardcodes_bfloat16")

    # The vendored actor model loader currently forces HF FlashAttention-2.
    # Turing compatibility therefore requires an explicit attention-path
    # redesign/validation rather than silently bypassing the target gate.
    if model_forces_flash_attention_2:
        blockers.append("actor_model_forces_flash_attention_2_on_turing")

    blockers = sorted(set(blockers))
    status = "BLOCKED" if blockers else "READY_FOR_DATA"
    return {
        "status": status,
        "gate": "kaggle_dual_t4_p1_compatibility",
        "training_started": False,
        "training_data_required_now": status == "READY_FOR_DATA",
        "training_data_inspected": False,
        "training_data_generated": False,
        "gpu_names": names,
        "gpu_compute_capabilities": capabilities,
        "bf16_supported": environment.get("bf16_supported"),
        "blockers": blockers,
        "next_gate": (
            "request_user_real_train_and_val_parquet"
            if status == "READY_FOR_DATA"
            else "resolve_t4_runtime_compatibility_before_requesting_data"
        ),
    }
