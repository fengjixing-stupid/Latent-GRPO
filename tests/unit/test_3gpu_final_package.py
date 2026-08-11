from __future__ import annotations

import json
import hashlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import yaml

from latent_grpo_runner.config import load_config
from tools.validate_3gpu_final import (
    _load_table,
    _validate_probe_worker_runtime_evidence,
    _validate_runtime_telemetry,
    evaluate_final_gate,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE = "53438ec07b804ebd1b670d6fe118199798350505"


def _load_yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_author_truth_files_capture_vendored_shell_values() -> None:
    low = _load_yaml("configs/author/latent_grpo_gsm8k_llama3.yaml")
    high = _load_yaml("configs/author/latent_grpo_math_qwen.yaml")

    assert low["provenance"]["source_file"] == "Latent-GRPO/Latent-GRPO-gsm8k-llama3.sh"
    assert high["provenance"]["source_file"] == "Latent-GRPO/Latent-GRPO-math500-qwen.sh"
    assert low["provenance"]["baseline_commit"] == BASELINE
    assert high["provenance"]["baseline_commit"] == BASELINE
    for truth in (low, high):
        source = ROOT / truth["provenance"]["source_file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == truth["provenance"]["source_sha256"]

    assert low["data"]["train_batch_size"] == 64
    assert low["data"]["val_batch_size"] == 128
    assert low["data"]["max_prompt_length"] == 192
    assert low["data"]["max_response_length"] == 128
    assert low["actor"]["lr"] == "1e-6"
    assert low["actor"]["ppo_mini_batch_size"] == 16
    assert low["actor"]["ppo_micro_batch_size_per_gpu"] == 2
    assert low["rollout"]["max_model_len"] == 1024
    assert low["rollout"]["max_num_batched_tokens"] == 2048
    assert low["rollout"]["gpu_memory_utilization"] == 0.6
    assert low["rollout"]["n"] == 8
    assert low["trainer"]["n_gpus_per_node"] == 8
    assert low["trainer"]["total_epochs"] == 10

    assert high["data"]["train_batch_size"] == 32
    assert high["data"]["val_batch_size"] == 500
    assert high["data"]["max_prompt_length"] == 1024
    assert high["data"]["max_response_length"] == 4096
    assert high["actor"]["use_kl_loss"] is True
    assert high["model"]["enable_gradient_checkpointing"] is True
    assert high["fsdp"]["param_offload"] is True
    assert high["rollout"]["max_model_len"] == 12000
    assert high["rollout"]["gpu_memory_utilization"] == 0.8
    assert high["rollout"]["latent_end_token_id"] == 522
    assert high["trainer"]["n_gpus_per_node"] == 8
    assert high["trainer"]["total_epochs"] == 5


def test_final_three_gpu_profiles_parse_and_preserve_author_frozen_values() -> None:
    final_low = load_config(ROOT / "configs/3gpu-final-low.yaml", workspace_root=ROOT)
    validation = load_config(ROOT / "configs/3gpu-final-validation.yaml", workspace_root=ROOT)

    for config in (final_low, validation):
        assert config.launcher.mode == "ray_direct"
        assert config.hardware.required_gpus == 3
        assert config.hardware.min_vram_gb == 40
        assert config.rollout.dtype == "bfloat16"
        assert config.batch.rollout_n == 8
        assert config.rollout.max_model_len == 1024
        assert config.rollout.max_num_batched_tokens == 2048
        assert config.rollout.gpu_memory_utilization == 0.6
        assert config.rollout.top_p == 0.95
        assert config.rollout.top_k == 30
        assert config.rollout.max_topk == 10
        assert config.rollout.temperature == 0.6
        assert config.rollout.gumbel_softmax_temperature == 1.0
        assert config.rollout.noise_scale == 1.0
        assert config.model.latent_end_token_id == 524
        assert config.model.use_remove_padding is True
        assert config.model.enable_gradient_checkpointing is False
        assert config.model.use_kl_loss is False
        assert config.features.metrics_enabled is True
        assert config.features.support_enabled is True
        assert config.features.checkpoint_probe_enabled is True
        assert config.features.credit_probe_enabled is True
        assert config.batch_arithmetic()[0] % config.batch_arithmetic()[1] == 0

        overrides = set(config.author_hydra_overrides())
        for author_value in (
            "data.val_batch_size=128",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048",
            "actor_rollout_ref.actor.freeze_embedding=true",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
            "actor_rollout_ref.rollout.val_kwargs.temperature=0.6",
            "actor_rollout_ref.ref.strategy=fsdp2",
            "algorithm.use_kl_in_reward=false",
            "algorithm.exclude_overlong_samples_from_advantage=false",
            "trainer.balance_batch=true",
        ):
            assert author_value in overrides

    assert final_low.profile_kind == "formal_training"
    assert final_low.batch.prompt_batch == 48
    assert final_low.batch.mini_prompt_batch == 12
    assert final_low.training.max_steps >= 2
    assert "trainer.total_epochs=10" in final_low.author_hydra_overrides()
    assert not any(
        item.startswith("trainer.total_training_steps=")
        for item in final_low.author_hydra_overrides()
    )
    assert validation.profile_kind == "final_runtime_validation"
    assert validation.batch.prompt_batch == 3
    assert validation.batch.mini_prompt_batch == 3
    assert validation.training.max_steps == 2
    assert "trainer.total_training_steps=2" in validation.author_hydra_overrides()
    assert "trainer.save_freq=1" in validation.author_hydra_overrides()


def test_high_final_profiles_parse_and_preserve_author_frozen_values() -> None:
    formal = load_config(ROOT / "configs/3gpu-final-high.yaml", workspace_root=ROOT)
    validation = load_config(ROOT / "configs/3gpu-final-high-validation.yaml", workspace_root=ROOT)

    for config in (formal, validation):
        assert config.launcher.mode == "ray_direct"
        assert config.hardware.required_gpus == 3
        assert config.hardware.min_vram_gb == 40
        assert config.batch.rollout_n == 8
        assert config.batch.actor_micro_batch_per_gpu == 1
        assert config.data.max_prompt_length == 1024
        assert config.data.max_response_length == 4096
        assert config.rollout.dtype == "bfloat16"
        assert config.rollout.max_model_len == 12000
        assert config.rollout.max_num_batched_tokens == 12000
        assert config.rollout.gpu_memory_utilization == 0.8
        assert config.model.latent_end_token_id == 522
        assert config.model.enable_gradient_checkpointing is True
        assert config.model.actor_param_offload is True
        assert config.model.actor_optimizer_offload is True
        assert config.model.ref_param_offload is False
        assert config.model.use_kl_loss is True
        assert config.features.metrics_enabled is True
        assert config.features.support_enabled is True
        assert config.features.checkpoint_probe_enabled is True
        assert config.features.credit_probe_enabled is True

        overrides = set(config.author_hydra_overrides())
        for author_value in (
            "data.val_batch_size=500",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480",
            "actor_rollout_ref.actor.freeze_embedding=true",
            "actor_rollout_ref.ref.strategy=fsdp2",
            "algorithm.use_kl_in_reward=false",
            "algorithm.exclude_overlong_samples_from_advantage=true",
            "trainer.balance_batch=true",
        ):
            assert author_value in overrides

    assert formal.profile_name == "3gpu-final-high"
    assert formal.profile_kind == "formal_training"
    assert formal.batch.prompt_batch == 12
    assert formal.batch.mini_prompt_batch == 12
    assert formal.batch_arithmetic()[:2] == (32, 32)
    assert "trainer.total_epochs=5" in formal.author_hydra_overrides()
    assert not any(
        item.startswith("trainer.total_training_steps=")
        for item in formal.author_hydra_overrides()
    )

    assert validation.profile_name == "3gpu-final-high-validation"
    assert validation.profile_kind == "final_runtime_validation"
    assert validation.batch.prompt_batch == 3
    assert validation.batch.mini_prompt_batch == 3
    assert validation.training.max_steps == 2
    assert "trainer.total_training_steps=2" in validation.author_hydra_overrides()
    assert "trainer.save_freq=1" in validation.author_hydra_overrides()


def test_final_wrappers_exist_are_syntax_checked_and_fail_closed(tmp_path: Path) -> None:
    wrappers = [
        "tools/prepare_3gpu_assets.sh",
        "tools/run_3gpu_preflight.sh",
        "tools/run_3gpu_final_validation.sh",
        "tools/run_3gpu_training.sh",
    ]
    for wrapper in wrappers:
        path = ROOT / wrapper
        completed = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=False)
        assert completed.returncode == 0, f"{wrapper}: {completed.stderr}"
        content = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in content
        assert "BLOCKED_REASON" in content
        assert "torchrun" not in content

    final_validation = (ROOT / "tools/run_3gpu_final_validation.sh").read_text(encoding="utf-8")
    for label in (
        "3GPU_PREFLIGHT_GATE",
        "3GPU_DISTRIBUTED_RUNTIME_GATE",
        "CORE_METRICS",
        "CUDA_RNG_ALL_DEVICES",
        "CHECKPOINT_GATE",
        "3GPU_FINAL_GATE",
    ):
        assert label in final_validation

    preflight = (ROOT / "tools/run_3gpu_preflight.sh").read_text(encoding="utf-8")
    assert "PYTHON_VERSION: PASS" in preflight
    assert "DISK_FREE: PASS" in preflight
    assert '--config) CONFIG="$2"' in preflight
    assert '[[ -f "${CONFIG}" ]]' in preflight
    assert '[[ -d "${MODEL_PATH}" ]]' in preflight
    assert "PATH_OR_HF_ID" not in preflight

    final_validation = (ROOT / "tools/run_3gpu_final_validation.sh").read_text(encoding="utf-8")
    assert '--config) CONFIG="$2"' in final_validation
    assert '--config "${CONFIG}"' in final_validation
    assert 'payload.get("profile_kind") == "final_runtime_validation"' in final_validation
    assert 'training.get("max_steps") == 2' in final_validation
    assert "validation_profile_required" in final_validation
    assert "PATH_OR_HF_ID" not in final_validation

    training = (ROOT / "tools/run_3gpu_training.sh").read_text(encoding="utf-8")
    assert "run_manifest.json" in training
    assert "resolved_config.yaml" in training
    assert "--dry-run" in training
    assert '[[ -d "${MODEL_PATH}" ]]' in training
    assert '[[ -s "${TRAIN_DATA}" ]]' in training
    assert '[[ -s "${VAL_DATA}" ]]' in training
    assert '"3gpu-final-high": "3gpu-final-high-validation"' in training
    assert "--profile-name 3gpu-final-low" not in training
    assert "PATH_OR_HF_ID" not in training

    low_report = tmp_path / "low-acceptance.json"
    low_report.write_text(
        json.dumps({"final_gate": "PASS", "profile_name": "3gpu-final-validation"}),
        encoding="utf-8",
    )
    output_root = tmp_path / "high-run"
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "tools/run_3gpu_training.sh"),
            "--config",
            "configs/3gpu-final-high.yaml",
            "--model-path",
            str(ROOT),
            "--train-data",
            str(ROOT / "requirements.txt"),
            "--val-data",
            str(ROOT / "requirements.txt"),
            "--output-root",
            str(output_root),
            "--gpus",
            "0,1,2",
            "--seed",
            "17",
            "--acceptance-report",
            str(low_report),
        ],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHON_BIN": sys.executable},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "final_validation_acceptance_missing_blocked_or_profile_mismatch" in completed.stdout
    assert not output_root.exists()


def test_final_acceptance_gate_requires_all_three_authoritative_workers() -> None:
    evidence = {
        "profile_name": "3gpu-final-high-validation",
        "git_commit": BASELINE,
        "gpu_count": 3,
        "target_precision": "bfloat16",
        "preflight": True,
        "distributed_runtime": True,
        "real_backward": True,
        "real_optimizer_step": True,
        "metrics_core_passed": 29,
        "raw_generated_token_count": True,
        "worker_driver_aggregation": True,
        "aggregation_worker_count": 3,
        "stage3_alignment": True,
        "stage4_probe": True,
        "cuda_rng_restored": True,
        "grad_pollution": False,
        "parameter_pollution_by_probe": False,
        "optimizer_state_pollution_by_probe": False,
        "checkpoint_write": True,
        "resume_compatibility": True,
        "gpu_memory_telemetry": True,
    }
    passed = evaluate_final_gate(evidence)
    assert passed["final_gate"] == "PASS"
    assert passed["metrics_core"] == "29/29"
    assert passed["profile_name"] == "3gpu-final-high-validation"
    assert "profile_name" in inspect.signature(_load_table).parameters

    evidence["aggregation_worker_count"] = 2
    blocked = evaluate_final_gate(evidence)
    assert blocked["final_gate"] == "BLOCKED"
    assert "aggregation_worker_count" in blocked["blockers"]


def test_final_docs_include_copy_paste_commands_and_runtime_deferral() -> None:
    docs = [
        "docs/3GPU_RUNBOOK.md",
        "docs/3GPU_ACCEPTANCE_CHECKLIST.md",
        "docs/AUTHOR_HYPERPARAMETER_AUDIT.md",
        "docs/3GPU_HYPERPARAMETER_DEVIATIONS.md",
    ]
    combined = "\n".join((ROOT / doc).read_text(encoding="utf-8") for doc in docs)

    for required in (
        BASELINE,
        "ray_direct",
        "TARGET_RUNTIME_EXECUTION_REQUIRED",
        "configs/author/latent_grpo_gsm8k_llama3.yaml",
        "configs/author/latent_grpo_math_qwen.yaml",
        "configs/3gpu-final-low.yaml",
        "configs/3gpu-final-validation.yaml",
        "configs/3gpu-final-high.yaml",
        "configs/3gpu-final-high-validation.yaml",
        "bash tools/run_3gpu_final_validation.sh",
        "bash tools/run_3gpu_training.sh",
        "3-GPU target-runtime / engineering adaptation",
        "0 silent deviations",
        "aggregation_worker_count",
        "CUDA_RNG_ALL_DEVICES",
    ):
        assert required in combined

    runbook = (ROOT / "docs/3GPU_RUNBOOK.md").read_text(encoding="utf-8")
    for required in (
        "https://huggingface.co/DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10",
        "https://huggingface.co/DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10",
        'export LOW_MODEL_PATH="$STORAGE_ROOT/models/LLaMA3.2-1B-Instruct-Latent-SFT-Top10"',
        'export HIGH_MODEL_PATH="$STORAGE_ROOT/models/Qwen2.5-Math-7B-Latent-SFT-4k-Top10"',
        'export LOW_TRAIN_DATA="$STORAGE_ROOT/data/GSM8k-Aug-oss-dup-all.parquet"',
        'export HIGH_TRAIN_DATA="$STORAGE_ROOT/data/DAPO-Math-17k-en-train.parquet"',
        "--config configs/3gpu-final-high-validation.yaml",
        "--config configs/3gpu-final-high.yaml",
        "local_model_directory_missing",
    ):
        assert required in runbook


def test_nvidia_smi_sample_includes_device_utilization() -> None:
    from scripts.target_machine.run_with_gpu_telemetry import sample

    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="0, GPU-a, A100, 123, 40960, 77\n",
        stderr="",
    )
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False), patch(
        "scripts.target_machine.run_with_gpu_telemetry.subprocess.run",
        return_value=completed,
    ):
        rows = sample()

    assert rows[0]["gpu_utilization_percent"] == 77


def test_nvidia_smi_sample_filters_selected_physical_indices() -> None:
    from scripts.target_machine.run_with_gpu_telemetry import sample

    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=(
            "0, GPU-0, L20, 30000, 46068, 100\n"
            "4, GPU-4, L20, 1, 46068, 0\n"
            "5, GPU-5, L20, 1, 46068, 0\n"
            "6, GPU-6, L20, 1, 46068, 0\n"
            "7, GPU-7, L20, 18063, 46068, 17\n"
        ),
        stderr="",
    )
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6"}, clear=False), patch(
        "scripts.target_machine.run_with_gpu_telemetry.subprocess.run",
        return_value=completed,
    ):
        rows = sample()

    assert [row["index"] for row in rows] == [4, 5, 6]
    assert all(row["memory_used_mib"] == 1 for row in rows)


def test_nvidia_smi_sample_preserves_cuda_visible_devices_order() -> None:
    from scripts.target_machine.run_with_gpu_telemetry import sample

    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=(
            "4, GPU-4, L20, 4, 46068, 10\n"
            "5, GPU-5, L20, 5, 46068, 20\n"
            "6, GPU-6, L20, 6, 46068, 30\n"
        ),
        stderr="",
    )
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "6,4,5"}, clear=False), patch(
        "scripts.target_machine.run_with_gpu_telemetry.subprocess.run",
        return_value=completed,
    ):
        rows = sample()

    assert [row["index"] for row in rows] == [6, 4, 5]
    assert [row["memory_used_mib"] for row in rows] == [6, 4, 5]


def test_runtime_telemetry_requires_two_complete_bounded_allocator_steps(tmp_path: Path) -> None:
    worker = lambda rank, reserved: {
        "worker_rank": rank,
        "device_index": 0,
        "current_allocated_bytes": reserved - 200,
        "current_reserved_bytes": reserved,
        "peak_allocated_bytes": reserved - 100,
        "peak_reserved_bytes": reserved,
    }
    (tmp_path / "gpu_runtime_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "gpu_runtime_metrics_v1",
                "steps": [
                    {"global_step": 1, "workers": [worker(rank, 1_000) for rank in range(3)]},
                    {"global_step": 2, "workers": [worker(rank, 1_100) for rank in range(3)]},
                ],
            }
        ),
        encoding="utf-8",
    )
    telemetry = {
        "schema_version": "gpu_telemetry_v2",
        "selected_physical_gpu_indices": [0, 1, 2],
        "exit_code": 0,
        "telemetry_error": None,
        "peak_memory_used_mib_by_gpu": {str(rank): 1000 for rank in range(3)},
        "memory_total_mib_by_gpu": {str(rank): 40960 for rank in range(3)},
        "peak_gpu_utilization_percent_by_gpu": {str(rank): 80 for rank in range(3)},
        "average_gpu_utilization_percent_by_gpu": {str(rank): 70 for rank in range(3)},
    }

    passed, details = _validate_runtime_telemetry(tmp_path, telemetry)

    assert passed is True
    assert details["allocator_step_count"] == 2
    assert details["reserved_growth_bytes_by_worker"] == {"0": 100, "1": 100, "2": 100}


def test_probe_runtime_evidence_requires_all_three_workers(tmp_path: Path) -> None:
    (tmp_path / "probe_worker_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": "probe_worker_runtime_v1",
                "checkpoints": [
                    {
                        "checkpoint_step": 2,
                        "workers": [
                            {
                                "worker_rank": rank,
                                "probe_extra_time_seconds": 0.1 + rank,
                                "probe_peak_memory_bytes": 1024 + rank,
                            }
                            for rank in range(3)
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    passed, details = _validate_probe_worker_runtime_evidence(tmp_path)

    assert passed is True
    assert len(details["workers"]) == 3


def test_active_cuda_dependency_metadata_is_internally_consistent() -> None:
    constraint = (ROOT / "constraints/linux-cu124-py311.txt").read_text(encoding="utf-8")
    runtime = (ROOT / "requirements/runtime-sglang.txt").read_text(encoding="utf-8")
    sglang_metadata = (
        ROOT / "Latent-GRPO/sglang_latent_reasoning_pkg/python/pyproject.toml"
    ).read_text(encoding="utf-8")
    installer = (ROOT / "scripts/target_machine/install_runtime.py").read_text(encoding="utf-8")

    for content in (constraint, runtime, sglang_metadata):
        assert "flashinfer-python==0.2.5" in content or "flashinfer_python==0.2.5" in content
        assert "0.2.3" not in content
    assert "sgl-kernel==0.1.1" in runtime
    assert "sgl-kernel==0.1.1" in sglang_metadata
    assert 'EXPECTED_SGL_KERNEL = "0.1.1"' in installer
    assert 'EXPECTED_FLASHINFER = "0.2.5"' in installer
    assert "requirements/tracking-optional.txt" in installer
    assert 'run("check")' in installer

    preflight = (ROOT / "tools/run_3gpu_preflight.sh").read_text(encoding="utf-8")
    assert "parquet.schema_arrow.names" in preflight
    assert "parquet.schema.names" not in preflight

    release_validator = (ROOT / "tools/validate_release_package.py").read_text(encoding="utf-8")
    assert "DEFERRED_TO_L20_SERVER" in release_validator
    assert "git_identity_and_clean_tree" in release_validator


def test_project_technical_handoff_is_model_ready() -> None:
    path = ROOT / "docs/PROJECT_TECHNICAL_HANDOFF.md"
    content = path.read_text(encoding="utf-8")
    for required in (
        "TARGET_RUNTIME_EXECUTION_REQUIRED",
        "ray_direct",
        "train_latent_grpo.py",
        "configs/3gpu-final-low.yaml",
        "configs/3gpu-final-validation.yaml",
        "29 个核心指标",
        "train/raw_generated_token_count",
        "3-GPU target-runtime / engineering adaptation",
        "tools/run_3gpu_final_validation.sh",
        "tools/run_3gpu_training.sh",
        "--acceptance-report",
        "给接手大模型的推荐提示词",
    ):
        assert required in content
    assert "/Users/" not in content
    for marker in ("T" + "BD", "T" + "ODO"):
        assert marker not in content
