#!/usr/bin/env python3
"""Data-free probe for the semantics-preserving dual-T4 runtime path."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_grpo_runner.environment import collect_environment
from latent_grpo_runner.validation.kaggle_t4_compatibility import assess_kaggle_t4_compatibility

CONFIG_PATH = ROOT / "latent_grpo_runner/config.py"
ACTOR_PATH = ROOT / "Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py"
FSDP_WORKER_PATH = ROOT / "Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py"
TORCH_FUNCTIONAL_PATH = ROOT / "Latent-GRPO/verl-0.4.x/verl/utils/torch_functional.py"
SGLANG_ROLLOUT_PATH = ROOT / "Latent-GRPO/verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="/kaggle/working/latent-grpo-p1-t4-compatibility.json",
    )
    return parser.parse_args(argv)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_identity():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return {
        "commit": completed.stdout.strip() if completed.returncode == 0 else None,
        "git_identity_available": completed.returncode == 0,
    }


def _write(path: Path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _runtime_import_checks():
    modules = (
        "sgl_kernel",
        "flashinfer",
        "cuda.bindings",
        "triton",
        "torchvision",
        "cachetools",
        "openai",
        "tiktoken",
        "torch_memory_saver",
        "sglang.srt.torch_memory_saver_adapter",
        "sglang.srt.layers.sampler",
        "sglang.srt.layers.attention.triton_backend",
        "verl.models.transformers.monkey_patch",
        "verl.workers.actor.dp_actor",
        "verl.workers.fsdp_workers",
        "verl.workers.rollout.async_server",
        "verl.trainer.ppo.ray_trainer",
    )
    expected_roots = {
        "sglang.srt.torch_memory_saver_adapter": ROOT / "Latent-GRPO/sglang_latent_reasoning_pkg/python",
        "sglang.srt.layers.sampler": ROOT / "Latent-GRPO/sglang_latent_reasoning_pkg/python",
        "sglang.srt.layers.attention.triton_backend": ROOT / "Latent-GRPO/sglang_latent_reasoning_pkg/python",
        "verl.models.transformers.monkey_patch": ROOT / "Latent-GRPO/verl-0.4.x",
        "verl.workers.actor.dp_actor": ROOT / "Latent-GRPO/verl-0.4.x",
        "verl.workers.fsdp_workers": ROOT / "Latent-GRPO/verl-0.4.x",
        "verl.workers.rollout.async_server": ROOT / "Latent-GRPO/verl-0.4.x",
        "verl.trainer.ppo.ray_trainer": ROOT / "Latent-GRPO/verl-0.4.x",
    }
    results = {}
    for module in modules:
        try:
            imported = importlib.import_module(module)
            module_file = getattr(imported, "__file__", None)
            path_ok = True
            expected_root = expected_roots.get(module)
            if expected_root is not None:
                if not module_file:
                    path_ok = False
                else:
                    try:
                        Path(module_file).resolve().relative_to(expected_root.resolve())
                    except ValueError:
                        path_ok = False
            results[module] = {
                "ok": path_ok,
                "error": None if path_ok else "imported_module_outside_repository_fork",
                "module_file": module_file,
            }
        except Exception as error:
            results[module] = {
                "ok": False,
                "error": f"{type(error).__name__}:{error}",
                "module_file": None,
            }
    return results


def main(argv=None):
    args = parse_args(argv)
    report_path = Path(args.report).expanduser().resolve()
    try:
        config_source = _read(CONFIG_PATH)
        actor = _read(ACTOR_PATH)
        fsdp = _read(FSDP_WORKER_PATH)
        torch_functional = _read(TORCH_FUNCTIONAL_PATH)
        sglang_rollout = _read(SGLANG_ROLLOUT_PATH)
        environment = collect_environment(
            mode="target",
            require_gpus=2,
            min_vram_gb=14,
            required_precision="float16",
            workspace_root=ROOT,
        )
        gumbel_start = torch_functional.index("def logprobs_from_logits_topk_gumbel")
        gumbel_end = torch_functional.index("def top_p_renorm_logprobs", gumbel_start)
        gumbel_source = torch_functional[gumbel_start:gumbel_end]
        import_checks = _runtime_import_checks()

        # Scope the HF attention check to the actor/reference builder contract.
        # The critic builder still contains a literal FlashAttention2 setting but
        # GRPO does not instantiate a critic, so scanning the entire file creates
        # a false positive for the T4 actor path.
        actor_attention_dispatch_ok = all(
            marker in fsdp
            for marker in (
                'attention_implementation = "flash_attention_2" if use_remove_padding else "sdpa"',
                "attn_implementation=attention_implementation",
            )
        )

        runtime_versions = {
            package: _version(package)
            for package in (
                "torch",
                "torchvision",
                "triton",
                "transformers",
                "sglang",
                "sgl-kernel",
                "compressed-tensors",
                "flashinfer-python",
                "cuda-python",
                "cuda-bindings",
                "ray",
                "pyarrow",
                "cachetools",
                "openai",
                "tiktoken",
                "torch-memory-saver",
            )
        }
        source_checks = {
            "runner_attention_backend_exposed": all(
                marker in config_source
                for marker in (
                    '"attention_backend"',
                    'actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend',
                )
            ),
            "actor_hardcodes_bfloat16": (
                "dtype=torch.bfloat16" in actor or ".to(torch.bfloat16)" in actor
            ),
            "model_forces_flash_attention_2": not actor_attention_dispatch_ok,
            "padded_latent_path_present": all(
                marker in actor
                for marker in (
                    "padded latent path: same Top-K/Gumbel semantics",
                    "inputs_embeds=topk_embs_final.detach()",
                    "next_topk_ids = rollout_topk_ids[:, -response_length:, :]",
                    "advantages=advantages",
                    "_latent_mixture_weights",
                )
            ),
            "gumbel_logprob_requires_flash_attention": (
                "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE" in gumbel_source
            ),
            "sglang_triton_attention_forwarded": (
                "attention_backend=attention_backend" in sglang_rollout
                and 'attention_backend = sglang_engine_kwargs.get("attention_backend", None)' in sglang_rollout
            ),
            "flashinfer_sampling_preserved": (
                "sampling_backend=self.config.get" in sglang_rollout
                and "'flashinfer'" in sglang_rollout
            ),
            "triton_version_ok": str(runtime_versions["triton"] or "").startswith("3.2.0"),
            "torchvision_version_ok": str(runtime_versions["torchvision"] or "").startswith("0.21.0+cu124"),
            "runtime_imports_ok": all(item["ok"] for item in import_checks.values()),
        }
        report = assess_kaggle_t4_compatibility(environment, **source_checks)
        report.update(
            {
                "environment_status": environment.get("status"),
                "environment_failure_reasons": environment.get("failure_reasons", []),
                "source_checks": source_checks,
                "runtime_versions": runtime_versions,
                "runtime_import_checks": import_checks,
                **_git_identity(),
            }
        )
    except Exception as error:
        report = {
            "status": "BLOCKED",
            "gate": "kaggle_dual_t4_p1_runtime_compatibility",
            "training_started": False,
            "training_data_required_now": False,
            "training_data_inspected": False,
            "training_data_generated": False,
            "blockers": [f"compatibility_probe_error:{error}"],
            "next_gate": "fix_probe_or_runtime_before_requesting_data",
            **_git_identity(),
        }
    _write(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "READY_FOR_DATA" else 3


if __name__ == "__main__":
    raise SystemExit(main())
