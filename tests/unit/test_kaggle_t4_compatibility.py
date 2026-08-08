from __future__ import annotations

import unittest

from latent_grpo_runner.validation.kaggle_t4_compatibility import assess_kaggle_t4_compatibility


class KaggleT4CompatibilityTests(unittest.TestCase):
    def _environment(self, *, fp16_supported: bool = True) -> dict[str, object]:
        versions = {
            "torch": "2.6.0+cu118",
            "ray": "test",
            "sglang": "0.4.6.post1",
            "sgl-kernel": "0.1.1",
            "flashinfer-python": "0.2.5+cu118torch2.6",
            "cuda-python": "11.8.6",
            "cuda-bindings": "11.8.6",
            "pyarrow": "test",
        }
        packages = {name: {"status": "present", "version": version} for name, version in versions.items()}
        return {
            "gpu_names": ["Tesla T4", "Tesla T4"],
            "gpu_compute_capabilities": ["7.5", "7.5"],
            "cuda_available": True,
            "nccl_available": True,
            "bf16_supported": False,
            "fp16_supported": fp16_supported,
            "cuda_runtime_version": "11.8",
            "dependency_check_status": packages,
        }

    def _assess(self, **overrides):
        kwargs = {
            "runner_attention_backend_exposed": True,
            "actor_hardcodes_bfloat16": False,
            "model_forces_flash_attention_2": False,
            "padded_latent_path_present": True,
            "gumbel_logprob_requires_flash_attention": False,
            "sglang_triton_attention_forwarded": True,
            "flashinfer_sampling_preserved": True,
            "runtime_imports_ok": True,
        }
        kwargs.update(overrides)
        return assess_kaggle_t4_compatibility(self._environment(), **kwargs)

    def test_ready_state_accepts_t4_fp16_without_bf16(self) -> None:
        report = self._assess()
        self.assertEqual(report["status"], "READY_FOR_DATA")
        self.assertTrue(report["training_data_required_now"])
        self.assertFalse(report["training_started"])
        self.assertEqual(report["runtime_precision"], "fp16")
        self.assertEqual(report["sampling_backend"], "flashinfer_preserved")
        self.assertEqual(report["blockers"], [])

    def test_semantic_backend_regression_fails_closed(self) -> None:
        report = self._assess(flashinfer_sampling_preserved=False)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("author_sampling_backend_not_preserved", report["blockers"])

    def test_missing_padded_latent_path_fails_closed(self) -> None:
        report = self._assess(padded_latent_path_present=False)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("padded_latent_actor_path_missing", report["blockers"])


if __name__ == "__main__":
    unittest.main()
