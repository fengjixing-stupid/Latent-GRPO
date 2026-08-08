from __future__ import annotations

import unittest

from latent_grpo_runner.validation.kaggle_t4_compatibility import assess_kaggle_t4_compatibility


class KaggleT4CompatibilityTests(unittest.TestCase):
    def _environment(self, *, bf16_supported: bool) -> dict[str, object]:
        return {
            "gpu_names": ["Tesla T4", "Tesla T4"],
            "gpu_compute_capabilities": ["7.5", "7.5"],
            "cuda_available": True,
            "nccl_available": True,
            "bf16_supported": bf16_supported,
        }

    def test_current_t4_path_fails_closed_before_training_data(self) -> None:
        report = assess_kaggle_t4_compatibility(
            self._environment(bf16_supported=False),
            actor_hardcodes_bfloat16=True,
            model_forces_flash_attention_2=True,
        )
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["training_started"])
        self.assertFalse(report["training_data_required_now"])
        self.assertFalse(report["training_data_inspected"])
        self.assertFalse(report["training_data_generated"])
        self.assertIn(
            "current_target_path_requires_bf16_but_t4_is_not_bf16_capable",
            report["blockers"],
        )
        self.assertIn("actor_forward_hardcodes_bfloat16", report["blockers"])
        self.assertIn("actor_model_forces_flash_attention_2_on_turing", report["blockers"])

    def test_ready_state_is_only_permission_to_request_real_data(self) -> None:
        report = assess_kaggle_t4_compatibility(
            self._environment(bf16_supported=True),
            actor_hardcodes_bfloat16=False,
            model_forces_flash_attention_2=False,
        )
        self.assertEqual(report["status"], "READY_FOR_DATA")
        self.assertTrue(report["training_data_required_now"])
        self.assertFalse(report["training_started"])
        self.assertEqual(report["blockers"], [])


if __name__ == "__main__":
    unittest.main()
