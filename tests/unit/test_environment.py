from __future__ import annotations

import unittest

from latent_grpo_runner.environment import build_report_envelope, collect_environment, validate_target_environment


class EnvironmentTests(unittest.TestCase):
    def test_development_mode_is_successful_and_explicitly_deferred(self) -> None:
        report = collect_environment(mode="development", platform_name="Darwin", machine="arm64")
        self.assertEqual(report["status"], "mac_development_check_passed")
        self.assertEqual(report["host_platform"], "macos_arm64")
        self.assertFalse(report["target_gpu_environment_available"])
        self.assertFalse(report["cuda_available"])
        self.assertEqual(report["training_runtime_validation"], "deferred_to_target_machine")

    def test_sensitive_values_are_hashed(self) -> None:
        report = collect_environment(
            mode="development",
            platform_name="Darwin",
            machine="arm64",
            hostname="developer-laptop",
            username="alice",
            python_executable="/Users/alice/private/venv/bin/python",
        )
        self.assertNotIn("developer-laptop", str(report))
        self.assertNotIn("alice", str(report))
        self.assertNotIn("/Users/alice", str(report))
        self.assertTrue(report["hostname_redacted"].startswith("sha256:"))

    def test_target_mode_rejects_missing_gpus_with_stable_reason(self) -> None:
        reasons = validate_target_environment(
            {
                "host_platform": "linux_x86_64",
                "cuda_available": False,
                "gpu_count": 0,
                "gpu_total_memory_bytes": [],
                "bf16_supported": False,
                "nccl_available": False,
            },
            require_gpus=3,
            min_vram_gb=40,
        )
        self.assertEqual(reasons[0], "gpu_count_below_requirement")
        self.assertIn("cuda_unavailable", reasons)
        self.assertIn("bf16_unsupported", reasons)
        self.assertIn("nccl_unavailable", reasons)

    def test_target_mode_requires_each_gpu_to_meet_vram_threshold(self) -> None:
        gib = 1024**3
        reasons = validate_target_environment(
            {
                "host_platform": "linux_x86_64",
                "cuda_available": True,
                "gpu_count": 3,
                "gpu_total_memory_bytes": [46 * gib, 39 * gib, 46 * gib],
                "bf16_supported": True,
                "nccl_available": True,
            },
            require_gpus=3,
            min_vram_gb=40,
        )
        self.assertEqual(reasons, ["gpu_vram_below_requirement"])

    def test_target_mode_reports_driver_wheel_incompatibility_stably(self) -> None:
        gib = 1024**3
        reasons = validate_target_environment(
            {
                "host_platform": "linux_x86_64",
                "cuda_available": True,
                "gpu_count": 3,
                "gpu_total_memory_bytes": [46 * gib, 46 * gib, 46 * gib],
                "bf16_supported": True,
                "nccl_available": True,
                "cuda_driver_wheel_compatible": False,
            },
            require_gpus=3,
            min_vram_gb=40,
        )
        self.assertEqual(reasons, ["cuda_driver_wheel_incompatible"])

    def test_t4_fp16_target_does_not_require_bf16(self) -> None:
        gib = 1024**3
        reasons = validate_target_environment(
            {
                "host_platform": "linux_x86_64",
                "cuda_available": True,
                "gpu_count": 2,
                "gpu_total_memory_bytes": [15 * gib, 15 * gib],
                "bf16_supported": False,
                "fp16_supported": True,
                "nccl_available": True,
            },
            require_gpus=2,
            min_vram_gb=14,
            required_precision="float16",
        )
        self.assertEqual(reasons, [])

    def test_target_report_envelope_has_required_machine_readable_fields(self) -> None:
        report = build_report_envelope(
            command=["python", "scripts/check_environment.py", "--mode", "target"],
            started_at="2026-08-02T00:00:00+00:00",
            finished_at="2026-08-02T00:00:01+00:00",
            exit_code=1,
            status="blocked",
            environment_summary={"gpu_count": 0},
            artifacts=["artifacts/target_machine/runtime_probe.json"],
            failure_reason="gpu_count_below_requirement",
        )
        self.assertEqual(
            set(report),
            {
                "command",
                "started_at",
                "finished_at",
                "exit_code",
                "status",
                "environment_summary",
                "stdout_log_path",
                "stderr_log_path",
                "artifacts",
                "failure_reason",
            },
        )
        self.assertEqual(report["failure_reason"], "gpu_count_below_requirement")


if __name__ == "__main__":
    unittest.main()
