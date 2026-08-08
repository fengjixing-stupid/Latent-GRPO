import unittest


class Stage1Tests(unittest.TestCase):
    def test_train_step_has_ten_core_metrics_definitions_and_exact_token_sum(self):
        from latent_grpo_runner.metrics.aggregators import SufficientStats
        from latent_grpo_runner.metrics.events import StepContext
        from latent_grpo_runner.metrics.stage1 import build_train_step_metrics

        stats = {
            "train/policy_loss": SufficientStats.from_values([1.0]),
            "train/entropy": SufficientStats.from_values([2.0]),
            "train/kl": SufficientStats.from_values([3.0]),
            "train/clip_fraction": SufficientStats.from_values([0.25], numerator_mask=[True]),
            "train/importance_ratio": SufficientStats.from_values([1.0, 3.0]),
            "train/response_length": SufficientStats.from_values([3.0, 5.0]),
            "train/latent_length": SufficientStats.from_values([1.0, 2.0]),
        }
        row = build_train_step_metrics(StepContext("smoke", 1, 4, 3, "post_update"), stats, [3, 5, 7], driver_step_time_seconds=0.5)
        core = {
            "train/policy_loss", "train/entropy", "train/kl", "train/clip_fraction",
            "train/importance_ratio_mean", "train/importance_ratio_std", "train/response_length",
            "train/latent_length", "train/generated_token_count", "train/step_time",
        }
        self.assertTrue(core.issubset(row))
        self.assertEqual(row["train/generated_token_count"], 15)
        self.assertEqual(row["generated_token_count_scope"], "final_training_rollout_trajectories")
        self.assertEqual(row["entropy_source"], "runtime_policy_entropy")
        self.assertIn("entropy_probability_space", row)
        self.assertIn("entropy_mask_definition", row)
        self.assertIn("entropy_definition_version", row)
        self.assertNotIn("train/gradient_norm", row)

    def test_step_time_uses_driver_whole_step_clock_not_worker_average(self):
        from latent_grpo_runner.metrics.aggregators import SufficientStats
        from latent_grpo_runner.metrics.events import StepContext
        from latent_grpo_runner.metrics.stage1 import build_train_step_metrics

        row = build_train_step_metrics(
            StepContext("smoke", 1, 4, 3, "post_update"),
            {"train/step_time": SufficientStats.from_values([1.0, 100.0])}, [1],
            driver_step_time_seconds=3.5,
        )
        self.assertEqual(row["train/step_time"], 3.5)
        self.assertNotIn("train/step_time_count", row)

    def test_empty_final_trajectory_domain_is_unavailable(self):
        from latent_grpo_runner.metrics.events import StepContext
        from latent_grpo_runner.metrics.stage1 import build_train_step_metrics

        row = build_train_step_metrics(StepContext("smoke", 1, 4, 3, "post_update"), {}, [], driver_step_time_seconds=1.0)
        self.assertIsNone(row["train/generated_token_count"])
        self.assertFalse(row["train/generated_token_count__available"])
        self.assertEqual(row["train/generated_token_count__unavailable_reason"], "empty_effective_mask")

    def test_no_stage2_runtime_interface_still_emits_schema_complete_null_fields(self):
        from latent_grpo_runner.metrics.events import StepContext
        from latent_grpo_runner.metrics.schemas import schema_manifest
        from latent_grpo_runner.metrics.stage1 import build_train_step_metrics

        row = build_train_step_metrics(StepContext("smoke", 1, 4, 3, "post_update"), {}, [1], driver_step_time_seconds=1.0)
        fields = {field["name"] for field in schema_manifest()["tables"]["train_step_metrics"]["fields"]}
        self.assertTrue(fields.issubset(row))
        self.assertIsNone(row["signal/reward_mean"])
        self.assertFalse(row["signal/reward_mean__available"])

    def test_train_step_emits_no_undeclared_fields_and_includes_quality_fields(self):
        from latent_grpo_runner.metrics.events import StepContext
        from latent_grpo_runner.metrics.schemas import schema_manifest
        from latent_grpo_runner.metrics.stage1 import build_train_step_metrics

        row = build_train_step_metrics(StepContext("smoke", 1, 4, 3, "post_update"), {}, [1], driver_step_time_seconds=1.0)
        declared = {field["name"] for field in schema_manifest()["tables"]["train_step_metrics"]["fields"]}
        self.assertEqual(set(row), declared)
        self.assertIn("train_core_available", row)
        self.assertIn("record_version", row)
        self.assertIn("is_resume_run", row)

    def test_missing_stat_is_null_and_has_metric_availability(self):
        from latent_grpo_runner.metrics.events import StepContext
        from latent_grpo_runner.metrics.stage1 import build_train_step_metrics

        row = build_train_step_metrics(StepContext("smoke", 1, 4, 3, "post_update"), {}, [])
        self.assertIsNone(row["train/policy_loss"])
        self.assertFalse(row["train/policy_loss__available"])
        self.assertEqual(row["train/policy_loss__unavailable_reason"], "missing_runtime_interface")
