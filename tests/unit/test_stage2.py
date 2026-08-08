import math
import unittest


class Stage2Tests(unittest.TestCase):
    def test_noisy_mixture_effective_k_and_top1_use_actual_weights(self):
        from latent_grpo_runner.metrics.stage2 import noisy_mixture_stats

        row = noisy_mixture_stats([[0.5, 0.5], [1.0, 0.0]])
        self.assertAlmostEqual(row["mixture/effective_k_noisy"], 1.5)
        self.assertEqual(row["mixture/top1_weight_noisy"], 0.75)
        self.assertEqual(row["mixture/noisy_count"], 2)

    def test_signal_stats_and_group_counts_keep_overlong_out_of_binary_total(self):
        from latent_grpo_runner.metrics.stage2 import build_group_metrics, signal_stats

        signals = signal_stats([1.0, 3.0], [1.0, -1.0, 0.0])
        self.assertEqual(signals["signal/reward_mean"], 2.0)
        self.assertEqual(signals["signal/advantage_std"], math.sqrt(2.0 / 3.0))
        row = build_group_metrics("g", [
            {"trajectory_class": "correct", "is_overlong_or_truncated_by_length": True, "generated_token_count": 8, "response_length": 8},
            {"trajectory_class": "non_correct", "is_overlong_or_truncated_by_length": False, "generated_token_count": 3, "response_length": 3},
        ])
        self.assertEqual((row["group/correct_trajectory_count"], row["group/non_correct_trajectory_count"], row["group/overlong_trajectory_count"]), (1, 1, 1))

    def test_gumbel_is_explicit_diagnostic_and_uses_distinct_denominators(self):
        from latent_grpo_runner.metrics.stage2 import gumbel_diagnostic

        disabled = gumbel_diagnostic([1.0], [0.0], enabled=False)
        self.assertFalse(disabled["gumbel_available"])
        self.assertEqual(disabled["gumbel_unavailable_reason"], "disabled_by_config")
        enabled = gumbel_diagnostic([-2.0, 2.0], [0.0, 1.0], enabled=True, lower_clip=-1.0, upper_clip=1.0)
        self.assertEqual((enabled["gumbel/raw_count"], enabled["gumbel/one_sided_count"]), (2, 2))
        self.assertEqual((enabled["gumbel/lower_clip_rate"], enabled["gumbel/upper_clip_rate"], enabled["gumbel/zero_rate"]), (0.5, 0.5, 0.5))

    def test_gumbel_nonfinite_values_are_excluded_from_each_rate_domain(self):
        from latent_grpo_runner.metrics.stage2 import gumbel_diagnostic

        row = gumbel_diagnostic([float("nan"), 2.0], [float("inf"), 0.0], enabled=True, lower_clip=0.0, upper_clip=1.0)
        self.assertEqual((row["gumbel/raw_count"], row["gumbel/one_sided_count"]), (1, 1))
        self.assertEqual((row["gumbel/upper_clip_rate"], row["gumbel/zero_rate"]), (1.0, 1.0))

    def test_mechanism_interface_returns_only_reduced_statistics(self):
        from latent_grpo_runner.metrics.stage2 import mechanism_stats

        row = mechanism_stats([-1.0, 0.0, 2.0], [True, True, False], [True, False, True], near_zero_threshold=0.1)
        self.assertEqual((row["count"], row["negative_count"], row["near_zero_count"], row["flipgrad_trigger_count"]), (2, 1, 1, 1))
        self.assertNotIn("surrogate_margin", row)

    def test_mechanism_nonfinite_margin_is_excluded_from_every_auxiliary_count(self):
        from latent_grpo_runner.metrics.stage2 import mechanism_stats

        row = mechanism_stats([float("nan"), -1.0], [True, True], [True, True], near_zero_threshold=0.1)
        self.assertEqual((row["count"], row["negative_count"], row["near_zero_count"], row["flipgrad_trigger_count"]), (1, 1, 0, 1))

    def test_stage2_worker_stats_merge_before_driver_record_and_attach_to_train_step(self):
        from latent_grpo_runner.metrics.events import StepContext
        from latent_grpo_runner.metrics.stage1 import build_train_step_metrics
        from latent_grpo_runner.metrics.stage2 import Stage2SufficientStats, build_stage2_metrics

        first = Stage2SufficientStats.from_local([[0.5, 0.5]], [0.0, 1.0], [True, True], [1.0], [1.0])
        second = Stage2SufficientStats.from_local([[1.0, 0.0]], [0.0], [True], [3.0], [-1.0])
        merged = first.merge(second)
        stage2 = build_stage2_metrics(merged)
        self.assertEqual(stage2["mixture/noisy_count"], 2)
        self.assertEqual(stage2["signal/reward_mean"], 2.0)
        self.assertTrue(stage2["signal/reward_mean__available"])
        row = build_train_step_metrics(StepContext("smoke", 1, 4, 3, "post_update"), {}, [2], driver_step_time_seconds=1.0, stage2_statistics=merged)
        self.assertEqual(row["mask/eligible_latent_token_count"], 3)
        self.assertIn("signal/advantage_std", row)

    def test_full_group_record_uses_context_and_same_memory_ocp_winner(self):
        from latent_grpo_runner.metrics.events import StepContext
        from latent_grpo_runner.metrics.stage2 import build_train_group_metrics, select_optimal_correct_path

        trajectories = [
            {"trajectory_id": 9, "trajectory_class": "correct", "is_overlong_or_truncated_by_length": True, "generated_token_count": 8, "response_length": 8, "reward": 2.0, "first_step_advantage": 1.0, "trajectory_mean_old_log_prob": -2.0},
            {"trajectory_id": 10, "trajectory_class": "non_correct", "is_overlong_or_truncated_by_length": False, "generated_token_count": 3, "response_length": 3, "reward": 2.0, "first_step_advantage": -1.0, "trajectory_mean_old_log_prob": -0.2},
        ]
        winner = select_optimal_correct_path(trajectories)
        row = build_train_group_metrics(StepContext("smoke", 1, 4, 3, "post_advantage_pre_update"), "group-1", "prompt-hash", trajectories, winner)
        self.assertEqual((row["profile_name"], row["prompt_id_or_hash"], row["optimal_correct_trajectory_id"]), ("smoke", "prompt-hash", 9))
        self.assertTrue(row["group/zero_variance_reward"])
        self.assertEqual(row["optimal_correct_mean_old_log_prob"], -2.0)
        self.assertEqual(row["observation_phase"], "post_advantage_pre_update")

    def test_train_group_emits_no_undeclared_fields_and_all_quality_context(self):
        from latent_grpo_runner.metrics.events import StepContext
        from latent_grpo_runner.metrics.schemas import schema_manifest
        from latent_grpo_runner.metrics.stage2 import build_train_group_metrics

        row = build_train_group_metrics(
            StepContext("smoke", 1, 4, 3, "post_advantage_pre_update"), "g", "prompt-hash", [], None,
        )
        declared = {field["name"] for field in schema_manifest()["tables"]["train_group_metrics"]["fields"]}
        self.assertEqual(set(row), declared)
        self.assertEqual(row["record_version"], "metrics_record_v1")
        self.assertIsNone(row["aggregation_worker_count"])
        self.assertIsNone(row["optimal_correct_trajectory_id"])
