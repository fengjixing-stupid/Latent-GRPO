from __future__ import annotations

import unittest

from latent_grpo_runner.metrics.aggregators import SufficientStats
from latent_grpo_runner.metrics.events import StepContext
from latent_grpo_runner.metrics.stage1 import build_train_step_metrics
from latent_grpo_runner.metrics.stage2 import Stage2SufficientStats


class PreBackwardMonitorMetricTests(unittest.TestCase):
    def test_probe_keeps_real_metrics_but_never_claims_step_time(self) -> None:
        context = StepContext(
            profile_name="kaggle-t4-monitor",
            seed=17,
            global_step=1,
            optimizer_step=0,
            observation_phase="pre_backward_probe",
        )
        stats = {
            "train/policy_loss": SufficientStats(sum=1.0, sum_sq=1.0, count=1),
            "train/entropy": SufficientStats(sum=2.0, sum_sq=4.0, count=1),
            "train/kl": SufficientStats(sum=0.1, sum_sq=0.01, count=1),
            "train/clip_fraction": SufficientStats(count=1, numerator_count=0),
            "train/importance_ratio": SufficientStats(sum=1.0, sum_sq=1.0, count=1),
            "train/response_length": SufficientStats(sum=8.0, sum_sq=64.0, count=1),
            "train/latent_length": SufficientStats(sum=3.0, sum_sq=9.0, count=1),
        }
        stage2 = Stage2SufficientStats(
            mixture_effective_k=SufficientStats(sum=2.0, sum_sq=4.0, count=1),
            mixture_top1_weight=SufficientStats(sum=0.7, sum_sq=0.49, count=1),
            zero_advantage=SufficientStats(count=1, numerator_count=0),
            reward=SufficientStats(sum=1.0, sum_sq=1.0, count=1),
            advantage=SufficientStats(sum=0.5, sum_sq=0.25, count=1),
        )
        row = build_train_step_metrics(
            context,
            stats,
            [8],
            driver_step_time_seconds=None,
            stage2_statistics=stage2,
            aggregation_worker_count=2,
            record_version="metrics_record_p1_pre_backward_probe_v1",
        )
        self.assertEqual(row["observation_phase"], "pre_backward_probe")
        self.assertEqual(row["optimizer_step"], 0)
        self.assertTrue(row["train/policy_loss__available"])
        self.assertFalse(row["train/step_time__available"])
        self.assertEqual(
            row["train/step_time__unavailable_reason"],
            "pre_backward_probe_no_actor_update",
        )
        self.assertFalse(row["train_core_available"])

    def test_probe_rejects_fabricated_post_update_step_time(self) -> None:
        context = StepContext(
            profile_name="kaggle-t4-monitor",
            seed=17,
            global_step=1,
            optimizer_step=0,
            observation_phase="pre_backward_probe",
        )
        with self.assertRaisesRegex(ValueError, "must not report post-update step_time"):
            build_train_step_metrics(context, {}, [], driver_step_time_seconds=1.0)


if __name__ == "__main__":
    unittest.main()
