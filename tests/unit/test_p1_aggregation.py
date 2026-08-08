from __future__ import annotations

import math
import unittest

from latent_grpo_runner.metrics.aggregators import SufficientStats
from latent_grpo_runner.metrics.events import StepContext
from latent_grpo_runner.metrics.p1 import (
    build_p1_train_step_metrics,
    merge_worker_p1_packets,
    sufficient_stats_to_record,
)


def rec(values, *, numerators=None, version="v1"):
    stat = SufficientStats.from_values(
        values,
        numerator_mask=([False] * len(values) if numerators is None else numerators),
    )
    return sufficient_stats_to_record(stat, definition_version=version)


def worker(rank, policy, kl, clip, ratio):
    return {
        "worker_rank": rank,
        "p1_sufficient_stats": {
            "train/policy_loss": rec(policy, version="policy_v1"),
            "train/kl": rec(kl, version="kl_v1"),
            "train/clip_fraction": rec(
                [float(value) for value in clip],
                numerators=[bool(value) for value in clip],
                version="clip_v1",
            ),
            "train/importance_ratio": rec(ratio, version="ratio_v1"),
        },
    }


class P1AggregationTests(unittest.TestCase):
    def test_two_worker_merge_is_global_not_mean_of_means(self):
        merged = merge_worker_p1_packets(
            [
                worker(0, [1.0, 3.0], [0.1, 0.3], [1, 0], [1.0, 2.0]),
                worker(1, [9.0], [0.9], [1], [10.0]),
            ],
            expected_worker_count=2,
        )
        self.assertTrue(merged["p1_worker_metrics_available"])
        stats = merged["p1_worker_sufficient_stats"]
        self.assertEqual(stats["train/policy_loss"]["count"], 3)
        self.assertAlmostEqual(stats["train/policy_loss"]["sum"] / 3, 13 / 3)
        # Worker means would be (2 + 9) / 2 == 5.5, which must not appear.
        self.assertNotAlmostEqual(stats["train/policy_loss"]["sum"] / 3, 5.5)
        self.assertEqual(stats["train/clip_fraction"]["numerator_count"], 2)
        self.assertEqual(stats["train/clip_fraction"]["count"], 3)
        ratio_mean = stats["train/importance_ratio"]["sum"] / 3
        ratio_var = stats["train/importance_ratio"]["sum_sq"] / 3 - ratio_mean**2
        self.assertAlmostEqual(ratio_mean, 13 / 3)
        self.assertAlmostEqual(math.sqrt(ratio_var), math.sqrt(((1-13/3)**2+(2-13/3)**2+(10-13/3)**2)/3))

    def test_three_worker_merge_preserves_exact_counts(self):
        merged = merge_worker_p1_packets(
            [
                worker(0, [1], [1], [0], [1]),
                worker(1, [2, 3], [2, 3], [1, 0], [2, 3]),
                worker(2, [4, 5, 6], [4, 5, 6], [1, 1, 0], [4, 5, 6]),
            ],
            expected_worker_count=3,
        )
        stats = merged["p1_worker_sufficient_stats"]
        self.assertEqual(stats["train/policy_loss"]["count"], 6)
        self.assertEqual(stats["train/policy_loss"]["sum"], 21.0)
        self.assertEqual(stats["train/clip_fraction"]["numerator_count"], 3)
        self.assertEqual(stats["train/clip_fraction"]["count"], 6)

    def test_incomplete_or_duplicate_worker_set_fails_closed(self):
        merged = merge_worker_p1_packets(
            [worker(0, [1], [1], [0], [1]), worker(0, [2], [2], [1], [2])],
            expected_worker_count=2,
        )
        self.assertFalse(merged["p1_worker_metrics_available"])
        self.assertEqual(
            merged["p1_worker_metrics_unavailable_reason"],
            "worker_packet_set_incomplete_or_duplicate",
        )

    def test_stage1_stage2_authoritative_row_uses_merged_stats(self):
        merged = merge_worker_p1_packets(
            [
                worker(0, [1.0, 3.0], [0.1, 0.3], [1, 0], [1.0, 2.0]),
                worker(1, [9.0], [0.9], [1], [10.0]),
            ],
            expected_worker_count=2,
        )
        driver = {
            "train/entropy": rec([0.2, 0.4, 0.6], version="entropy_v1"),
            "train/response_length": rec([2, 3, 4], version="response_v1"),
            "train/latent_length": rec([1, 2, 1], version="latent_v1"),
            "mixture/effective_k_noisy": rec([2, 4], version="mixture_v1"),
            "mixture/top1_weight_noisy": rec([0.75, 0.5], version="mixture_v1"),
            "mask/zero_advantage_rate": rec([0, 1, 0, 1], numerators=[True, False, True, False], version="zero_v1"),
            "signal/reward": rec([1, 0, 2], version="reward_v1"),
            "signal/advantage": rec([-1, 0, 1, 2], version="adv_v1"),
        }
        row = build_p1_train_step_metrics(
            context=StepContext("smoke", 17, 5, 8, "post_update"),
            worker_statistics=merged["p1_worker_sufficient_stats"],
            driver_statistics=driver,
            final_training_trajectory_lengths=[2, 3, 4],
            driver_step_time_seconds=1.25,
            aggregation_worker_count=2,
        )
        self.assertTrue(row["record_available"])
        self.assertTrue(row["train_core_available"])
        self.assertTrue(row["stage2_available"])
        self.assertAlmostEqual(row["train/policy_loss"], 13 / 3)
        self.assertAlmostEqual(row["train/clip_fraction"], 2 / 3)
        self.assertEqual(row["train/generated_token_count"], 9)
        self.assertEqual(row["final_training_trajectory_count"], 3)
        self.assertAlmostEqual(row["mask/zero_advantage_rate"], 0.5)
        self.assertEqual(row["aggregation_worker_count"], 2)
        self.assertEqual(row["record_version"], "metrics_record_p1_v1")


if __name__ == "__main__":
    unittest.main()
