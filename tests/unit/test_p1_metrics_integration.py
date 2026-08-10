from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from latent_grpo_runner.metrics.aggregators import SufficientStats
from latent_grpo_runner.metrics.integration import DurableMetricsObserver
from latent_grpo_runner.metrics.p1 import (
    merge_worker_p1_packets,
    sufficient_stats_to_record,
)
from tests.unit.test_storage import JsonBackend


def _record(values, *, numerators=None, version):
    stats = SufficientStats.from_values(
        values,
        numerator_mask=([False] * len(values) if numerators is None else numerators),
    )
    return sufficient_stats_to_record(stats, definition_version=version)


def _worker(rank, offset):
    return {
        "worker_rank": rank,
        "p1_sufficient_stats": {
            "train/policy_loss": _record([1.0 + offset], version="policy_v1"),
            "train/kl": _record([0.1 + offset], version="kl_v1"),
            "train/clip_fraction": _record([1.0], numerators=[rank == 0], version="clip_v1"),
            "train/importance_ratio": _record([1.0 + offset], version="ratio_v1"),
        },
    }


class P1DurableIntegrationTests(unittest.TestCase):
    def test_p1_actor_update_writes_one_authoritative_available_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observer = DurableMetricsObserver(
                output_root=root,
                profile_name="smoke",
                seed=17,
                config_hash="cfg",
                backend=JsonBackend(),
            )
            observer.start_run()
            workers = merge_worker_p1_packets(
                [_worker(0, 0.0), _worker(1, 1.0)], expected_worker_count=2
            )
            driver = {
                "train/entropy": _record([0.2, 0.4], version="entropy_v1"),
                "train/response_length": _record([2, 3], version="response_v1"),
                "train/latent_length": _record([1, 1], version="latent_v1"),
                "mixture/effective_k_noisy": _record([2, 2], version="mix_v1"),
                "mixture/top1_weight_noisy": _record([0.5, 0.5], version="mix_v1"),
                "mask/zero_advantage_rate": _record(
                    [0, 1], numerators=[True, False], version="zero_v1"
                ),
                "signal/reward": _record([1, 0], version="reward_v1"),
                "signal/advantage": _record([-1, 1], version="adv_v1"),
            }
            observer.emit(
                "actor_update",
                {
                    "global_step": 1,
                    "observation_phase": "post_update",
                    "aggregation_worker_count": 2,
                    "optimizer_update_available": True,
                    "update_count": 1,
                    "did_update": True,
                    **workers,
                    "p1_driver_metrics_available": True,
                    "p1_driver_sufficient_stats": driver,
                    "final_training_trajectory_lengths": [2, 3],
                    "raw_generated_trajectory_lengths": [2, 3, 4],
                    "driver_step_time_seconds": 1.5,
                    "metrics_compute_time": 0.01,
                    "learning_rate": 1e-6,
                },
            )
            parts = list((root / "train_step_metrics").glob("part-*.parquet"))
            self.assertEqual(len(parts), 1)
            row = JsonBackend().read(parts[0])["rows"][0]
            self.assertTrue(row["record_available"])
            self.assertTrue(row["train_core_available"])
            self.assertTrue(row["stage2_available"])
            self.assertEqual(row["record_version"], "metrics_record_p1_v1")
            self.assertEqual(row["train/generated_token_count"], 5)
            self.assertEqual(row["train/raw_generated_token_count"], 9)
            self.assertAlmostEqual(row["train/policy_loss"], 1.5)
            self.assertAlmostEqual(row["train/clip_fraction"], 0.5)
            self.assertEqual(row["optimizer_step"], 1)
            observer.close()


if __name__ == "__main__":
    unittest.main()
