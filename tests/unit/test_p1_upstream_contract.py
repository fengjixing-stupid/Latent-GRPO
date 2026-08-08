from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class P1UpstreamContractTests(unittest.TestCase):
    def test_policy_loss_exposes_existing_intermediates_only_when_requested(self):
        source = (ROOT / "Latent-GRPO/verl-0.4.x/verl/trainer/ppo/core_algos.py").read_text(encoding="utf-8")
        self.assertIn("return_observer_tensors: bool = False", source)
        self.assertIn('"policy_loss_elements": pg_losses', source)
        self.assertIn('"importance_ratio": ratio', source)
        self.assertIn('"clip_mask": clip_mask', source)

    def test_actor_handoff_contains_bounded_p1_sufficient_stats(self):
        source = (ROOT / "Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
        self.assertIn('"p1_sufficient_stats"', source)
        self.assertIn("sufficient_stats_from_tensor", source)
        self.assertIn("merge_serialized_sufficient_stats", source)
        self.assertNotIn('self._latent_grpo_observer_facts["p1_raw_tensors"]', source)

    def test_driver_merges_worker_stats_and_collects_final_batch_before_write(self):
        source = (ROOT / "Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
        self.assertIn("merge_worker_p1_packets", source)
        self.assertIn("collect_driver_p1_statistics", source)
        self.assertIn("p1_driver_compute_time", source)
        reduce_index = source.index('actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])')
        emit = source.index('self._latent_grpo_observer.emit("actor_update", actor_update_event)')
        self.assertLess(reduce_index, emit)

    def test_runtime_collector_does_not_train(self):
        source = (ROOT / "latent_grpo_runner/metrics/torch_collectors.py").read_text(encoding="utf-8")
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("manual_seed", source)
        self.assertNotIn("rand(", source)
        self.assertNotIn("randn(", source)


if __name__ == "__main__":
    unittest.main()
