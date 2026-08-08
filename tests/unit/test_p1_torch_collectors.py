from __future__ import annotations

import importlib.util
import math
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
class P1TorchCollectorTests(unittest.TestCase):
    def test_final_batch_lengths_signal_and_noisy_mixture(self):
        import torch
        from latent_grpo_runner.metrics.torch_collectors import collect_driver_p1_from_tensors

        # Two trajectories, response width 3.  Positions (0,0), (0,2), (1,1)
        # are latent; the others are hard/padded.
        response_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
        topk_ids = torch.tensor(
            [
                [[10, 11], [20, -100], [30, 31]],
                [[40, -100], [50, 51], [-100, -100]],
            ]
        )
        gumbels = torch.tensor(
            [
                [[0.0, 0.0], [3.0, 0.0], [0.0, math.log(3.0)]],
                [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        payload = collect_driver_p1_from_tensors(
            response_mask=response_mask,
            rollout_topk_ids=topk_ids,
            rollout_topk_gumbels=gumbels,
            gumbel_temperature=torch.tensor([1.0, 1.0]),
            entropies=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]]),
            token_level_rewards=torch.tensor([[0.0, 0.0, 1.0], [0.0, 2.0, 0.0]]),
            advantages=torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.0]]),
            exclude_overlong_samples_from_advantage=True,
        )
        self.assertEqual(payload["final_training_trajectory_lengths"], [3, 2])
        stats = payload["p1_driver_sufficient_stats"]
        self.assertEqual(stats["train/response_length"]["sum"], 5.0)
        self.assertEqual(stats["train/latent_length"]["sum"], 3.0)
        self.assertEqual(stats["signal/reward"]["sum"], 3.0)
        self.assertEqual(stats["signal/reward"]["count"], 2)
        self.assertEqual(stats["mask/zero_advantage_rate"]["count"], 3)
        self.assertEqual(stats["mask/zero_advantage_rate"]["numerator_count"], 1)
        self.assertEqual(stats["mixture/effective_k_noisy"]["count"], 3)
        self.assertEqual(stats["mixture/top1_weight_noisy"]["count"], 3)

    def test_actor_overlong_zeroing_is_mirrored_without_mutating_input(self):
        import torch
        from latent_grpo_runner.metrics.torch_collectors import collect_driver_p1_from_tensors

        advantages = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        original = advantages.clone()
        payload = collect_driver_p1_from_tensors(
            response_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            rollout_topk_ids=torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, -100]]]),
            rollout_topk_gumbels=torch.zeros((2, 2, 2)),
            gumbel_temperature=torch.ones(2),
            entropies=torch.ones((2, 2)),
            token_level_rewards=torch.zeros((2, 2)),
            advantages=advantages,
            exclude_overlong_samples_from_advantage=False,
        )
        self.assertTrue(torch.equal(advantages, original))
        stats = payload["p1_driver_sufficient_stats"]
        # First trajectory is full-width/clipped and is zeroed only in the detached metric view.
        self.assertEqual(stats["signal/advantage"]["sum"], 3.0)
        self.assertEqual(stats["mask/zero_advantage_rate"]["numerator_count"], 2)


if __name__ == "__main__":
    unittest.main()
