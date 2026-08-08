import unittest


class MaskTests(unittest.TestCase):
    def test_latent_position_mask_excludes_prompt_padding_hard_and_loss_excluded(self):
        from latent_grpo_runner.metrics.masks import valid_latent_position_mask

        result = valid_latent_position_mask(
            topk_token_ids=[[11, 12], [-100, -100], [13, 14], [15, 16]],
            response_mask=[True, True, False, True],
            attention_mask=[True, True, True, False],
            loss_mask=[True, True, True, True],
        )
        self.assertEqual(result, [True, False, False, False])

    def test_component_mask_and_zero_advantage_share_eligible_domain(self):
        from latent_grpo_runner.metrics.masks import (
            valid_latent_component_mask, zero_advantage_stats,
        )

        component_mask = valid_latent_component_mask(
            [[11, -100], [12, 13]], [True, False]
        )
        self.assertEqual(component_mask, [[True, False], [False, False]])
        stats = zero_advantage_stats([0.0, 0.01, 0.0], [True, False, True], zero_threshold=0.0)
        self.assertEqual((stats.numerator_count, stats.count), (2, 2))
        self.assertEqual(stats.rate(), 1.0)

    def test_zero_advantage_excludes_nonfinite_values_from_eligible_denominator(self):
        from latent_grpo_runner.metrics.masks import zero_advantage_stats

        stats = zero_advantage_stats([0.0, float("nan"), 1.0, float("inf")], [True, True, True, True])
        self.assertEqual((stats.count, stats.numerator_count, stats.nan_count), (2, 1, 2))
        self.assertEqual(stats.rate(), 0.5)
