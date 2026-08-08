import math
import unittest


class SufficientStatsTests(unittest.TestCase):
    def test_stats_tracks_mask_nan_extrema_and_rate_numerator(self):
        from latent_grpo_runner.metrics.aggregators import SufficientStats

        stats = SufficientStats.from_values([2.0, float("nan"), 4.0, 99.0], [True, True, True, False], [False, True, True, True])
        self.assertEqual((stats.sum, stats.sum_sq, stats.count), (6.0, 20.0, 2))
        self.assertEqual((stats.nan_count, stats.masked_count, stats.numerator_count), (1, 1, 1))
        self.assertEqual((stats.min, stats.max), (2.0, 4.0))
        self.assertEqual(stats.mean(), 3.0)
        self.assertEqual(stats.std(), 1.0)
        self.assertEqual(stats.rate(), 0.5)

    def test_merge_uses_global_sums_not_a_mean_of_worker_means(self):
        from latent_grpo_runner.metrics.aggregators import SufficientStats

        left = SufficientStats.from_values([1.0])
        right = SufficientStats.from_values([10.0, 10.0, 10.0])
        merged = left.merge(right)
        self.assertEqual(merged.mean(), 7.75)
        self.assertNotEqual(merged.mean(), (left.mean() + right.mean()) / 2)

    def test_empty_or_all_nan_stats_are_unavailable(self):
        from latent_grpo_runner.metrics.aggregators import SufficientStats

        empty = SufficientStats.from_values([float("nan")], [True])
        self.assertFalse(empty.available)
        self.assertEqual(empty.unavailable_reason, "empty_effective_mask")
        self.assertTrue(math.isnan(empty.mean()))
        self.assertEqual(empty.nan_count, 1)
