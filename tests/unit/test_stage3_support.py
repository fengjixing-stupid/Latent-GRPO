import math
import unittest


class Stage3SupportTests(unittest.TestCase):
    def test_support_retention_handles_full_partial_none_and_top1_membership(self) -> None:
        from latent_grpo_runner.metrics.support import collect_support_metrics

        rows, benchmark = collect_support_metrics(
            profile_name="smoke",
            seed=7,
            global_step=3,
            optimizer_step_at_observation=5,
            group_ids=["g", "g", "h"],
            trajectory_ids=[0, 1, 0],
            trajectory_classes=["correct", "non_correct", "non_correct"],
            trajectory_mean_old_log_probs=[-2.0, -0.2, -0.7],
            response_mask=[[True, True], [True, True], [True, True]],
            rollout_topk_ids=[
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ],
            old_topk_indices=[
                [[1, 2], [4, 3]],
                [[6, 50], [70, 80]],
                [[90, 100], [110, 120]],
            ],
        )

        self.assertEqual([(row["group_id"], row["trajectory_id"], row["trajectory_class"]) for row in rows],
                         [("g", 0, "correct"), ("g", 1, "non_correct"), ("h", 0, "non_correct")])
        self.assertEqual(rows[0]["support/retention_rate"], 1.0)
        self.assertEqual(rows[0]["support/top1_retention_rate"], 1.0)
        self.assertEqual(rows[1]["support/retention_rate"], 0.25)
        self.assertEqual(rows[1]["support/top1_retention_rate"], 0.0)
        self.assertEqual(rows[2]["support/retention_rate"], 0.0)
        self.assertEqual(rows[2]["support/top1_retention_rate"], 0.0)
        self.assertEqual(benchmark["support_selected_trajectory_count"], 3)
        self.assertEqual(benchmark["support_benchmark/total_effective_position_count"], 6)

    def test_support_excludes_padding_hard_tokens_and_overlong_candidates(self) -> None:
        from latent_grpo_runner.metrics.support import collect_support_metrics

        rows, benchmark = collect_support_metrics(
            profile_name="smoke",
            seed=7,
            global_step=3,
            optimizer_step_at_observation=5,
            group_ids=["g", "g"],
            trajectory_ids=[0, 1],
            trajectory_classes=["correct", "non_correct"],
            trajectory_mean_old_log_probs=[-1.0, -0.1],
            is_overlong_or_truncated_by_length=[False, True],
            response_mask=[[True, False, True], [True, True, True]],
            rollout_topk_ids=[[[1, 2], [3, 4], [5, -100]], [[7, 8], [9, 10], [11, 12]]],
            old_topk_indices=[[[2, 1], [3, 4], [5, 6]], [[8, 7], [10, 9], [12, 11]]],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trajectory_id"], 0)
        self.assertEqual(rows[0]["support/effective_position_count"], 1)
        self.assertEqual(rows[0]["support/retention_rate"], 1.0)
        self.assertEqual(benchmark["support_candidate_trajectory_count"], 1)

    def test_support_fail_closed_for_shape_k_and_order_mismatch(self) -> None:
        from latent_grpo_runner.metrics.support import collect_support_metrics

        base = dict(
            profile_name="smoke",
            seed=7,
            global_step=3,
            optimizer_step_at_observation=5,
            group_ids=["g"],
            trajectory_ids=[0],
            trajectory_classes=["correct"],
            trajectory_mean_old_log_probs=[-1.0],
            response_mask=[[True]],
        )
        for bad in (
            {"rollout_topk_ids": [[[1, 2]]], "old_topk_indices": [[[1, 2, 3]]]},
            {"rollout_topk_ids": [[[1, 2], [3, 4]]], "old_topk_indices": [[[1, 2], [3, 4]]]},
            {"rollout_topk_ids": [[[1, 2]]], "old_topk_indices": [[[1, -100]]]},
        ):
            rows, benchmark = collect_support_metrics(**base, **bad)
            self.assertEqual(rows, [])
            self.assertFalse(benchmark["support_available"])
            self.assertIsNotNone(benchmark["support_unavailable_reason"])

    def test_support_tie_selects_smallest_stable_trajectory_id(self) -> None:
        from latent_grpo_runner.metrics.support import collect_support_metrics

        rows, _ = collect_support_metrics(
            profile_name="smoke",
            seed=7,
            global_step=3,
            optimizer_step_at_observation=5,
            group_ids=["g", "g"],
            trajectory_ids=[3, 2],
            trajectory_classes=["non_correct", "non_correct"],
            trajectory_mean_old_log_probs=[-1.0, -1.0],
            response_mask=[[True], [True]],
            rollout_topk_ids=[[[1, 2]], [[3, 4]]],
            old_topk_indices=[[[1, 2]], [[3, 4]]],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trajectory_id"], 2)
        self.assertTrue(math.isfinite(rows[0]["support/retention_rate"]))


if __name__ == "__main__":
    unittest.main()
