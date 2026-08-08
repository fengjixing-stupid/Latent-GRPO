import unittest


class IdentityTests(unittest.TestCase):
    def test_ids_are_created_after_repeat_and_survive_reorder_and_filter(self):
        from latent_grpo_runner.metrics.identity import attach_stable_trajectory_ids, stable_group_id

        repeated = [
            {"prompt_identity": "question-A", "payload": "a0"},
            {"prompt_identity": "question-A", "payload": "a1"},
            {"prompt_identity": "question-B", "payload": "b0"},
        ]
        assigned = attach_stable_trajectory_ids(repeated, global_step=8)
        self.assertEqual([row["trajectory_id"] for row in assigned], [0, 1, 0])
        self.assertEqual(assigned[0]["group_id"], stable_group_id(8, "question-A"))
        filtered_reordered = [assigned[1], assigned[2]]
        self.assertEqual([row["trajectory_id"] for row in filtered_reordered], [1, 0])

    def test_group_id_is_repeatable_and_trajectory_class_is_binary_with_overlong_overlap(self):
        from latent_grpo_runner.metrics.identity import classify_trajectory, stable_group_id

        self.assertEqual(stable_group_id(8, "question-A"), stable_group_id(8, "question-A"))
        item = classify_trajectory(is_correct=True, response_length=9, max_response_length=8)
        self.assertEqual(item, {"trajectory_class": "correct", "is_overlong_or_truncated_by_length": True})
        self.assertEqual(classify_trajectory(False, 3, 8)["trajectory_class"], "non_correct")

    def test_group_id_rejects_unstable_non_string_prompt_identity(self):
        from latent_grpo_runner.metrics.identity import stable_group_id

        with self.assertRaises(ValueError):
            stable_group_id(8, {"prompt": "question-A"})
