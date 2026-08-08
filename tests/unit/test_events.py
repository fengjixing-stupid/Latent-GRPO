import dataclasses
import unittest


class StepContextTests(unittest.TestCase):
    def test_context_is_immutable_and_serializes_dynamic_fields(self):
        from latent_grpo_runner.metrics.events import StepContext

        context = StepContext(
            profile_name="smoke", seed=7, global_step=12, optimizer_step=9,
            observation_phase="post_update", learning_rate=0.0003,
            wall_clock_seconds=4.5,
        )
        self.assertTrue(dataclasses.is_dataclass(context))
        self.assertEqual(context.to_record()["global_step"], 12)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            context.global_step = 13

    def test_context_rejects_checkpoint_fields_for_ordinary_training(self):
        from latent_grpo_runner.metrics.events import StepContext

        with self.assertRaises(ValueError):
            StepContext("smoke", 7, 1, 1, "post_update", checkpoint_step=1)
