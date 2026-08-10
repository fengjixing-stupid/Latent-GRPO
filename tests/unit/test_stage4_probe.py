import math
import random
import unittest


class Stage4ProbeTests(unittest.TestCase):
    def test_onesided_probe_metrics_use_reduced_stats_and_exact_p05(self) -> None:
        from latent_grpo_runner.metrics.probe import build_probe_metric_row

        row = build_probe_metric_row(
            profile_name="smoke",
            seed=1,
            global_step=4,
            optimizer_step=3,
            checkpoint_step=4,
            probe_batch_id="probe-a",
            deltas=[-2.0, 0.0, 2.0, 100.0],
            valid_delta_mask=[True, True, True, False],
            flipgrad_trigger_mask=[True, False, False, True],
            credit=None,
            near_zero_threshold=0.0,
        )

        self.assertEqual(row["onesided/delta_mean"], 0.0)
        self.assertAlmostEqual(row["onesided/delta_std"], math.sqrt(8.0 / 3.0))
        self.assertAlmostEqual(row["onesided/delta_p05"], -1.8)
        self.assertEqual(row["onesided/delta_min"], -2.0)
        self.assertEqual(row["onesided/delta_negative_rate"], 1 / 3)
        self.assertEqual(row["onesided/delta_near_zero_rate"], 1 / 3)
        self.assertEqual(row["onesided/flipgrad_rate"], 1 / 3)
        self.assertEqual(row["onesided/delta_count"], 3)
        self.assertEqual(row["onesided/flipgrad_count"], 1)

    def test_onesided_probe_fail_closed_for_empty_mask_single_and_all_zero(self) -> None:
        from latent_grpo_runner.metrics.probe import build_probe_metric_row

        empty = build_probe_metric_row(
            profile_name="smoke",
            seed=1,
            global_step=4,
            optimizer_step=3,
            checkpoint_step=4,
            probe_batch_id="probe-a",
            deltas=[1.0],
            valid_delta_mask=[False],
            flipgrad_trigger_mask=[True],
            credit=None,
        )
        self.assertFalse(empty["onesided_available"])
        self.assertEqual(empty["onesided_unavailable_reason"], "empty_effective_mask")

        single = build_probe_metric_row(
            profile_name="smoke",
            seed=1,
            global_step=4,
            optimizer_step=3,
            checkpoint_step=4,
            probe_batch_id="probe-a",
            deltas=[5.0],
            valid_delta_mask=[True],
            flipgrad_trigger_mask=[False],
            credit=None,
        )
        self.assertEqual(single["onesided/delta_std"], 0.0)

        zeros = build_probe_metric_row(
            profile_name="smoke",
            seed=1,
            global_step=4,
            optimizer_step=3,
            checkpoint_step=4,
            probe_batch_id="probe-a",
            deltas=[0.0, 0.0],
            valid_delta_mask=[True, True],
            flipgrad_trigger_mask=[False, False],
            credit=None,
            near_zero_threshold=0.0,
        )
        self.assertEqual(zeros["onesided/delta_near_zero_rate"], 1.0)
        self.assertEqual(zeros["onesided/delta_min"], 0.0)

    def test_credit_autograd_computes_u_q_concentration_spearman_and_alignment(self) -> None:
        import torch

        from latent_grpo_runner.metrics.probe import collect_credit_from_autograd

        topk_log_probs = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        coefficients = torch.tensor([[3.0, 1.0, -2.0]])
        policy_loss = -(topk_log_probs * coefficients).sum()

        credit = collect_credit_from_autograd(
            policy_loss=policy_loss,
            topk_log_probs=topk_log_probs,
            mixture_weights=torch.tensor([[0.9, 0.5, 0.1]]),
            valid_component_mask=torch.tensor([[True, True, True]]),
            advantages=torch.tensor([[1.0, 1.0, -1.0]]),
        )

        self.assertTrue(credit.concentration_available)
        self.assertAlmostEqual(credit.top1_share, 0.5)
        expected_effective_k = math.exp(-sum(q * math.log(q) for q in (0.5, 1 / 6, 1 / 3)))
        self.assertAlmostEqual(credit.effective_k, expected_effective_k)
        self.assertAlmostEqual(credit.weight_credit_spearman, 0.5)
        self.assertEqual(credit.surrogate_alignment_rate, 1.0)
        self.assertEqual(credit.alignment_count, 3)

    def test_credit_degenerate_spearman_and_zero_gradient_fail_closed_per_family(self) -> None:
        import torch

        from latent_grpo_runner.metrics.probe import collect_credit_from_autograd

        topk_log_probs = torch.tensor([[0.0, 0.0]], requires_grad=True)
        policy_loss = (topk_log_probs * 0.0).sum()
        credit = collect_credit_from_autograd(
            policy_loss=policy_loss,
            topk_log_probs=topk_log_probs,
            mixture_weights=torch.tensor([[0.5, 0.5]]),
            valid_component_mask=torch.tensor([[True, True]]),
            advantages=torch.tensor([[1.0, -1.0]]),
        )

        self.assertFalse(credit.concentration_available)
        self.assertEqual(credit.concentration_unavailable_reason, "empty_credit_mass")
        self.assertFalse(credit.spearman_available)
        self.assertEqual(credit.spearman_unavailable_reason, "constant_rank")
        self.assertFalse(credit.alignment_available)
        self.assertEqual(credit.alignment_unavailable_reason, "zero_gradient_direction")

    def test_credit_probe_preserves_model_optimizer_grad_rng_and_training_mode(self) -> None:
        import numpy as np
        import torch

        from latent_grpo_runner.metrics.probe import run_preserving_training_state

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        model.train(False)
        output = model(torch.ones(1, 2)).sum()
        output.backward()
        grad_before = [parameter.grad.clone() for parameter in model.parameters()]
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        def probe() -> str:
            random.random()
            np.random.rand()
            torch.rand(1)
            model.train(True)
            for parameter in model.parameters():
                parameter.grad = None
            return "ok"

        result = run_preserving_training_state(model, optimizer, probe)

        self.assertEqual(result.value, "ok")
        self.assertTrue(result.state_restored)
        self.assertFalse(model.training)
        for before, parameter in zip(grad_before, model.parameters()):
            self.assertTrue(torch.equal(before, parameter.grad))
        self.assertEqual(random.random(), 0.052363598850944326)
        self.assertAlmostEqual(float(np.random.rand()), 0.6964691855978616)
        self.assertAlmostEqual(float(torch.rand(1).item()), 0.29611194133758545)


if __name__ == "__main__":
    unittest.main()
