import math
import random
import unittest


class Stage4ProbeTests(unittest.TestCase):
    def test_packed_probe_tensors_restore_response_layout_and_keep_autograd(self) -> None:
        import torch

        from latent_grpo_runner.metrics.probe import restore_packed_probe_tensors

        values = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            requires_grad=True,
        )
        packed = {
            "topk_log_probs": values,
            "raw_diff": values + 10.0,
            "flipgrad_trigger_mask": torch.tensor(
                [[False, True], [True, False], [False, False], [True, True]]
            ),
        }
        restored = restore_packed_probe_tensors(
            packed,
            indices=torch.tensor([0, 1, 3, 4]),
            batch=2,
            seqlen=3,
            response_length=1,
        )

        self.assertTrue(torch.equal(restored["topk_log_probs"], torch.tensor([[[3.0, 4.0]], [[7.0, 8.0]]])))
        self.assertTrue(torch.equal(restored["raw_diff"], torch.tensor([[[13.0, 14.0]], [[17.0, 18.0]]])))
        restored["topk_log_probs"].sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])))

    def test_checkpoint_packet_uses_packed_autograd_target_then_restores_credit(self) -> None:
        import torch

        from latent_grpo_runner.metrics.probe import (
            collect_checkpoint_probe_packet,
            restore_packed_probe_tensors,
        )

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        packed = torch.tensor(
            [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6], [-0.7, -0.8]],
            requires_grad=True,
        )
        coefficients = torch.tensor(
            [[11.0, 12.0], [21.0, 22.0], [31.0, 32.0], [41.0, 42.0]]
        )
        policy_loss = -(packed * coefficients).sum()
        indices = torch.tensor([0, 1, 3, 4])
        restore_spec = {
            "indices": indices,
            "batch": 2,
            "seqlen": 3,
            "response_length": 1,
        }
        restored = restore_packed_probe_tensors(
            {
                "topk_log_probs": packed,
                "raw_diff": packed.detach() + 1.0,
                "flipgrad_trigger_mask": torch.zeros_like(packed, dtype=torch.bool),
            },
            **restore_spec,
        )
        shape = restored["topk_log_probs"].shape
        packet = collect_checkpoint_probe_packet(
            policy_loss=policy_loss,
            topk_log_probs=restored["topk_log_probs"],
            deltas=restored["raw_diff"],
            mixture_weights=torch.full(shape, 0.5),
            valid_component_mask=torch.ones(shape, dtype=torch.bool),
            advantages=torch.ones(shape),
            trajectory_masks={"all": torch.ones(shape, dtype=torch.bool)},
            position_masks={"all": torch.ones(shape, dtype=torch.bool)},
            model=model,
            optimizer=optimizer,
            flipgrad_trigger_mask=restored["flipgrad_trigger_mask"],
            autograd_topk_log_probs=packed,
            autograd_restore_spec=restore_spec,
            retain_graph=True,
        )

        self.assertEqual(packet["credit"], [21.0, 22.0, 41.0, 42.0])
        self.assertTrue(packet["credit_autograd_executed"])
        self.assertTrue(all(packet["state_preservation"].values()))
        policy_loss.backward()

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
