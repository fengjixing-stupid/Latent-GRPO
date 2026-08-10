from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "latent_grpo_runner/config.py"
ACTOR = ROOT / "Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py"
TORCH_FUNCTIONAL = ROOT / "Latent-GRPO/verl-0.4.x/verl/utils/torch_functional.py"
CORE_ALGOS = ROOT / "Latent-GRPO/verl-0.4.x/verl/trainer/ppo/core_algos.py"
FSDP = ROOT / "Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py"
SGLANG_ROLLOUT = ROOT / "Latent-GRPO/verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py"
SGLANG_SAMPLER = ROOT / "Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py"


def _extract_function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class T4RuntimeSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import torch.nn.functional as F
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(f"torch unavailable: {error}")
        cls.torch = torch
        cls.F = F

    def test_shared_mixture_is_shape_equivalent_to_packed_formula(self):
        torch = self.torch
        fn = _extract_function(ACTOR, "_latent_mixture_weights", {"torch": torch})
        ids = torch.tensor([
            [[3, -100, -100], [2, 4, 6]],
            [[7, -100, -100], [1, 5, 8]],
        ])
        scores = torch.tensor([
            [[0.5, -3.0, -4.0], [0.2, 0.1, -0.4]],
            [[0.7, -2.0, -2.5], [-0.1, 0.4, 0.2]],
        ])
        hard_padded, weights_padded = fn(ids, scores, 1.0)
        hard_flat, weights_flat = fn(ids.reshape(-1, 3), scores.reshape(-1, 3), 1.0)
        self.assertTrue(torch.equal(hard_padded.reshape(-1), hard_flat))
        self.assertTrue(torch.allclose(weights_padded.reshape(-1, 3), weights_flat, atol=0, rtol=0))

    def test_flipgrad_keeps_forward_value_and_changes_gradient_route(self):
        torch = self.torch
        fn = _extract_function(
            TORCH_FUNCTIONAL,
            "logprobs_from_logits_topk_gumbel",
            {"torch": torch, "F": self.F},
        )
        base = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
        ids = torch.tensor([[0, 1]])
        with torch.no_grad():
            topk_logp = base.detach().float().log_softmax(-1).gather(-1, ids)
            scores = topk_logp - 0.5
        labels = torch.tensor([0])
        standard = fn(base, ids, scores, labels, 1.0, 1.0, advantages=None)
        standard.sum().backward()
        grad_standard = base.grad.detach().clone()

        flipped_logits = base.detach().clone().requires_grad_(True)
        flipped = fn(
            flipped_logits, ids, scores, labels, 1.0, 1.0,
            advantages=torch.tensor([-1.0]),
        )
        flipped.sum().backward()
        self.assertTrue(torch.allclose(standard.detach(), flipped.detach(), atol=0, rtol=0))
        self.assertFalse(torch.allclose(grad_standard, flipped_logits.grad))

    def test_gumbel_logprob_excludes_masked_nan_inf_before_nonlinear_math(self):
        torch = self.torch
        fn = _extract_function(
            TORCH_FUNCTIONAL,
            "logprobs_from_logits_topk_gumbel",
            {"torch": torch, "F": self.F},
        )
        logits = torch.tensor(
            [[2.0, 1.0, 0.0], [float("nan"), float("inf"), -float("inf")]],
            requires_grad=True,
        )
        ids = torch.tensor([[0, 1], [-100, -100]])
        gumbels = torch.tensor([[0.2, -0.1], [float("nan"), float("inf")]])
        labels = torch.tensor([0, 0])
        valid_mask = torch.tensor([True, False])
        baseline_logits = logits[:1].detach().clone().requires_grad_(True)
        baseline = fn(
            baseline_logits,
            ids[:1],
            gumbels[:1],
            labels[:1],
            1.0,
            1.0,
            advantages=torch.tensor([1.0]),
        )
        baseline.sum().backward()

        output = fn(
            logits,
            ids,
            gumbels,
            labels,
            1.0,
            1.0,
            advantages=torch.tensor([1.0, float("nan")]),
            valid_mask=valid_mask,
        )

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output[:1], baseline.detach()))
        self.assertEqual(output[1].item(), 0.0)
        output.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[:1], baseline_logits.grad))
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1])))

    def test_ppo_excludes_masked_nan_inf_before_ratio_and_loss(self):
        torch = self.torch
        masked_mean = _extract_function(
            TORCH_FUNCTIONAL,
            "masked_mean",
            {"torch": torch},
        )
        verl_f = SimpleNamespace(masked_mean=masked_mean)
        agg_loss = _extract_function(
            CORE_ALGOS,
            "agg_loss",
            {"torch": torch, "verl_F": verl_f},
        )
        compute_policy_loss = _extract_function(
            CORE_ALGOS,
            "compute_policy_loss",
            {"torch": torch, "verl_F": verl_f, "agg_loss": agg_loss},
        )
        old_log_prob = torch.tensor([[-1.0, -2.0, -float("inf"), float("nan")]])
        log_prob = torch.tensor(
            [[-0.9, -2.1, -float("inf"), float("inf")]],
            requires_grad=True,
        )
        advantages = torch.tensor([[1.0, -0.5, float("nan"), float("inf")]])
        response_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)

        pg_loss, _, ppo_kl, _ = compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            cliprange=0.2,
        )

        self.assertTrue(torch.isfinite(pg_loss))
        self.assertTrue(torch.isfinite(ppo_kl))
        pg_loss.backward()
        self.assertTrue(torch.isfinite(log_prob.grad).all())
        self.assertTrue(
            torch.equal(log_prob.grad[~response_mask], torch.zeros_like(log_prob.grad[~response_mask]))
        )

    def test_masked_mean_excludes_nonfinite_values_and_gradients(self):
        torch = self.torch
        masked_mean = _extract_function(
            TORCH_FUNCTIONAL,
            "masked_mean",
            {"torch": torch},
        )
        values = torch.tensor(
            [2.0, float("nan"), float("inf"), -float("inf")],
            requires_grad=True,
        )
        mask = torch.tensor([1, 0, 0, 0], dtype=torch.bool)

        result = masked_mean(values, mask)

        self.assertEqual(result.item(), 2.0)
        result.backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(torch.equal(values.grad[1:], torch.zeros_like(values.grad[1:])))

    def test_padded_actor_reuses_same_latent_and_flipgrad_inputs(self):
        source = ACTOR.read_text(encoding="utf-8")
        self.assertIn("_latent_mixture_weights", source)
        self.assertIn("padded latent path: same Top-K/Gumbel semantics", source)
        self.assertIn("inputs_embeds=topk_embs_final.detach()", source)
        self.assertIn("next_topk_ids = rollout_topk_ids[:, -response_length:, :]", source)
        self.assertIn("next_topk_gumbels = rollout_topk_gumbels[:, -response_length:, :]", source)
        self.assertIn("advantages=advantages", source)
        self.assertIn("valid_mask=response_mask", source)
        self.assertIn("component_log_probs = full_current_topk_logits.detach().float()", source)

    def test_embedding_lookup_bounds_fsdp_unshard_and_uses_one_padded_lookup(self):
        source = ACTOR.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lookup = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "safe_lookup_embeddings"
        )
        lookup_source = ast.get_source_segment(source, lookup)
        forward = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_forward_micro_batch"
        )
        forward_source = ast.get_source_segment(source, forward)
        padded_source = forward_source.split(
            "padded latent path: same Top-K/Gumbel semantics", 1
        )[1]

        self.assertIn("recurse=False", lookup_source)
        self.assertIn("get_torch_device().empty_cache()", lookup_source)
        self.assertIn("embs = embed(_input_ids).detach()", lookup_source)
        self.assertEqual(padded_source.count("safe_lookup_embeddings("), 1)
        self.assertIn("all_first_embs = topk_embs_all[..., 0, :]", padded_source)

    def test_t4_changes_attention_not_sampling_backend(self):
        config_source = CONFIG.read_text(encoding="utf-8")
        fsdp = FSDP.read_text(encoding="utf-8")
        rollout = SGLANG_ROLLOUT.read_text(encoding="utf-8")
        self.assertIn("actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend", config_source)
        self.assertNotIn("actor_rollout_ref.rollout.sampling_backend", config_source)
        self.assertIn('attention_implementation = "flash_attention_2" if use_remove_padding else "sdpa"', fsdp)
        self.assertIn("attention_backend=attention_backend", rollout)
        self.assertIn("sampling_backend=self.config.get", rollout)
        self.assertIn("'flashinfer'", rollout)
        self.assertNotIn("sampling_backend='pytorch'", rollout)

    def test_author_latent_sampler_precedes_backend_and_overrides_latent_ids(self):
        source = SGLANG_SAMPLER.read_text(encoding="utf-8")
        latent = source.index("if enable_latent:")
        backend = source.index('global_server_args_dict["sampling_backend"] == "flashinfer"')
        override = source.index("sampling_info.latent_modes,", backend)
        self.assertLess(latent, backend)
        self.assertLess(backend, override)
        self.assertIn("latent_batch_next_token_ids", source[latent:override + 200])


if __name__ == "__main__":
    unittest.main()
