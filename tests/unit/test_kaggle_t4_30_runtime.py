from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTOR = ROOT / "Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py"
FUNCTIONAL = ROOT / "Latent-GRPO/verl-0.4.x/verl/utils/torch_functional.py"
TRAINER = ROOT / "Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py"
RUNNER = ROOT / "tools/run_kaggle_t4_30_metric_validation.py"
VALIDATOR = ROOT / "tools/validate_kaggle_t4_30_metrics.py"
NOTEBOOK_BUILDER = ROOT / "tools/build_kaggle_29_metric_notebook.py"


def _function(path: Path, name: str, namespace: dict[str, object]):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace[name]


class KaggleT430RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            import torch.nn.functional as F
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(f"torch unavailable: {error}")
        cls.torch = torch
        cls.F = F

    def test_full_metric_profile_is_a_one_step_dual_t4_checkpoint_run(self) -> None:
        from latent_grpo_runner.config import ConfigError, load_config

        config = load_config(ROOT / "configs/kaggle-t4-30-metric.yaml", workspace_root=ROOT)
        self.assertEqual(config.profile_name, "kaggle-t4-30-metric")
        self.assertEqual(config.hardware.required_gpus, 2)
        self.assertEqual(config.batch.rollout_n, 4)
        self.assertEqual(config.training.max_steps, 1)
        self.assertEqual(config.training.filter_groups_max_num_gen_batches, 10)
        self.assertFalse(config.training.pre_backward_monitor_probe)
        self.assertTrue(config.features.metrics_enabled)
        self.assertTrue(config.features.support_enabled)
        self.assertTrue(config.features.checkpoint_probe_enabled)
        self.assertTrue(config.features.credit_probe_enabled)
        overrides = config.author_hydra_overrides()
        self.assertIn("actor_rollout_ref.rollout.n=4", overrides)
        self.assertIn("algorithm.filter_groups.max_num_gen_batches=10", overrides)
        self.assertIn("trainer.save_freq=1", overrides)
        self.assertIn("trainer.test_freq=-1", overrides)
        self.assertIn(
            "actor_rollout_ref.actor.checkpoint.contents=[model,extra]", overrides
        )
        self.assertIn("trainer.resume_mode=disable", overrides)
        with self.assertRaisesRegex(ConfigError, "does not support checkpoint resume"):
            config.with_runtime_overrides(resume_from=Path("global_step_1"))

    def test_gumbel_function_exposes_real_graph_intermediates_only_when_requested(self) -> None:
        torch = self.torch
        function = _function(
            FUNCTIONAL,
            "logprobs_from_logits_topk_gumbel",
            {"torch": torch, "F": self.F},
        )
        logits = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
        ids = torch.tensor([[0, 1]])
        with torch.no_grad():
            gumbels = logits.detach().log_softmax(-1).gather(-1, ids) - 0.25
        labels = torch.tensor([0])

        plain = function(logits, ids, gumbels, labels, 1.0, 1.0, advantages=torch.tensor([-1.0]))
        observed, details = function(
            logits,
            ids,
            gumbels,
            labels,
            1.0,
            1.0,
            advantages=torch.tensor([-1.0]),
            return_probe_tensors=True,
        )

        self.assertTrue(torch.equal(plain, observed))
        self.assertTrue(details["topk_log_probs"].requires_grad)
        self.assertEqual(details["topk_log_probs"].shape, ids.shape)
        self.assertEqual(details["raw_diff"].shape, ids.shape)
        gradient = torch.autograd.grad(observed.sum(), details["topk_log_probs"])[0]
        self.assertEqual(gradient.shape, ids.shape)

    def test_checkpoint_packet_uses_one_autograd_and_preserves_training_state(self) -> None:
        torch = self.torch
        from latent_grpo_runner.metrics.probe import collect_checkpoint_probe_packet

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        topk_log_probs = torch.tensor([[-0.2, -1.2]], requires_grad=True)
        coefficients = torch.tensor([[2.0, -1.0]])
        policy_loss = -(topk_log_probs * coefficients).sum()
        packet = collect_checkpoint_probe_packet(
            policy_loss=policy_loss,
            topk_log_probs=topk_log_probs,
            deltas=torch.tensor([[0.1, -0.2]]),
            mixture_weights=torch.tensor([[0.7, 0.3]]),
            valid_component_mask=torch.tensor([[True, True]]),
            advantages=torch.tensor([[1.0, 1.0]]),
            trajectory_masks={"all": torch.tensor([[True, True]])},
            position_masks={"all": torch.tensor([[True, True]])},
            model=model,
            optimizer=optimizer,
            retain_graph=True,
        )

        self.assertTrue(packet["credit_autograd_executed"])
        self.assertEqual(packet["credit"], [2.0, -1.0])
        self.assertTrue(all(packet["state_preservation"].values()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        policy_loss.backward()

    def test_driver_builds_formal_probe_rows_from_all_worker_packets(self) -> None:
        from latent_grpo_runner.metrics.probe import build_checkpoint_probe_event

        state = {
            "parameters_unchanged": True,
            "optimizer_state_unchanged": True,
            "training_grads_unchanged": True,
            "cpu_rng_restored": True,
            "cuda_rng_restored": True,
            "python_rng_restored": True,
            "numpy_rng_restored": True,
            "module_mode_restored": True,
        }
        packet = {
            "worker_rank": 0,
            "available": True,
            "deltas": [0.1, -0.2],
            "credit": [2.0, -1.0],
            "mixture_weights": [0.7, 0.3],
            "advantages": [1.0, 1.0],
            "valid_component_mask": [True, True],
            "flipgrad_trigger_mask": [False, True],
            "trajectory_masks": {"all": [True, True]},
            "position_masks": {"all": [True, True]},
            "trajectory_count": 1,
            "latent_position_count": 1,
            "credit_autograd_executed": True,
            "state_preservation": state,
            "probe_extra_time_seconds": 0.01,
            "probe_peak_memory_bytes": 10,
        }
        event = build_checkpoint_probe_event(
            [packet],
            expected_worker_count=1,
            profile_name="kaggle-t4-30-metric",
            seed=17,
            global_step=1,
            optimizer_step=1,
            checkpoint_step=1,
            probe_batch_id="checkpoint-1",
        )

        self.assertEqual(len(event["rows"]), 1)
        self.assertEqual(event["rows"][0]["trajectory_group"], "all")
        self.assertTrue(event["rows"][0]["credit_concentration_available"])
        self.assertTrue(event["benchmark"]["credit_autograd_executed"])
        self.assertTrue(event["benchmark"]["parameters_unchanged"])
        self.assertEqual(
            event["worker_runtime"],
            [{"worker_rank": 0, "probe_extra_time_seconds": 0.01, "probe_peak_memory_bytes": 10}],
        )

    def test_runtime_hook_and_named_argument_runner_are_formal_entrypoints(self) -> None:
        actor = ACTOR.read_text(encoding="utf-8")
        trainer = TRAINER.read_text(encoding="utf-8")
        self.assertIn("collect_checkpoint_probe_packet", actor)
        self.assertIn("checkpoint_probe_requested", actor)
        self.assertIn("retain_graph=True", actor)
        self.assertIn('batch.meta_info["checkpoint_probe"]', trainer)
        self.assertIn('emit("checkpoint_probe"', trainer)
        self.assertTrue(RUNNER.is_file())
        runner = RUNNER.read_text(encoding="utf-8")
        for flag in ("--model-path", "--train-path", "--val-path", "--output-root"):
            self.assertIn(flag, runner)
        self.assertIn("configs/kaggle-t4-30-metric.yaml", runner)

    def test_validator_and_notebook_builder_cover_29_core_plus_raw(self) -> None:
        from tools.validate_kaggle_t4_30_metrics import (
            CORE_METRICS,
            STATE_FIELDS,
            _stage123_source_contract,
        )

        self.assertEqual(len(CORE_METRICS), 29)
        self.assertEqual(len({row[0] for row in CORE_METRICS}), 29)
        self.assertIn("cuda_rng_restored", STATE_FIELDS)
        self.assertTrue(_stage123_source_contract())
        validator = VALIDATOR.read_text(encoding="utf-8")
        self.assertIn("KAGGLE_T4_30_RUNTIME_GATE:", validator)
        self.assertIn("train/raw_generated_token_count", validator)

        builder = NOTEBOOK_BUILDER.read_text(encoding="utf-8")
        self.assertIn("Latent_GRPO_Kaggle_2xT4_30_Metric_Runtime_Validation.ipynb", builder)
        self.assertIn("LATENT_GRPO_EXPECTED_COMMIT", builder)
        self.assertIn("tools/run_kaggle_t4_30_metric_validation.py", builder)
        self.assertNotIn('FORMAL_STAGE12_RUNNER = Path("tools/run_kaggle_t4_monitor_probe.sh")', builder)


if __name__ == "__main__":
    unittest.main()
