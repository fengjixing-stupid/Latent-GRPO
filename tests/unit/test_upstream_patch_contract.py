"""Static author-patch contracts that do not import CUDA/Ray/SGLang."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "Latent-GRPO"


def _source(relative_path: str) -> str:
    return (UPSTREAM / relative_path).read_text(encoding="utf-8")


def _function(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


class UpstreamPatchContractTests(unittest.TestCase):
    def test_stable_ids_are_attached_after_repeat_before_balance_or_filter(self) -> None:
        source = _source("verl-0.4.x/verl/trainer/ppo/ray_trainer.py")
        repeat = source.index("batch = batch.repeat(")
        attach = source.index("attach_stable_ids_to_batch(", repeat)
        union = source.index("batch = batch.union(gen_batch_output)", repeat)
        balance = source.index("self._balance_batch(batch", union)
        filtering = source.index("if self.config.algorithm.filter_groups.enable", union)

        self.assertLess(repeat, attach)
        self.assertLess(attach, union)
        self.assertLess(attach, balance)
        self.assertLess(attach, filtering)
        init = source.index("self._latent_grpo_observer = (")
        self.assertLess(init, repeat)
        self.assertIn("observer=self._latent_grpo_observer", source[attach : attach + 400])

    def test_ocp_default_return_stays_compatible_and_observer_return_is_opt_in(self) -> None:
        source = _source("verl-0.4.x/verl/trainer/ppo/core_algos.py")
        for name in (
            "compute_latent_grpo_outcome_advantage_firstmask_best_exclude_advantage",
            "compute_latent_grpo_outcome_advantage_firstmask_best_include_advantage",
        ):
            node = _function(source, name)
            args = [arg.arg for arg in node.args.args]
            self.assertIn("trajectory_ids", args)
            self.assertIn("return_observer_data", args)
            defaults = dict(zip(args[-len(node.args.defaults) :], node.args.defaults))
            self.assertIsInstance(defaults["return_observer_data"], ast.Constant)
            self.assertFalse(defaults["return_observer_data"].value)
        self.assertIn('"mean_old_log_prob"', source)
        self.assertIn('"trajectory_id"', source)

    def test_ocp_opt_in_facts_capture_the_selected_candidate_without_changing_formula(self) -> None:
        source = _source("verl-0.4.x/verl/trainer/ppo/core_algos.py")
        for name in (
            "compute_latent_grpo_outcome_advantage_firstmask_best_exclude_advantage",
            "compute_latent_grpo_outcome_advantage_firstmask_best_include_advantage",
        ):
            function_source = ast.get_source_segment(source, _function(source, name))
            self.assertIn("_build_ocp_observer_fact(", function_source)
            self.assertIn("winner_idx=winner_idx", function_source)
            self.assertIn("stable_group_ids=stable_group_ids", function_source)
            self.assertIn("mean_old_log_prob=", function_source)
            self.assertIn("if return_observer_data", function_source)

    def test_ocp_helper_returns_stable_group_and_winner_identity(self) -> None:
        source = _source("verl-0.4.x/verl/trainer/ppo/core_algos.py")
        tree = ast.parse(source)
        helper = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_ocp_observer_fact"),
            None,
        )
        self.assertIsNotNone(helper, "dependency-free OCP fact helper is missing")
        module = ast.fix_missing_locations(
            ast.Module(
                body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), helper],
                type_ignores=[],
            )
        )
        namespace: dict[str, object] = {}
        exec(compile(module, "<ocp-helper>", "exec"), namespace)
        build_fact = namespace["_build_ocp_observer_fact"]

        fact = build_fact(
            upstream_group_id="random-uid",
            batch_indices=[0, 1, 2],
            winner_idx=1,
            stable_group_ids=["stable-group", "stable-group", "stable-group"],
            trajectory_ids=[0, 1, 2],
            mean_old_log_prob=-0.2,
        )

        self.assertEqual(
            fact,
            {
                "group_id": "stable-group",
                "winner_local_index": 1,
                "trajectory_id": 1,
                "mean_old_log_prob": -0.2,
            },
        )
        with self.assertRaisesRegex(ValueError, "one stable group_id"):
            build_fact(
                upstream_group_id="random-uid",
                batch_indices=[0, 1],
                winner_idx=1,
                stable_group_ids=["stable-a", "stable-b"],
                trajectory_ids=[0, 1],
                mean_old_log_prob=-0.2,
            )

    def test_real_trainer_opts_in_only_for_enabled_persistent_observer(self) -> None:
        source = _source("verl-0.4.x/verl/trainer/ppo/ray_trainer.py")
        compute_source = ast.get_source_segment(source, _function(source, "compute_advantage"))
        fit_source = ast.get_source_segment(source, _function(source, "fit"))

        self.assertIn('observer = kwargs.get("observer")', compute_source)
        self.assertIn('stable_group_ids=data.non_tensor_batch["group_id"]', compute_source)
        self.assertIn('trajectory_ids=data.non_tensor_batch["trajectory_id"]', compute_source)
        self.assertIn("return_observer_data=True", compute_source)
        self.assertIn('observer.emit("ocp_selection"', compute_source)
        self.assertIn("observer=self._latent_grpo_observer", fit_source)

    def test_ocp_uses_response_masked_mean_for_the_same_winner(self) -> None:
        source = _source("verl-0.4.x/verl/trainer/ppo/core_algos.py")
        for name in (
            "compute_latent_grpo_outcome_advantage_firstmask_best_exclude_advantage",
            "compute_latent_grpo_outcome_advantage_firstmask_best_include_advantage",
        ):
            function_source = ast.get_source_segment(source, _function(source, name))
            self.assertIn("log_prob_sums = (old_log_probs * response_mask).sum(dim=-1)", function_source)
            self.assertIn("log_prob_means = log_prob_sums / (lengths + 1e-8)", function_source)
            self.assertIn("candidate_log_probs = [log_prob_means[c].item()", function_source)
            self.assertIn("mean_old_log_prob=", function_source)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("numpy", "torch", "ray", "tensordict", "pandas", "transformers")),
        "upstream CPU dependencies unavailable; numerical OCP equivalence is target-machine deferred",
    )
    def test_ocp_default_outputs_equal_opt_in_outputs_and_winner_facts(self) -> None:
        upstream_python = str(UPSTREAM / "verl-0.4.x")
        sys.path.insert(0, upstream_python)
        try:
            import numpy as np
            import torch

            core_algos = importlib.import_module("verl.trainer.ppo.core_algos")
            rewards = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
            mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 1, 1]])
            old_log_probs = torch.tensor([[-1.0, -1.0, 99.0], [-0.2, 99.0, 99.0], [99.0, -0.5, -0.3]])
            uids = np.array(["uid", "uid", "uid"], dtype=object)
            group_ids = np.array(["stable-group"] * 3, dtype=object)
            trajectory_ids = np.array([0, 1, 2])

            for name in (
                "compute_latent_grpo_outcome_advantage_firstmask_best_exclude_advantage",
                "compute_latent_grpo_outcome_advantage_firstmask_best_include_advantage",
            ):
                function = getattr(core_algos, name)
                common = {
                    "token_level_rewards": rewards,
                    "response_mask": mask,
                    "index": uids,
                    "old_log_probs": old_log_probs,
                    "norm_adv_by_std_in_grpo": False,
                }
                if "exclude" in name:
                    common["max_response_length"] = 3
                default = function(**common)
                observed = function(
                    **common,
                    stable_group_ids=group_ids,
                    trajectory_ids=trajectory_ids,
                    return_observer_data=True,
                )

                self.assertEqual(len(default), 3)
                self.assertEqual(len(observed), 4)
                self.assertTrue(torch.equal(default[0], observed[0]))
                self.assertTrue(torch.equal(default[1], observed[1]))
                self.assertEqual(default[2].keys(), observed[2].keys())
                for group_id in default[2]:
                    self.assertTrue(torch.equal(default[2][group_id], observed[2][group_id]))
                self.assertEqual(len(observed[3]), 1)
                self.assertEqual(observed[3][0]["group_id"], "stable-group")
                self.assertEqual(observed[3][0]["winner_local_index"], 1)
                self.assertEqual(observed[3][0]["trajectory_id"], 1)
                self.assertAlmostEqual(observed[3][0]["mean_old_log_prob"], -0.2, places=6)
        finally:
            sys.path.remove(upstream_python)

    def test_topk_logits_are_concatenated_from_logits_not_ids(self) -> None:
        source = _source("verl-0.4.x/verl/workers/actor/dp_actor.py")
        function_source = ast.get_source_segment(source, _function(source, "compute_log_prob"))

        self.assertIn("topk_logits = torch.concat(topk_logits_lst, dim=0)", function_source)
        self.assertNotIn("topk_logits = torch.concat(topk_ids_lst, dim=0)", function_source)

    def test_optimizer_reports_real_success_and_scheduler_skips_nonfinite_update(self) -> None:
        actor_source = _source("verl-0.4.x/verl/workers/actor/dp_actor.py")
        worker_source = _source("verl-0.4.x/verl/workers/fsdp_workers.py")

        optimizer_source = ast.get_source_segment(actor_source, _function(actor_source, "_optimizer_step"))
        update_source = ast.get_source_segment(actor_source, _function(actor_source, "update_policy"))
        worker_update_source = ast.get_source_segment(worker_source, _function(worker_source, "update_actor"))
        scheduler_source = ast.get_source_segment(
            worker_source, _function(worker_source, "_advance_actor_scheduler_after_update")
        )
        self.assertIn("did_step", optimizer_source)
        self.assertIn("update_count", update_source)
        self.assertIn("did_update", update_source)
        self.assertIn("_advance_actor_scheduler_after_update", worker_update_source)
        self.assertIn("if update_count > 0", scheduler_source)
        self.assertIn("scheduler.step()", scheduler_source)

    def test_latent_end_is_runtime_configured_and_missing_value_fails_closed(self) -> None:
        source = _source("sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py")

        self.assertNotIn("topk_indices[:,0] != 524", source)
        self.assertIn('global_server_args_dict.get("latent_end_token_id")', source)
        self.assertIn("latent_end_token_id is required", source)

    def test_dynamic_batch_and_fused_latent_paths_fail_safe_and_flash_is_required(self) -> None:
        trainer_source = _source("verl-0.4.x/verl/trainer/ppo/ray_trainer.py")
        actor_source = _source("verl-0.4.x/verl/workers/actor/dp_actor.py")

        validate_source = ast.get_source_segment(trainer_source, _function(trainer_source, "_validate_config"))
        guard_source = ast.get_source_segment(
            trainer_source,
            _function(trainer_source, "_validate_latent_instrumentation_config"),
        )
        self.assertIn("_validate_latent_instrumentation_config(", validate_source)
        self.assertIn("latent instrumentation requires actor.use_dynamic_bsz=false", guard_source)
        self.assertIn("latent instrumentation requires rollout.log_prob_use_dynamic_bsz=false", guard_source)
        self.assertIn("latent instrumentation requires use_fused_kernels=false", guard_source)
        self.assertIn("latent instrumentation requires model.use_remove_padding=true", guard_source)
        self.assertIn("FlashAttention cross entropy is required for latent Gumbel log-prob", actor_source)


if __name__ == "__main__":
    unittest.main()
