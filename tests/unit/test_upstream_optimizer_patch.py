"""Mac-safe contract tests for the narrow actor optimizer observer patch."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTOR_PATH = ROOT / "Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py"
WORKER_PATH = ROOT / "Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py"


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"function {function_name!r} not found in {path}")


def _load_standalone_function(path: Path, function_name: str):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function_name)
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[function_name]


class UpstreamOptimizerPatchTests(unittest.TestCase):
    def test_compute_log_prob_concatenates_logits_not_token_ids(self) -> None:
        source = _function_source(ACTOR_PATH, "compute_log_prob")

        self.assertIn("topk_logits = torch.concat(topk_logits_lst, dim=0)", source)
        self.assertNotIn("topk_logits = torch.concat(topk_ids_lst, dim=0)", source)

    def test_optimizer_step_preserves_return_and_records_actual_outcome(self) -> None:
        source = _function_source(ACTOR_PATH, "_optimizer_step")

        self.assertIn("return grad_norm", source)
        self.assertIn("did_step = False", source)
        self.assertIn("did_step = True", source)
        self.assertIn("self.actor_optimizer.step()", source)
        self.assertIn("_record_optimizer_step_outcome", source)

    def test_optimizer_step_synthetic_finite_and_nonfinite_outcomes(self) -> None:
        source = ACTOR_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_optimizer_step")

        class NeverMatches:
            pass

        class Torch:
            class nn:
                class utils:
                    @staticmethod
                    def clip_grad_norm_(parameters, max_norm):
                        del max_norm
                        return parameters[0]

            class distributed:
                @staticmethod
                def get_rank():
                    return 0

            @staticmethod
            def isfinite(value):
                return value.finite

        namespace = {
            "torch": Torch,
            "FSDP": NeverMatches,
            "FSDPModule": NeverMatches,
            "fsdp2_clip_grad_norm_": None,
            "logger": type("Logger", (), {"warning": staticmethod(lambda *args: None)})(),
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(ACTOR_PATH), "exec"), namespace)
        optimizer_step = namespace["_optimizer_step"]

        class Optimizer:
            def __init__(self) -> None:
                self.steps = 0
                self.zeroes = 0

            def step(self) -> None:
                self.steps += 1

            def zero_grad(self) -> None:
                self.zeroes += 1

        class ActorModule:
            def __init__(self, grad_norm) -> None:
                self.grad_norm = grad_norm

            def parameters(self):
                return [self.grad_norm]

        class Subject:
            def __init__(self, finite: bool) -> None:
                self.config = type("Config", (), {"grad_clip": 1.0})()
                self.actor_module = ActorModule(type("Norm", (), {"finite": finite})())
                self.actor_optimizer = Optimizer()
                self.last_update_did_step = False
                self.recorded = []

            def _record_optimizer_step_outcome(self, did_step, grad_norm) -> None:
                self.recorded.append((did_step, grad_norm))

        finite = Subject(True)
        optimizer_step(finite)
        self.assertEqual((finite.actor_optimizer.steps, finite.actor_optimizer.zeroes), (1, 0))
        self.assertEqual(finite.recorded[0][0], True)

        nonfinite = Subject(False)
        optimizer_step(nonfinite)
        self.assertEqual((nonfinite.actor_optimizer.steps, nonfinite.actor_optimizer.zeroes), (0, 1))
        self.assertEqual(nonfinite.recorded[0][0], False)

    def test_component_observer_is_opt_in_and_reduces_before_export(self) -> None:
        forward_source = _function_source(ACTOR_PATH, "_forward_micro_batch")
        record_source = _function_source(ACTOR_PATH, "_record_component_sufficient_stats")

        self.assertIn("collect_component_stats=False", forward_source)
        self.assertIn("_record_component_sufficient_stats", forward_source)
        self.assertIn(".detach()", record_source)
        self.assertIn(".item()", record_source)
        self.assertNotIn("state_dict", record_source)
        self.assertNotIn(".grad", record_source)

    def test_plain_component_reference_reduces_to_scalars(self) -> None:
        from latent_grpo_runner.upstream_adapter import build_component_sufficient_stats

        facts = build_component_sufficient_stats(
            [-0.25, 0.0, 0.5, float("nan")],
            [True, True, False, True],
            [True, False, True, False],
            near_zero_threshold=0.01,
        )
        self.assertEqual((facts["count"], facts["negative_count"], facts["near_zero_count"]), (2, 1, 1))
        self.assertTrue(all(value is None or isinstance(value, (int, float)) for value in facts.values()))

    def test_three_worker_packets_merge_sums_counts_and_min_without_worker_means(self) -> None:
        from latent_grpo_runner.upstream_adapter import merge_worker_observer_packets

        def packet(rank: int, stats: dict) -> dict:
            return {"worker_rank": rank, "did_update": True, "update_count": 1,
                    "optimizer_steps": [{"did_step": True}], "component_sufficient_stats": [stats]}

        common = {"near_zero_threshold": 1e-6, "definition_version": "stage2_surrogate_v1"}
        packets = [
            packet(0, {**common, "sum": 2.0, "sum_sq": 4.0, "count": 1, "nan_count": 0,
                       "masked_count": 2, "min": 2.0, "negative_count": 0, "near_zero_count": 0,
                       "flipgrad_trigger_count": 0}),
            packet(1, {**common, "sum": -3.0, "sum_sq": 9.0, "count": 1, "nan_count": 1,
                       "masked_count": 0, "min": -3.0, "negative_count": 1, "near_zero_count": 0,
                       "flipgrad_trigger_count": 1}),
            packet(2, {**common, "sum": 0.0, "sum_sq": 0.0, "count": 2, "nan_count": 0,
                       "masked_count": 1, "min": 0.0, "negative_count": 0, "near_zero_count": 2,
                       "flipgrad_trigger_count": 0}),
        ]

        merged = merge_worker_observer_packets(packets, expected_worker_count=3)

        self.assertTrue(merged["optimizer_update_available"])
        self.assertEqual((merged["update_count"], merged["aggregation_worker_count"]), (1, 3))
        self.assertEqual(
            merged["component_sufficient_stats"],
            {**common, "sum": -1.0, "sum_sq": 13.0, "count": 4, "nan_count": 1,
             "masked_count": 3, "min": -3.0, "negative_count": 1, "near_zero_count": 2,
             "flipgrad_trigger_count": 1},
        )

    def test_worker_outcome_disagreement_fails_closed(self) -> None:
        from latent_grpo_runner.upstream_adapter import merge_worker_observer_packets

        merged = merge_worker_observer_packets([
            {"worker_rank": 0, "did_update": True, "update_count": 1,
             "optimizer_steps": [{"did_step": True}], "component_sufficient_stats": []},
            {"worker_rank": 1, "did_update": False, "update_count": 0,
             "optimizer_steps": [{"did_step": False}], "component_sufficient_stats": []},
        ], expected_worker_count=2)

        self.assertFalse(merged["optimizer_update_available"])
        self.assertIsNone(merged["update_count"])
        self.assertEqual(merged["optimizer_update_unavailable_reason"], "optimizer_outcome_rank_disagreement")

    def test_worker_exports_plain_observer_payload_only_when_nonempty(self) -> None:
        source = _function_source(WORKER_PATH, "update_actor")

        self.assertIn("update_count = self.actor.last_update_count", source)
        self.assertIn("_advance_actor_scheduler_after_update(self.actor_lr_scheduler, update_count)", source)
        self.assertIn("consume_latent_grpo_observer_facts", source)
        self.assertIn('output.meta_info["latent_grpo_observer"]', source)
        self.assertRegex(source, r"if\s+observer_facts\s*:")

    def test_scheduler_all_skipped_success_and_mixed_semantics(self) -> None:
        advance = _load_standalone_function(WORKER_PATH, "_advance_actor_scheduler_after_update")

        class Scheduler:
            def __init__(self) -> None:
                self.calls = 0

            def step(self) -> None:
                self.calls += 1

        skipped = Scheduler()
        self.assertFalse(advance(skipped, 0))
        self.assertEqual(skipped.calls, 0)
        for update_count in (1, 2):  # all-success and success/skip mixed outer updates
            scheduler = Scheduler()
            self.assertTrue(advance(scheduler, update_count))
            self.assertEqual(scheduler.calls, 1)

    def test_worker_collector_preserves_every_rank_packet_and_driver_consumes_it(self) -> None:
        worker_source = WORKER_PATH.read_text(encoding="utf-8")
        trainer_source = (ROOT / "Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("collect_dp_compute_data_proto_with_observer", worker_source)
        self.assertIn('"latent_grpo_worker_observers"', worker_source)
        self.assertIn("merge_worker_observer_packets", trainer_source)
        self.assertIn('"latent_grpo_worker_observers", []', trainer_source)
        self.assertIn('self._latent_grpo_observer.emit("actor_update"', trainer_source)

    def test_worker_collector_synthetically_preserves_rank_one_and_two(self) -> None:
        collector_source = _function_source(WORKER_PATH, "collect_dp_compute_data_proto_with_observer")

        class FakeDataProto:
            def __init__(self, meta_info: dict) -> None:
                self.meta_info = meta_info

            @staticmethod
            def concat(rows):
                return FakeDataProto(dict(rows[0].meta_info))

        namespace = {
            "DataProto": FakeDataProto,
            "collect_dp_compute": lambda _group, output: output,
        }
        exec(collector_source, namespace)
        outputs = [
            FakeDataProto({"metrics": {}, "latent_grpo_observer": {"update_count": rank + 1}})
            for rank in range(3)
        ]

        merged = namespace["collect_dp_compute_data_proto_with_observer"](object(), outputs)

        self.assertEqual(
            merged.meta_info["latent_grpo_worker_observers"],
            [
                {"worker_rank": 0, "update_count": 1},
                {"worker_rank": 1, "update_count": 2},
                {"worker_rank": 2, "update_count": 3},
            ],
        )

    def test_component_capture_uses_final_advantage_and_response_or_loss_mask(self) -> None:
        update_source = _function_source(ACTOR_PATH, "update_policy")
        forward_source = _function_source(ACTOR_PATH, "_forward_micro_batch")
        self.assertLess(
            update_source.index("is_clipped = cur_response_length == response_length"),
            update_source.index("self._forward_micro_batch"),
        )
        self.assertIn("component_response_mask=response_mask", update_source)
        self.assertIn("component_response_mask=None", forward_source)
        self.assertIn("current_response_mask", forward_source)
        self.assertIn("component_alignment_shape_mismatch", forward_source)

    def test_observer_defaults_disabled(self) -> None:
        init_source = _function_source(ACTOR_PATH, "__init__")
        self.assertIn('os.getenv("LATENT_GRPO_OBSERVER_ENABLED", "0")', init_source)

    def test_upstream_files_still_parse_without_importing_verl_or_cuda(self) -> None:
        actor_source = ACTOR_PATH.read_text(encoding="utf-8")
        worker_source = WORKER_PATH.read_text(encoding="utf-8")
        ast.parse(actor_source)
        ast.parse(worker_source)
        self.assertNotIn("import latent_grpo_runner", actor_source)
        self.assertNotIn("import latent_grpo_runner", worker_source)


if __name__ == "__main__":
    unittest.main()
