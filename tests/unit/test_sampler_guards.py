"""CPU-only contracts for latent sampler configuration and target guards."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from latent_grpo_runner.upstream_adapter import (
    BufferedObserver,
    EvalQuestionFacts,
    NoOpObserver,
    emit_eval_question_facts,
    eval_question_event,
    load_observer_from_env,
)


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "Latent-GRPO"


def _source(relative_path: str) -> str:
    return (UPSTREAM / relative_path).read_text(encoding="utf-8")


def _load_function(relative_path: str, name: str, globals_: dict[str, object] | None = None):
    """Compile one dependency-free function without importing CUDA modules."""
    source = _source(relative_path)
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {} if globals_ is None else dict(globals_)
    exec(compile(ast.fix_missing_locations(module), relative_path, "exec"), namespace)
    return namespace[name]


class _Config(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class SamplerGuardTests(unittest.TestCase):
    def test_flash_attention_guard_rejects_only_latent_gumbel_path(self) -> None:
        require = _load_function(
            "verl-0.4.x/verl/workers/actor/dp_actor.py",
            "_require_flash_attention_cross_entropy",
            {"verl_F": SimpleNamespace(FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE=False)},
        )
        require(add_noise_gumbel_softmax=False)
        with self.assertRaisesRegex(
            RuntimeError,
            "FlashAttention cross entropy is required for latent Gumbel log-prob",
        ):
            require(add_noise_gumbel_softmax=True)

        source = _source("verl-0.4.x/verl/utils/torch_functional.py")
        tree = ast.parse(source)
        latent_logprob = next(
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.FunctionDef) and item.name == "logprobs_from_logits_topk_gumbel"
        )
        function_source = ast.get_source_segment(source, latent_logprob)
        self.assertNotIn("output = logprobs_from_logits_v2(logits, labels)", function_source)
        self.assertIn("FlashAttention cross entropy is required for latent Gumbel log-prob", function_source)

    def test_latent_instrumentation_config_rejects_each_unsupported_path(self) -> None:
        validate = _load_function(
            "verl-0.4.x/verl/trainer/ppo/ray_trainer.py",
            "_validate_latent_instrumentation_config",
        )

        def config(*, latent=True, actor_dynamic=False, rollout_dynamic=False, fused=False):
            return SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(
                    actor=_Config(use_dynamic_bsz=actor_dynamic, use_fused_kernels=fused),
                    rollout=_Config(enable_latent=latent, log_prob_use_dynamic_bsz=rollout_dynamic),
                )
            )

        with self.assertRaisesRegex(ValueError, "actor.use_dynamic_bsz=false"):
            validate(config(actor_dynamic=True), observer_enabled=False)
        with self.assertRaisesRegex(ValueError, "rollout.log_prob_use_dynamic_bsz=false"):
            validate(config(rollout_dynamic=True), observer_enabled=False)
        with self.assertRaisesRegex(ValueError, "use_fused_kernels=false"):
            validate(config(fused=True), observer_enabled=False)

        validate(config(latent=False, actor_dynamic=True, rollout_dynamic=True, fused=True), observer_enabled=False)
        with self.assertRaisesRegex(ValueError, "actor.use_dynamic_bsz=false"):
            validate(config(latent=False, actor_dynamic=True), observer_enabled=True)

    def test_production_observer_requires_explicit_durable_sink(self) -> None:
        class MissingEnabledSink:
            durable = True

            def emit(self, event_type, facts):
                del event_type, facts

        class DurableSink:
            enabled = True
            durable = True

            def emit(self, event_type, facts):
                del event_type, facts

        enabled = {"LATENT_GRPO_OBSERVER_ENABLED": "1"}
        with self.assertRaisesRegex(RuntimeError, "authoritative observer sink"):
            load_observer_from_env(enabled)
        with self.assertRaisesRegex(TypeError, "BufferedObserver is synthetic-only"):
            load_observer_from_env(enabled, sink=BufferedObserver())
        with self.assertRaisesRegex(TypeError, "enabled=true"):
            load_observer_from_env(enabled, sink=MissingEnabledSink())
        sink = DurableSink()
        self.assertIs(load_observer_from_env(enabled, sink=sink), sink)
        self.assertIsInstance(load_observer_from_env({"LATENT_GRPO_OBSERVER_ENABLED": "0"}), NoOpObserver)

    def test_eval_raw_fact_event_preserves_identity_and_explicit_unavailability(self) -> None:
        event = eval_question_event(
            EvalQuestionFacts(
                data_source="math500",
                question_id="math500:17",
                generation_id=2,
                predicted_answer="\\boxed{4}",
                reference_answer="4",
                reward=1.0,
                is_correct=None,
                correctness_unavailable_reason="reward_extra_info.acc_missing",
            )
        )

        self.assertEqual(
            event,
            {
                "event_type": "eval_question",
                "data_source": "math500",
                "question_id": "math500:17",
                "generation_id": 2,
                "predicted_answer": "\\boxed{4}",
                "reference_answer": "4",
                "reward": 1.0,
                "is_correct": None,
                "correctness_unavailable_reason": "reward_extra_info.acc_missing",
            },
        )

    def test_validation_emits_raw_events_before_aggregate_metrics(self) -> None:
        source = _source("verl-0.4.x/verl/trainer/ppo/ray_trainer.py")
        tree = ast.parse(source)
        validate = next(
            item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == "_validate"
        )
        function_source = ast.get_source_segment(source, validate)

        emit = function_source.index("emit_eval_question_facts(")
        aggregate = function_source.index("process_validation_metrics(")
        self.assertLess(emit, aggregate)

    def test_eval_batch_adapter_assigns_generation_ids_per_question(self) -> None:
        observer = BufferedObserver()
        emitted = emit_eval_question_facts(
            observer,
            data_sources=["gsm8k", "gsm8k", "math500"],
            extra_infos=[{"index": 8}, {"index": 8}, {"index": 3}],
            reward_models=[
                {"ground_truth": "2"},
                {"ground_truth": "2"},
                {"ground_truth": "7"},
            ],
            outputs=["a", "b", "c"],
            scores=[1.0, 0.0, -1.0],
            correctness=[True, False, None],
        )

        events = observer.drain()
        self.assertEqual(emitted, 3)
        self.assertEqual([event["question_id"] for event in events], ["gsm8k:8", "gsm8k:8", "math500:3"])
        self.assertEqual([event["generation_id"] for event in events], [0, 1, 0])
        self.assertEqual(events[2]["correctness_unavailable_reason"], "reward_extra_info.acc_missing")

    def test_eval_generation_ids_continue_across_batches_and_accuracy_is_normalized(self) -> None:
        observer = BufferedObserver()
        ordinals: dict[str, int] = {}
        common = {
            "observer": observer,
            "data_sources": ["gsm8k"],
            "extra_infos": [{"index": 8}],
            "reward_models": [{"ground_truth": "2"}],
            "scores": [1.0],
            "generation_ordinals": ordinals,
        }
        emit_eval_question_facts(outputs=["a"], correctness=[1], **common)
        emit_eval_question_facts(outputs=["b"], correctness=[0.0], **common)

        events = observer.drain()
        self.assertEqual([event["generation_id"] for event in events], [0, 1])
        self.assertEqual([event["is_correct"] for event in events], [True, False])
        with self.assertRaisesRegex(ValueError, "correctness must be bool, 0, 1, or null"):
            emit_eval_question_facts(outputs=["c"], correctness=[0.5], **common)

    def test_runtime_guard_accepts_each_profile_id_and_rejects_missing_or_out_of_range(self) -> None:
        validate = _load_function(
            "verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py",
            "_validate_latent_runtime_config",
        )
        model = SimpleNamespace(vocab_size=1000)

        self.assertEqual(
            validate(_Config(enable_latent=True, latent_end_token_id=522), model),
            {"enable_latent": True, "latent_end_token_id": 522, "vocab_size": 1000},
        )
        self.assertEqual(validate(_Config(enable_latent=False), model)["enable_latent"], False)
        with self.assertRaisesRegex(ValueError, "latent_end_token_id is required"):
            validate(_Config(enable_latent=True), model)
        with self.assertRaisesRegex(ValueError, "outside model vocabulary"):
            validate(_Config(enable_latent=True, latent_end_token_id=1000), model)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            validate(_Config(enable_latent=True, latent_end_token_id=True), model)

    def test_sampler_guard_reads_runtime_id_and_fails_closed(self) -> None:
        relative = "sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py"
        require = _load_function(relative, "_require_latent_end_token_id", {"global_server_args_dict": {}})
        with self.assertRaisesRegex(RuntimeError, "latent_end_token_id is required"):
            require()

        require = _load_function(
            relative,
            "_require_latent_end_token_id",
            {"global_server_args_dict": {"latent_end_token_id": 522}},
        )
        self.assertEqual(require(), 522)

    def test_rollout_passes_validated_value_to_server_without_legacy_fallback(self) -> None:
        source = _source("verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py")
        tree = ast.parse(source)
        init_engine = next(
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.FunctionDef) and item.name == "_init_inference_engine"
        )
        function_source = ast.get_source_segment(source, init_engine)

        self.assertIn('enable_latent=self.config.get("enable_latent", False)', function_source)
        self.assertIn('latent_end_token_id=self.config.get("latent_end_token_id")', function_source)
        self.assertNotIn('latent_end_token_id=self.config.get("latent_end_token_id", 524)', function_source)

    def test_server_cli_uses_the_server_args_field_name_with_legacy_alias(self) -> None:
        source = _source("sglang_latent_reasoning_pkg/python/sglang/srt/server_args.py")

        self.assertIn('"--latent-end-token-id"', source)
        self.assertIn('"--latent-end-str-id"', source)
        self.assertIn('dest="latent_end_token_id"', source)

    def test_configuration_reaches_sampler_through_existing_server_and_request_path(self) -> None:
        server_args = _source("sglang_latent_reasoning_pkg/python/sglang/srt/server_args.py")
        model_config = _source("sglang_latent_reasoning_pkg/python/sglang/srt/configs/model_config.py")
        model_runner = _source("sglang_latent_reasoning_pkg/python/sglang/srt/model_executor/model_runner.py")
        scheduler = _source("sglang_latent_reasoning_pkg/python/sglang/srt/managers/scheduler.py")
        schedule_batch = _source("sglang_latent_reasoning_pkg/python/sglang/srt/managers/schedule_batch.py")
        sampler = _source("sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py")

        self.assertIn("latent_end_token_id: int = None", server_args)
        self.assertIn("self.latent_end_token_id = latent_end_token_id", model_config)
        self.assertIn('"latent_end_token_id": server_args.latent_end_token_id', model_runner)
        self.assertIn("latent_end_token_id = self.latent_end_token_id", scheduler)
        self.assertIn("self.sampling_params.latent_end_token_id = latent_end_token_id", schedule_batch)
        self.assertIn('global_server_args_dict.get("latent_end_token_id")', sampler)

    def test_sglang_target_backend_keeps_explicit_package_fail_fast_checks(self) -> None:
        source = _source("verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py")
        tree = ast.parse(source)
        set_envs = next(
            item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == "_set_envs_and_config"
        )
        function_source = ast.get_source_segment(source, set_envs)

        self.assertIn('assert_pkg_version(\n            "flashinfer_python",', function_source)
        self.assertIn('assert_pkg_version(\n            "sgl-kernel",', function_source)


if __name__ == "__main__":
    unittest.main()
