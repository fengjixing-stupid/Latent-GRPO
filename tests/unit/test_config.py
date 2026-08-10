from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from latent_grpo_runner.config import ConfigError, load_config, validate_latent_end_token


ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def test_all_supported_profiles_parse_and_have_deterministic_hashes(self) -> None:
        for name in (
            "smoke",
            "3gpu-low",
            "3gpu-high-smoke",
            "kaggle-t4-monitor",
            "kaggle-t4-30-metric",
        ):
            config = load_config(ROOT / "configs" / f"{name}.yaml", workspace_root=ROOT)
            self.assertEqual(config.profile_name, name)
            self.assertEqual(len(config.config_hash), 64)
            self.assertEqual(config.config_hash, load_config(
                ROOT / "configs" / f"{name}.yaml", workspace_root=ROOT
            ).config_hash)
            self.assertTrue(config.features.metrics_enabled)

    def test_kaggle_monitor_profile_is_fail_closed_before_backward(self) -> None:
        config = load_config(ROOT / "configs" / "kaggle-t4-monitor.yaml", workspace_root=ROOT)
        self.assertEqual(config.hardware.required_gpus, 2)
        self.assertEqual(config.batch_arithmetic(), (2, 1, 1, 2))
        self.assertTrue(config.training.pre_backward_monitor_probe)
        self.assertEqual(config.rollout.dtype, "float16")
        self.assertEqual(config.rollout.attention_backend, "triton")
        self.assertFalse(config.model.use_remove_padding)
        self.assertTrue(config.model.enable_gradient_checkpointing)
        self.assertTrue(config.model.actor_param_offload)
        self.assertTrue(config.model.actor_optimizer_offload)
        self.assertTrue(config.model.ref_param_offload)
        self.assertEqual(config.data.max_prompt_length, 256)
        self.assertEqual(config.data.max_response_length, 32)
        self.assertIs(config.data.filter_overlong_prompts, True)
        self.assertEqual(config.data.filter_overlong_prompts_workers, 1)
        self.assertEqual(config.rollout.max_model_len, 288)
        self.assertEqual(config.rollout.max_num_batched_tokens, 288)
        overrides = config.author_hydra_overrides()
        self.assertIn("data.train_batch_size=2", overrides)
        self.assertIn("data.filter_overlong_prompts=true", overrides)
        self.assertIn("data.filter_overlong_prompts_workers=1", overrides)
        self.assertIn("trainer.pre_backward_monitor_probe=true", overrides)
        self.assertIn("trainer.val_before_train=false", overrides)
        self.assertIn("trainer.test_freq=-1", overrides)
        self.assertIn("trainer.save_freq=-1", overrides)
        self.assertFalse(any("4bit" in item or "8bit" in item or "quant" in item for item in overrides))

        contents = (ROOT / "configs" / "kaggle-t4-monitor.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "unsafe-kaggle.yaml"
            path.write_text(
                contents.replace("pre_backward_monitor_probe: true", "pre_backward_monitor_probe: false"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "must stop before backward"):
                load_config(path, workspace_root=ROOT)

        with self.assertRaisesRegex(ConfigError, "cannot override max_steps"):
            config.with_runtime_overrides(max_steps=2)
        with self.assertRaisesRegex(ConfigError, "does not support checkpoint resume"):
            config.with_runtime_overrides(resume_from=Path("fake/global_step_1"))
        with self.assertRaisesRegex(ConfigError, "requires metrics_enabled"):
            config.with_runtime_overrides(metrics_enabled=False)

    def test_non_kaggle_profiles_preserve_upstream_prompt_filter_defaults(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml", workspace_root=ROOT)
        self.assertIsNone(config.data.filter_overlong_prompts)
        self.assertIsNone(config.data.filter_overlong_prompts_workers)
        overrides = config.author_hydra_overrides()
        self.assertFalse(any(item.startswith("data.filter_overlong_prompts=") for item in overrides))
        self.assertFalse(any(item.startswith("data.filter_overlong_prompts_workers=") for item in overrides))
        self.assertFalse(any(item.startswith("actor_rollout_ref.actor.checkpoint.contents=") for item in overrides))
        self.assertFalse(any(item == "trainer.resume_mode=disable" for item in overrides))

    def test_three_gpu_profiles_satisfy_full_per_rank_batch_arithmetic(self) -> None:
        expected = {
            "3gpu-low": (8, 4, 2, 2),
            "3gpu-high-smoke": (4, 4, 4, 1),
        }
        for name, values in expected.items():
            config = load_config(ROOT / "configs" / f"{name}.yaml", workspace_root=ROOT)
            self.assertEqual(config.batch_arithmetic(), values)

    def test_resume_compatibility_hash_excludes_only_resume_locator(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml", workspace_root=ROOT)
        resumed = config.with_runtime_overrides(
            resume_from=Path("artifacts/runs/smoke/global_step_1")
        )
        self.assertNotEqual(config.config_hash, resumed.config_hash)
        self.assertEqual(config.resume_compatibility_hash, resumed.resume_compatibility_hash)

    def test_credit_probe_requires_checkpoint_probe(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml", workspace_root=ROOT)
        self.assertFalse(config.features.credit_probe_enabled)
        with self.assertRaisesRegex(ConfigError, "requires checkpoint_probe_enabled=true"):
            config.with_runtime_overrides(credit_probe_enabled=True)
        enabled = config.with_runtime_overrides(
            checkpoint_probe_enabled=True,
            credit_probe_enabled=True,
        )
        self.assertTrue(enabled.features.checkpoint_probe_enabled)
        self.assertTrue(enabled.features.credit_probe_enabled)

    def test_unknown_field_is_rejected(self) -> None:
        contents = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "unknown.yaml"
            path.write_text(contents + "\nunknown_field: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "unknown field"):
                load_config(path, workspace_root=ROOT)

    def test_three_gpu_profile_rejects_non_three_gpu_target(self) -> None:
        contents = (ROOT / "configs" / "3gpu-low.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad-gpus.yaml"
            path.write_text(contents.replace("required_gpus: 3", "required_gpus: 2"), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "requires exactly 3 GPUs"):
                load_config(path, workspace_root=ROOT)

    def test_batch_divisibility_is_rejected_before_upstream_launch(self) -> None:
        contents = (ROOT / "configs" / "3gpu-low.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad-batch.yaml"
            path.write_text(contents.replace("mini_prompt_batch: 3", "mini_prompt_batch: 2"), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "mini_prompt_batch.*rollout_n"):
                load_config(path, workspace_root=ROOT)

    def test_paper_profiles_are_not_accepted_as_engineering_profiles(self) -> None:
        contents = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "paper.yaml"
            path.write_text(contents.replace("profile_name: smoke", "profile_name: paper-low"), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "paper profiles"):
                load_config(path, workspace_root=ROOT)

    def test_hydra_overrides_map_to_observed_upstream_fields(self) -> None:
        config = load_config(ROOT / "configs" / "3gpu-low.yaml", workspace_root=ROOT)
        import yaml

        upstream = yaml.safe_load(
            (ROOT / "Latent-GRPO" / "verl-0.4.x" / "verl" / "trainer" / "config" / "ppo_trainer.yaml").read_text(
                encoding="utf-8"
            )
        )
        for override in config.author_hydra_overrides():
            key, _ = override.split("=", 1)
            current = upstream
            for component in key.split("."):
                self.assertIn(component, current, f"upstream Hydra config has no key for {key}")
                current = current[component]

    def test_optional_sglang_attention_backend_is_exposed_without_sampling_override(self) -> None:
        contents = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
        contents = contents.replace("  dtype: bfloat16\n", "  dtype: bfloat16\n  attention_backend: triton\n", 1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "t4-attention.yaml"
            path.write_text(contents, encoding="utf-8")
            config = load_config(path, workspace_root=ROOT)
        overrides = config.author_hydra_overrides()
        self.assertIn(
            "actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton",
            overrides,
        )
        self.assertFalse(any("sampling_backend" in item for item in overrides))

    def test_schema_version_and_bool_integer_fields_are_rejected(self) -> None:
        contents = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad-schema.yaml"
            path.write_text(contents.replace("schema_version: 1", "schema_version: 2"), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "schema_version"):
                load_config(path, workspace_root=ROOT)
            path.write_text(contents.replace("latent_end_token_id: 524", "latent_end_token_id: true"), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "latent_end_token_id must be an integer"):
                load_config(path, workspace_root=ROOT)

    def test_launch_time_latent_end_validation_checks_token_and_vocab(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml", workspace_root=ROOT)

        class Tokenizer:
            vocab_size = 600

            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                if text == "</think>" and add_special_tokens is False:
                    return [524, 125, 29]
                return [1]

        class ModelMetadata:
            vocab_size = 600

        result = validate_latent_end_token(config.model, Tokenizer(), ModelMetadata())
        self.assertEqual(result["latent_end_token_id"], 524)
        self.assertEqual(result["latent_end_token"], "</think>")
        self.assertEqual(result["latent_end_validation_status"], "validated")

        class WrongTokenizer(Tokenizer):
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [525, 125, 29]

        with self.assertRaisesRegex(ConfigError, "first tokenizer ID"):
            validate_latent_end_token(config.model, WrongTokenizer(), ModelMetadata())

        class SmallModelMetadata:
            vocab_size = 100

        with self.assertRaisesRegex(ConfigError, "outside tokenizer/model vocabulary"):
            validate_latent_end_token(config.model, Tokenizer(), SmallModelMetadata())


if __name__ == "__main__":
    unittest.main()
