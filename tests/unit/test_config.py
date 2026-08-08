from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from latent_grpo_runner.config import ConfigError, load_config, validate_latent_end_token


ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def test_all_supported_profiles_parse_and_have_deterministic_hashes(self) -> None:
        for name in ("smoke", "3gpu-low", "3gpu-high-smoke"):
            config = load_config(ROOT / "configs" / f"{name}.yaml", workspace_root=ROOT)
            self.assertEqual(config.profile_name, name)
            self.assertEqual(len(config.config_hash), 64)
            self.assertEqual(config.config_hash, load_config(
                ROOT / "configs" / f"{name}.yaml", workspace_root=ROOT
            ).config_hash)
            self.assertTrue(config.features.metrics_enabled)

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

            def convert_ids_to_tokens(self, token_id: int) -> str:
                return "<|latent_end|>" if token_id == 524 else "<other>"

        class ModelMetadata:
            vocab_size = 600

        result = validate_latent_end_token(config.model, Tokenizer(), ModelMetadata())
        self.assertEqual(result["latent_end_token_id"], 524)
        self.assertEqual(result["latent_end_validation_status"], "validated")

        class WrongTokenizer(Tokenizer):
            def convert_ids_to_tokens(self, token_id: int) -> str:
                return "<wrong>"

        with self.assertRaisesRegex(ConfigError, "does not match"):
            validate_latent_end_token(config.model, WrongTokenizer(), ModelMetadata())

        class SmallModelMetadata:
            vocab_size = 100

        with self.assertRaisesRegex(ConfigError, "outside tokenizer/model vocabulary"):
            validate_latent_end_token(config.model, Tokenizer(), SmallModelMetadata())


if __name__ == "__main__":
    unittest.main()
