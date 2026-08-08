import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OutputValidatorTests(unittest.TestCase):
    def test_validator_rejects_empty_schema_and_missing_required_static_files(self):
        from latent_grpo_runner.validation.output_validator import validate_output_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schema_manifest.json").write_text(json.dumps({"metrics_schema_version": "metrics_schema_v1", "tables": {}}))
            result = validate_output_directory(root)
            self.assertFalse(result.ok)
            self.assertTrue(any("required static" in error or "required dynamic" in error for error in result.errors))
            self.assertTrue((root / "validation" / "schema_validation.json").exists())

    def test_validator_rejects_undeclared_metric_table_and_declared_missing_manifest(self):
        from latent_grpo_runner.validation.output_validator import validate_output_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("run_config.json", "platform_config_snapshot.json", "run_status.json"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "schema_manifest.json").write_text(json.dumps({"metrics_schema_version": "metrics_schema_v1", "tables": {
                "train_step_metrics": {"primary_key": ["profile_name", "seed", "global_step"]},
            }}), encoding="utf-8")
            (root / "metrics" / "train_step_metrics").mkdir(parents=True)
            (root / "metrics" / "undeclared_metrics").mkdir()
            result = validate_output_directory(root)
            self.assertFalse(result.ok)
            self.assertTrue(any("missing success-part manifest" in error for error in result.errors))
            self.assertTrue(any("undeclared dynamic table" in error for error in result.errors))

    def test_validator_rejects_unavailable_value_without_reason_and_forbidden_field(self):
        from latent_grpo_runner.validation.output_validator import validate_records

        rows = [{"profile_name": "s", "seed": 1, "global_step": 1, "train/policy_loss": 0.5,
                 "train/policy_loss__available": False, "train/policy_loss__unavailable_reason": None,
                 "train/gradient_norm": 1.0}]
        errors = validate_records("train_step_metrics", rows, {"primary_key": ["profile_name", "seed", "global_step"]})
        self.assertTrue(any("unavailable" in error for error in errors))
        self.assertTrue(any("prohibited" in error for error in errors))

    def test_validator_rejects_missing_family_reason_checkpoint_step_and_invalid_topk_values(self):
        from latent_grpo_runner.validation.output_validator import validate_records

        rows = [{"profile_name": "s", "seed": 1, "global_step": 1, "checkpoint_step": 1,
                 "train_core_available": False, "model_export_path": "s3://model",
                 "clean_topk_token_ids": ["not-an-int"], "clean_topk_probs": [-0.1], "clean_topk_k": 1}]
        errors = validate_records("train_step_metrics", rows, {"primary_key": ["profile_name", "seed", "global_step"]})
        self.assertTrue(any("checkpoint_step" in error for error in errors))
        self.assertTrue(any("family" in error for error in errors))
        self.assertTrue(any("prohibited" in error for error in errors))
        self.assertTrue(any("top-k" in error for error in errors))

    def test_validator_enforces_manifest_columns_nullability_and_physical_types(self):
        from latent_grpo_runner.validation.output_validator import validate_records

        schema = {"primary_key": ["profile_name", "seed", "global_step"], "fields": [
            {"name": "profile_name", "physical_type": "string", "nullable": False},
            {"name": "seed", "physical_type": "int64", "nullable": False},
            {"name": "global_step", "physical_type": "int64", "nullable": False},
            {"name": "value", "physical_type": "float64", "nullable": True},
        ]}
        errors = validate_records("train_step_metrics", [{"profile_name": None, "seed": "wrong", "global_step": 1, "extra": True}], schema)
        self.assertTrue(any("missing required field" in error for error in errors))
        self.assertTrue(any("unexpected field" in error for error in errors))
        self.assertTrue(any("non-nullable" in error or "physical type" in error for error in errors))

    def test_validator_enforces_cumulative_tokens_and_excludes_probe_eval_from_training_counter(self):
        from latent_grpo_runner.validation.output_validator import validate_records

        train_rows = [
            {"profile_name": "s", "seed": 1, "global_step": 1, "train/generated_token_count": 3, "cumulative_rollout_tokens": 3},
            {"profile_name": "s", "seed": 1, "global_step": 2, "train/generated_token_count": 4, "cumulative_rollout_tokens": 10},
        ]
        errors = validate_records("train_step_metrics", train_rows, {"primary_key": ["profile_name", "seed", "global_step"]})
        self.assertTrue(any("cumulative_rollout_tokens" in error for error in errors))
        probe_errors = validate_records("probe_metrics", [{"profile_name": "s", "seed": 1, "checkpoint_step": 1, "probe_batch_id": "p", "trajectory_group": "g", "latent_position_group": "l", "cumulative_rollout_tokens": 1}],
                                      {"primary_key": ["profile_name", "seed", "checkpoint_step", "probe_batch_id", "trajectory_group", "latent_position_group"]})
        self.assertTrue(any("cumulative_rollout_tokens" in error for error in probe_errors))

    def test_validator_reports_duplicate_primary_keys(self):
        from latent_grpo_runner.validation.output_validator import validate_records

        rows = [{"profile_name": "s", "seed": 1, "global_step": 1}, {"profile_name": "s", "seed": 1, "global_step": 1}]
        errors = validate_records("train_step_metrics", rows, {"primary_key": ["profile_name", "seed", "global_step"]})
        self.assertTrue(any("duplicate primary key" in error for error in errors))

    def test_validator_rejects_unknown_record_version_when_schema_declares_it(self):
        from latent_grpo_runner.validation.output_validator import validate_records

        errors = validate_records(
            "train_step_metrics", [{"profile_name": "s", "seed": 1, "global_step": 1, "record_version": "future_record_v99"}],
            {"primary_key": ["profile_name", "seed", "global_step"], "record_version": "metrics_record_v1"},
        )
        self.assertTrue(any("record version" in error for error in errors))

    def test_train_step_records_must_not_move_backwards_within_one_run(self):
        from latent_grpo_runner.validation.output_validator import validate_records

        errors = validate_records(
            "train_step_metrics", [{"profile_name": "s", "seed": 1, "global_step": 2}, {"profile_name": "s", "seed": 1, "global_step": 1}],
            {"primary_key": ["profile_name", "seed", "global_step"]},
        )
        self.assertTrue(any("not monotonic" in error for error in errors))

    def test_cli_validates_the_checked_in_sample_run_without_claiming_gpu_execution(self):
        result = subprocess.run(
            [sys.executable, "scripts/validate_outputs.py", "--input", "tests/fixtures/sample_run"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = json.loads((Path("tests/fixtures/sample_run") / "platform_config_snapshot.json").read_text())
        self.assertFalse(snapshot["target_gpu_environment_available"])
