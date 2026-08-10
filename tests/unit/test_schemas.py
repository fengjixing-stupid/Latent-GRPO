import unittest


class SchemaTests(unittest.TestCase):
    def test_gumbel_and_eval_manifest_schema_match_rtm_contract(self):
        from pathlib import Path

        from latent_grpo_runner.metrics.schemas import schema_manifest

        rtm_fields = {}
        for line in Path("docs/requirements_traceability_matrix.md").read_text(encoding="utf-8").splitlines():
            if not line.startswith("| `"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) == 14:
                rtm_fields[(cells[7].strip("`"), cells[0].strip("`"))] = cells[8]

        tables = schema_manifest()["tables"]
        gumbel = {field["name"]: field for field in tables["gumbel_diagnostics"]["fields"]}
        expected_gumbel = {
            "gumbel_diagnostics_enabled": ("bool", False),
            "gumbel_compute_time_seconds": ("float64", True),
            "record_available": ("bool", False),
            "record_unavailable_reason": ("string", True),
        }
        for name, (physical_type, nullable) in expected_gumbel.items():
            self.assertEqual(rtm_fields[("gumbel_diagnostics", name)], physical_type)
            self.assertEqual(gumbel[name]["physical_type"], physical_type)
            self.assertEqual(gumbel[name]["nullable"], nullable)

        eval_table = tables["eval_dataset_manifest.parquet"]
        eval_fields = {field["name"]: field for field in eval_table["fields"]}
        expected_eval_names = {
            "eval_dataset_name", "eval_dataset_version", "question_id", "prompt_hash",
            "reference_answer", "reference_answer_hash",
        }
        self.assertEqual(set(eval_fields), expected_eval_names)
        self.assertEqual(eval_table["primary_key"], ["eval_dataset_name", "eval_dataset_version", "question_id"])
        for name in expected_eval_names:
            self.assertEqual(rtm_fields[("eval_dataset_manifest.parquet", name)], "string")
            self.assertEqual(eval_fields[name]["physical_type"], "string")
            self.assertFalse(eval_fields[name]["nullable"])

    def test_manifest_lists_stage12_fields_availability_counts_and_deferred_stage_interfaces(self):
        from latent_grpo_runner.metrics.schemas import schema_manifest

        manifest = schema_manifest()
        step_names = {field["name"] for field in manifest["tables"]["train_step_metrics"]["fields"]}
        self.assertIn("train/generated_token_count", step_names)
        self.assertIn("train/raw_generated_token_count", step_names)
        self.assertIn("train/policy_loss__available", step_names)
        self.assertIn("train/importance_ratio_count", step_names)
        self.assertNotIn("train/gradient_norm", step_names)
        self.assertEqual(manifest["stages"]["stage3"]["status"], "deferred")
        self.assertEqual(manifest["stages"]["stage4"]["status"], "disabled")

    def test_manifest_covers_gumbel_stage2_families_without_memory_only_mechanism_tensors(self):
        from latent_grpo_runner.metrics.schemas import schema_manifest

        manifest = schema_manifest()
        gumbel_names = {field["name"] for field in manifest["tables"]["gumbel_diagnostics"]["fields"]}
        stage2_names = {field["name"] for field in manifest["tables"]["train_step_metrics"]["fields"]}
        self.assertIn("gumbel/raw_count", gumbel_names)
        self.assertIn("gumbel/one_sided_count", gumbel_names)
        self.assertIn("gumbel_available", gumbel_names)
        self.assertIn("mask/eligible_latent_token_count", stage2_names)
        self.assertIn("signal/reward_count", stage2_names)
        self.assertNotIn("surrogate_margin", stage2_names)

    def test_stage2_is_in_authoritative_train_step_schema_with_shared_counts_only(self):
        from latent_grpo_runner.metrics.schemas import schema_manifest

        fields = {field["name"]: field for field in schema_manifest()["tables"]["train_step_metrics"]["fields"]}
        self.assertIn("signal/reward_mean", fields)
        self.assertEqual(fields["train/generated_token_count"]["logical_type"], "int64")
        self.assertEqual(fields["train/raw_generated_token_count"]["logical_type"], "int64")
        self.assertNotIn("train/raw_generated_token_count_count", fields)
        self.assertNotIn("stage2_metric_fields", schema_manifest()["tables"])
        self.assertNotIn("train/generated_token_count_count", fields)
        self.assertIn("signal/reward_count", fields)

    def test_deferred_runtime_tables_are_still_declared_with_explicit_reason(self):
        from latent_grpo_runner.metrics.schemas import schema_manifest

        tables = schema_manifest()["tables"]
        for name in ("eval_question_results", "eval_clean_topk", "support_metrics", "support_benchmark_metrics", "probe_metrics", "probe_benchmark_metrics"):
            self.assertEqual(tables[name]["status"], "target_machine_test_deferred")
            self.assertIsInstance(tables[name]["deferred_reason"], str)

    def test_schema_forbids_tensor_and_gradient_persistence(self):
        from latent_grpo_runner.metrics.schemas import persistent_field_is_allowed

        self.assertTrue(persistent_field_is_allowed("signal/reward_mean"))
        self.assertFalse(persistent_field_is_allowed("full_logits"))
        self.assertFalse(persistent_field_is_allowed("gradient_norm"))
        self.assertFalse(persistent_field_is_allowed("component_log_probs"))
