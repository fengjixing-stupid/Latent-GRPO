import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TargetCoverageTests(unittest.TestCase):
    def test_authoritative_extractor_uses_declared_sources_and_quarantines_ambiguity(self):
        from latent_grpo_runner.validation.target_coverage import extract_target_fields

        document = """# Target variables

| 字段 | 定义 |
|---|---|
| `table_metric` | canonical table field |

完整字段框架：

```text
fenced_field
assigned_field="value_constant"
formula_field = exp(-sum_i q_i)
<family>_available
```

权威字段：`inline_field`。

明确不记录：

```text
forbidden_field
```

值枚举：

```text
trajectory_class:
  correct
  non_correct
```

参考示例：

```text
uncertain_example
```
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "target_variables.md"
            path.write_text(document, encoding="utf-8")
            result = extract_target_fields(path)

        self.assertEqual(result["fields"], ["assigned_field", "fenced_field", "formula_field", "inline_field", "table_metric"])
        self.assertEqual(result["ambiguous_tokens"], ["uncertain_example"])
        self.assertNotIn("value_constant", result["fields"])
        self.assertNotIn("correct", result["fields"])
        self.assertNotIn("<family>_available", result["fields"])
        self.assertNotIn("forbidden_field", result["fields"])

    def test_coverage_computes_real_unique_field_differences(self):
        from latent_grpo_runner.validation.target_coverage import build_target_coverage

        def row(field: str, table: str) -> str:
            cells = [f"`{field}`", "1", "raw_fact", "definition", "phase", "n/a", "copy", f"`{table}`",
                     "string", "module", "T-1", "true", "reason", "planned"]
            return "| " + " | ".join(cells) + " |"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "spec").mkdir()
            (root / "docs").mkdir()
            (root / "spec" / "target_variables.md").write_text(
                "字段清单：\n\n```text\nshared_field\nspec_only_field\n```\n", encoding="utf-8"
            )
            rtm = root / "docs" / "requirements_traceability_matrix.md"
            rtm.write_text("\n".join((row("shared_field", "table_a"), row("rtm_only_field", "table_b"))), encoding="utf-8")
            report = build_target_coverage(rtm)

        self.assertEqual(report["spec_extracted_fields"], ["shared_field", "spec_only_field"])
        self.assertEqual(report["rtm_unique_fields"], ["rtm_only_field", "shared_field"])
        self.assertEqual(report["missing_from_rtm"], ["spec_only_field"])
        self.assertEqual(report["extra_in_rtm"], ["rtm_only_field"])
        self.assertEqual(report["missing_fields"], ["spec_only_field"])
        self.assertEqual(report["ambiguous_tokens"], [])
        self.assertEqual(report["coverage_mode"], "rtm_table_qualified_with_spec_unique_crosscheck")

    def test_rtm_extraction_preserves_table_qualified_coverage_and_allowed_statuses(self):
        from latent_grpo_runner.validation.target_coverage import build_target_coverage

        report = build_target_coverage(Path("docs/requirements_traceability_matrix.md"))
        self.assertEqual(report["total_target_fields"], 420)
        self.assertEqual(report["unique_target_fields"], 330)
        self.assertEqual(report["implemented_fields"], 0)
        self.assertGreater(report["schema_declared_fields"], 0)
        self.assertEqual(report["verified_fields"], 0)
        self.assertEqual(report["unavailable_with_reason_fields"], 0)
        self.assertGreater(report["target_machine_test_deferred_fields"], 0)
        self.assertEqual(len(report["inventory"]), 420)
        self.assertEqual(report["source_counts"]["rtm_table_qualified_records"], 420)
        self.assertEqual(report["source_counts"]["target_variables_core_metrics"], 29)
        self.assertTrue(all({"storage_table", "field_name", "declared", "implemented", "test_status"} <= set(item) for item in report["inventory"]))
        self.assertGreater(report["source_counts"]["target_variables_canonical_unique_fields"], 0)
        self.assertEqual(report["missing_fields"], [])
        self.assertEqual(report["missing_fields"], report["missing_from_rtm"])
        self.assertIn("implemented", report["allowed_statuses"])

    def test_coverage_rejects_unknown_status_instead_of_claiming_it_implemented(self):
        from latent_grpo_runner.validation.target_coverage import CoverageStatusError, build_target_coverage

        path = Path("docs/requirements_traceability_matrix.md")
        content = path.read_text(encoding="utf-8").replace(
            "| true | null plus stable family reason | planned |",
            "| true | null plus stable family reason | imaginary_passed |", 1,
        )
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "rtm.md"
            copy.write_text(content, encoding="utf-8")
            with self.assertRaises(CoverageStatusError):
                build_target_coverage(copy)

    def test_planned_status_is_deferred_not_an_unavailable_claim(self):
        from latent_grpo_runner.validation.target_coverage import build_target_coverage

        report = build_target_coverage(Path("docs/requirements_traceability_matrix.md"))
        self.assertEqual(report["unavailable_with_reason_fields"], 0)
