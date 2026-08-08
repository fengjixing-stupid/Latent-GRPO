import json
import tempfile
import unittest
from pathlib import Path


class JsonBackend:
    """A small backend double: Parquet mechanics remain integration-tested with PyArrow."""

    def write(self, path, rows, schema):
        Path(path).write_text(json.dumps({"rows": rows, "schema": schema}), encoding="utf-8")

    def read(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))


class WrongSchemaBackend(JsonBackend):
    def read(self, path):
        decoded = super().read(path)
        decoded["schema"] = {"version": "wrong", "fields": []}
        return decoded


class StrictFakeBackend(JsonBackend):
    """Exercises the schema contract without importing the optional PyArrow wheel."""

    def read(self, path):
        decoded = super().read(path)
        decoded["schema"] = {"fields": [
            {"name": "profile_name", "physical_type": "string", "nullable": False},
            {"name": "seed", "physical_type": "int64", "nullable": False},
            {"name": "global_step", "physical_type": "int64", "nullable": False},
            {"name": "value", "physical_type": "float64", "nullable": True},
        ]}
        return decoded


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.schema = {"version": "metrics_schema_v1", "fields": ["profile_name", "seed", "global_step", "value"]}

    def tearDown(self):
        self.directory.cleanup()

    def test_atomic_json_replace_never_leaves_partial_json(self):
        from latent_grpo_runner.metrics.storage import atomic_write_json

        target = self.root / "run_status.json"
        atomic_write_json(target, {"status": "running", "value": None}, fsync=True)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"status": "running", "value": None})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_writer_commits_closed_readable_part_then_manifest_and_checkpoint(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter

        writer = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        part = writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 2, "value": None}])
        self.assertTrue(part.exists())
        manifest = json.loads((part.parent / "_SUCCESS_PARTS.json").read_text())
        self.assertEqual(manifest["parts"][0]["rows"], 1)
        self.assertEqual(writer.writer_checkpoint()["last_part_number"], 0)

    def test_fake_backend_round_trip_preserves_list_and_null_availability_fields(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter

        schema = {"version": "metrics_schema_v1", "fields": ["profile_name", "seed", "global_step", "clean_topk_token_ids", "metric", "metric__available", "metric__unavailable_reason"]}
        writer = AppendOnlyPartWriter(self.root, "train_step_metrics", schema, backend=JsonBackend())
        part = writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 2, "clean_topk_token_ids": [1, 3], "metric": None,
                               "metric__available": False, "metric__unavailable_reason": "disabled_by_config"}])
        decoded = JsonBackend().read(part)["rows"][0]
        self.assertEqual(decoded["clean_topk_token_ids"], [1, 3])
        self.assertIsNone(decoded["metric"])

    def test_duplicate_primary_key_is_rejected_in_batch_pending_and_previous_part(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter, DuplicatePrimaryKeyError

        writer = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        row = {"profile_name": "smoke", "seed": 1, "global_step": 2, "value": 1.0}
        with self.assertRaises(DuplicatePrimaryKeyError):
            writer.append([row, dict(row)])
        writer.append([row])
        with self.assertRaises(DuplicatePrimaryKeyError):
            writer.append([dict(row)])

    def test_temp_files_are_ignored_and_manifest_rebuilds_from_committed_parts(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter

        writer = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 2, "value": 1.0}])
        table = self.root / "train_step_metrics"
        (table / "orphan.parquet.tmp").write_text("incomplete", encoding="utf-8")
        (table / "_SUCCESS_PARTS.json").unlink()
        writer.close()
        restored = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        self.assertEqual(restored.writer_checkpoint()["last_part_number"], 0)
        self.assertEqual(len(json.loads((table / "_SUCCESS_PARTS.json").read_text())["parts"]), 1)

    def test_schema_mismatch_and_non_driver_writer_are_rejected(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter, SchemaMismatchError, WriterAuthorityError

        with self.assertRaises(WriterAuthorityError):
            AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend(), writer_rank=1)
        AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        with self.assertRaises(SchemaMismatchError):
            AppendOnlyPartWriter(self.root, "train_step_metrics", {"version": "other", "fields": []}, backend=JsonBackend())

    def test_readback_schema_mismatch_prevents_part_publication(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter, SchemaMismatchError

        writer = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=WrongSchemaBackend())
        with self.assertRaises(SchemaMismatchError):
            writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 2, "value": 1.0}])
        self.assertEqual(list((self.root / "train_step_metrics").glob("part-*.parquet")), [])

    def test_stale_manifest_is_reconciled_with_readable_renamed_orphan_part(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter, DuplicatePrimaryKeyError

        writer = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 1, "value": 1.0}])
        table = self.root / "train_step_metrics"
        before_second_commit = (table / "_SUCCESS_PARTS.json").read_text(encoding="utf-8")
        writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 2, "value": 2.0}])
        (table / "_SUCCESS_PARTS.json").write_text(before_second_commit, encoding="utf-8")
        writer.close()
        resumed = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        self.assertEqual(len(resumed.manifest["parts"]), 2)
        with self.assertRaises(DuplicatePrimaryKeyError):
            resumed.append([{"profile_name": "smoke", "seed": 1, "global_step": 2, "value": 2.0}])

    def test_bad_orphan_part_is_quarantined_during_reconciliation(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter

        writer = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        table = self.root / "train_step_metrics"
        (table / "part-000000-bad.parquet").write_text("not a closed part", encoding="utf-8")
        writer.close()
        resumed = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        self.assertEqual(resumed.manifest["parts"], [])
        self.assertEqual(len(list((table / "quarantine").glob("unreadable-part-*.parquet"))), 1)

    def test_readback_requires_physical_type_and_nullability_match_manifest(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter, SchemaMismatchError

        schema = {"version": "metrics_schema_v1", "fields": [
            {"name": "profile_name", "physical_type": "string", "nullable": False},
            {"name": "seed", "physical_type": "int64", "nullable": False},
            {"name": "global_step", "physical_type": "int64", "nullable": False},
            {"name": "value", "physical_type": "float32", "nullable": True},
        ]}
        writer = AppendOnlyPartWriter(self.root, "train_step_metrics", schema, backend=StrictFakeBackend())
        with self.assertRaises(SchemaMismatchError):
            writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 1, "value": 1.0}])

    def test_schema_contract_rejects_missing_extra_and_wrongly_typed_columns_before_write(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter, SchemaMismatchError

        schema = {"version": "metrics_schema_v1", "fields": [
            {"name": "profile_name", "physical_type": "string", "nullable": False},
            {"name": "seed", "physical_type": "int64", "nullable": False},
            {"name": "global_step", "physical_type": "int64", "nullable": False},
            {"name": "value", "physical_type": "float64", "nullable": True},
        ]}
        writer = AppendOnlyPartWriter(self.root, "train_step_metrics", schema, backend=StrictFakeBackend())
        with self.assertRaises(SchemaMismatchError):
            writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 1}])
        with self.assertRaises(SchemaMismatchError):
            writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 1, "value": 1.0, "extra": True}])
        with self.assertRaises(SchemaMismatchError):
            writer.append([{"profile_name": "smoke", "seed": "one", "global_step": 1, "value": 1.0}])

    def test_writer_lock_rejects_second_process_writer_until_close(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter, WriterAuthorityError

        first = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        with self.assertRaisesRegex(WriterAuthorityError, "writer lock"):
            AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        first.close()
        with AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend()) as reopened:
            self.assertEqual(reopened.writer_checkpoint()["committed_part_count"], 0)

    def test_resume_initialization_quarantines_future_committed_part(self):
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter

        writer = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend())
        writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 1, "value": 1.0}])
        writer.append([{"profile_name": "smoke", "seed": 1, "global_step": 2, "value": 2.0}])
        writer.close()
        resumed = AppendOnlyPartWriter(self.root, "train_step_metrics", self.schema, backend=JsonBackend(), resume_checkpoint_step=1)
        self.assertEqual(len(resumed.manifest["parts"]), 1)
        self.assertEqual(len(list((self.root / "train_step_metrics" / "quarantine").glob("future-step-*.parquet"))), 1)
        resumed.close()
