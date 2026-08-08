import tempfile
import unittest
from pathlib import Path


class ResumeTests(unittest.TestCase):
    def test_sidecar_rejects_future_records_and_schema_mismatch(self):
        from latent_grpo_runner.checkpointing import CheckpointCompatibilityError, build_checkpoint_sidecar, validate_resume

        sidecar = build_checkpoint_sidecar(global_step=2, optimizer_step=2, config_hash="cfg", schema_version="metrics_schema_v1", upstream_commit="abc")
        with self.assertRaises(CheckpointCompatibilityError):
            validate_resume(sidecar, config_hash="cfg", schema_version="metrics_schema_v1", committed_steps={"train_step_metrics": 3})
        with self.assertRaises(CheckpointCompatibilityError):
            validate_resume(sidecar, config_hash="cfg", schema_version="metrics_schema_v2", committed_steps={})

    def test_sidecar_round_trip_is_atomic_and_resume_context_is_explicit(self):
        from latent_grpo_runner.checkpointing import build_checkpoint_sidecar, read_checkpoint_sidecar, write_checkpoint_sidecar

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_sidecar.json"
            sidecar = build_checkpoint_sidecar(global_step=2, optimizer_step=1, config_hash="cfg", schema_version="metrics_schema_v1", upstream_commit="abc")
            write_checkpoint_sidecar(path, sidecar)
            self.assertEqual(read_checkpoint_sidecar(path)["global_step"], 2)
            self.assertEqual(read_checkpoint_sidecar(path)["resume_context"], {"is_resume_run": True, "resume_from_step": 2})

    def test_future_part_is_quarantined_instead_of_deleted(self):
        from latent_grpo_runner.checkpointing import quarantine_future_part

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            part = root / "part-000003-deadbeef.parquet"
            part.write_text("preserve this", encoding="utf-8")
            quarantined = quarantine_future_part(part, root / "quarantine", "beyond_checkpoint_step")
            self.assertFalse(part.exists())
            self.assertEqual(quarantined.read_text(encoding="utf-8"), "preserve this")
            self.assertTrue(quarantined.name.startswith("beyond_checkpoint_step-"))

    def test_resume_factory_reopens_writer_with_sidecar_step_and_quarantines_future_parts(self):
        from latent_grpo_runner.checkpointing import resume_metric_writers_from_sidecar
        from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter
        from tests.unit.test_storage import JsonBackend

        schema = {"version": "metrics_schema_v1", "fields": ["profile_name", "seed", "global_step", "value"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AppendOnlyPartWriter(root, "train_step_metrics", schema, backend=JsonBackend())
            writer.append([{"profile_name": "s", "seed": 1, "global_step": 1, "value": 1.0}])
            writer.append([{"profile_name": "s", "seed": 1, "global_step": 2, "value": 2.0}])
            writer.close()
            sidecar = {"global_step": 1, "metrics_schema_version": "metrics_schema_v1", "writer_manifests": {"train_step_metrics": {}}}
            writers = resume_metric_writers_from_sidecar(root, sidecar, {"train_step_metrics": schema}, backend=JsonBackend())
            self.assertEqual(len(writers["train_step_metrics"].manifest["parts"]), 1)
            writers["train_step_metrics"].close()
