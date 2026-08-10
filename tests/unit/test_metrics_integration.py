from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.unit.test_storage import JsonBackend


class DurableMetricsIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _observer(self, **kwargs):
        from latent_grpo_runner.metrics.integration import DurableMetricsObserver

        return DurableMetricsObserver(
            output_root=self.root,
            profile_name="smoke",
            seed=17,
            config_hash="cfg",
            backend=JsonBackend(),
            **kwargs,
        )

    @staticmethod
    def _actor_update(global_step: int, *, update_count: int = 1, workers: int = 2) -> dict[str, object]:
        return {
            "global_step": global_step,
            "observation_phase": "post_update",
            "aggregation_worker_count": workers,
            "optimizer_update_available": True,
            "update_count": update_count,
            "did_update": update_count > 0,
            "component_available": False,
            "component_unavailable_reason": "p1_not_connected",
            "component_sufficient_stats": None,
        }

    def test_actor_update_commits_one_schema_complete_explicitly_unavailable_row(self) -> None:
        observer = self._observer()
        observer.start_run()
        observer.emit("post_repeat_ids", {"global_step": 1, "trajectory_count": 4, "group_count": 2})
        observer.emit("ocp_selection", {"group_id": "g", "trajectory_id": 0})
        observer.emit("actor_update", self._actor_update(1, update_count=2, workers=2))

        parts = list((self.root / "train_step_metrics").glob("part-*.parquet"))
        self.assertEqual(len(parts), 1)
        row = JsonBackend().read(parts[0])["rows"][0]
        self.assertEqual(row["profile_name"], "smoke")
        self.assertEqual(row["seed"], 17)
        self.assertEqual(row["global_step"], 1)
        self.assertEqual(row["optimizer_step"], 2)
        self.assertEqual(row["aggregation_worker_count"], 2)
        self.assertFalse(row["record_available"])
        self.assertEqual(row["record_unavailable_reason"], "p1_metric_collectors_not_integrated")
        self.assertFalse(row["train/policy_loss__available"])
        self.assertIsNone(row["train/policy_loss"])
        from latent_grpo_runner.metrics.schemas import schema_manifest
        from latent_grpo_runner.validation.output_validator import validate_records
        self.assertEqual(
            validate_records(
                "train_step_metrics",
                [row],
                schema_manifest()["tables"]["train_step_metrics"],
            ),
            [],
        )
        self.assertEqual(observer.deferred_event_counts, {"post_repeat_ids": 1, "ocp_selection": 1})
        observer.close()

    def test_duplicate_step_and_non_driver_writer_fail_closed(self) -> None:
        from latent_grpo_runner.metrics.storage import DuplicatePrimaryKeyError, WriterAuthorityError

        observer = self._observer()
        observer.start_run()
        observer.emit("actor_update", self._actor_update(1))
        with self.assertRaises(DuplicatePrimaryKeyError):
            observer.emit("actor_update", self._actor_update(1))
        observer.close()

        non_driver = self._observer(writer_rank=1)
        with self.assertRaises(WriterAuthorityError):
            non_driver.start_run()
        non_driver.close()

    def test_checkpoint_resume_restores_optimizer_step_and_quarantines_future_part(self) -> None:
        observer = self._observer()
        observer.start_run()
        observer.emit("actor_update", self._actor_update(1, update_count=2))
        sidecar_path = observer.checkpoint(global_step=1)
        observer.emit("actor_update", self._actor_update(2, update_count=3))
        observer.close()

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(sidecar["optimizer_step"], 2)
        self.assertIn("train_step_metrics", sidecar["writer_manifests"])
        self.assertIn("train_group_metrics", sidecar["writer_manifests"])

        resumed = self._observer()
        resumed.start_run(resume_checkpoint_step=1)
        self.assertEqual(resumed.optimizer_step, 2)
        quarantine = self.root / "train_step_metrics" / "quarantine"
        self.assertEqual(len(list(quarantine.glob("future-step-*.parquet"))), 1)
        resumed.emit("actor_update", self._actor_update(2, update_count=1))
        self.assertEqual(resumed.optimizer_step, 3)
        self.assertEqual(resumed.writer_checkpoint("train_step_metrics")["committed_part_count"], 2)
        resumed.close()

    def test_support_and_checkpoint_probe_events_write_authoritative_tables(self) -> None:
        from latent_grpo_runner.metrics.probe import build_probe_benchmark_row, build_probe_metric_row
        from latent_grpo_runner.metrics.support import collect_support_metrics

        observer = self._observer()
        observer.start_run()
        support_rows, support_benchmark = collect_support_metrics(
            profile_name="smoke",
            seed=17,
            global_step=1,
            optimizer_step_at_observation=0,
            group_ids=["g"],
            trajectory_ids=[0],
            trajectory_classes=["correct"],
            trajectory_mean_old_log_probs=[-0.1],
            response_mask=[[True]],
            rollout_topk_ids=[[[1, 2]]],
            old_topk_indices=[[[2, 1]]],
        )
        observer.emit("support_metrics", {"rows": support_rows, "benchmark": support_benchmark})
        probe_row = build_probe_metric_row(
            profile_name="smoke",
            seed=17,
            global_step=1,
            optimizer_step=0,
            checkpoint_step=1,
            probe_batch_id="probe-a",
            deltas=[0.0],
            valid_delta_mask=[True],
            flipgrad_trigger_mask=[False],
            credit=None,
        )
        probe_benchmark = build_probe_benchmark_row(
            profile_name="smoke",
            seed=17,
            global_step=1,
            checkpoint_step=1,
            probe_batch_id="probe-a",
            probe_trajectory_count=1,
            probe_latent_position_count=1,
            credit_autograd_executed=False,
            probe_rng_restore_succeeded=True,
        )
        observer.emit("checkpoint_probe", {"rows": [probe_row], "benchmark": probe_benchmark})

        support_part = next((self.root / "support_metrics").glob("part-*.parquet"))
        support_record = JsonBackend().read(support_part)["rows"][0]
        self.assertEqual(support_record["support/retention_rate"], 1.0)
        probe_part = next((self.root / "probe_metrics").glob("part-*.parquet"))
        probe_record = JsonBackend().read(probe_part)["rows"][0]
        self.assertEqual(probe_record["observation_phase"], "checkpoint_probe")
        self.assertEqual(probe_record["credit_concentration_unavailable_reason"], "disabled_by_config")
        observer.close()

    def test_resume_requires_matching_sidecar_and_schema(self) -> None:
        from latent_grpo_runner.checkpointing import CheckpointCompatibilityError
        from latent_grpo_runner.metrics.storage import SchemaMismatchError

        missing_sidecar = self._observer()
        with self.assertRaises(CheckpointCompatibilityError):
            missing_sidecar.start_run(resume_checkpoint_step=1)
        missing_sidecar.close()

        observer = self._observer()
        observer.start_run()
        observer.emit("actor_update", self._actor_update(1))
        observer.close()
        schema_path = self.root / "train_step_metrics" / "_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["fields"] = schema["fields"][:-1]
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        reopened = self._observer()
        with self.assertRaises(SchemaMismatchError):
            reopened.start_run()
        reopened.close()

    def test_factory_is_disabled_by_default_and_requires_launcher_metadata(self) -> None:
        from latent_grpo_runner.metrics.integration import ObserverIntegrationError, create_observer_sink_from_env

        self.assertIsNone(create_observer_sink_from_env({"LATENT_GRPO_OBSERVER_ENABLED": "0"}, backend=JsonBackend()))
        with self.assertRaisesRegex(ObserverIntegrationError, "missing launcher metadata"):
            create_observer_sink_from_env({"LATENT_GRPO_OBSERVER_ENABLED": "1"}, backend=JsonBackend())
        sink = create_observer_sink_from_env(
            {
                "LATENT_GRPO_OBSERVER_ENABLED": "1",
                "LATENT_GRPO_OBSERVER_OUTPUT_ROOT": str(self.root),
                "LATENT_GRPO_OBSERVER_PROFILE_NAME": "smoke",
                "LATENT_GRPO_OBSERVER_SEED": "17",
                "LATENT_GRPO_OBSERVER_CONFIG_HASH": "cfg",
            },
            backend=JsonBackend(),
        )
        self.assertIsNotNone(sink)
        sink.close()


if __name__ == "__main__":
    unittest.main()
