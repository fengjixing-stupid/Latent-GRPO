import json
import tempfile
import unittest
from pathlib import Path


class RunMetadataTests(unittest.TestCase):
    def test_start_metadata_writes_required_static_files_and_defers_standalone_gumbel(self):
        from latent_grpo_runner.run_metadata import write_run_start_metadata

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run_start_metadata(
                output_root=root,
                profile_name="3gpu-final-validation",
                profile_kind="final_runtime_validation",
                seed=17,
                config_hash="config-hash",
                resume_compatibility_hash="resume-hash",
                resolved_config={"profile_name": "3gpu-final-validation"},
                platform_snapshot={"cuda_available": True, "failure_reasons": []},
            )
            for name in (
                "run_config.json",
                "platform_config_snapshot.json",
                "schema_manifest.json",
                "run_status.json",
            ):
                self.assertTrue((root / name).is_file(), name)
            manifest = json.loads((root / "schema_manifest.json").read_text())
            self.assertEqual(
                manifest["tables"]["gumbel_diagnostics"]["status"],
                "target_machine_test_deferred",
            )
            self.assertTrue(
                manifest["tables"]["gumbel_diagnostics"]["deferred_reason"]
            )
            status = json.loads((root / "run_status.json").read_text())
            self.assertEqual(status["status"], "running")

    def test_terminal_status_records_checkpoint_pointer_without_inventing_optimizer_step(self):
        from latent_grpo_runner.run_metadata import write_run_terminal_status

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latest_checkpointed_iteration.txt").write_text("2\n")
            write_run_terminal_status(output_root=root, status="completed")
            status = json.loads((root / "run_status.json").read_text())
            self.assertEqual(status["last_checkpoint_step"], 2)
            self.assertEqual(status["last_committed_global_step"], 2)
            self.assertIsNone(status["last_committed_optimizer_step"])
            self.assertEqual(status["status"], "completed")


if __name__ == "__main__":
    unittest.main()
