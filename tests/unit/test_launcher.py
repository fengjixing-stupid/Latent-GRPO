from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from latent_grpo_runner.config import load_config
from latent_grpo_runner.distributed import build_launcher_plan, launch
from scripts.probe_ray_distributed import collect_ray_placement_evidence


ROOT = Path(__file__).resolve().parents[2]


class LauncherTests(unittest.TestCase):
    def test_metrics_config_authoritatively_sets_observer_environment(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml", workspace_root=ROOT)
        seen: list[dict[str, str]] = []

        def fake_run(_command: tuple[str, ...], **kwargs: object) -> int:
            seen.append(dict(kwargs["env"]))
            return 0

        launch(config, run_command=fake_run, environment={"LATENT_GRPO_OBSERVER_ENABLED": "0"})
        self.assertEqual(seen[-1]["LATENT_GRPO_OBSERVER_ENABLED"], "1")
        self.assertIn(str(ROOT), seen[-1]["PYTHONPATH"].split(os.pathsep))

        disabled = config.with_runtime_overrides(metrics_enabled=False)
        launch(disabled, run_command=fake_run, environment={"LATENT_GRPO_OBSERVER_ENABLED": "1"})
        self.assertEqual(seen[-1]["LATENT_GRPO_OBSERVER_ENABLED"], "0")

    def test_resume_keeps_the_same_authoritative_metrics_mapping(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml", workspace_root=ROOT).with_runtime_overrides(
            resume_from=Path("artifacts/checkpoint")
        )
        seen: list[dict[str, str]] = []

        def fake_run(_command: tuple[str, ...], **kwargs: object) -> int:
            seen.append(dict(kwargs["env"]))
            return 0

        launch(config, run_command=fake_run, environment={})
        self.assertEqual(seen[0]["LATENT_GRPO_OBSERVER_ENABLED"], "1")

    def test_real_metrics_launch_fails_before_starting_without_parquet_sink(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml", workspace_root=ROOT)
        with self.assertRaisesRegex(RuntimeError, "durable AppendOnlyPartWriter/Stage1/2 Parquet sink"):
            launch(config, environment={})

    def test_default_launcher_is_single_ray_direct_driver(self) -> None:
        config = load_config(ROOT / "configs" / "3gpu-low.yaml", workspace_root=ROOT)
        plan = build_launcher_plan(config)
        self.assertEqual(plan.mode, "ray_direct")
        self.assertEqual(plan.command[:3], (sys.executable, "-m", "verl.trainer.main_ppo"))
        self.assertFalse(plan.control_rank_only)
        self.assertIn("trainer.n_gpus_per_node=3", plan.command)

    def test_dry_run_imports_neither_ray_nor_sglang_nor_torch(self) -> None:
        command = [
            sys.executable,
            "train_latent_grpo.py",
            "--config",
            "configs/smoke.yaml",
            "--dry-run",
            "--validate-config",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            Path(temporary_directory, "sitecustomize.py").write_text(
                "import builtins\n"
                "original_import = builtins.__import__\n"
                "def guarded_import(name, *args, **kwargs):\n"
                "    if name.split('.')[0] in {'ray', 'sglang', 'torch', 'flash_attn', 'flashinfer'}:\n"
                "        raise AssertionError('dry-run imported forbidden runtime: ' + name)\n"
                "    return original_import(name, *args, **kwargs)\n"
                "builtins.__import__ = guarded_import\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = temporary_directory + os.pathsep + str(ROOT)
            completed = subprocess.run(
                command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("launcher_plan", completed.stdout)
        self.assertIn("target_machine_test_deferred", completed.stdout)

    def test_torchrun_control_only_allows_rank_zero_to_launch(self) -> None:
        config = load_config(ROOT / "configs" / "3gpu-low.yaml", workspace_root=ROOT)
        config = config.with_launcher_mode("torchrun_control")
        seen: list[tuple[str, ...]] = []

        def fake_run(command: tuple[str, ...], **_: object) -> int:
            seen.append(command)
            return 0

        with patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "3"}, clear=False):
            self.assertEqual(launch(config, run_command=fake_run), 0)
        self.assertEqual(seen, [])

        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "3"}, clear=False):
            self.assertEqual(launch(config, run_command=fake_run), 0)
        self.assertEqual(len(seen), 1)

    def test_ray_probe_submits_one_gpu_workers_and_proves_bindings_and_error_propagation(self) -> None:
        class ObjectRef:
            def __init__(self, function, args):
                self.function = function
                self.args = args

        class RemoteFunction:
            def __init__(self, ray, function, options):
                self.ray = ray
                self.function = function
                self.options = options

            def remote(self, *args):
                self.ray.submitted_task_options.append(self.options)
                return ObjectRef(self.function, args)

        class FakeRay:
            def __init__(self):
                self.submitted_options = []
                self.submitted_task_options = []
                self.current_gpu = []

            def get_gpu_ids(self):
                return self.current_gpu

            def remote(self, **options):
                def decorate(function):
                    self.submitted_options.append(options)
                    return RemoteFunction(self, function, options)

                return decorate

            def get(self, refs):
                if isinstance(refs, list):
                    return [self.get(reference) for reference in refs]
                self.current_gpu = [float(refs.args[0])] if refs.args else [2.0]
                try:
                    return refs.function(*refs.args)
                finally:
                    self.current_gpu = []

        ray = FakeRay()
        evidence = collect_ray_placement_evidence(ray, num_gpus=3)
        self.assertTrue(evidence["binding_validation_passed"])
        self.assertEqual(evidence["driver_ray_gpu_ids"], [])
        self.assertTrue(evidence["worker_exception_propagated"])
        self.assertEqual(len(evidence["worker_bindings"]), 3)
        self.assertEqual([options["num_gpus"] for options in ray.submitted_task_options], [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
