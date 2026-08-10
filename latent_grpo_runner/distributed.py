"""Launch planning with a single Ray driver by default.

No CUDA, Ray, SGLang, or torch imports belong in this module.  The actual
upstream process imports its training runtime only after the target gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping

from .config import ResolvedConfig


@dataclass(frozen=True)
class LauncherPlan:
    mode: str
    command: tuple[str, ...]
    working_directory: Path
    control_rank_only: bool
    target_status: str = "target_machine_test_deferred"
    metrics_enabled: bool = False
    metrics_sink_status: str = "disabled"

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "command": list(self._redacted_command()),
            "working_directory": "<upstream>/verl-0.4.x",
            "control_rank_only": self.control_rank_only,
            "target_status": self.target_status,
            "metrics_enabled": self.metrics_enabled,
            "metrics_sink_status": self.metrics_sink_status,
        }

    def _redacted_command(self) -> tuple[str, ...]:
        redacted: list[str] = []
        for index, item in enumerate(self.command):
            if index == 0:
                redacted.append("<python>")
            if "trainer.default_local_dir=" in item or "trainer.resume_from_path=" in item:
                key, _ = item.split("=", 1)
                redacted.append(f"{key}=<configured-path>")
            elif index != 0:
                redacted.append(item)
        return tuple(redacted)


def build_launcher_plan(config: ResolvedConfig) -> LauncherPlan:
    upstream_root = config.paths["upstream_repo_path"]
    working_directory = upstream_root / "verl-0.4.x"
    command = (sys.executable, "-m", "verl.trainer.main_ppo", *config.author_hydra_overrides())
    return LauncherPlan(
        mode=config.launcher.mode,
        command=command,
        working_directory=working_directory,
        control_rank_only=config.launcher.mode == "torchrun_control",
        metrics_enabled=config.features.metrics_enabled,
        metrics_sink_status=(
            "driver_append_only_p1_ready" if config.features.metrics_enabled else "disabled"
        ),
    )


def launch(
    config: ResolvedConfig,
    *,
    run_command: Callable[..., int] | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Start exactly one upstream driver under the selected launcher mode."""
    plan = build_launcher_plan(config)
    runtime_environment = dict(os.environ if environment is None else environment)
    runtime_environment["LATENT_GRPO_OBSERVER_ENABLED"] = "1" if config.features.metrics_enabled else "0"
    runtime_environment["LATENT_GRPO_OBSERVER_OUTPUT_ROOT"] = str(config.paths["output_root"])
    runtime_environment["LATENT_GRPO_OBSERVER_PROFILE_NAME"] = config.profile_name
    runtime_environment["LATENT_GRPO_OBSERVER_SEED"] = str(config.training.seed)
    runtime_environment["LATENT_GRPO_OBSERVER_CONFIG_HASH"] = config.resume_compatibility_hash
    runtime_environment["LATENT_GRPO_SUPPORT_ENABLED"] = "1" if config.features.support_enabled else "0"
    runtime_environment["LATENT_GRPO_CHECKPOINT_PROBE_ENABLED"] = "1" if config.features.checkpoint_probe_enabled else "0"
    runtime_environment["LATENT_GRPO_CREDIT_PROBE_ENABLED"] = "1" if config.features.credit_probe_enabled else "0"
    if config.features.metrics_enabled:
        workspace_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = runtime_environment.get("PYTHONPATH", "")
        python_paths = [path for path in existing_pythonpath.split(os.pathsep) if path != workspace_root]
        runtime_environment["PYTHONPATH"] = os.pathsep.join([workspace_root, *python_paths])
    if plan.control_rank_only and int(runtime_environment.get("RANK", "0")) != 0:
        return 0
    if plan.control_rank_only:
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            runtime_environment.pop(key, None)
    if run_command is not None:
        return int(run_command(plan.command, cwd=plan.working_directory, env=runtime_environment))
    completed = subprocess.run(
        plan.command,
        cwd=plan.working_directory,
        env=runtime_environment,
        check=False,
    )
    return completed.returncode
