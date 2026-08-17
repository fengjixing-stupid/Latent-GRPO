"""Mac-safe Latent-GRPO profile validator and upstream launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from latent_grpo_runner.config import ConfigError, load_config, validate_latent_end_token
from latent_grpo_runner.distributed import build_launcher_plan, launch
from latent_grpo_runner.environment import collect_environment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Runner YAML profile")
    parser.add_argument("--profile-name", help="Must match the profile declared by --config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--model-path", help="HF repo ID or local author SFT model directory")
    parser.add_argument("--train-files", help="Training parquet path for an engineering probe")
    parser.add_argument("--val-files", help="Validation/test parquet path for an engineering probe")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--enable-support", dest="support_enabled", action="store_true")
    parser.add_argument("--disable-support", dest="support_enabled", action="store_false")
    parser.add_argument("--enable-metrics", dest="metrics_enabled", action="store_true")
    parser.add_argument("--disable-metrics", dest="metrics_enabled", action="store_false")
    parser.add_argument("--enable-checkpoint-probe", dest="checkpoint_probe_enabled", action="store_true")
    parser.add_argument("--disable-checkpoint-probe", dest="checkpoint_probe_enabled", action="store_false")
    parser.add_argument("--enable-credit-probe", dest="credit_probe_enabled", action="store_true")
    parser.add_argument("--allow-hardware-mismatch", action="store_true")
    parser.set_defaults(metrics_enabled=None, support_enabled=None, checkpoint_probe_enabled=None, credit_probe_enabled=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace_root = Path(__file__).resolve().parent
    try:
        config = load_config(args.config, workspace_root=workspace_root)
        if args.profile_name and args.profile_name != config.profile_name:
            raise ConfigError("--profile-name must match the profile declared by --config")
        config = config.with_runtime_overrides(
            seed=args.seed,
            output_root=Path(args.output_root) if args.output_root else None,
            max_steps=args.max_steps,
            resume_from=Path(args.resume_from) if args.resume_from else None,
            model_path=args.model_path,
            train_files=args.train_files,
            val_files=args.val_files,
            metrics_enabled=args.metrics_enabled,
            support_enabled=args.support_enabled,
            checkpoint_probe_enabled=args.checkpoint_probe_enabled,
            credit_probe_enabled=args.credit_probe_enabled,
        )
    except (ConfigError, ValueError) as error:
        print(f"configuration_error: {error}", file=sys.stderr)
        return 2

    plan = build_launcher_plan(config)
    if args.dry_run or args.validate_config:
        # This branch intentionally performs no target CUDA/Ray/SGLang/torch import.
        config.paths["output_root"].mkdir(parents=True, exist_ok=True)
        environment = collect_environment(mode="development", workspace_root=workspace_root)
        print(
            json.dumps(
                {
                    "status": "mac_development_check_passed",
                    "config_hash": config.config_hash,
                    "validation": "static_check_passed",
                    "launcher_plan": plan.as_dict(),
                    "target_environment": "target_machine_test_deferred",
                    "environment": environment,
                },
                sort_keys=True,
            )
        )
        return 0

    target_report = collect_environment(
        mode="target",
        require_gpus=config.hardware.required_gpus,
        min_vram_gb=config.hardware.min_vram_gb,
        required_precision=config.rollout.dtype,
        workspace_root=workspace_root,
    )
    if target_report["failure_reasons"] and not args.allow_hardware_mismatch:
        print(json.dumps(target_report, sort_keys=True), file=sys.stderr)
        return 3
    try:
        from transformers import AutoConfig, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model.path)
        model_metadata = AutoConfig.from_pretrained(config.model.path)
        if config.profile_name in {"kaggle-t4-monitor", "kaggle-t4-30-metric"} and getattr(model_metadata, "quantization_config", None):
            raise ConfigError("Kaggle T4 profiles require the unquantized author SFT model")
        latent_end_validation = validate_latent_end_token(config.model, tokenizer, model_metadata)
        print(json.dumps({"latent_end_validation": latent_end_validation}, sort_keys=True))
    except (ConfigError, ModuleNotFoundError, OSError, ValueError) as error:
        print(f"latent_end_validation_error: {error}", file=sys.stderr)
        return 4
    if config.features.metrics_enabled:
        from latent_grpo_runner.run_metadata import write_run_start_metadata

        write_run_start_metadata(
            output_root=config.paths["output_root"],
            profile_name=config.profile_name,
            profile_kind=config.profile_kind,
            seed=config.training.seed,
            config_hash=config.config_hash,
            resume_compatibility_hash=config.resume_compatibility_hash,
            resolved_config=config._hashable_mapping(),
            platform_snapshot=target_report,
        )
    try:
        exit_code = launch(config)
    except RuntimeError as error:
        if config.features.metrics_enabled:
            from latent_grpo_runner.run_metadata import write_run_terminal_status

            write_run_terminal_status(
                output_root=config.paths["output_root"],
                status="failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
        print(f"launch_blocked: {error}", file=sys.stderr)
        return 5
    if config.features.metrics_enabled:
        from latent_grpo_runner.run_metadata import write_run_terminal_status

        write_run_terminal_status(
            output_root=config.paths["output_root"],
            status="completed" if exit_code == 0 else "failed",
            error_type=None if exit_code == 0 else "UpstreamExitCode",
            error_message=None if exit_code == 0 else f"upstream exited with code {exit_code}",
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
