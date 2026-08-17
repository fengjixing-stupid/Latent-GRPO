"""Authoritative static run metadata for durable metrics validation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from latent_grpo_runner.metrics.schemas import schema_manifest
from latent_grpo_runner.metrics.storage import atomic_write_json

_METRICS_SCHEMA_VERSION = "metrics_schema_v1"
_GUMBEL_DEFERRED_REASON = "standalone_gumbel_diagnostic_not_enabled_for_training_profile"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def runtime_schema_manifest() -> dict[str, Any]:
    """Return the persisted manifest for a normal training/validation run.

    The standalone Gumbel diagnostic table is schema-declared but intentionally
    not collected on the formal training path.  Persist that fact explicitly so
    the output validator does not confuse an intentionally disabled diagnostic
    with a missing authoritative training table.
    """
    manifest = deepcopy(schema_manifest())
    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise RuntimeError("schema manifest tables are unavailable")
    gumbel = tables.get("gumbel_diagnostics")
    if not isinstance(gumbel, dict):
        raise RuntimeError("gumbel_diagnostics schema is unavailable")
    gumbel["status"] = "target_machine_test_deferred"
    gumbel["deferred_reason"] = _GUMBEL_DEFERRED_REASON
    return manifest


def write_run_start_metadata(
    *,
    output_root: str | Path,
    profile_name: str,
    profile_kind: str,
    seed: int,
    config_hash: str,
    resume_compatibility_hash: str,
    resolved_config: Mapping[str, Any],
    platform_snapshot: Mapping[str, Any],
) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    now = _utc_now()

    atomic_write_json(
        root / "run_config.json",
        {
            "metrics_schema_version": _METRICS_SCHEMA_VERSION,
            "profile_name": profile_name,
            "profile_kind": profile_kind,
            "seed": seed,
            "config_hash": config_hash,
            "resume_compatibility_hash": resume_compatibility_hash,
            "status": "running",
            "resolved_config": dict(resolved_config),
        },
    )
    atomic_write_json(root / "platform_config_snapshot.json", dict(platform_snapshot))
    atomic_write_json(root / "schema_manifest.json", runtime_schema_manifest())
    atomic_write_json(
        root / "run_status.json",
        {
            "last_committed_global_step": None,
            "last_committed_optimizer_step": None,
            "last_checkpoint_step": None,
            "last_error_message": None,
            "last_error_type": None,
            "status": "running",
            "updated_at": now,
        },
    )


def write_run_terminal_status(
    *,
    output_root: str | Path,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError("terminal run status must be completed or failed")
    root = Path(output_root)
    checkpoint_step: int | None = None
    pointer = root / "latest_checkpointed_iteration.txt"
    try:
        checkpoint_step = int(pointer.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        checkpoint_step = None
    atomic_write_json(
        root / "run_status.json",
        {
            "last_committed_global_step": checkpoint_step,
            "last_committed_optimizer_step": None,
            "last_checkpoint_step": checkpoint_step,
            "last_error_message": error_message,
            "last_error_type": error_type,
            "status": status,
            "updated_at": _utc_now(),
        },
    )
