"""Checkpoint sidecar metadata and strict resume compatibility checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .metrics.storage import atomic_write_json


class CheckpointCompatibilityError(RuntimeError):
    pass


def build_checkpoint_sidecar(*, global_step: int, optimizer_step: int, config_hash: str, schema_version: str, upstream_commit: str,
                             writer_manifests: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"sidecar_version": "checkpoint_sidecar_v1", "global_step": global_step, "optimizer_step": optimizer_step,
            "config_hash": config_hash, "metrics_schema_version": schema_version, "upstream_commit": upstream_commit,
            "writer_manifests": dict(writer_manifests or {}),
            "resume_context": {"is_resume_run": True, "resume_from_step": global_step}}


def write_checkpoint_sidecar(path: str | Path, sidecar: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(sidecar))


def read_checkpoint_sidecar(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointCompatibilityError(f"invalid checkpoint sidecar: {path}") from error


def validate_resume(sidecar: Mapping[str, Any], *, config_hash: str, schema_version: str,
                    committed_steps: Mapping[str, int]) -> None:
    if sidecar.get("config_hash") != config_hash:
        raise CheckpointCompatibilityError("resume config hash mismatch")
    if sidecar.get("metrics_schema_version") != schema_version:
        raise CheckpointCompatibilityError("resume metrics schema version mismatch")
    checkpoint_step = sidecar.get("global_step")
    if not isinstance(checkpoint_step, int):
        raise CheckpointCompatibilityError("resume checkpoint is missing integer global_step")
    future = {table: step for table, step in committed_steps.items() if step > checkpoint_step}
    if future:
        raise CheckpointCompatibilityError(f"future committed records require quarantine: {future}")


def quarantine_future_part(part_path: str | Path, quarantine_directory: str | Path, reason: str) -> Path:
    """Move an untrusted future part aside; never delete crash/recovery evidence."""
    source = Path(part_path)
    target_directory = Path(quarantine_directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / f"{reason}-{source.name}"
    if target.exists():
        raise CheckpointCompatibilityError(f"quarantine collision: {target}")
    os.replace(source, target)
    return target


def resume_metric_writers_from_sidecar(output_root: str | Path, sidecar: Mapping[str, Any], table_schemas: Mapping[str, Mapping[str, Any]], *, backend: Any = None) -> dict[str, Any]:
    """Production resume factory: sidecar step drives writer reconciliation/quarantine."""
    from .metrics.storage import AppendOnlyPartWriter

    checkpoint_step = sidecar.get("global_step")
    if not isinstance(checkpoint_step, int):
        raise CheckpointCompatibilityError("resume sidecar missing integer global_step")
    if sidecar.get("metrics_schema_version") != "metrics_schema_v1":
        raise CheckpointCompatibilityError("resume sidecar metrics schema version mismatch")
    requested = sidecar.get("writer_manifests", {})
    if not isinstance(requested, Mapping):
        raise CheckpointCompatibilityError("resume sidecar writer_manifests must be an object")
    writers = {}
    try:
        for table_name, schema in table_schemas.items():
            if table_name not in requested:
                raise CheckpointCompatibilityError(f"resume sidecar missing writer manifest: {table_name}")
            writers[table_name] = AppendOnlyPartWriter(output_root, table_name, schema, backend=backend, resume_checkpoint_step=checkpoint_step)
        return writers
    except Exception:
        for writer in writers.values():
            writer.close()
        raise
