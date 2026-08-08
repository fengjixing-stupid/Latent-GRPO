"""Driver-owned durable observer integration for the upstream VERL trainer.

P0 deliberately owns transport, writer authority, lifecycle, checkpoint/resume,
and explicit unavailable rows.  Stage 1/2 metric formulas remain in their
existing builders and are connected in P1; this module must not duplicate
training math or trigger model work.
"""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
from typing import Any, Mapping

from latent_grpo_runner.checkpointing import (
    CheckpointCompatibilityError,
    build_checkpoint_sidecar,
    read_checkpoint_sidecar,
    resume_metric_writers_from_sidecar,
    write_checkpoint_sidecar,
)
from latent_grpo_runner.metrics.events import StepContext
from latent_grpo_runner.metrics.p1 import P1AggregationError, build_p1_train_step_metrics
from latent_grpo_runner.metrics.schemas import schema_manifest
from latent_grpo_runner.metrics.storage import AppendOnlyPartWriter, PartBackend


_METRICS_SCHEMA_VERSION = "metrics_schema_v1"
_P0_UNAVAILABLE_REASON = "p1_metric_collectors_not_integrated"
_CHECKPOINT_SIDECAR_NAME = "latent_grpo_metrics_sidecar.json"
_WRITER_TABLES = ("train_step_metrics", "train_group_metrics")


class ObserverIntegrationError(RuntimeError):
    """Fail-closed error raised by the authoritative metrics integration."""


class DurableMetricsObserver:
    """Single-coordinator observer backed by append-only Parquet writers.

    P0 writes one authoritative, schema-complete ``train_step_metrics`` row
    when the already-existing ``actor_update`` event arrives.  The 16 Stage
    1/2 values are intentionally marked unavailable until P1 connects their
    sufficient-statistic collectors/builders.  This makes the persistence path
    real without inventing metric values or changing training math.
    """

    durable = True
    enabled = True

    def __init__(
        self,
        *,
        output_root: str | Path,
        profile_name: str,
        seed: int,
        config_hash: str,
        upstream_commit: str = "unknown",
        backend: PartBackend | None = None,
        writer_rank: int = 0,
    ) -> None:
        if not profile_name:
            raise ObserverIntegrationError("observer profile_name must be non-empty")
        if type(seed) is not int:
            raise ObserverIntegrationError("observer seed must be an integer")
        if not config_hash:
            raise ObserverIntegrationError("observer config_hash must be non-empty")
        self.output_root = Path(output_root)
        self.profile_name = profile_name
        self.seed = seed
        self.config_hash = config_hash
        self.upstream_commit = upstream_commit or "unknown"
        self.backend = backend
        self.writer_rank = writer_rank
        self._writers: dict[str, AppendOnlyPartWriter] = {}
        self._started = False
        self._closed = False
        self._is_resume_run = False
        self._resume_from_step: int | None = None
        self._optimizer_step = 0
        self._deferred_event_counts: Counter[str] = Counter()

        manifest = schema_manifest()
        if manifest.get("metrics_schema_version") != _METRICS_SCHEMA_VERSION:
            raise ObserverIntegrationError("unexpected metrics schema version")
        tables = manifest.get("tables")
        if not isinstance(tables, Mapping):
            raise ObserverIntegrationError("metrics schema tables are unavailable")
        self._table_schemas = {name: dict(tables[name]) for name in _WRITER_TABLES}

    @property
    def started(self) -> bool:
        return self._started

    @property
    def optimizer_step(self) -> int:
        return self._optimizer_step

    @property
    def deferred_event_counts(self) -> dict[str, int]:
        return dict(self._deferred_event_counts)

    def start_run(self, *, resume_checkpoint_step: int | None = None) -> None:
        """Open the only authoritative writers after upstream resume is known."""
        if self._closed:
            raise ObserverIntegrationError("cannot start a closed observer")
        if self._started:
            expected = self._resume_from_step if self._is_resume_run else None
            if resume_checkpoint_step != expected:
                raise ObserverIntegrationError("observer start_run called with conflicting resume step")
            return
        if resume_checkpoint_step is not None and (
            type(resume_checkpoint_step) is not int or resume_checkpoint_step < 1
        ):
            raise ObserverIntegrationError("resume_checkpoint_step must be a positive integer or null")

        self.output_root.mkdir(parents=True, exist_ok=True)
        if resume_checkpoint_step is None:
            self._writers = self._open_new_writers()
        else:
            sidecar_path = self._checkpoint_sidecar_path(resume_checkpoint_step)
            if not sidecar_path.is_file():
                raise CheckpointCompatibilityError(
                    f"metrics-enabled resume requires checkpoint sidecar: {sidecar_path}"
                )
            sidecar = read_checkpoint_sidecar(sidecar_path)
            if sidecar.get("config_hash") != self.config_hash:
                raise CheckpointCompatibilityError("resume config hash mismatch")
            if sidecar.get("metrics_schema_version") != _METRICS_SCHEMA_VERSION:
                raise CheckpointCompatibilityError("resume metrics schema version mismatch")
            optimizer_step = sidecar.get("optimizer_step")
            if type(optimizer_step) is not int or optimizer_step < 0:
                raise CheckpointCompatibilityError("resume sidecar has invalid optimizer_step")
            self._writers = resume_metric_writers_from_sidecar(
                self.output_root,
                sidecar,
                self._table_schemas,
                backend=self.backend,
            )
            self._optimizer_step = optimizer_step
            self._is_resume_run = True
            self._resume_from_step = resume_checkpoint_step
        self._started = True

    def emit(self, event_type: str, facts: Mapping[str, object]) -> None:
        """Consume detached coordinator events without introducing model work."""
        self._require_started()
        if not isinstance(event_type, str) or not event_type:
            raise ObserverIntegrationError("observer event_type must be a non-empty string")
        if not isinstance(facts, Mapping):
            raise ObserverIntegrationError("observer facts must be a mapping")

        if event_type == "actor_update":
            self._commit_train_step(facts)
            return

        # These existing hooks are intentionally accepted at P0 so enabling the
        # durable sink does not block the training path.  Their schema-complete
        # persistence belongs to P1/group/eval integration and no raw tensor is
        # retained here.
        if event_type in {"post_repeat_ids", "ocp_selection", "eval_question"}:
            self._deferred_event_counts[event_type] += 1
            return
        raise ObserverIntegrationError(f"unsupported observer event_type: {event_type}")

    def checkpoint(self, *, global_step: int, checkpoint_dir: str | Path | None = None) -> Path:
        """Publish metrics sidecar before upstream marks a checkpoint latest."""
        self._require_started()
        if type(global_step) is not int or global_step < 1:
            raise ObserverIntegrationError("checkpoint global_step must be a positive integer")
        expected_dir = self.output_root / f"global_step_{global_step}"
        target_dir = expected_dir if checkpoint_dir is None else Path(checkpoint_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        sidecar = build_checkpoint_sidecar(
            global_step=global_step,
            optimizer_step=self._optimizer_step,
            config_hash=self.config_hash,
            schema_version=_METRICS_SCHEMA_VERSION,
            upstream_commit=self.upstream_commit,
            writer_manifests={name: writer.writer_checkpoint() for name, writer in self._writers.items()},
        )
        sidecar["observer_deferred_event_counts"] = self.deferred_event_counts
        path = target_dir / _CHECKPOINT_SIDECAR_NAME
        write_checkpoint_sidecar(path, sidecar)
        return path

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as error:  # close every writer before surfacing the first failure
                errors.append(error)
        self._writers.clear()
        self._closed = True
        if errors:
            raise ObserverIntegrationError("failed to close one or more metrics writers") from errors[0]

    def writer_checkpoint(self, table_name: str) -> dict[str, Any]:
        self._require_started()
        try:
            return self._writers[table_name].writer_checkpoint()
        except KeyError as error:
            raise ObserverIntegrationError(f"unknown writer table: {table_name}") from error

    def _open_new_writers(self) -> dict[str, AppendOnlyPartWriter]:
        writers: dict[str, AppendOnlyPartWriter] = {}
        try:
            for name, schema in self._table_schemas.items():
                writers[name] = AppendOnlyPartWriter(
                    self.output_root,
                    name,
                    schema,
                    backend=self.backend,
                    writer_rank=self.writer_rank,
                    primary_key=schema["primary_key"],
                )
            return writers
        except Exception:
            for writer in writers.values():
                writer.close()
            raise

    def _commit_train_step(self, facts: Mapping[str, object]) -> None:
        global_step = facts.get("global_step")
        update_count = facts.get("update_count")
        aggregation_worker_count = facts.get("aggregation_worker_count")
        if type(global_step) is not int or global_step < 1:
            raise ObserverIntegrationError("actor_update requires positive integer global_step")
        if facts.get("observation_phase") != "post_update":
            raise ObserverIntegrationError("actor_update must be emitted at post_update")
        if facts.get("optimizer_update_available") is not True:
            raise ObserverIntegrationError("authoritative optimizer_step unavailable from worker consensus")
        if type(update_count) is not int or update_count < 0:
            raise ObserverIntegrationError("actor_update requires non-negative integer update_count")
        if type(aggregation_worker_count) is not int or aggregation_worker_count < 1:
            raise ObserverIntegrationError("actor_update requires positive aggregation_worker_count")

        next_optimizer_step = self._optimizer_step + update_count
        p1_declared = (
            "p1_worker_metrics_available" in facts
            or "p1_driver_metrics_available" in facts
        )
        if p1_declared:
            if facts.get("p1_worker_metrics_available") is not True:
                raise ObserverIntegrationError(
                    "P1 worker statistics unavailable: "
                    + str(facts.get("p1_worker_metrics_unavailable_reason") or "unknown")
                )
            if facts.get("p1_driver_metrics_available") is not True:
                raise ObserverIntegrationError("P1 driver statistics unavailable")
            worker_statistics = facts.get("p1_worker_sufficient_stats")
            driver_statistics = facts.get("p1_driver_sufficient_stats")
            trajectory_lengths = facts.get("final_training_trajectory_lengths")
            step_time = facts.get("driver_step_time_seconds")
            if not isinstance(worker_statistics, Mapping) or not isinstance(driver_statistics, Mapping):
                raise ObserverIntegrationError("P1 sufficient statistics must be mappings")
            if not isinstance(trajectory_lengths, (list, tuple)):
                raise ObserverIntegrationError("P1 final trajectory lengths must be a sequence")
            if type(step_time) not in {int, float}:
                raise ObserverIntegrationError("P1 driver step time must be numeric")
            learning_rate = facts.get("learning_rate")
            metrics_compute_time = facts.get("metrics_compute_time")
            context = StepContext(
                profile_name=self.profile_name,
                seed=self.seed,
                global_step=global_step,
                optimizer_step=next_optimizer_step,
                observation_phase="post_update",
                learning_rate=(
                    float(learning_rate) if type(learning_rate) in {int, float} else None
                ),
                is_resume_run=self._is_resume_run,
                resume_from_step=self._resume_from_step,
            )
            try:
                row = build_p1_train_step_metrics(
                    context=context,
                    worker_statistics=worker_statistics,
                    driver_statistics=driver_statistics,
                    final_training_trajectory_lengths=trajectory_lengths,
                    driver_step_time_seconds=float(step_time),
                    aggregation_worker_count=aggregation_worker_count,
                    metrics_compute_time=(
                        float(metrics_compute_time)
                        if type(metrics_compute_time) in {int, float}
                        else None
                    ),
                )
            except (P1AggregationError, ValueError, TypeError) as error:
                raise ObserverIntegrationError("invalid P1 train-step payload") from error
        else:
            # Backward-compatible P0 synthetic fixtures remain explicitly
            # unavailable. The production P1 trainer always declares P1 fields.
            row = self._p0_unavailable_train_step_row(
                global_step=global_step,
                optimizer_step=next_optimizer_step,
                aggregation_worker_count=aggregation_worker_count,
            )
        self._writers["train_step_metrics"].append([row])
        self._optimizer_step = next_optimizer_step

    def _p0_unavailable_train_step_row(
        self, *, global_step: int, optimizer_step: int, aggregation_worker_count: int
    ) -> dict[str, object]:
        schema = self._table_schemas["train_step_metrics"]
        row: dict[str, object] = {field["name"]: None for field in schema["fields"]}
        row.update(
            {
                "profile_name": self.profile_name,
                "seed": self.seed,
                "metric_scope": "train_step",
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "observation_phase": "post_update",
                "is_resume_run": self._is_resume_run,
                "resume_from_step": self._resume_from_step,
                "aggregation_worker_count": aggregation_worker_count,
                "record_version": "p0_transport_v1",
                "record_available": False,
                "record_unavailable_reason": _P0_UNAVAILABLE_REASON,
                "train_core_available": False,
                "train_core_unavailable_reason": _P0_UNAVAILABLE_REASON,
                "mixture_available": False,
                "mixture_unavailable_reason": _P0_UNAVAILABLE_REASON,
                "mask_available": False,
                "mask_unavailable_reason": _P0_UNAVAILABLE_REASON,
                "signal_available": False,
                "signal_unavailable_reason": _P0_UNAVAILABLE_REASON,
                "stage2_available": False,
                "stage2_unavailable_reason": _P0_UNAVAILABLE_REASON,
            }
        )
        for field in schema["fields"]:
            name = field["name"]
            if name.endswith("__available"):
                row[name] = False
            elif name.endswith("__unavailable_reason"):
                row[name] = _P0_UNAVAILABLE_REASON
            elif field.get("nullable") is False and row[name] is None:
                physical_type = field.get("physical_type", field.get("logical_type"))
                if physical_type in {"int64", "int32"}:
                    row[name] = 0
                elif physical_type == "bool":
                    row[name] = False
                elif physical_type == "string":
                    raise ObserverIntegrationError(f"P0 row lacks required string field: {name}")
                elif physical_type in {"float64", "float32"}:
                    row[name] = 0.0
                else:
                    raise ObserverIntegrationError(f"unsupported required P0 field type: {name}")
        return row

    def _checkpoint_sidecar_path(self, global_step: int) -> Path:
        return self.output_root / f"global_step_{global_step}" / _CHECKPOINT_SIDECAR_NAME

    def _require_started(self) -> None:
        if self._closed:
            raise ObserverIntegrationError("observer is closed")
        if not self._started:
            raise ObserverIntegrationError("observer must start_run before use")


def create_observer_sink_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    backend: PartBackend | None = None,
) -> DurableMetricsObserver | None:
    """Create the TaskRunner-owned sink from launcher-authoritative metadata."""
    values = os.environ if environ is None else environ
    enabled = values.get("LATENT_GRPO_OBSERVER_ENABLED", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None

    required = {
        "output_root": values.get("LATENT_GRPO_OBSERVER_OUTPUT_ROOT"),
        "profile_name": values.get("LATENT_GRPO_OBSERVER_PROFILE_NAME"),
        "seed": values.get("LATENT_GRPO_OBSERVER_SEED"),
        "config_hash": values.get("LATENT_GRPO_OBSERVER_CONFIG_HASH"),
    }
    missing = sorted(name for name, value in required.items() if value in {None, ""})
    if missing:
        raise ObserverIntegrationError(
            "enabled observer is missing launcher metadata: " + ", ".join(missing)
        )
    try:
        seed = int(str(required["seed"]))
    except ValueError as error:
        raise ObserverIntegrationError("LATENT_GRPO_OBSERVER_SEED must be an integer") from error

    return DurableMetricsObserver(
        output_root=str(required["output_root"]),
        profile_name=str(required["profile_name"]),
        seed=seed,
        config_hash=str(required["config_hash"]),
        upstream_commit=values.get("LATENT_GRPO_OBSERVER_UPSTREAM_COMMIT", "unknown"),
        backend=backend,
    )
