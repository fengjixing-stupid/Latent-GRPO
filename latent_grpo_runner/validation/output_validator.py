"""Validate committed output artifacts without importing PyArrow at module load."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..metrics.schemas import persistent_field_is_allowed
from ..metrics.storage import PyArrowBackend, _field_specs, _schema_matches, _value_matches, atomic_write_json


REQUIRED_STATIC_FILES = ("run_config.json", "platform_config_snapshot.json", "schema_manifest.json", "run_status.json")
REQUIRED_DYNAMIC_TABLES = frozenset({"train_step_metrics", "train_group_metrics", "eval_question_results", "eval_clean_topk",
                                     "gumbel_diagnostics", "support_metrics", "support_benchmark_metrics",
                                     "probe_metrics", "probe_benchmark_metrics"})


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


def validate_records(table_name: str, rows: Sequence[Mapping[str, Any]], table_schema: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    primary_key = tuple(table_schema.get("primary_key", ()))
    seen: set[tuple[Any, ...]] = set()
    last_step_by_run: dict[tuple[Any, Any], int] = {}
    last_tokens_by_run: dict[tuple[Any, Any], int] = {}
    for index, row in enumerate(rows):
        for field in _field_specs(table_schema):
            name = field["name"]
            if name not in row:
                errors.append(f"{table_name}[{index}] missing required field: {name}")
                continue
            value = row[name]
            if value is None and not field.get("nullable", True):
                errors.append(f"{table_name}[{index}] non-nullable field is null: {name}")
            elif value is not None and field.get("physical_type") and not _value_matches(value, field["physical_type"]):
                errors.append(f"{table_name}[{index}] physical type mismatch: {name}")
        expected_names = {field["name"] for field in _field_specs(table_schema)}
        for name in set(row) - expected_names:
            if expected_names:
                errors.append(f"{table_name}[{index}] unexpected field: {name}")
        expected_record_version = table_schema.get("record_version")
        if expected_record_version is not None and row.get("record_version") != expected_record_version:
            errors.append(f"{table_name}[{index}] unsupported record version: {row.get('record_version')!r}")
        for name, value in row.items():
            if not persistent_field_is_allowed(name):
                errors.append(f"{table_name}[{index}] prohibited persistent field: {name}")
            if name.endswith("__available") and value is False:
                metric = name[:-11]
                if row.get(metric) is not None or not isinstance(row.get(f"{metric}__unavailable_reason"), str):
                    errors.append(f"{table_name}[{index}] unavailable metric must be null with a reason: {metric}")
            if name.endswith("_available") and not name.endswith("__available") and value is False:
                family = name[:-10]
                reason = row.get(f"{family}_unavailable_reason")
                if not isinstance(reason, str) or not reason:
                    errors.append(f"{table_name}[{index}] unavailable family must have a reason: {family}")
            if (name.endswith("_count") or name == "count") and value is not None and (not isinstance(value, int) or value < 0):
                errors.append(f"{table_name}[{index}] negative or invalid count: {name}")
            if (name.endswith("_rate") or name.endswith("_fraction")) and value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
                errors.append(f"{table_name}[{index}] rate outside [0,1]: {name}")
            if name.endswith("_std") and value is not None and (not isinstance(value, (int, float)) or value < 0):
                errors.append(f"{table_name}[{index}] negative std: {name}")
        if primary_key:
            missing = [name for name in primary_key if name not in row]
            if missing:
                errors.append(f"{table_name}[{index}] missing primary-key fields: {', '.join(missing)}")
            else:
                key = tuple(row[name] for name in primary_key)
                if key in seen:
                    errors.append(f"{table_name}[{index}] duplicate primary key: {key!r}")
                seen.add(key)
        if table_name in {"train_step_metrics", "train_group_metrics"}:
            run_key = (row.get("profile_name"), row.get("seed"))
            step = row.get("global_step")
            if isinstance(step, int):
                prior = last_step_by_run.get(run_key)
                if prior is not None and step < prior:
                    errors.append(f"{table_name}[{index}] global_step is not monotonic for {run_key!r}")
                last_step_by_run[run_key] = step if prior is None else max(prior, step)
            if "checkpoint_step" in row:
                errors.append(f"{table_name}[{index}] ordinary training tables must not contain checkpoint_step")
        if table_name == "train_step_metrics" and all(name in row for name in ("profile_name", "seed", "train/generated_token_count", "cumulative_rollout_tokens")):
            run_key = (row["profile_name"], row["seed"])
            generated, cumulative = row["train/generated_token_count"], row["cumulative_rollout_tokens"]
            if isinstance(generated, int) and isinstance(cumulative, int):
                prior = last_tokens_by_run.get(run_key)
                if prior is not None and cumulative != prior + generated:
                    errors.append(f"{table_name}[{index}] cumulative_rollout_tokens is not continuous")
                last_tokens_by_run[run_key] = cumulative
        if table_name in {"probe_metrics", "probe_benchmark_metrics", "eval_question_results", "eval_clean_topk"} and "cumulative_rollout_tokens" in row:
            errors.append(f"{table_name}[{index}] cumulative_rollout_tokens is training-only")
        if table_name == "train_group_metrics":
            correct, incorrect, total = (row.get("group/correct_trajectory_count"), row.get("group/non_correct_trajectory_count"), row.get("group/trajectory_count"))
            if all(isinstance(value, int) for value in (correct, incorrect, total)) and correct + incorrect != total:
                errors.append(f"{table_name}[{index}] correct + non_correct must equal trajectory_count")
        if table_name == "eval_clean_topk":
            ids, probs, k = row.get("clean_topk_token_ids"), row.get("clean_topk_probs"), row.get("clean_topk_k")
            if ids is not None and (not isinstance(ids, list) or not isinstance(probs, list) or len(ids) != len(probs) or len(ids) != k):
                errors.append(f"{table_name}[{index}] clean top-k lists do not match clean_topk_k")
            elif ids is not None and (any(type(token) is not int for token in ids) or any(type(probability) not in {int, float} or not math.isfinite(probability) or probability < 0 for probability in probs)):
                errors.append(f"{table_name}[{index}] clean top-k token IDs/probabilities are invalid")
        if "clean_topk_token_ids" in row:
            ids, probs, k = row.get("clean_topk_token_ids"), row.get("clean_topk_probs"), row.get("clean_topk_k")
            if ids is not None and (not isinstance(ids, list) or not isinstance(probs, list) or len(ids) != len(probs) or len(ids) != k
                                    or any(type(token) is not int for token in ids)
                                    or any(type(probability) not in {int, float} or not math.isfinite(probability) or probability < 0 for probability in probs)):
                errors.append(f"{table_name}[{index}] clean top-k token IDs/probabilities are invalid")
    return errors


def _read_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid JSON {path}: {error}")
        return None


def _find_table_directory(root: Path, table_name: str) -> Path | None:
    candidates = [root / table_name, root / "metrics" / table_name]
    candidates.extend(root.glob(f"eval/**/{table_name}"))
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def _discovered_dynamic_tables(root: Path) -> set[str]:
    discovered = set()
    metrics = root / "metrics"
    if metrics.is_dir():
        discovered.update(path.name for path in metrics.iterdir() if path.is_dir())
    for table_name in REQUIRED_DYNAMIC_TABLES:
        if (root / table_name).is_dir() or any((root / "eval").glob(f"**/{table_name}")):
            discovered.add(table_name)
    return discovered


def _is_explicitly_deferred(table_schema: Mapping[str, Any]) -> bool:
    return table_schema.get("status") == "target_machine_test_deferred" and isinstance(table_schema.get("deferred_reason"), str) and bool(table_schema["deferred_reason"])


def validate_output_directory(root: str | Path) -> ValidationResult:
    root_path = Path(root)
    errors: list[str] = []
    for filename in REQUIRED_STATIC_FILES:
        json_path = root_path / filename
        if not json_path.exists():
            errors.append(f"missing required static file: {filename}")
        else:
            _read_json(json_path, errors)
    manifest = _read_json(root_path / "schema_manifest.json", errors) if (root_path / "schema_manifest.json").exists() else {"tables": {}}
    tables = manifest.get("tables", {}) if isinstance(manifest, dict) else {}
    if not isinstance(tables, dict):
        errors.append("schema manifest tables must be an object")
        tables = {}
    for table_name in REQUIRED_DYNAMIC_TABLES:
        if table_name not in tables:
            errors.append(f"schema manifest missing required dynamic table: {table_name}")
    for table_name in _discovered_dynamic_tables(root_path) - set(tables):
        errors.append(f"undeclared dynamic table: {table_name}")
    for csv_path in root_path.rglob("*.csv"):
        errors.append(f"CSV cannot be an authoritative metrics table: {csv_path.relative_to(root_path)}")
    backend = PyArrowBackend()
    for table_name, table_schema in tables.items():
        if not isinstance(table_schema, Mapping):
            errors.append(f"{table_name}: schema must be an object")
            continue
        directory = _find_table_directory(root_path, table_name)
        if directory is None:
            if table_name in REQUIRED_DYNAMIC_TABLES and not _is_explicitly_deferred(table_schema):
                errors.append(f"{table_name}: missing required dynamic table directory")
            continue
        if not (directory / "_schema.json").exists():
            errors.append(f"{table_name}: missing table schema")
        part_manifest_path = directory / "_SUCCESS_PARTS.json"
        part_manifest = _read_json(part_manifest_path, errors) if part_manifest_path.exists() else None
        if not isinstance(part_manifest, dict):
            errors.append(f"{table_name}: missing success-part manifest")
            continue
        actual_files = {path.name for path in directory.glob("part-*.parquet")}
        declared_files = {part.get("file") for part in part_manifest.get("parts", [])}
        if actual_files != declared_files:
            errors.append(f"{table_name}: manifest does not match committed parts")
        rows: list[Mapping[str, Any]] = []
        for part in part_manifest.get("parts", []):
            try:
                decoded = backend.read(directory / part["file"])
                if not _schema_matches(table_schema, decoded.get("schema")):
                    errors.append(f"{table_name}: part schema does not match table manifest")
                rows.extend(decoded.get("rows", []))
            except Exception as error:  # read corruption and optional-backend absence are validation failures
                errors.append(f"{table_name}: unreadable Parquet part {part.get('file')}: {error}")
        errors.extend(validate_records(table_name, rows, table_schema))
    reports = {
        "schema_validation.json": {"ok": not errors, "errors": errors},
        "primary_key_validation.json": {"ok": not any("primary key" in error for error in errors), "errors": [error for error in errors if "primary key" in error]},
        "completeness_validation.json": {"ok": not errors, "errors": errors},
    }
    for name, report in reports.items():
        atomic_write_json(root_path / "validation" / name, report)
    return ValidationResult(not errors, errors)
