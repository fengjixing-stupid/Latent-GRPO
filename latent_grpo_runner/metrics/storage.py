"""Atomic JSON and append-only Parquet part storage.

PyArrow is deliberately imported only by :class:`PyArrowBackend`, so config,
schema, and CPU unit-test workflows remain usable on development machines.
"""

from __future__ import annotations

import json
import os
import uuid
import fcntl
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class StorageError(RuntimeError):
    pass


class DuplicatePrimaryKeyError(StorageError):
    pass


class SchemaMismatchError(StorageError):
    pass


class WriterAuthorityError(StorageError):
    pass


class ParquetBackendUnavailable(StorageError):
    """Stable error for callers that need the optional PyArrow backend."""


class PartBackend(Protocol):
    def write(self, path: Path, rows: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> None: ...

    def read(self, path: Path) -> Mapping[str, Any]: ...


def atomic_write_json(path: str | Path, value: Any, *, fsync: bool = True) -> None:
    """Write a JSON document through a sibling temp file and atomic replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(temporary, target)
        if fsync:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class PyArrowBackend:
    """Production backend; import and native I/O occur only on first use."""

    @staticmethod
    def _modules():
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError as error:
            raise ParquetBackendUnavailable("PyArrow is required for Parquet metrics storage; install pyarrow to write or validate parts") from error
        return pa, pq

    def write(self, path: Path, rows: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> None:
        pa, pq = self._modules()
        field_specs = _field_specs(schema)
        _validate_rows_against_schema(rows, field_specs)
        table = pa.Table.from_pylist(list(rows), schema=_arrow_schema(pa, field_specs))
        pq.write_table(table, path)

    def read(self, path: Path) -> Mapping[str, Any]:
        pa, pq = self._modules()
        table = pq.read_table(path)
        return {"rows": table.to_pylist(), "schema": {"fields": [
            {"name": field.name, "physical_type": _physical_type_name(pa, field.type), "nullable": field.nullable}
            for field in table.schema
        ]}}


_PRIMARY_KEYS = {
    "train_step_metrics": ("profile_name", "seed", "global_step"),
    "train_group_metrics": ("profile_name", "seed", "global_step", "group_id"),
    "support_metrics": ("profile_name", "seed", "global_step", "group_id", "trajectory_id", "trajectory_class"),
    "probe_metrics": ("profile_name", "seed", "checkpoint_step", "probe_batch_id", "trajectory_group", "latent_position_group"),
    "probe_benchmark_metrics": ("profile_name", "seed", "checkpoint_step", "probe_batch_id"),
    "gumbel_diagnostics": ("profile_name", "seed", "diagnostic_run_id", "diagnostic_batch_index"),
}


def _canonical_schema(schema: Mapping[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _part_number(path: Path) -> int:
    return int(path.name.split("-", 2)[1])


def _field_specs(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = schema.get("fields", [])
    if not isinstance(fields, list):
        raise SchemaMismatchError("schema fields must be a list")
    return [dict(field) if isinstance(field, Mapping) else {"name": field, "nullable": True} for field in fields]


def _schema_matches(expected: Mapping[str, Any], actual: Any) -> bool:
    if not isinstance(actual, Mapping):
        return False
    if "version" in actual:
        return _canonical_schema(actual) == _canonical_schema(expected)
    actual_fields = actual.get("fields")
    if not isinstance(actual_fields, list):
        return False
    expected_fields = _field_specs(expected)
    if len(expected_fields) != len(actual_fields):
        return False
    for expected_field, actual_field in zip(expected_fields, actual_fields):
        if not isinstance(actual_field, Mapping) or actual_field.get("name") != expected_field["name"]:
            return False
        for property_name in ("physical_type", "nullable"):
            if property_name in expected_field and actual_field.get(property_name) != expected_field[property_name]:
                return False
    return True


def _physical_type_name(pa: Any, arrow_type: Any) -> str:
    if pa.types.is_string(arrow_type):
        return "string"
    if pa.types.is_boolean(arrow_type):
        return "bool"
    if pa.types.is_int64(arrow_type):
        return "int64"
    if pa.types.is_int32(arrow_type):
        return "int32"
    if pa.types.is_float64(arrow_type):
        return "float64"
    if pa.types.is_float32(arrow_type):
        return "float32"
    if pa.types.is_list(arrow_type):
        return f"list<{_physical_type_name(pa, arrow_type.value_type)}>"
    return str(arrow_type)


def _arrow_type(pa: Any, physical_type: str) -> Any:
    scalars = {"string": pa.string, "bool": pa.bool_, "int64": pa.int64, "int32": pa.int32,
               "float64": pa.float64, "float32": pa.float32}
    if physical_type in scalars:
        return scalars[physical_type]()
    if physical_type.startswith("list<") and physical_type.endswith(">"):
        return pa.list_(_arrow_type(pa, physical_type[5:-1]))
    raise SchemaMismatchError(f"unsupported Parquet physical type: {physical_type}")


def _arrow_schema(pa: Any, fields: Sequence[Mapping[str, Any]]) -> Any:
    try:
        return pa.schema([pa.field(field["name"], _arrow_type(pa, field.get("physical_type", field.get("logical_type"))),
                                   nullable=field.get("nullable", True)) for field in fields])
    except KeyError as error:
        raise SchemaMismatchError("Parquet schema field missing name") from error


def _value_matches(value: Any, physical_type: str) -> bool:
    if physical_type == "string":
        return isinstance(value, str)
    if physical_type == "bool":
        return type(value) is bool
    if physical_type in {"int64", "int32"}:
        return type(value) is int
    if physical_type in {"float64", "float32"}:
        return type(value) in {int, float}
    if physical_type.startswith("list<") and physical_type.endswith(">"):
        return isinstance(value, list) and all(_value_matches(item, physical_type[5:-1]) for item in value)
    return False


def _validate_rows_against_schema(rows: Sequence[Mapping[str, Any]], fields: Sequence[Mapping[str, Any]]) -> None:
    expected_names = {field["name"] for field in fields}
    for index, row in enumerate(rows):
        names = set(row)
        if names != expected_names:
            missing, extra = sorted(expected_names - names), sorted(names - expected_names)
            raise SchemaMismatchError(f"row {index} schema columns mismatch; missing={missing}, extra={extra}")
        for field in fields:
            value = row[field["name"]]
            if value is None:
                if not field.get("nullable", True):
                    raise SchemaMismatchError(f"row {index} non-nullable field is null: {field['name']}")
                continue
            physical_type = field.get("physical_type", field.get("logical_type"))
            if physical_type is not None and not _value_matches(value, physical_type):
                raise SchemaMismatchError(f"row {index} type mismatch for {field['name']}: expected {physical_type}")


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AppendOnlyPartWriter:
    """Single-driver, append-only writer with crash-safe part publication."""

    def __init__(
        self,
        output_root: str | Path,
        table_name: str,
        schema: Mapping[str, Any],
        *,
        backend: PartBackend | None = None,
        writer_rank: int = 0,
        primary_key: Sequence[str] | None = None,
        fsync: bool = True,
        resume_checkpoint_step: int | None = None,
    ) -> None:
        if writer_rank != 0:
            raise WriterAuthorityError("only driver/rank 0 may create a metrics writer")
        self.table_name = table_name
        self.schema = dict(schema)
        self.backend = backend or PyArrowBackend()
        self.primary_key = tuple(primary_key or _PRIMARY_KEYS.get(table_name, ("profile_name", "seed", "global_step")))
        self.fsync = fsync
        self.resume_checkpoint_step = resume_checkpoint_step
        self.table_dir = Path(output_root) / table_name
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.table_dir / ".writer.lock"
        self._lock_handle = None
        self._acquire_writer_lock()
        self.schema_path = self.table_dir / "_schema.json"
        self.manifest_path = self.table_dir / "_SUCCESS_PARTS.json"
        self.checkpoint_path = self.table_dir / "_WRITER_CHECKPOINT.json"
        try:
            self._verify_or_write_schema()
            self.manifest = self._load_or_rebuild_manifest()
            self._keys = self._committed_keys()
        except Exception:
            self.close()
            raise

    def _acquire_writer_lock(self) -> None:
        if self.lock_path in self._held_lock_paths:
            raise WriterAuthorityError(f"writer lock already held for table {self.table_name}")
        handle = self.lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise WriterAuthorityError(f"writer lock already held for table {self.table_name}") from error
        self._held_lock_paths.add(self.lock_path)
        self._lock_handle = handle

    def close(self) -> None:
        if getattr(self, "_lock_handle", None) is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None
            self._held_lock_paths.discard(self.lock_path)

    def __enter__(self) -> "AppendOnlyPartWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _verify_or_write_schema(self) -> None:
        if self.schema_path.exists():
            stored = json.loads(self.schema_path.read_text(encoding="utf-8"))
            if _canonical_schema(stored) != _canonical_schema(self.schema):
                raise SchemaMismatchError(f"schema mismatch for table {self.table_name}")
        else:
            atomic_write_json(self.schema_path, self.schema, fsync=self.fsync)

    def _load_or_rebuild_manifest(self) -> dict[str, Any]:
        existing: dict[str, Any] | None = None
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if _canonical_schema(existing.get("schema")) != _canonical_schema(self.schema):
                raise SchemaMismatchError(f"manifest schema mismatch for table {self.table_name}")
        parts = []
        for path in sorted(self.table_dir.glob("part-*.parquet"), key=_part_number):
            try:
                decoded = self.backend.read(path)
                rows = list(decoded.get("rows", []))
                if self.resume_checkpoint_step is not None and any(
                    isinstance(row.get("checkpoint_step", row.get("global_step")), int)
                    and row.get("checkpoint_step", row.get("global_step")) > self.resume_checkpoint_step
                    for row in rows
                ):
                    quarantine = self.table_dir / "quarantine"
                    quarantine.mkdir(exist_ok=True)
                    os.replace(path, quarantine / f"future-step-{path.name}")
                    continue
                _validate_rows_against_schema(rows, _field_specs(self.schema))
                if not _schema_matches(self.schema, decoded.get("schema")):
                    raise SchemaMismatchError("readback schema does not match table schema")
                parts.append(self._part_metadata(path.name, rows))
            except Exception:
                quarantine = self.table_dir / "quarantine"
                quarantine.mkdir(exist_ok=True)
                os.replace(path, quarantine / f"unreadable-{path.name}")
        manifest = {"table_name": self.table_name, "schema": self.schema, "primary_key": list(self.primary_key), "parts": parts}
        if existing != manifest:
            atomic_write_json(self.manifest_path, manifest, fsync=self.fsync)
        return manifest

    def _part_metadata(self, filename: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        keys = [self._key(row) for row in rows]
        return {"file": filename, "part_number": _part_number(Path(filename)), "rows": len(rows),
                "min_key": list(min(keys)) if keys else None, "max_key": list(max(keys)) if keys else None}

    def _key(self, row: Mapping[str, Any]) -> tuple[Any, ...]:
        missing = [name for name in self.primary_key if name not in row]
        if missing:
            raise DuplicatePrimaryKeyError(f"missing primary-key fields for {self.table_name}: {', '.join(missing)}")
        return tuple(row[name] for name in self.primary_key)

    def _committed_keys(self) -> set[tuple[Any, ...]]:
        keys: set[tuple[Any, ...]] = set()
        for part in self.manifest["parts"]:
            decoded = self.backend.read(self.table_dir / part["file"])
            for row in decoded.get("rows", []):
                key = self._key(row)
                if key in keys:
                    raise DuplicatePrimaryKeyError(f"duplicate primary key in committed parts: {key!r}")
                keys.add(key)
        return keys

    def append(self, rows: Sequence[Mapping[str, Any]]) -> Path:
        materialized = [dict(row) for row in rows]
        if not materialized:
            raise StorageError("cannot commit an empty Parquet part")
        batch_keys = [self._key(row) for row in materialized]
        if len(set(batch_keys)) != len(batch_keys):
            raise DuplicatePrimaryKeyError("duplicate primary key within pending batch")
        duplicate = next((key for key in batch_keys if key in self._keys), None)
        if duplicate is not None:
            raise DuplicatePrimaryKeyError(f"duplicate primary key against committed part: {duplicate!r}")
        part_number = (max((part["part_number"] for part in self.manifest["parts"]), default=-1) + 1)
        final_name = f"part-{part_number:06d}-{uuid.uuid4().hex}.parquet"
        final_path = self.table_dir / final_name
        temporary = self.table_dir / f".{final_name}.tmp"
        try:
            _validate_rows_against_schema(materialized, _field_specs(self.schema))
            self.backend.write(temporary, materialized, self.schema)
            if self.fsync:
                _fsync_file(temporary)
            # A backend must close before returning. The readback is deliberately
            # before rename, making incomplete tmp files non-authoritative.
            decoded = self.backend.read(temporary)
            read_rows = list(decoded.get("rows", []))
            if not _schema_matches(self.schema, decoded.get("schema")):
                raise SchemaMismatchError(f"part readback schema mismatch for table {self.table_name}")
            if len(read_rows) != len(materialized):
                raise StorageError(f"part readback row count mismatch for {self.table_name}")
            if set(self._key(row) for row in read_rows) != set(batch_keys):
                raise StorageError(f"part readback primary keys mismatch for {self.table_name}")
            os.replace(temporary, final_path)
            if self.fsync:
                _fsync_directory(self.table_dir)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        metadata = self._part_metadata(final_name, materialized)
        self.manifest["parts"].append(metadata)
        atomic_write_json(self.manifest_path, self.manifest, fsync=self.fsync)
        self._keys.update(batch_keys)
        atomic_write_json(self.checkpoint_path, self.writer_checkpoint(), fsync=self.fsync)
        return final_path

    def writer_checkpoint(self) -> dict[str, Any]:
        parts = self.manifest["parts"]
        return {"table_name": self.table_name, "last_part_number": max((part["part_number"] for part in parts), default=-1),
                "committed_part_count": len(parts), "primary_key": list(self.primary_key)}
    _held_lock_paths: set[Path] = set()
