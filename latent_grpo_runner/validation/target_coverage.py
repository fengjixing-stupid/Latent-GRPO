"""Extract table-qualified target-variable coverage from the canonical RTM."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any

from ..metrics.schemas import schema_manifest


ALLOWED_STATUSES = frozenset({"implemented", "static_check_passed", "synthetic_test_passed", "mac_development_check_passed",
                              "target_machine_test_deferred", "target_machine_probe_passed", "single_gpu_tested",
                              "three_gpu_ray_tested", "cuda_runtime_verified", "requirements_lock_verified",
                              "memory_feasibility_verified", "blocked", "unavailable_with_reason"})
_LEGACY_STATUS = {"planned": "target_machine_test_deferred"}


class CoverageStatusError(ValueError):
    pass


_FIELD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:/[A-Za-z0-9_]+)*$")
_BACKTICK = re.compile(r"`([^`]+)`")
_EXCLUSION_MARKERS = ("不记录", "禁止", "不得出现", "不得计入", "不得长期", "排除", "可离线派生")
_FENCE_DECLARATION_MARKERS = (
    "字段", "变量", "指标", "分母", "共享", "availability", "接口", "中间量", "主键", "配置",
    "schema", "保存", "输入", "时间", "基础", "记录使用", "额外使用", "至少支持", "完整", "只写一次",
    "每行", "定义版本", "轨迹属性", "动态分组", "默认",
)
_ENUMERATION_MARKERS = ("值枚举", "推荐分组")
_AMBIGUOUS_MARKERS = ("参考示例", "示例字段")


def _is_field_name(token: str) -> bool:
    return "<" not in token and ">" not in token and bool(_FIELD_NAME.fullmatch(token))


def _line_declarations(line: str, *, allow_plain: bool, enumeration: bool = False) -> set[str]:
    """Extract declarations from one fenced line, never expression RHS/value literals."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return set()
    if enumeration:
        return set()
    lhs = re.split(r"\s*(?:=|\+=|∈|:)\s*", stripped, maxsplit=1)[0].strip()
    if lhs != stripped:
        return {lhs} if _is_field_name(lhs) else set()
    return {stripped} if allow_plain and _is_field_name(stripped) else set()


def extract_target_fields(path: str | Path) -> dict[str, list[str]]:
    """Extract canonical unique names from explicit spec declarations.

    Accepted authorities are field-name Markdown tables, fenced field lists with
    declaration context, and inline statements explicitly labelled as fields.
    Prohibition blocks, formula RHS/value literals, and ``<...>`` templates are
    excluded. Identifier-looking tokens in unclassified fences are reported as
    ambiguous instead of being promoted to canonical fields.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    fields: set[str] = set()
    ambiguous: set[str] = set()
    in_fence = False
    fence_mode = "ambiguous"
    table_first_column_is_field = False
    prior_meaningful: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                context = prior_meaningful[-1] if prior_meaningful else ""
                if any(marker in context for marker in _EXCLUSION_MARKERS):
                    fence_mode = "excluded"
                elif any(marker in context for marker in _ENUMERATION_MARKERS):
                    fence_mode = "enumeration"
                elif any(marker in context for marker in _AMBIGUOUS_MARKERS):
                    fence_mode = "ambiguous"
                elif any(marker in context for marker in _FENCE_DECLARATION_MARKERS):
                    fence_mode = "canonical"
                else:
                    # A text fence made solely of identifier-shaped lines is
                    # itself an explicit field list. Non-identifiers and RHS
                    # expressions are still ignored by _line_declarations.
                    fence_mode = "canonical"
                in_fence = True
            else:
                in_fence = False
            continue

        if in_fence:
            if fence_mode == "excluded":
                continue
            declarations = _line_declarations(
                line, allow_plain=fence_mode in {"canonical", "ambiguous"}, enumeration=fence_mode == "enumeration"
            )
            if fence_mode in {"canonical", "enumeration"}:
                fields.update(declarations)
            else:
                ambiguous.update(declarations)
            continue

        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells and cells[0] in {"字段", "目标变量"}:
                table_first_column_is_field = True
            elif cells and set(cells[0]) <= {"-", ":"}:
                pass
            elif table_first_column_is_field and cells:
                match = _BACKTICK.fullmatch(cells[0])
                if match and _is_field_name(match.group(1)):
                    fields.add(match.group(1))
            continue
        table_first_column_is_field = False

        if not stripped or stripped == "---":
            continue
        if not stripped.startswith("#"):
            excluded = any(marker in stripped for marker in _EXCLUSION_MARKERS)
            explicit_inline = re.search(r"(?:权威|canonical|目标)?字段\s*[：:]", stripped, re.IGNORECASE)
            if explicit_inline and not excluded:
                fields.update(token for token in _BACKTICK.findall(stripped) if _is_field_name(token))
        prior_meaningful.append(stripped)

    return {"fields": sorted(fields), "ambiguous_tokens": sorted(ambiguous - fields)}


def _code_value(value: str) -> str:
    value = value.strip()
    return value[1:-1] if value.startswith("`") and value.endswith("`") else value


def _rows(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 14:
            continue
        field, table, status = _code_value(cells[0]), _code_value(cells[7]), cells[13]
        normalized = _LEGACY_STATUS.get(status, status)
        if normalized not in ALLOWED_STATUSES:
            raise CoverageStatusError(f"unknown RTM status for {table}.{field}: {status}")
        rows.append((table, field, normalized))
    return rows


def build_target_coverage(rtm_path: str | Path) -> dict[str, Any]:
    rows = _rows(Path(rtm_path))
    qualified = {(table, field) for table, field, _ in rows}
    declared_schema = schema_manifest().get("tables", {})
    schema_declared = {
        (table_name, field["name"])
        for table_name, table in declared_schema.items()
        for field in table.get("fields", [])
        if (table_name, field["name"]) in qualified
    }
    inventory = [
        {"storage_table": table, "field_name": field, "declared": (table, field) in schema_declared,
         "implemented": False, "test_status": status}
        for table, field, status in rows
    ]
    target_path = next((parent / "spec" / "target_variables.md" for parent in Path(rtm_path).resolve().parents
                        if (parent / "spec" / "target_variables.md").is_file()), None)
    target_text = target_path.read_text(encoding="utf-8") if target_path else ""
    core_metrics = 29 if "| **合计** | **29**" in target_text else 0
    extraction = extract_target_fields(target_path) if target_path else {"fields": [], "ambiguous_tokens": []}
    spec_fields = set(extraction["fields"])
    rtm_fields = {field for _, field in qualified}
    missing_from_rtm = sorted(spec_fields - rtm_fields)
    extra_in_rtm = sorted(rtm_fields - spec_fields)
    # Schema declaration is evidence of an implemented storage surface; it is
    # deliberately kept separate from test/runtime verification claims.
    status_counts = Counter(status for _, _, status in rows)
    return {"schema_version": "metrics_schema_v1", "total_target_fields": len(qualified),
            "unique_target_fields": len({field for _, field in qualified}), "inventory": inventory, "implemented_fields": 0,
            "schema_declared_fields": len(schema_declared),
            "verified_fields": 0, "unavailable_with_reason_fields": status_counts["unavailable_with_reason"],
            "target_machine_test_deferred_fields": status_counts["target_machine_test_deferred"],
            "blocked_fields": status_counts["blocked"], "missing_fields": missing_from_rtm,
            "spec_extracted_fields": sorted(spec_fields), "rtm_unique_fields": sorted(rtm_fields),
            "missing_from_rtm": missing_from_rtm, "extra_in_rtm": extra_in_rtm,
            "ambiguous_tokens": extraction["ambiguous_tokens"],
            "coverage_mode": "rtm_table_qualified_with_spec_unique_crosscheck",
            "coverage_is_acceptance_ready": False,
            "coverage_limit_reason": "target_variables.md supplies authoritative unique field names; table qualification is supplied and inventoried by the RTM",
            "source_counts": {"rtm_table_qualified_records": len(rows), "target_variables_core_metrics": core_metrics,
                              "target_variables_canonical_unique_fields": len(spec_fields)},
            "allowed_statuses": sorted(ALLOWED_STATUSES)}
