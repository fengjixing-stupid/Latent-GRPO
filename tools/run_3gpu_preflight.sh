#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
BLOCKED_REASON=""
CONFIG="configs/3gpu-final-validation.yaml"
MODEL_PATH=""
TRAIN_DATA=""
VAL_DATA=""
OUTPUT_ROOT=""
GPUS=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_PREEXISTING_GPU_MEMORY_MIB="${LATENT_GRPO_MAX_PREEXISTING_GPU_MEMORY_MIB:-2048}"
MAX_PREEXISTING_GPU_UTILIZATION="${LATENT_GRPO_MAX_PREEXISTING_GPU_UTILIZATION:-20}"

blocked() {
  BLOCKED_REASON="$1"
  echo "3GPU_PREFLIGHT_GATE: BLOCKED"
  echo "BLOCKED_REASON: ${BLOCKED_REASON}"
  echo "LOG_PATH: ${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/validation/3gpu-final}/logs/preflight.log"
  echo "NEXT_ACTION: fix the named blocker and rerun this command"
  exit 2
}

while (($#)); do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --val-data) VAL_DATA="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--config configs/3gpu-final-validation.yaml] --model-path LOCAL_MODEL_DIR --train-data LOCAL_FILE --val-data LOCAL_FILE --output-root DIR --gpus 4,5,6"; exit 0 ;;
    *) blocked "unknown_argument:$1" ;;
  esac
done

[[ -n "${MODEL_PATH}" && -n "${TRAIN_DATA}" && -n "${VAL_DATA}" && -n "${OUTPUT_ROOT}" && -n "${GPUS}" ]] || blocked "required_argument_missing"
[[ -f "${CONFIG}" ]] || blocked "config_missing:${CONFIG}"
[[ -d "${MODEL_PATH}" ]] || blocked "local_model_directory_missing:${MODEL_PATH}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
[[ ${#GPU_IDS[@]} -eq 3 ]] || blocked "selected_gpu_count_must_equal_3"
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || blocked "gpu_id_must_be_numeric:${gpu_id}"
done
[[ "$(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l | tr -d ' ')" == "3" ]] || blocked "selected_gpu_ids_must_be_unique"
[[ "$(uname -s)" == "Linux" ]] || blocked "target_platform_not_linux"
# Target-machine setup scripts may update tracked historical report templates.
# They are runtime evidence, not source/config changes, so exclude only that
# generated subtree while keeping the rest of the repository fail-closed.
SOURCE_DIRTY="$(git status --short -- . \
  ':(exclude)artifacts/target_machine' \
  ':(exclude)artifacts/target_machine/**')"
[[ -z "${SOURCE_DIRTY}" ]] || blocked "git_working_tree_not_clean"
[[ -s "${TRAIN_DATA}" ]] || blocked "train_data_missing:${TRAIN_DATA}"
[[ -s "${VAL_DATA}" ]] || blocked "val_data_missing:${VAL_DATA}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || blocked "python_missing:${PYTHON_BIN}"
command -v nvidia-smi >/dev/null 2>&1 || blocked "nvidia_smi_missing"

mkdir -p "${OUTPUT_ROOT}/logs"
"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY' || blocked "python_version_or_disk_space_gate_failed"
from pathlib import Path
import shutil
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"python_3_11_required:found={sys.version.split()[0]}")
free = shutil.disk_usage(Path(sys.argv[1])).free
minimum = 20 * 1024**3
if free < minimum:
    raise SystemExit(f"disk_free_below_20_gib:found_bytes={free}")
print(f"python={sys.version.split()[0]} disk_free_bytes={free}")
PY

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/Latent-GRPO/verl-0.4.x:${PROJECT_ROOT}/Latent-GRPO/sglang_latent_reasoning_pkg/python${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_REPORT="${OUTPUT_ROOT}/runtime_probe.json"
ASSET_REPORT="${OUTPUT_ROOT}/asset_probe.json"
RAY_REPORT="${OUTPUT_ROOT}/ray_gpu_probe.json"
PREFLIGHT_REPORT="${OUTPUT_ROOT}/preflight.json"

"${PYTHON_BIN}" scripts/check_environment.py --mode target --require-gpus 3 --min-vram-gb 40 --output "${RUNTIME_REPORT}" \
  >>"${OUTPUT_ROOT}/logs/preflight.log" 2>&1 || blocked "cuda_bf16_nccl_or_vram_gate_failed"

"${PYTHON_BIN}" - "${RUNTIME_REPORT}" "${GPUS}" "${MAX_PREEXISTING_GPU_MEMORY_MIB}" "${MAX_PREEXISTING_GPU_UTILIZATION}" <<'PY' \
  >>"${OUTPUT_ROOT}/logs/preflight.log" 2>&1 || blocked "selected_gpu_not_idle_or_mapping_failed"
import json
from pathlib import Path
import sys

report_path, selected, max_memory_mib, max_util = sys.argv[1:]
envelope = json.loads(Path(report_path).read_text(encoding="utf-8"))
environment = envelope.get("environment_summary", {})
expected = [int(value) for value in selected.split(",")]
indices = environment.get("gpu_indices", [])
used_bytes = environment.get("gpu_memory_used_bytes", [])
utilization = environment.get("gpu_utilization_percent", [])
if indices != expected:
    raise SystemExit(f"gpu_mapping_mismatch:expected={expected}:observed={indices}")
if len(used_bytes) != 3 or len(utilization) != 3:
    raise SystemExit("selected_gpu_runtime_metrics_incomplete")
limit_bytes = int(max_memory_mib) * 1024 * 1024
for index, used, util in zip(indices, used_bytes, utilization):
    if int(used) > limit_bytes:
        raise SystemExit(f"gpu_preexisting_memory_too_high:index={index}:used_bytes={used}:limit={limit_bytes}")
    if int(util) > int(max_util):
        raise SystemExit(f"gpu_preexisting_utilization_too_high:index={index}:util={util}:limit={max_util}")
print(json.dumps({"selected_gpu_idle_gate": "PASS", "indices": indices, "used_bytes": used_bytes, "utilization": utilization}, sort_keys=True))
PY

"${PYTHON_BIN}" scripts/target_machine/import_check.py \
  >>"${OUTPUT_ROOT}/logs/preflight.log" 2>&1 || blocked "runtime_import_or_abi_gate_failed"
"${PYTHON_BIN}" -m pip check \
  >>"${OUTPUT_ROOT}/logs/preflight.log" 2>&1 || blocked "pip_dependency_consistency_gate_failed"
"${PYTHON_BIN}" train_latent_grpo.py --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" --train-files "${TRAIN_DATA}" --val-files "${VAL_DATA}" \
  --output-root "${OUTPUT_ROOT}/config-dry-run" --dry-run --validate-config \
  >>"${OUTPUT_ROOT}/logs/preflight.log" 2>&1 || blocked "config_schema_gate_failed"

"${PYTHON_BIN}" - "${CONFIG}" "${MODEL_PATH}" "${TRAIN_DATA}" "${VAL_DATA}" "${ASSET_REPORT}" <<'PY' \
  >>"${OUTPUT_ROOT}/logs/preflight.log" 2>&1 || blocked "asset_tokenizer_gate_failed"
import json
from pathlib import Path
import sys
import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer
from latent_grpo_runner.config import load_config, validate_latent_end_token

config_path, model_path, train_path, val_path, output = sys.argv[1:]
required_columns = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
data_reports = []
for path_string in (train_path, val_path):
    path = Path(path_string)
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    missing = sorted(required_columns - columns)
    if parquet.metadata.num_rows < 1 or missing:
        raise SystemExit(f"invalid_parquet_schema:{path}:rows={parquet.metadata.num_rows}:missing={missing}")
    rows = pq.read_table(path).slice(0, min(8, parquet.metadata.num_rows)).to_pylist()
    for row_index, row in enumerate(rows):
        prompt = row.get("prompt")
        reward = row.get("reward_model")
        extra = row.get("extra_info")
        if not isinstance(row.get("data_source"), str) or not row["data_source"]:
            raise SystemExit(f"invalid_data_source:{path}:row={row_index}")
        if not isinstance(prompt, list) or not prompt:
            raise SystemExit(f"invalid_prompt:{path}:row={row_index}")
        if any(not isinstance(message, dict) or not message.get("role") or not message.get("content") for message in prompt):
            raise SystemExit(f"invalid_prompt_message:{path}:row={row_index}")
        if not isinstance(reward, dict) or not reward.get("style") or reward.get("ground_truth") is None:
            raise SystemExit(f"invalid_reward_model:{path}:row={row_index}")
        if not isinstance(extra, dict) or "index" not in extra or "split" not in extra:
            raise SystemExit(f"invalid_extra_info:{path}:row={row_index}")
    data_reports.append({"path": str(path.resolve()), "rows": parquet.metadata.num_rows, "columns": sorted(columns)})

model_dir = Path(model_path)
weight_markers = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin")) + list(model_dir.glob("*.index.json"))
if not weight_markers:
    raise SystemExit(f"model_weight_files_missing:{model_dir}")
config = load_config(config_path, workspace_root=Path.cwd())
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
token = validate_latent_end_token(config.model, tokenizer, model_config)
payload = {
    "status": "PASS",
    "profile_name": config.profile_name,
    "model_path": str(model_dir.resolve()),
    "model_weight_file_count": len(weight_markers),
    "data": data_reports,
    "latent_end": token,
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

"${PYTHON_BIN}" scripts/probe_ray_distributed.py --num-gpus 3 --output "${RAY_REPORT}" \
  >>"${OUTPUT_ROOT}/logs/preflight.log" 2>&1 || blocked "ray_three_worker_binding_gate_failed"

"${PYTHON_BIN}" - "${RUNTIME_REPORT}" "${ASSET_REPORT}" "${RAY_REPORT}" "${PREFLIGHT_REPORT}" <<'PY'
import json
from pathlib import Path
import sys
runtime, asset, ray, output = map(Path, sys.argv[1:])
r = json.loads(runtime.read_text())
a = json.loads(asset.read_text())
y = json.loads(ray.read_text())
environment = r.get("environment_summary", {})
payload = {
    "status": "PASS",
    "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "gpu_count": 3,
    "selected_gpus": environment.get("visible_devices"),
    "cuda_device_order": __import__("os").environ.get("CUDA_DEVICE_ORDER"),
    "gpu_indices": environment.get("gpu_indices"),
    "gpu_names": environment.get("gpu_names"),
    "gpu_total_memory_bytes": environment.get("gpu_total_memory_bytes"),
    "gpu_memory_used_bytes_at_preflight": environment.get("gpu_memory_used_bytes"),
    "gpu_utilization_percent_at_preflight": environment.get("gpu_utilization_percent"),
    "gpu_compute_capabilities": environment.get("gpu_compute_capabilities"),
    "cuda_runtime_version": environment.get("cuda_runtime_version"),
    "nccl_version": environment.get("nccl_version"),
    "bf16_supported": environment.get("bf16_supported"),
    "asset_gate": a.get("status"),
    "ray_gate": y.get("status"),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "GPU_COUNT: PASS (3)"
echo "GPU_MAPPING: PASS (${GPUS})"
echo "GPU_IDLE: PASS"
echo "PYTHON_VERSION: PASS (3.11)"
echo "DISK_FREE: PASS (>=20 GiB)"
echo "GPU_VRAM: PASS"
echo "BF16: PASS"
echo "NCCL: PASS"
echo "DEPENDENCY_CONSISTENCY: PASS"
echo "MODEL_ASSET: PASS"
echo "DATA_ASSET: PASS"
echo "LATENT_END_TOKEN: PASS"
echo "3GPU_PREFLIGHT_GATE: PASS"
