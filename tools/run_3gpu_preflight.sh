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
    -h|--help) echo "Usage: $0 [--config configs/3gpu-final-validation.yaml] --model-path LOCAL_MODEL_DIR --train-data LOCAL_FILE --val-data LOCAL_FILE --output-root DIR --gpus 0,1,2"; exit 0 ;;
    *) blocked "unknown_argument:$1" ;;
  esac
done

[[ -n "${MODEL_PATH}" && -n "${TRAIN_DATA}" && -n "${VAL_DATA}" && -n "${OUTPUT_ROOT}" && -n "${GPUS}" ]] || blocked "required_argument_missing"
[[ -f "${CONFIG}" ]] || blocked "config_missing:${CONFIG}"
[[ -d "${MODEL_PATH}" ]] || blocked "local_model_directory_missing:${MODEL_PATH}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
[[ ${#GPU_IDS[@]} -eq 3 ]] || blocked "selected_gpu_count_must_equal_3"
[[ "$(uname -s)" == "Linux" ]] || blocked "target_platform_not_linux"
[[ -z "$(git status --short)" ]] || blocked "git_working_tree_not_clean"
[[ -s "${TRAIN_DATA}" ]] || blocked "train_data_missing:${TRAIN_DATA}"
[[ -s "${VAL_DATA}" ]] || blocked "val_data_missing:${VAL_DATA}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || blocked "python_missing:${PYTHON_BIN}"

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
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/Latent-GRPO/verl-0.4.x:${PROJECT_ROOT}/Latent-GRPO/sglang_latent_reasoning_pkg/python${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_REPORT="${OUTPUT_ROOT}/runtime_probe.json"
ASSET_REPORT="${OUTPUT_ROOT}/asset_probe.json"
RAY_REPORT="${OUTPUT_ROOT}/ray_gpu_probe.json"
PREFLIGHT_REPORT="${OUTPUT_ROOT}/preflight.json"

"${PYTHON_BIN}" scripts/check_environment.py --mode target --require-gpus 3 --min-vram-gb 40 --output "${RUNTIME_REPORT}" \
  >>"${OUTPUT_ROOT}/logs/preflight.log" 2>&1 || blocked "cuda_bf16_nccl_or_vram_gate_failed"
"${PYTHON_BIN}" scripts/target_machine/import_check.py \
  >>"${OUTPUT_ROOT}/logs/preflight.log" 2>&1 || blocked "runtime_import_or_abi_gate_failed"
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
for path in (train_path, val_path):
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows < 1 or not parquet.schema.names:
        raise SystemExit(f"invalid_or_empty_parquet:{path}")
config = load_config(config_path, workspace_root=Path.cwd())
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_config = AutoConfig.from_pretrained(model_path)
token = validate_latent_end_token(config.model, tokenizer, model_config)
payload = {"status": "PASS", "profile_name": config.profile_name, "model_path": model_path, "train_data": train_path, "val_data": val_path, "latent_end": token}
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
    "gpu_names": environment.get("gpu_names"),
    "gpu_total_memory_bytes": environment.get("gpu_total_memory_bytes"),
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
echo "PYTHON_VERSION: PASS (3.11)"
echo "DISK_FREE: PASS (>=20 GiB)"
echo "GPU_VRAM: PASS"
echo "BF16: PASS"
echo "NCCL: PASS"
echo "MODEL_ASSET: PASS"
echo "DATA_ASSET: PASS"
echo "LATENT_END_TOKEN: PASS"
echo "3GPU_PREFLIGHT_GATE: PASS"
