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
SEED=17
PYTHON_BIN="${PYTHON_BIN:-python3}"
# tools/validate_3gpu_final.py emits CORE_METRICS, CUDA_RNG_ALL_DEVICES,
# CHECKPOINT_GATE, and 3GPU_FINAL_GATE after reading the durable artifacts.

blocked() {
  BLOCKED_REASON="$1"
  local log_path="${2:-${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/validation/3gpu-final}/logs/runtime.log}"
  echo "3GPU_FINAL_GATE: BLOCKED"
  echo "BLOCKED_REASON: ${BLOCKED_REASON}"
  echo "LOG_PATH: ${log_path}"
  echo "NEXT_ACTION: inspect the log, fix this gate, then rerun validation"
  exit 3
}

while (($#)); do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --val-data) VAL_DATA="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--config configs/3gpu-final-validation.yaml] --model-path LOCAL_MODEL_DIR --train-data LOCAL_FILE --val-data LOCAL_FILE --output-root DIR --gpus 0,1,2 [--seed 17]"; exit 0 ;;
    *) blocked "unknown_argument:$1" ;;
  esac
done

[[ -n "${MODEL_PATH}" && -n "${TRAIN_DATA}" && -n "${VAL_DATA}" && -n "${OUTPUT_ROOT}" && -n "${GPUS}" ]] || blocked "required_argument_missing"
RUN_ROOT="${OUTPUT_ROOT}/run"
LOG_ROOT="${OUTPUT_ROOT}/logs"
[[ ! -e "${RUN_ROOT}" && ! -e "${OUTPUT_ROOT}/acceptance.json" ]] || blocked "validation_output_already_exists:${OUTPUT_ROOT}"
mkdir -p "${LOG_ROOT}"

bash tools/run_3gpu_preflight.sh --model-path "${MODEL_PATH}" --train-data "${TRAIN_DATA}" \
  --config "${CONFIG}" --val-data "${VAL_DATA}" --output-root "${OUTPUT_ROOT}" --gpus "${GPUS}" \
  >"${LOG_ROOT}/preflight.log" 2>&1 || blocked "preflight_failed" "${LOG_ROOT}/preflight.log"
echo "3GPU_PREFLIGHT_GATE: PASS"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/Latent-GRPO/verl-0.4.x:${PROJECT_ROOT}/Latent-GRPO/sglang_latent_reasoning_pkg/python${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" scripts/target_machine/run_with_gpu_telemetry.py --output "${OUTPUT_ROOT}/gpu_telemetry.json" -- \
  "${PYTHON_BIN}" train_latent_grpo.py --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" --train-files "${TRAIN_DATA}" \
  --val-files "${VAL_DATA}" --output-root "${RUN_ROOT}" --seed "${SEED}" \
  >"${LOG_ROOT}/runtime.log" 2>&1 || blocked "distributed_training_or_optimizer_update_failed" "${LOG_ROOT}/runtime.log"
echo "3GPU_DISTRIBUTED_RUNTIME_GATE: PASS"

"${PYTHON_BIN}" scripts/validate_outputs.py --input "${RUN_ROOT}" \
  >"${LOG_ROOT}/metrics.log" 2>&1 || blocked "output_schema_or_primary_key_gate_failed" "${LOG_ROOT}/metrics.log"

LATEST_STEP="$(tr -d '[:space:]' < "${RUN_ROOT}/latest_checkpointed_iteration.txt")"
CHECKPOINT_PATH="${RUN_ROOT}/global_step_${LATEST_STEP}"
"${PYTHON_BIN}" train_latent_grpo.py --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" --train-files "${TRAIN_DATA}" \
  --val-files "${VAL_DATA}" --output-root "${RUN_ROOT}" --seed "${SEED}" \
  --resume-from "${CHECKPOINT_PATH}" \
  >"${LOG_ROOT}/checkpoint.log" 2>&1 || blocked "checkpoint_resume_gate_failed" "${LOG_ROOT}/checkpoint.log"
"${PYTHON_BIN}" - "${OUTPUT_ROOT}/resume_gate.json" "${CHECKPOINT_PATH}" <<'PY'
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps({"status": "PASS", "checkpoint": sys.argv[2]}, indent=2) + "\n")
PY

"${PYTHON_BIN}" tools/validate_3gpu_final.py --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" --train-data "${TRAIN_DATA}" --val-data "${VAL_DATA}" \
  --run-root "${RUN_ROOT}" --output-root "${OUTPUT_ROOT}" --seed "${SEED}" \
  --preflight-report "${OUTPUT_ROOT}/preflight.json" --ray-report "${OUTPUT_ROOT}/ray_gpu_probe.json" \
  --telemetry-report "${OUTPUT_ROOT}/gpu_telemetry.json" --resume-report "${OUTPUT_ROOT}/resume_gate.json" \
  >"${LOG_ROOT}/probe.log" 2>&1 || {
    tail -n 20 "${LOG_ROOT}/probe.log"
    blocked "metrics_probe_rng_or_checkpoint_acceptance_failed" "${LOG_ROOT}/probe.log"
  }
cat "${LOG_ROOT}/probe.log"
