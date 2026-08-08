#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KAGGLE_T4_PYTHON:-/kaggle/working/latent-t4-cu124/bin/python}"

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "usage: $0 MODEL_PATH TRAIN_PARQUET VAL_PARQUET [OUTPUT_ROOT]" >&2
  exit 2
fi
MODEL_PATH="$1"
TRAIN_FILE="$2"
VAL_FILE="$3"
OUTPUT_ROOT="${4:-${ROOT}/artifacts/runs/kaggle-t4-monitor}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "BLOCKED: validated Kaggle T4 Python not found: ${PYTHON_BIN}" >&2
  exit 3
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "BLOCKED: train parquet not found: ${TRAIN_FILE}" >&2
  exit 4
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "BLOCKED: val/test parquet not found: ${VAL_FILE}" >&2
  exit 5
fi
if [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == .* ]]; then
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "BLOCKED: local author SFT model directory not found: ${MODEL_PATH}" >&2
    exit 6
  fi
  if ! find "${MODEL_PATH}" -maxdepth 1 -type f \
      \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .; then
    echo "BLOCKED: local model directory has no model weight files: ${MODEL_PATH}" >&2
    exit 6
  fi
fi
if [[ -e "${OUTPUT_ROOT}" ]] && find "${OUTPUT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
  echo "BLOCKED: output root is not empty; choose a fresh path: ${OUTPUT_ROOT}" >&2
  exit 7
fi

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
if [[ "${#GPU_ROWS[@]}" -ne 2 ]]; then
  echo "BLOCKED: expected exactly two GPUs" >&2
  exit 8
fi
for row in "${GPU_ROWS[@]}"; do
  if [[ "${row}" != *"T4"* || "${row}" != *"7.5"* ]]; then
    echo "BLOCKED: expected dual T4 compute capability 7.5, got: ${row}" >&2
    exit 9
  fi
done

cd "${ROOT}"
"${PYTHON_BIN}" train_latent_grpo.py \
  --config configs/kaggle-t4-monitor.yaml \
  --profile-name kaggle-t4-monitor \
  --model-path "${MODEL_PATH}" \
  --train-files "${TRAIN_FILE}" \
  --val-files "${VAL_FILE}" \
  --output-root "${OUTPUT_ROOT}"

"${PYTHON_BIN}" tools/validate_kaggle_t4_monitor_probe.py "${OUTPUT_ROOT}"
echo "KAGGLE_T4_REAL_MONITOR_PROBE: PASS"
