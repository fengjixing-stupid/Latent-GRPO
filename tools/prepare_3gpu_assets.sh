#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCKED_REASON=""
MODEL_SOURCE=""
MODEL_PATH=""
TRAIN_SOURCE=""
TRAIN_DATA=""
VAL_SOURCE=""
VAL_DATA=""

blocked() {
  BLOCKED_REASON="$1"
  echo "ASSET_GATE: BLOCKED"
  echo "BLOCKED_REASON: ${BLOCKED_REASON}"
  echo "LOG_PATH: ${PROJECT_ROOT}/artifacts/validation/3gpu-final/logs/preflight.log"
  echo "NEXT_ACTION: provide the missing local asset or an explicit source"
  exit 2
}

usage() {
  echo "Usage: $0 --model-source <HF_ID|local_dir> --model-path <dir> --train-data <file> [--train-source <local|https>] --val-data <file> [--val-source <local|https>]"
}

while (($#)); do
  case "$1" in
    --model-source) MODEL_SOURCE="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --train-source) TRAIN_SOURCE="$2"; shift 2 ;;
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --val-source) VAL_SOURCE="$2"; shift 2 ;;
    --val-data) VAL_DATA="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) blocked "unknown_argument:$1" ;;
  esac
done

[[ -n "${MODEL_PATH}" && -n "${TRAIN_DATA}" && -n "${VAL_DATA}" ]] || blocked "required_asset_path_missing"

if [[ ! -d "${MODEL_PATH}" ]]; then
  [[ -n "${MODEL_SOURCE}" ]] || blocked "model_missing:${MODEL_PATH}"
  mkdir -p "$(dirname "${MODEL_PATH}")"
  if [[ -d "${MODEL_SOURCE}" ]]; then
    mkdir -p "${MODEL_PATH}"
    cp -R -n "${MODEL_SOURCE}/." "${MODEL_PATH}/"
  else
    command -v hf >/dev/null 2>&1 || blocked "hf_cli_missing_for_model_download"
    hf download "${MODEL_SOURCE}" --local-dir "${MODEL_PATH}" || blocked "model_download_failed:${MODEL_SOURCE}"
  fi
fi

prepare_file() {
  local label="$1"
  local destination="$2"
  local source="$3"
  [[ -s "${destination}" ]] && return 0
  [[ -n "${source}" ]] || blocked "${label}_missing:${destination}"
  mkdir -p "$(dirname "${destination}")"
  if [[ -s "${source}" ]]; then
    cp -n "${source}" "${destination}"
  elif [[ "${source}" == http://* || "${source}" == https://* ]]; then
    command -v curl >/dev/null 2>&1 || blocked "curl_missing_for_${label}_download"
    curl --fail --location --output "${destination}.partial" "${source}" || blocked "${label}_download_failed"
    mv "${destination}.partial" "${destination}"
  else
    blocked "unsupported_${label}_source:${source}"
  fi
  [[ -s "${destination}" ]] || blocked "${label}_integrity_failed:${destination}"
}

prepare_file "train_data" "${TRAIN_DATA}" "${TRAIN_SOURCE}"
prepare_file "val_data" "${VAL_DATA}" "${VAL_SOURCE}"
[[ -n "$(find "${MODEL_PATH}" -mindepth 1 -maxdepth 2 -type f -print -quit)" ]] || blocked "model_directory_empty:${MODEL_PATH}"

echo "MODEL_PATH: $(cd "${MODEL_PATH}" && pwd)"
echo "TRAIN_DATA: $(cd "$(dirname "${TRAIN_DATA}")" && pwd)/$(basename "${TRAIN_DATA}")"
echo "VAL_DATA: $(cd "$(dirname "${VAL_DATA}")" && pwd)/$(basename "${VAL_DATA}")"
echo "ASSET_GATE: PASS"
