#!/usr/bin/env bash
set -u

TARGET_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TARGET_SCRIPT_DIR}/../.." && pwd)"
TARGET_REPORT_DIR="${TARGET_REPORT_DIR:-${PROJECT_ROOT}/artifacts/target_machine}"
TARGET_LOG_DIR="${TARGET_REPORT_DIR}/logs"
TARGET_RUN_DIR="${TARGET_REPORT_DIR}/runs"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3}"
TARGET_VENV="${TARGET_VENV:-${PROJECT_ROOT}/.venv-target}"
TARGET_PYTHON="${TARGET_PYTHON:-${TARGET_VENV}/bin/python}"
REPORT_RUNNER="${PROJECT_ROOT}/scripts/target_machine/run_reported.py"

mkdir -p "${TARGET_REPORT_DIR}" "${TARGET_LOG_DIR}" "${TARGET_RUN_DIR}"
cd "${PROJECT_ROOT}"

