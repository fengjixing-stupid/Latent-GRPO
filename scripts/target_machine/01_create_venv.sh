#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

"${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/venv_creation.json" \
  --stdout-log "${TARGET_LOG_DIR}/01_create_venv.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/01_create_venv.stderr.log" \
  --success-status target_machine_probe_passed \
  --artifact "${TARGET_VENV}" \
  -- "${SYSTEM_PYTHON}" scripts/target_machine/create_venv.py --path "${TARGET_VENV}"

