#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

"${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/import_check.json" \
  --stdout-log "${TARGET_LOG_DIR}/04_import_check.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/04_import_check.stderr.log" \
  --success-status cuda_runtime_verified \
  -- "${TARGET_PYTHON}" scripts/target_machine/import_check.py
IMPORT_EXIT=$?

if [ "${IMPORT_EXIT}" -eq 0 ]; then
  "${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
    --report "${TARGET_REPORT_DIR}/requirements_validation.json" \
    --stdout-log "${TARGET_LOG_DIR}/04_pip_check.stdout.log" \
    --stderr-log "${TARGET_LOG_DIR}/04_pip_check.stderr.log" \
    --success-status requirements_lock_verified \
    -- "${TARGET_PYTHON}" -m pip check
  REQUIREMENTS_EXIT=$?
else
  "${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
    --report "${TARGET_REPORT_DIR}/requirements_validation.json" \
    --stdout-log "${TARGET_LOG_DIR}/04_pip_check.stdout.log" \
    --stderr-log "${TARGET_LOG_DIR}/04_pip_check.stderr.log" \
    --success-status requirements_lock_verified \
    -- "${SYSTEM_PYTHON}" -c "import sys; print('import/ABI gate failed; pip check not promoted to lock verification', file=sys.stderr); sys.exit(1)"
  REQUIREMENTS_EXIT=$?
fi

if [ "${IMPORT_EXIT}" -ne 0 ]; then
  exit "${IMPORT_EXIT}"
fi
exit "${REQUIREMENTS_EXIT}"

