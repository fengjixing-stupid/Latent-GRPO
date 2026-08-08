#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

SINGLE_RUN="${TARGET_RUN_DIR}/single_gpu"
"${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/single_gpu_output_validation.json" \
  --stdout-log "${TARGET_LOG_DIR}/07_validate_single_gpu_output.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/07_validate_single_gpu_output.stderr.log" \
  --success-status single_gpu_tested \
  --artifact "${SINGLE_RUN}" \
  -- "${TARGET_PYTHON}" scripts/validate_outputs.py --input "${SINGLE_RUN}"

