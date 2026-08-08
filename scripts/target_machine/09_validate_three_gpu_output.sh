#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

THREE_RUN="${TARGET_RUN_DIR}/three_gpu"
"${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/three_gpu_output_validation.json" \
  --stdout-log "${TARGET_LOG_DIR}/09_validate_three_gpu_output.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/09_validate_three_gpu_output.stderr.log" \
  --success-status three_gpu_ray_tested \
  --artifact "${THREE_RUN}" \
  -- "${TARGET_PYTHON}" scripts/validate_outputs.py --input "${THREE_RUN}"

