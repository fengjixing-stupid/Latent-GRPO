#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

RESUME_RUN="${TARGET_RUN_DIR}/resume_gpu"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
if [ -z "${RESUME_CHECKPOINT}" ]; then
  "${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
    --report "${TARGET_REPORT_DIR}/resume_smoke.json" \
    --stdout-log "${TARGET_LOG_DIR}/10_resume_smoke.stdout.log" \
    --stderr-log "${TARGET_LOG_DIR}/10_resume_smoke.stderr.log" \
    --success-status single_gpu_tested \
    -- "${SYSTEM_PYTHON}" -c "import sys; print('Set RESUME_CHECKPOINT to a validated global_step_<N> checkpoint with N < 2', file=sys.stderr); sys.exit(2)"
  exit $?
fi

CUDA_VISIBLE_DEVICES=0 "${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/resume_smoke.json" \
  --stdout-log "${TARGET_LOG_DIR}/10_resume_smoke.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/10_resume_smoke.stderr.log" \
  --success-status single_gpu_tested \
  --artifact "${RESUME_RUN}" \
  --artifact "${RESUME_CHECKPOINT}" \
  -- "${TARGET_PYTHON}" train_latent_grpo.py --config configs/smoke.yaml \
  --output-root "${RESUME_RUN}" --resume-from "${RESUME_CHECKPOINT}" --max-steps 2

