#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

SINGLE_RUN="${TARGET_RUN_DIR}/single_gpu"
MEMORY_REPORT="${TARGET_REPORT_DIR}/single_gpu_memory.json"
CUDA_VISIBLE_DEVICES=0 "${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/single_gpu_smoke.json" \
  --stdout-log "${TARGET_LOG_DIR}/06_single_gpu_smoke.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/06_single_gpu_smoke.stderr.log" \
  --success-status single_gpu_tested \
  --artifact "${SINGLE_RUN}" \
  --artifact "${MEMORY_REPORT}" \
  -- "${TARGET_PYTHON}" scripts/target_machine/run_with_gpu_telemetry.py --output "${MEMORY_REPORT}" -- \
  "${TARGET_PYTHON}" train_latent_grpo.py --config configs/smoke.yaml \
  --output-root "${SINGLE_RUN}" --max-steps 2
