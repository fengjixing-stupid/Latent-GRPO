#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

THREE_RUN="${TARGET_RUN_DIR}/three_gpu"
MEMORY_REPORT="${TARGET_REPORT_DIR}/three_gpu_memory.json"
# ray_direct is intentional: one Python driver asks Ray for three GPU workers.
CUDA_VISIBLE_DEVICES=0,1,2 "${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/three_gpu_smoke.json" \
  --stdout-log "${TARGET_LOG_DIR}/08_three_gpu_smoke.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/08_three_gpu_smoke.stderr.log" \
  --success-status three_gpu_ray_tested \
  --artifact "${THREE_RUN}" \
  --artifact "${MEMORY_REPORT}" \
  -- "${TARGET_PYTHON}" scripts/target_machine/run_with_gpu_telemetry.py --output "${MEMORY_REPORT}" -- \
  "${TARGET_PYTHON}" train_latent_grpo.py --config configs/3gpu-low.yaml \
  --output-root "${THREE_RUN}" --max-steps 2
