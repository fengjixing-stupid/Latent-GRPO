#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DETAIL_REPORT="${TARGET_REPORT_DIR}/ray_gpu_probe_details.json"
CUDA_VISIBLE_DEVICES=0,1,2 "${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/ray_gpu_probe.json" \
  --stdout-log "${TARGET_LOG_DIR}/05_probe_ray_gpus.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/05_probe_ray_gpus.stderr.log" \
  --success-status target_machine_probe_passed \
  --environment-summary-json "${DETAIL_REPORT}" \
  --artifact "${DETAIL_REPORT}" \
  -- "${TARGET_PYTHON}" scripts/probe_ray_distributed.py --num-gpus 3 --output "${DETAIL_REPORT}"

