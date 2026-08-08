#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DETAIL_REPORT="${TARGET_REPORT_DIR}/runtime_probe_details.json"
"${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/runtime_probe.json" \
  --stdout-log "${TARGET_LOG_DIR}/00_probe_environment.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/00_probe_environment.stderr.log" \
  --success-status target_machine_probe_passed \
  --environment-summary-json "${DETAIL_REPORT}" \
  --artifact "${DETAIL_REPORT}" \
  -- "${SYSTEM_PYTHON}" scripts/probe_target_machine.py \
  --require-gpus 3 --min-vram-gb 40 --output "${DETAIL_REPORT}"

