#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

"${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/pytorch_install.json" \
  --stdout-log "${TARGET_LOG_DIR}/02_install_pytorch.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/02_install_pytorch.stderr.log" \
  --success-status target_machine_probe_passed \
  -- "${TARGET_PYTHON}" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0

