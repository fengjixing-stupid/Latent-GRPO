#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# SGL_KERNEL_VERSION is deliberately mandatory: vendored metadata says 0.1.0
# while the author README says 0.1.1.  The target operator resolves that conflict.
"${SYSTEM_PYTHON}" "${REPORT_RUNNER}" \
  --report "${TARGET_REPORT_DIR}/runtime_install.json" \
  --stdout-log "${TARGET_LOG_DIR}/03_install_runtime.stdout.log" \
  --stderr-log "${TARGET_LOG_DIR}/03_install_runtime.stderr.log" \
  --success-status target_machine_probe_passed \
  --artifact "${PROJECT_ROOT}/Latent-GRPO/verl-0.4.x" \
  --artifact "${PROJECT_ROOT}/Latent-GRPO/sglang_latent_reasoning_pkg" \
  -- "${TARGET_PYTHON}" scripts/target_machine/install_runtime.py

