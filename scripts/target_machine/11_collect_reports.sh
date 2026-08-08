#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MANIFEST_PYTHON="${TARGET_PYTHON}"
if [ ! -x "${MANIFEST_PYTHON}" ]; then
  MANIFEST_PYTHON="${SYSTEM_PYTHON}"
fi
"${MANIFEST_PYTHON}" scripts/target_machine/build_report_manifest.py \
  >"${TARGET_LOG_DIR}/11_collect_reports.stdout.log" \
  2>"${TARGET_LOG_DIR}/11_collect_reports.stderr.log"
exit $?

