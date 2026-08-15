#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
BLOCKED_REASON=""
CONFIG="configs/3gpu-final-low.yaml"
MODEL_PATH=""
TRAIN_DATA=""
VAL_DATA=""
OUTPUT_ROOT=""
GPUS=""
SEED=""
ACCEPTANCE_REPORT="${PROJECT_ROOT}/artifacts/validation/3gpu-final/acceptance.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ORIGINAL_COMMAND="$0 $*"

blocked() {
  BLOCKED_REASON="$1"
  echo "3GPU_TRAINING_GATE: BLOCKED"
  echo "BLOCKED_REASON: ${BLOCKED_REASON}"
  echo "LOG_PATH: ${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/runs}/logs/training.log"
  echo "NEXT_ACTION: satisfy the named prerequisite and retry"
  exit 2
}

while (($#)); do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --val-data) VAL_DATA="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --acceptance-report) ACCEPTANCE_REPORT="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 --config configs/3gpu-final-low.yaml --model-path LOCAL_MODEL_DIR --train-data LOCAL_FILE --val-data LOCAL_FILE --output-root DIR --gpus 0,1,2 --seed 17 [--acceptance-report FILE]"; exit 0 ;;
    *) blocked "unknown_argument:$1" ;;
  esac
done

[[ -n "${MODEL_PATH}" && -n "${TRAIN_DATA}" && -n "${VAL_DATA}" && -n "${OUTPUT_ROOT}" && -n "${GPUS}" && -n "${SEED}" ]] || blocked "required_argument_missing"
[[ -f "${CONFIG}" ]] || blocked "config_missing:${CONFIG}"
[[ -d "${MODEL_PATH}" ]] || blocked "local_model_directory_missing:${MODEL_PATH}"
[[ -s "${TRAIN_DATA}" ]] || blocked "train_data_missing:${TRAIN_DATA}"
[[ -s "${VAL_DATA}" ]] || blocked "val_data_missing:${VAL_DATA}"
[[ ! -e "${OUTPUT_ROOT}" ]] || blocked "training_output_already_exists:${OUTPUT_ROOT}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
[[ ${#GPU_IDS[@]} -eq 3 ]] || blocked "selected_gpu_count_must_equal_3"
"${PYTHON_BIN}" - "${CONFIG}" "${ACCEPTANCE_REPORT}" <<'PY' || blocked "final_validation_acceptance_missing_blocked_or_profile_mismatch"
import json
from pathlib import Path
import sys
import yaml

config_path, report_path = map(Path, sys.argv[1:])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
required_validation_profile = {
    "3gpu-final-low": "3gpu-final-validation",
    "3gpu-final-high": "3gpu-final-high-validation",
}.get(config.get("profile_name"))
path = report_path
payload = json.loads(path.read_text()) if path.is_file() else {}
accepted = (
    config.get("profile_kind") == "formal_training"
    and required_validation_profile is not None
    and payload.get("final_gate") == "PASS"
    and payload.get("profile_name") == required_validation_profile
)
raise SystemExit(0 if accepted else 1)
PY

mkdir -p "${OUTPUT_ROOT}/logs"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/Latent-GRPO/verl-0.4.x:${PROJECT_ROOT}/Latent-GRPO/sglang_latent_reasoning_pkg/python${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" train_latent_grpo.py --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" --train-files "${TRAIN_DATA}" --val-files "${VAL_DATA}" \
  --output-root "${OUTPUT_ROOT}" --seed "${SEED}" --dry-run --validate-config \
  >"${OUTPUT_ROOT}/logs/config-dry-run.log" 2>&1 || blocked "formal_training_config_gate_failed"

"${PYTHON_BIN}" - "${CONFIG}" "${MODEL_PATH}" "${TRAIN_DATA}" "${VAL_DATA}" "${OUTPUT_ROOT}" "${SEED}" "${GPUS}" "${ORIGINAL_COMMAND}" <<'PY'
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
from latent_grpo_runner.config import load_config

config_path, model, train, val, output, seed, gpus, command = sys.argv[1:]
root = Path.cwd()
output_path = Path(output).resolve()
config = load_config(config_path, workspace_root=root).with_runtime_overrides(
    model_path=model, train_files=train, val_files=val, output_root=output_path, seed=int(seed)
)
def identity(value: str) -> dict[str, object]:
    path = Path(value).expanduser()
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"kind": "file", "path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}
    if path.is_dir():
        marker = path / "config.json"
        digest = hashlib.sha256(marker.read_bytes()).hexdigest() if marker.is_file() else None
        return {"kind": "directory", "path": str(path.resolve()), "config_sha256": digest}
    return {"kind": "hugging_face_id", "id": value}
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
resolved = {
    "profile_name": config.profile_name,
    "profile_kind": config.profile_kind,
    "seed": config.training.seed,
    "launcher": asdict(config.launcher),
    "target_hardware": asdict(config.hardware),
    "batch": asdict(config.batch),
    "model": asdict(config.model),
    "data": asdict(config.data),
    "rollout": asdict(config.rollout),
    "training": asdict(config.training),
    "features": asdict(config.features),
    "paths": {key: str(value) for key, value in config.paths.items()},
    "upstream_overrides": dict(config.upstream_overrides),
    "hydra_overrides": list(config.author_hydra_overrides()),
    "config_hash": config.resume_compatibility_hash,
}
(output_path / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
manifest = {
    "profile_name": config.profile_name,
    "seed": config.training.seed,
    "git_commit": commit,
    "model_path": model,
    "model_identity": identity(model),
    "train_data_identity": identity(train),
    "val_data_identity": identity(val),
    "resolved_config": "resolved_config.yaml",
    "three_gpu_deviation_list": "docs/3GPU_HYPERPARAMETER_DEVIATIONS.md",
    "launcher": "ray_direct",
    "selected_gpus": gpus,
    "start_time": datetime.now(timezone.utc).isoformat(),
    "launch_command": command,
}
(output_path / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"${PYTHON_BIN}" train_latent_grpo.py --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" --train-files "${TRAIN_DATA}" --val-files "${VAL_DATA}" \
  --output-root "${OUTPUT_ROOT}" --seed "${SEED}" \
  >"${OUTPUT_ROOT}/logs/training.log" 2>&1 || blocked "formal_training_runtime_failed"
echo "3GPU_TRAINING_GATE: PASS"
