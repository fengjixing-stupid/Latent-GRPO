#!/usr/bin/env bash
set -euo pipefail

# Kaggle/T4-only runtime stack. It intentionally does not touch training data.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "BLOCKED: Kaggle T4 runtime stack requires Linux" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "BLOCKED: nvidia-smi is unavailable" >&2
  exit 3
fi
mapfile -t T4_GPU_ROWS < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
if [[ "${#T4_GPU_ROWS[@]}" -ne 2 ]]; then
  echo "BLOCKED: expected exactly two GPUs, found ${#T4_GPU_ROWS[@]}" >&2
  exit 4
fi
for row in "${T4_GPU_ROWS[@]}"; do
  if [[ "${row}" != *"T4"* || "${row}" != *"7.5"* ]]; then
    echo "BLOCKED: expected dual T4 compute capability 7.5, got: ${row}" >&2
    exit 5
  fi
done

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu118
"${PYTHON_BIN}" -m pip install cuda-bindings==11.8.6 cuda-python==11.8.6
"${PYTHON_BIN}" -m pip install \
  flashinfer-python==0.2.5 \
  --index-url https://flashinfer.ai/whl/cu118/torch2.6
"${PYTHON_BIN}" -m pip install \
  sgl-kernel==0.1.1 \
  --index-url https://docs.sglang.ai/whl/cu118

"${PYTHON_BIN}" -m pip install \
  -r "${ROOT}/requirements/runtime-core.txt" \
  -r "${ROOT}/requirements/metrics.txt" \
  -r "${ROOT}/requirements/reward-math.txt"

TMP_REQ="$(mktemp)"
trap 'rm -f "${TMP_REQ}"' EXIT
grep -Ev '^(flashinfer-python|sgl-kernel|cuda-python|cuda-bindings)([<=> ]|$)' \
  "${ROOT}/requirements/runtime-sglang.txt" > "${TMP_REQ}"
"${PYTHON_BIN}" -m pip install -r "${TMP_REQ}"

# Install the repository forks without allowing pip to replace the guarded T4 wheels.
"${PYTHON_BIN}" -m pip install -e \
  "${ROOT}/Latent-GRPO/sglang_latent_reasoning_pkg/python" --no-deps
"${PYTHON_BIN}" -m pip install -e \
  "${ROOT}/Latent-GRPO/verl-0.4.x" --no-deps

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("bf16_supported", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
for package in ("sglang", "sgl-kernel", "flashinfer-python", "cuda-python", "cuda-bindings", "ray", "pyarrow"):
    print(package, md.version(package))
PY

echo "T4_RUNTIME_STACK_INSTALLED: no training data read or generated"
