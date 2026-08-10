#!/usr/bin/env bash
set -euo pipefail

# Kaggle/T4-only runtime stack. It intentionally does not touch training data.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-python}"
VENV_DIR="${KAGGLE_T4_VENV:-/kaggle/working/latent-t4-cu124}"

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

# Build an isolated Python 3.10 runtime so Kaggle's system Python/Torch cannot
# leak into the validated SGLang 0.4.6-era stack.
"${BOOTSTRAP_PYTHON}" -m pip install -q uv
UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" ]]; then
  echo "BLOCKED: uv was installed but its executable is not on PATH" >&2
  exit 6
fi
"${UV_BIN}" python install 3.10
rm -rf "${VENV_DIR}"
"${UV_BIN}" venv --python 3.10 --seed "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
"${PYTHON_BIN}" -m pip install --no-deps --no-cache-dir --only-binary=:all: sgl-kernel==0.1.1
"${PYTHON_BIN}" -m pip install --no-deps --no-cache-dir \
  flashinfer-python==0.2.5 \
  --index-url https://flashinfer.ai/whl/cu124/torch2.6
"${PYTHON_BIN}" -m pip install cuda-python==12.9.0 cuda-bindings==12.9.0
"${PYTHON_BIN}" -m pip install --no-deps torchao==0.10.0
"${PYTHON_BIN}" -m pip install --no-deps --no-cache-dir --only-binary=:all: \
  torch-memory-saver==0.0.8

TMP_REQ="$(mktemp)"
TMP_CONSTRAINTS="$(mktemp)"
trap 'rm -f "${TMP_REQ}" "${TMP_CONSTRAINTS}"' EXIT
cat > "${TMP_CONSTRAINTS}" <<'CONSTRAINTS'
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
triton==3.2.0
transformers==4.51.1
sgl-kernel==0.1.1
compressed-tensors==0.9.3
flashinfer-python==0.2.5
cuda-python==12.9.0
cuda-bindings==12.9.0
torchao==0.10.0
cachetools==5.5.2
openai==1.109.1
tiktoken==0.13.0
torch-memory-saver==0.0.8
CONSTRAINTS

# reward-math is intentionally excluded: this gate is data-free and only
# establishes the execution/runtime compatibility boundary.
"${PYTHON_BIN}" -m pip install -c "${TMP_CONSTRAINTS}" \
  -r "${ROOT}/requirements/runtime-core.txt" \
  -r "${ROOT}/requirements/metrics.txt"

grep -Ev '^(flashinfer-python|sgl-kernel|torch-memory-saver|cuda-python|cuda-bindings|torchao)([<=> ]|$)' \
  "${ROOT}/requirements/runtime-sglang.txt" > "${TMP_REQ}"
"${PYTHON_BIN}" -m pip install -c "${TMP_CONSTRAINTS}" -r "${TMP_REQ}"

# Install repository forks without allowing pip to replace guarded wheels.
"${PYTHON_BIN}" -m pip install -e \
  "${ROOT}/Latent-GRPO/sglang_latent_reasoning_pkg/python" --no-deps
"${PYTHON_BIN}" -m pip install -e \
  "${ROOT}/Latent-GRPO/verl-0.4.x" --no-deps

"${PYTHON_BIN}" -c 'import importlib.metadata as md, sys, torch, triton, torchvision, sgl_kernel, flashinfer; assert sys.version_info[:2] == (3, 10), sys.version; assert torch.__version__.startswith("2.6.0+cu124"), torch.__version__; assert torch.version.cuda == "12.4", torch.version.cuda; assert triton.__version__.startswith("3.2.0"), triton.__version__; assert torchvision.__version__.startswith("0.21.0+cu124"), torchvision.__version__; assert md.version("compressed-tensors").startswith("0.9.3"), md.version("compressed-tensors"); assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2, torch.cuda.device_count(); assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(2)); print("T4_BASE_RUNTIME: PASS"); print("python", sys.executable); print("torch", torch.__version__, "cuda", torch.version.cuda); print("triton", triton.__version__); print("torchvision", torchvision.__version__); print("sgl-kernel", md.version("sgl-kernel")); print("compressed-tensors", md.version("compressed-tensors")); print("flashinfer-python", md.version("flashinfer-python")); print("cuda-python", md.version("cuda-python")); print("cuda-bindings", md.version("cuda-bindings"))'

echo "T4_RUNTIME_STACK_INSTALLED: ${VENV_DIR}; no training data read or generated"
echo "NEXT: ${PYTHON_BIN} ${ROOT}/tools/probe_kaggle_p1_t4_compatibility.py"
