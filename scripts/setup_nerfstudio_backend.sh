#!/usr/bin/env bash
set -euo pipefail

# Phase 8.7E2: create the isolated Nerfstudio/Nerfacto qualification runtime.
# This script never installs Nerfstudio, tiny-cuda-nn, or their CUDA stack into
# the production `depth` environment.

ENV_NAME="eia-nerf-nerfstudio"
ROOT_DIR="${HOME}/Desktop/eia-nerf-backends"
RECREATE=0
RESUME=0
CUDA_ARCH="75"
MAX_JOBS_VALUE="${MAX_JOBS:-4}"

PYTHON_VERSION="3.10"
TORCH_VERSION="2.1.2+cu118"
TORCHVISION_VERSION="0.16.2+cu118"
CUDA_TOOLKIT_VERSION="11.8.0"
NERFSTUDIO_TAG="v1.1.5"
NERFSTUDIO_REVISION_PREFIX="6b60855"
TCNN_REVISION="b3473c81396fe927293bdfd5a6be32df8769927c"

usage() {
  cat <<'EOF'
Usage: scripts/setup_nerfstudio_backend.sh [options]

Options:
  --root PATH         External checkout root (default: ~/Desktop/eia-nerf-backends)
  --env-name NAME     Isolated conda environment name (default: eia-nerf-nerfstudio)
  --cuda-arch XX      tiny-cuda-nn compute capability (default: 75 for RTX 2060)
  --recreate           Remove/recreate the isolated env and checkouts
  --resume             Resume a verified partial E2 setup after a build failure
  -h, --help           Show this help

This E2 setup is intentionally targeted at real RTX 2060 qualification.
The production `depth` environment is never modified.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT_DIR="${2:?missing root path}"; shift 2 ;;
    --env-name) ENV_NAME="${2:?missing environment name}"; shift 2 ;;
    --cuda-arch) CUDA_ARCH="${2:?missing CUDA architecture}"; shift 2 ;;
    --recreate) RECREATE=1; shift ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$RECREATE" -eq 1 && "$RESUME" -eq 1 ]]; then
  echo "ERROR: --recreate and --resume are mutually exclusive" >&2
  exit 2
fi

[[ "$CUDA_ARCH" =~ ^[0-9]{2,3}$ ]] || { echo "ERROR: --cuda-arch must be numeric (for RTX 2060 use 75)" >&2; exit 2; }
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found" >&2; exit 1; }

ROOT_DIR="$(mkdir -p "$ROOT_DIR" && cd "$ROOT_DIR" && pwd)"
NERFSTUDIO_ROOT="$ROOT_DIR/nerfstudio"
TCNN_ROOT="$ROOT_DIR/tiny-cuda-nn"
MANIFEST="$ROOT_DIR/e2_environment_manifest.json"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
  echo "ERROR: do not run setup while isolated environment '$ENV_NAME' is active" >&2
  exit 1
fi

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

if [[ "$RECREATE" -eq 1 ]]; then
  if env_exists "$ENV_NAME"; then
    conda env remove -y -n "$ENV_NAME"
  fi
  rm -rf "$NERFSTUDIO_ROOT" "$TCNN_ROOT" "$MANIFEST"
fi

if [[ "$RESUME" -eq 0 ]]; then
  if env_exists "$ENV_NAME"; then
    echo "ERROR: conda env '$ENV_NAME' already exists. Use --resume after a verified partial setup or --recreate for a controlled rebuild." >&2
    exit 1
  fi
  if [[ -e "$NERFSTUDIO_ROOT" || -e "$TCNN_ROOT" ]]; then
    echo "ERROR: E2 checkout already exists. Use --resume after a verified partial setup, --recreate, or choose another --root." >&2
    exit 1
  fi

  echo "=== Create isolated Nerfstudio environment ==="
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip wheel "setuptools<70"

  # CUDA 11.8 follows the pinned Nerfstudio v1.1.5 installation/runtime recipe.
  conda install -y -n "$ENV_NAME" -c "nvidia/label/cuda-11.8.0" "cuda-toolkit=${CUDA_TOOLKIT_VERSION}"
  # Ubuntu 24 hosts commonly ship a newer system GCC than CUDA 11.8 accepts.
  # Keep the compatibility compiler isolated inside this environment as well.
  conda install -y -n "$ENV_NAME" -c conda-forge "gcc_linux-64=11" "gxx_linux-64=11" ninja cmake

  conda run -n "$ENV_NAME" python -m pip install --upgrade pip "setuptools<70" wheel
  conda run -n "$ENV_NAME" python -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --extra-index-url https://download.pytorch.org/whl/cu118
  conda run -n "$ENV_NAME" python -m pip install "numpy==1.26.4" ninja packaging

  echo "=== Checkout pinned Nerfstudio ==="
  git clone --filter=blob:none --no-checkout https://github.com/nerfstudio-project/nerfstudio.git "$NERFSTUDIO_ROOT"
  git -C "$NERFSTUDIO_ROOT" fetch --depth 1 origin "refs/tags/${NERFSTUDIO_TAG}:refs/tags/${NERFSTUDIO_TAG}"
  git -C "$NERFSTUDIO_ROOT" checkout --detach "$NERFSTUDIO_TAG"

  echo "=== Checkout pinned tiny-cuda-nn ==="
  git clone --filter=blob:none --no-checkout https://github.com/NVlabs/tiny-cuda-nn.git "$TCNN_ROOT"
  git -C "$TCNN_ROOT" fetch --depth 1 origin "$TCNN_REVISION"
  git -C "$TCNN_ROOT" checkout --detach "$TCNN_REVISION"
  git -C "$TCNN_ROOT" submodule update --init --recursive --depth 1
else
  echo "=== Resume verified partial Nerfstudio environment ==="
  env_exists "$ENV_NAME" || { echo "ERROR: --resume requires existing conda env '$ENV_NAME'" >&2; exit 1; }
  [[ -d "$NERFSTUDIO_ROOT/.git" ]] || { echo "ERROR: --resume requires pinned Nerfstudio checkout at $NERFSTUDIO_ROOT" >&2; exit 1; }
  [[ -d "$TCNN_ROOT/.git" ]] || { echo "ERROR: --resume requires pinned tiny-cuda-nn checkout at $TCNN_ROOT" >&2; exit 1; }
fi

NERFSTUDIO_REVISION="$(git -C "$NERFSTUDIO_ROOT" rev-parse HEAD)"
[[ "$NERFSTUDIO_REVISION" == ${NERFSTUDIO_REVISION_PREFIX}* ]] || {
  echo "ERROR: Nerfstudio ${NERFSTUDIO_TAG} resolved to ${NERFSTUDIO_REVISION}, expected prefix ${NERFSTUDIO_REVISION_PREFIX}" >&2
  exit 1
}
ACTUAL_TCNN_REVISION="$(git -C "$TCNN_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_TCNN_REVISION" == "$TCNN_REVISION" ]] || {
  echo "ERROR: tiny-cuda-nn revision mismatch" >&2
  exit 1
}

# conda run may emit surrounding blank lines; use sys.prefix and strip blanks so
# CUDA_HOME can never silently become empty.
ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)' | sed '/^[[:space:]]*$/d' | tail -n 1)"
[[ -n "$ENV_PREFIX" && -d "$ENV_PREFIX" ]] || {
  echo "ERROR: could not resolve isolated environment prefix" >&2
  exit 1
}
ENV_PYTHON="$ENV_PREFIX/bin/python"
[[ -x "$ENV_PYTHON" ]] || { echo "ERROR: isolated Python missing at $ENV_PYTHON" >&2; exit 1; }

# Verify the partial runtime before resuming an expensive native build.
conda run -n "$ENV_NAME" python - <<'PY'
import sys
import torch
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"unexpected Python: {sys.version}")
if not torch.__version__.startswith("2.1.2"):
    raise SystemExit(f"unexpected torch: {torch.__version__}")
if torch.version.cuda != "11.8":
    raise SystemExit(f"unexpected torch CUDA runtime: {torch.version.cuda}")
print("resume_runtime_prerequisites=PASS")
PY

# The CUDA toolkit ships a link-time libcuda stub; the real libcuda.so.1 is
# provided at runtime by the installed NVIDIA driver. Auto-discover the stub
# and expose it only through LIBRARY_PATH for this build. Never add stubs to
# LD_LIBRARY_PATH because that would be incorrect for runtime execution.
CUDA_STUB_DIR=""
for candidate in \
  "$ENV_PREFIX/targets/x86_64-linux/lib/stubs" \
  "$ENV_PREFIX/lib64/stubs" \
  "$ENV_PREFIX/lib/stubs"; do
  if [[ -f "$candidate/libcuda.so" ]]; then
    CUDA_STUB_DIR="$candidate"
    break
  fi
done
if [[ -z "$CUDA_STUB_DIR" ]]; then
  CUDA_STUB_DIR="$(find "$ENV_PREFIX" -type f -name libcuda.so -printf '%h\n' 2>/dev/null | head -n 1 || true)"
fi
[[ -n "$CUDA_STUB_DIR" && -f "$CUDA_STUB_DIR/libcuda.so" ]] || {
  echo "ERROR: CUDA driver link stub libcuda.so not found inside isolated CUDA toolkit" >&2
  exit 1
}
echo "isolated_env_prefix=$ENV_PREFIX"
echo "cuda_driver_stub_dir=$CUDA_STUB_DIR"

# Build the torch bindings for the target GPU only. This keeps E2 compile time
# bounded and matches the real RTX 2060 qualification target (compute 7.5).
# On --resume, do not rebuild a binding that already imports successfully.
if conda run -n "$ENV_NAME" python -c 'import tinycudann' >/dev/null 2>&1; then
  echo "tiny_cuda_nn_runtime=already_installed"
else
  echo "=== Build pinned tiny-cuda-nn torch bindings for SM ${CUDA_ARCH} ==="
  conda run -n "$ENV_NAME" env \
    CUDA_HOME="$ENV_PREFIX" \
    LIBRARY_PATH="$CUDA_STUB_DIR" \
    TCNN_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    MAX_JOBS="$MAX_JOBS_VALUE" \
    python -m pip install --no-build-isolation "$TCNN_ROOT/bindings/torch"
fi

# Install exact Nerfstudio source after torch/tcnn so pip keeps the pinned
# runtime. Do NOT disable PEP 517 build isolation here: transitive source
# dependencies such as fpsample declare their own build-system requirements
# (scikit-build-core + pybind11), which pip must provision in temporary build
# environments. numpy remains pinned for this Nerfstudio/PyTorch stack.
echo "=== Install pinned Nerfstudio ==="
conda run -n "$ENV_NAME" env MAX_JOBS="$MAX_JOBS_VALUE" \
  python -m pip install "$NERFSTUDIO_ROOT" "numpy==1.26.4"

conda run -n "$ENV_NAME" python -m pip check

# Fail setup if the real CUDA/tcnn/Nerfstudio runtime cannot import.
conda run -n "$ENV_NAME" python - <<'PY'
import importlib.metadata as md
import torch
import tinycudann
import nerfstudio
print("python_runtime=ready")
print("nerfstudio=", md.version("nerfstudio"))
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in isolated Nerfstudio environment")
print("gpu=", torch.cuda.get_device_name(0))
print("compute_capability=", torch.cuda.get_device_capability(0))
print("total_vram_mib=", round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1))
PY

[[ -x "$ENV_PREFIX/bin/ns-train" ]] || { echo "ERROR: ns-train missing" >&2; exit 1; }
[[ -x "$ENV_PREFIX/bin/ns-render" ]] || { echo "ERROR: ns-render missing" >&2; exit 1; }

"$ENV_PYTHON" - <<PY
import json
from pathlib import Path
payload = {
    "environment_name": ${ENV_NAME@Q},
    "python_executable": ${ENV_PYTHON@Q},
    "python_version": ${PYTHON_VERSION@Q},
    "nerfstudio_root": ${NERFSTUDIO_ROOT@Q},
    "nerfstudio_tag": ${NERFSTUDIO_TAG@Q},
    "nerfstudio_revision": ${NERFSTUDIO_REVISION@Q},
    "tiny_cuda_nn_root": ${TCNN_ROOT@Q},
    "tiny_cuda_nn_revision": ${ACTUAL_TCNN_REVISION@Q},
    "torch_version": ${TORCH_VERSION@Q},
    "torchvision_version": ${TORCHVISION_VERSION@Q},
    "cuda_toolkit": ${CUDA_TOOLKIT_VERSION@Q},
    "tcnn_cuda_architectures": ${CUDA_ARCH@Q},
    "production_depth_environment_modified": False,
    "phase": "8.7E2",
}
Path(${MANIFEST@Q}).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "Phase 8.7E2 isolated Nerfstudio setup: PASS"
echo "python=$ENV_PYTHON"
echo "nerfstudio_root=$NERFSTUDIO_ROOT"
echo "tiny_cuda_nn_root=$TCNN_ROOT"
echo "manifest=$MANIFEST"
echo "production depth environment was not modified"
