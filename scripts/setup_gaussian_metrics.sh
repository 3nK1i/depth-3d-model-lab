#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="eia-gaussian-metrics"
PYTHON_VERSION="3.10"
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
LPIPS_VERSION="0.1.4"
NUMPY_VERSION="1.26.4"
PILLOW_VERSION="10.4.0"
RECREATE=0

usage() {
  cat <<EOF
Usage: $0 [--recreate]

Create the isolated Phase 8.7D3 image-quality environment.
This environment is separate from production 'depth' and from both Gaussian backends.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate) RECREATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  if [[ "$RECREATE" -eq 1 ]]; then
    conda env remove -n "$ENV_NAME" -y
  else
    echo "Environment '$ENV_NAME' already exists; updating pinned benchmark packages."
  fi
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -n "$ENV_NAME" -y "python=${PYTHON_VERSION}" pip
fi

conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
conda run -n "$ENV_NAME" python -m pip install \
  "lpips==${LPIPS_VERSION}" "numpy==${NUMPY_VERSION}" "Pillow==${PILLOW_VERSION}"

# Instantiate the exact LPIPS network once so its standard AlexNet weights are
# resolved/cached during setup rather than unexpectedly during a benchmark run.
conda run --no-capture-output -n "$ENV_NAME" python -c '
import importlib.metadata as md
import lpips
import torch
import torchvision
model = lpips.LPIPS(net="alex", version="0.1", verbose=False)
model.eval()
print("torch=", torch.__version__)
print("torchvision=", torchvision.__version__)
print("lpips_package=", md.version("lpips"))
print("LPIPS_ALEX_V0_1=READY")
'

echo "Phase 8.7D3 metrics environment setup: PASS"
