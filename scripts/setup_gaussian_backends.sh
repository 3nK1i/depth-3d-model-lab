#!/usr/bin/env bash
set -euo pipefail

# Phase 8.7D2: create isolated real Gaussian environments.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script never installs Graphdeco/gsplat into the production `depth` env.

ROOT_DIR="${HOME}/Desktop/eia-gaussian-backends"
BACKEND="both"
RECREATE=0
GSPLAT_ENV="eia-gaussian-gsplat"
GRAPHDECO_ENV="eia-gaussian-graphdeco"
GSPLAT_TAG="v1.5.3"
GSPLAT_REVISION_SHORT="937e299"
GRAPHDECO_REVISION="54c035f7834b564019656c3e3fcc3646292f727d"
GRAPHDECO_CSTDINT_PATCH_ID="diff-gaussian-rasterization-pr49-cstdint"
GRAPHDECO_CSTDINT_PATCH="$SCRIPT_DIR/gaussian_patches/graphdeco_diff_gaussian_rasterization_cstdint.patch"
GRAPHDECO_CSTDINT_PATCH_SHA256="2bb9a353d3dc6646c43507dcbd02718b819c94216368af9fd22b396220deed79"
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
CUDA_VERSION="12.8"
PYTHON_VERSION="3.10"

usage() {
  cat <<'EOF'
Usage: scripts/setup_gaussian_backends.sh [options]

Options:
  --backend gsplat|graphdeco|both   Backend(s) to create (default: both)
  --root PATH                       External checkout root
  --recreate                        Remove/recreate the selected external env/checkouts
  -h, --help                        Show this help

The stable production conda environment `depth` is never modified.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="${2:?missing backend value}"; shift 2 ;;
    --root)
      ROOT_DIR="${2:?missing root value}"; shift 2 ;;
    --recreate)
      RECREATE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$BACKEND" != "gsplat" && "$BACKEND" != "graphdeco" && "$BACKEND" != "both" ]]; then
  echo "ERROR: --backend must be gsplat, graphdeco, or both" >&2
  exit 2
fi

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found" >&2; exit 1; }
command -v g++ >/dev/null 2>&1 || { echo "ERROR: g++ not found; install a compatible C++ compiler first" >&2; exit 1; }

mkdir -p "$ROOT_DIR"
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

remove_if_recreate() {
  local env_name="$1"
  local checkout="$2"
  if [[ "$RECREATE" -eq 1 ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
      echo "ERROR: cannot recreate active conda env '$env_name'" >&2
      exit 1
    fi
    if env_exists "$env_name"; then
      conda env remove -y -n "$env_name"
    fi
    rm -rf "$checkout"
  fi
}

create_base_env() {
  local env_name="$1"
  if env_exists "$env_name"; then
    echo "ERROR: conda env '$env_name' already exists. Use --recreate for a controlled rebuild." >&2
    exit 1
  fi
  conda create -y -n "$env_name" "python=${PYTHON_VERSION}" pip setuptools wheel
  conda install -y -n "$env_name" -c nvidia "cuda-toolkit=${CUDA_VERSION}"
  conda run -n "$env_name" python -m pip install --upgrade pip
  conda run -n "$env_name" python -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu128
  conda run -n "$env_name" python -m pip install \
    "numpy==1.26.4" pillow packaging ninja cmake
}

env_prefix() {
  conda run -n "$1" python -c 'import os; print(os.environ["CONDA_PREFIX"])' | tail -n 1
}

apply_graphdeco_compatibility_patch() {
  local checkout="$1"
  local rasterizer="$checkout/submodules/diff-gaussian-rasterization"

  [[ -f "$GRAPHDECO_CSTDINT_PATCH" ]] || {
    echo "ERROR: missing Graphdeco compatibility patch: $GRAPHDECO_CSTDINT_PATCH" >&2
    exit 1
  }

  local actual_sha256
  actual_sha256="$(sha256sum "$GRAPHDECO_CSTDINT_PATCH" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$GRAPHDECO_CSTDINT_PATCH_SHA256" ]]; then
    echo "ERROR: Graphdeco compatibility patch checksum mismatch" >&2
    exit 1
  fi

  if git -C "$rasterizer" apply --unidiff-zero --check "$GRAPHDECO_CSTDINT_PATCH"; then
    git -C "$rasterizer" apply --unidiff-zero "$GRAPHDECO_CSTDINT_PATCH"
    echo "Applied Graphdeco compatibility patch: $GRAPHDECO_CSTDINT_PATCH_ID"
  elif git -C "$rasterizer" apply --unidiff-zero --reverse --check "$GRAPHDECO_CSTDINT_PATCH"; then
    echo "Graphdeco compatibility patch already applied: $GRAPHDECO_CSTDINT_PATCH_ID"
  else
    echo "ERROR: Graphdeco compatibility patch does not apply cleanly to pinned rasterizer source" >&2
    exit 1
  fi

  grep -Fq '#include <cstdint>' "$rasterizer/cuda_rasterizer/rasterizer_impl.h" || {
    echo "ERROR: Graphdeco compatibility patch verification failed" >&2
    exit 1
  }
}

setup_gsplat() {
  local checkout="$ROOT_DIR/gsplat"
  remove_if_recreate "$GSPLAT_ENV" "$checkout"
  create_base_env "$GSPLAT_ENV"

  git clone --recursive --depth 1 --branch "$GSPLAT_TAG" \
    https://github.com/nerfstudio-project/gsplat.git "$checkout"
  local actual_revision
  actual_revision="$(git -C "$checkout" rev-parse --short=7 HEAD)"
  if [[ "$actual_revision" != "$GSPLAT_REVISION_SHORT" ]]; then
    echo "ERROR: gsplat $GSPLAT_TAG resolved to $actual_revision, expected $GSPLAT_REVISION_SHORT" >&2
    exit 1
  fi
  local prefix
  prefix="$(env_prefix "$GSPLAT_ENV")"
  CUDA_HOME="$prefix" MAX_JOBS="${MAX_JOBS:-4}" \
    conda run -n "$GSPLAT_ENV" python -m pip install --no-build-isolation "$checkout"

  echo "=== gsplat isolated environment ==="
  conda run -n "$GSPLAT_ENV" python -c \
    'import importlib.metadata as md, torch; print("torch=", torch.__version__); print("torch_cuda=", torch.version.cuda); print("cuda_available=", torch.cuda.is_available()); print("gsplat=", md.version("gsplat"))'
  echo "python=$(conda run -n "$GSPLAT_ENV" python -c 'import sys; print(sys.executable)' | tail -n 1)"
  echo "checkout=$checkout"
}

setup_graphdeco() {
  local checkout="$ROOT_DIR/gaussian-splatting"
  remove_if_recreate "$GRAPHDECO_ENV" "$checkout"
  create_base_env "$GRAPHDECO_ENV"

  conda run -n "$GRAPHDECO_ENV" python -m pip install \
    plyfile tqdm opencv-python joblib

  git clone --filter=blob:none --no-checkout \
    https://github.com/graphdeco-inria/gaussian-splatting.git "$checkout"
  git -C "$checkout" fetch --depth 1 origin "$GRAPHDECO_REVISION"
  git -C "$checkout" checkout --detach "$GRAPHDECO_REVISION"
  git -C "$checkout" submodule update --init --recursive --depth 1
  apply_graphdeco_compatibility_patch "$checkout"

  local prefix
  prefix="$(env_prefix "$GRAPHDECO_ENV")"
  CUDA_HOME="$prefix" MAX_JOBS="${MAX_JOBS:-4}" \
    conda run -n "$GRAPHDECO_ENV" python -m pip install --no-build-isolation \
      "$checkout/submodules/diff-gaussian-rasterization" \
      "$checkout/submodules/simple-knn" \
      "$checkout/submodules/fused-ssim"

  echo "=== Graphdeco isolated environment ==="
  conda run -n "$GRAPHDECO_ENV" python -c \
    "import sys; sys.path.insert(0, r'$checkout'); import torch, diff_gaussian_rasterization, simple_knn._C; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('graphdeco_extensions=ready')"
  echo "revision=$(git -C "$checkout" rev-parse HEAD)"
  echo "compatibility_patch=$GRAPHDECO_CSTDINT_PATCH_ID"
  echo "python=$(conda run -n "$GRAPHDECO_ENV" python -c 'import sys; print(sys.executable)' | tail -n 1)"
  echo "checkout=$checkout"
}

if [[ "$BACKEND" == "gsplat" || "$BACKEND" == "both" ]]; then
  setup_gsplat
fi
if [[ "$BACKEND" == "graphdeco" || "$BACKEND" == "both" ]]; then
  setup_graphdeco
fi

echo "Phase 8.7D2 external Gaussian setup complete. Production env 'depth' was not modified."
