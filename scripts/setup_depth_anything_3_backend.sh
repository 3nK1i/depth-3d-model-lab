#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/eiatech/anaconda3}"
ENV_NAME="${DA3_ENV_NAME:-eia-da3}"
BACKEND_ROOT="${DA3_BACKEND_ROOT:-/home/eiatech/Desktop/eia-da3-backends}"
REPO_ROOT="${BACKEND_ROOT}/depth-anything-3"
UPSTREAM="https://github.com/ByteDance-Seed/Depth-Anything-3.git"
REVISION="${DA3_REVISION:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${PROJECT_ROOT}/scripts/da3_patches/turing_native_bf16.patch"
CONDA="${CONDA_ROOT}/bin/conda"
PYTHON="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"

if [[ ! -x "${CONDA}" ]]; then
  echo "ERROR: conda not found: ${CONDA}" >&2
  exit 2
fi
if [[ ! -s "${PATCH_FILE}" ]]; then
  echo "ERROR: compatibility patch missing: ${PATCH_FILE}" >&2
  exit 2
fi
if [[ -z "${REVISION}" ]]; then
  echo "ERROR: DA3_REVISION must be an explicit 40-character upstream Git commit." >&2
  exit 2
fi
if [[ ! "${REVISION}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: DA3_REVISION must be a full 40-character Git commit, got: ${REVISION}" >&2
  exit 2
fi

mkdir -p "${BACKEND_ROOT}"
if [[ -e "${REPO_ROOT}" ]]; then
  echo "ERROR: ${REPO_ROOT} already exists. Remove it explicitly before a fresh qualification install." >&2
  exit 2
fi

git clone "${UPSTREAM}" "${REPO_ROOT}"
git -C "${REPO_ROOT}" checkout "${REVISION}"
SOURCE_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

API_FILE="${REPO_ROOT}/src/depth_anything_3/api.py"
if grep -Fq 'is_bf16_supported(including_emulation=False)' "${API_FILE}"; then
  echo "DA3 upstream already contains native-bf16-only Turing guard; local compatibility patch not required."
else
  git -C "${REPO_ROOT}" apply --check "${PATCH_FILE}"
  git -C "${REPO_ROOT}" apply "${PATCH_FILE}"
fi
git -C "${REPO_ROOT}" diff --check

"${CONDA}" create -y -n "${ENV_NAME}" python=3.11
"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install \
  torch==2.7.1 torchvision==0.22.1 xformers==0.0.31 \
  --index-url https://download.pytorch.org/whl/cu128
"${PYTHON}" -m pip install numpy==1.26.4 addict
"${PYTHON}" -m pip install -e "${REPO_ROOT}"

PATCH_SHA256="$(sha256sum "${PATCH_FILE}" | awk '{print $1}')"
export SOURCE_REVISION PATCH_SHA256 REPO_ROOT
"${PYTHON}" - <<'PY'
import json, os, platform
from pathlib import Path
import torch, torchvision, xformers
import depth_anything_3.api  # noqa: F401
import addict  # noqa: F401

expected = {
    "python_prefix": "3.11.",
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "xformers": "0.0.31",
    "cuda": "12.8",
}
actual = {
    "python": platform.python_version(),
    "torch": str(torch.__version__).split("+", 1)[0],
    "torchvision": str(torchvision.__version__).split("+", 1)[0],
    "xformers": str(xformers.__version__),
    "cuda": str(torch.version.cuda or ""),
}
if not actual["python"].startswith(expected["python_prefix"]):
    raise SystemExit(f"ERROR: unexpected Python runtime: {actual['python']}")
for key in ("torch", "torchvision", "xformers", "cuda"):
    if actual[key] != expected[key]:
        raise SystemExit(f"ERROR: runtime drift for {key}: expected {expected[key]}, got {actual[key]}")

payload = {
    "schema_version": 1,
    "source_revision": os.environ["SOURCE_REVISION"],
    "patch_sha256": os.environ["PATCH_SHA256"],
    "python_version": platform.python_version(),
    "torch_version": str(torch.__version__),
    "torchvision_version": str(torchvision.__version__),
    "xformers_version": str(xformers.__version__),
    "torch_cuda_version": str(torch.version.cuda or ""),
    "compatibility_patch": "turing_native_bf16_autocast_pr269_equivalent",
}
path = Path(os.environ["REPO_ROOT"]) / ".eia_da3_runtime_lock.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(path)
print(json.dumps(payload, indent=2, sort_keys=True))
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu=", torch.cuda.get_device_name(0))
    print("compute_capability=", torch.cuda.get_device_capability(0))
PY

"${PYTHON}" -m pip check

echo "Phase 8.7F2 isolated DA3 environment setup: PASS"
echo "environment=${CONDA_ROOT}/envs/${ENV_NAME}"
echo "repository=${REPO_ROOT}"
echo "source_revision=${SOURCE_REVISION}"
echo "production_depth_environment_untouched=True"
