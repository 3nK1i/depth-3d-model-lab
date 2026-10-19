#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="depth3d-rollingdepth"
ROOT="${HOME}/Desktop/depth3d-rollingdepth-backends"
REVISION=""
DOWNLOAD_WEIGHTS=0
ACCEPT_MODEL_LICENSE=0
FINALIZE_EXISTING=0
REPO_URL="https://github.com/prs-eth/RollingDepth.git"

usage() {
  cat <<'EOF'
Usage: setup_rolling_depth_backend.sh --revision <40-hex> [options]

Options:
  --revision <sha>          Required exact RollingDepth git revision.
  --env-name <name>         Conda env name (default: depth3d-rollingdepth).
  --root <path>             Backend root (default: ~/Desktop/depth3d-rollingdepth-backends).
  --download-weights        Run the official model-weight download script.
  --accept-model-license    Confirm acceptance of RollingDepth model LICENSE-MODEL.
  --finalize-existing       Verify/finalize an already installed runtime without remote fetch or dependency reinstall.
  -h, --help                Show this help.

The code checkout is Apache-2.0. The model weights use the repository's
LICENSE-MODEL (RAIL++-M); --download-weights requires explicit acceptance.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --revision) REVISION="${2:-}"; shift 2 ;;
    --env-name) ENV_NAME="${2:-}"; shift 2 ;;
    --root) ROOT="${2:-}"; shift 2 ;;
    --download-weights) DOWNLOAD_WEIGHTS=1; shift ;;
    --accept-model-license) ACCEPT_MODEL_LICENSE=1; shift ;;
    --finalize-existing) FINALIZE_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: --revision must be an exact 40-character git SHA." >&2
  exit 2
fi
REVISION="${REVISION,,}"

if [[ "$DOWNLOAD_WEIGHTS" -eq 1 && "$ACCEPT_MODEL_LICENSE" -ne 1 ]]; then
  echo "ERROR: --download-weights requires --accept-model-license." >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not available." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is required by RollingDepth/pyav." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

ROOT="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$ROOT")"
REPO="$ROOT/RollingDepth"
mkdir -p "$ROOT"

if [[ "$FINALIZE_EXISTING" -eq 1 ]]; then
  if [[ ! -d "$REPO/.git" ]]; then
    echo "ERROR: --finalize-existing requires an existing RollingDepth checkout at $REPO." >&2
    exit 1
  fi
  ACTUAL_REVISION="$(git -C "$REPO" rev-parse HEAD)"
  if [[ "$ACTUAL_REVISION" != "$REVISION" ]]; then
    echo "ERROR: existing RollingDepth checkout revision does not match --revision." >&2
    exit 1
  fi
  echo "RollingDepth finalize-existing mode: reusing pinned checkout revision=$ACTUAL_REVISION"
else
  if [[ ! -d "$REPO/.git" ]]; then
    git clone "$REPO_URL" "$REPO"
  else
    if [[ "$(git -C "$REPO" remote get-url origin)" != "$REPO_URL" ]]; then
      echo "ERROR: existing checkout origin is not the official RollingDepth repository." >&2
      exit 1
    fi
  fi

  git -C "$REPO" fetch --tags origin
  git -C "$REPO" cat-file -e "${REVISION}^{commit}"
  git -C "$REPO" checkout --detach "$REVISION"
  ACTUAL_REVISION="$(git -C "$REPO" rev-parse HEAD)"
  if [[ "$ACTUAL_REVISION" != "$REVISION" ]]; then
    echo "ERROR: checkout revision mismatch." >&2
    exit 1
  fi
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  if [[ "$FINALIZE_EXISTING" -eq 1 ]]; then
    echo "ERROR: --finalize-existing requires existing Conda environment '$ENV_NAME'." >&2
    exit 1
  fi
  conda create -y -n "$ENV_NAME" python=3.12
fi

resolve_env_python() {
  local output resolved
  if ! output="$(
    PYTHONNOUSERSITE=1 conda run -n "$ENV_NAME" \
      python -c 'import os,sys; print(os.path.realpath(sys.executable))' 2>&1
  )"; then
    echo "ERROR: conda could not execute Python in environment '$ENV_NAME'." >&2
    printf '%s\n' "$output" >&2
    return 1
  fi
  resolved="$(printf '%s\n' "$output" | awk '/^\// {path=$0} END {print path}')"
  if [[ -z "$resolved" || ! -x "$resolved" ]]; then
    echo "ERROR: could not resolve RollingDepth Python executable from sys.executable." >&2
    printf '%s\n' "$output" >&2
    return 1
  fi
  printf '%s\n' "$resolved"
}

PYTHON_BIN="$(resolve_env_python)"

if [[ "$FINALIZE_EXISTING" -eq 0 ]]; then
  PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m pip install --upgrade pip

  NORMALIZED_REQUIREMENTS="$(mktemp)"
  cleanup_requirements() {
    rm -f "$NORMALIZED_REQUIREMENTS"
  }
  trap cleanup_requirements EXIT

  python - "$REPO/requirements.txt" "$NORMALIZED_REQUIREMENTS" <<'PYREQ'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
normalized = []
replacements = 0
for line in lines:
    replaced, count = re.subn(
        r"^(\s*)pyav(?=(?:[<>=!~]|\s|$))",
        r"\1av",
        line,
        count=1,
        flags=re.IGNORECASE,
    )
    normalized.append(replaced)
    replacements += count

destination.write_text("".join(normalized), encoding="utf-8")
print(f"RollingDepth dependency normalization: pyav->av replacements={replacements}")
PYREQ

  # Upstream calls the FFmpeg binding "pyav", but the installable PyPI
  # distribution is named "av". Normalize only the temporary install input.
  PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m pip install -r "$NORMALIZED_REQUIREMENTS"
  (
    cd "$REPO"
    PYTHONNOUSERSITE=1 conda run -n "$ENV_NAME" bash script/install_diffusers_dev.sh
  )
else
  echo "RollingDepth finalize-existing mode: skipping dependency reinstall"
fi

CHECKPOINT_DIR="$REPO/checkpoint/rollingdepth-v1-0"
CHECKPOINT_MARKER="$CHECKPOINT_DIR/model_index.json"
if [[ "$DOWNLOAD_WEIGHTS" -eq 1 ]]; then
  if [[ -f "$CHECKPOINT_MARKER" ]]; then
    echo "RollingDepth checkpoint already present; skipping weight download: $CHECKPOINT_DIR"
  elif [[ "$FINALIZE_EXISTING" -eq 1 ]]; then
    echo "ERROR: --finalize-existing requires existing checkpoint marker $CHECKPOINT_MARKER." >&2
    exit 1
  else
    (
      cd "$REPO"
      PYTHONNOUSERSITE=1 conda run -n "$ENV_NAME" bash script/download_weight.sh
    )
  fi
  if [[ ! -f "$CHECKPOINT_MARKER" ]]; then
    echo "ERROR: RollingDepth checkpoint download/extraction did not produce $CHECKPOINT_MARKER." >&2
    exit 1
  fi
fi

PROBE_FILE="$(mktemp)"
trap 'rm -f "$PROBE_FILE" "${NORMALIZED_REQUIREMENTS:-}"' EXIT

PYTHONNOUSERSITE=1 "$PYTHON_BIN" - "$REPO" "$PROBE_FILE" <<'PY'
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
probe_path = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(repo))

payload = {
    "python_version": sys.version.split()[0],
    "upstream_importable": False,
    "modified_diffusers_ready": False,
    "torch_available": False,
    "cuda_available": False,
    "device_names": [],
}

from rollingdepth import RollingDepthPipeline
payload["upstream_importable"] = RollingDepthPipeline is not None

import diffusers
payload["diffusers_version"] = str(getattr(diffusers, "__version__", ""))
diffusers_root = pathlib.Path(diffusers.__file__).resolve().parent
marker = "Modified in RollingDepth"
payload["modified_diffusers_ready"] = any(
    marker in path.read_text(encoding="utf-8", errors="ignore")
    for path in diffusers_root.rglob("*.py")
)

import torch
payload["torch_available"] = True
payload["torch_version"] = str(torch.__version__)
payload["torch_cuda_version"] = str(torch.version.cuda or "")
payload["cuda_available"] = bool(torch.cuda.is_available())
if payload["cuda_available"]:
    payload["device_names"] = [
        str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())
    ]

probe_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
print(f"RollingDepth runtime probe written: {probe_path}")
PY

PROBE_JSON="$(cat "$PROBE_FILE")"

LOCK="$REPO/.depth3d_rolling_depth_runtime_lock.json"
PYTHONNOUSERSITE=1 "$PYTHON_BIN" - "$LOCK" "$REVISION" "$ENV_NAME" "$PYTHON_BIN" "$PROBE_JSON" <<'PY'
import json
import pathlib
import sys

lock_path = pathlib.Path(sys.argv[1])
revision = sys.argv[2]
env_name = sys.argv[3]
python_executable = sys.argv[4]
probe = json.loads(sys.argv[5])

payload = {
    "schema_version": 1,
    "backend": "rolling_depth",
    "repository": "https://github.com/prs-eth/RollingDepth.git",
    "source_revision": revision,
    "conda_env": env_name,
    "python_executable": python_executable,
    **probe,
}
lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(lock_path)
PY

if [[ "$(PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c 'import json,sys; p=json.loads(sys.argv[1]); print("1" if p.get("upstream_importable") and p.get("modified_diffusers_ready") and p.get("torch_available") else "0")' "$PROBE_JSON")" != "1" ]]; then
  echo "ERROR: RollingDepth runtime verification failed: $PROBE_JSON" >&2
  exit 1
fi

cat <<EOF
RollingDepth isolated runtime: READY
repository=$REPO
revision=$REVISION
python=$PYTHON_BIN
runtime_lock=$LOCK
checkpoint=$REPO/checkpoint/rollingdepth-v1-0
probe=$PROBE_JSON
EOF
