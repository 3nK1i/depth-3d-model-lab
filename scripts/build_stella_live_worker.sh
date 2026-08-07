#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STELLA_PREFIX="${EIA_STELLA_PREFIX:-$HOME/Desktop/eia-stella-vslam-probe}"
BUILD_ROOT="${EIA_STELLA_LIVE_WORKER_BUILD:-$HOME/Desktop/eia-stella-live-worker-build}"
INSTALL_ROOT="${EIA_STELLA_LIVE_WORKER_PREFIX:-$HOME/Desktop/eia-stella-live-worker}"
METADATA="$STELLA_PREFIX/EIA_STELLA_PROBE_INFO.txt"
QUALIFIED_VERSION="0.3.13"
QUALIFIED_COMMIT="7902892d445843563d2d3149505fd3e1a118a141"
QUALIFIED_FBOW="7753c2c17dd7ce8bdf4a7a3fb1262b78d5b9b9b5"

read_meta() {
    local key="$1"
    awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$METADATA"
}

if [[ ! -f "$METADATA" ]]; then
    echo "ERROR: missing qualified stella probe metadata: $METADATA" >&2
    exit 1
fi
if [[ "$(read_meta probe_status)" != "PASS" ]]; then
    echo "ERROR: stella capability probe is not PASS" >&2
    exit 1
fi
STELLA_VERSION="$(read_meta stella_vslam_version)"
STELLA_COMMIT="$(read_meta stella_vslam_commit)"
FBOW_COMMIT="$(read_meta stella_FBoW_submodule_commit)"
if [[ "$STELLA_VERSION" != "$QUALIFIED_VERSION" || "$STELLA_COMMIT" != "$QUALIFIED_COMMIT" || "$FBOW_COMMIT" != "$QUALIFIED_FBOW" ]]; then
    echo "ERROR: isolated stella/FBoW build does not match the qualified Phase 8.4 backend" >&2
    printf 'stella_version=%s\nstella_commit=%s\nfbow_commit=%s\n' "$STELLA_VERSION" "$STELLA_COMMIT" "$FBOW_COMMIT" >&2
    exit 1
fi
if [[ ! -f "$STELLA_PREFIX/lib/libstella_vslam.so" ]]; then
    echo "ERROR: $STELLA_PREFIX/lib/libstella_vslam.so is missing" >&2
    exit 1
fi

# Avoid accidentally resolving build dependencies through the production Conda environment.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONPATH || true
export CMAKE_PREFIX_PATH="$STELLA_PREFIX"
export LIBRARY_PATH="$STELLA_PREFIX/lib:$STELLA_PREFIX/lib64"
export LD_LIBRARY_PATH="$STELLA_PREFIX/lib:$STELLA_PREFIX/lib64"

rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT" "$INSTALL_ROOT"

cmake \
    -S "$REPO_ROOT/native/live_slam_worker" \
    -B "$BUILD_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_ROOT" \
    -DCMAKE_PREFIX_PATH="$STELLA_PREFIX" \
    -DCMAKE_BUILD_RPATH="$STELLA_PREFIX/lib;$STELLA_PREFIX/lib64" \
    -DCMAKE_INSTALL_RPATH="$STELLA_PREFIX/lib;$STELLA_PREFIX/lib64" \
    -DEIA_STELLA_VERSION="$STELLA_VERSION" \
    -DEIA_STELLA_COMMIT="$STELLA_COMMIT"

cmake --build "$BUILD_ROOT" --parallel "${EIA_BUILD_JOBS:-4}"
cmake --install "$BUILD_ROOT"

WORKER="$INSTALL_ROOT/bin/eia_stella_live_worker"
if [[ ! -x "$WORKER" ]]; then
    echo "ERROR: worker was not installed: $WORKER" >&2
    exit 1
fi

if LD_LIBRARY_PATH="$STELLA_PREFIX/lib:$STELLA_PREFIX/lib64" ldd "$WORKER" | grep -q 'not found'; then
    echo "ERROR: unresolved live worker shared library:" >&2
    LD_LIBRARY_PATH="$STELLA_PREFIX/lib:$STELLA_PREFIX/lib64" ldd "$WORKER" | grep 'not found' >&2
    exit 1
fi

LD_LIBRARY_PATH="$STELLA_PREFIX/lib:$STELLA_PREFIX/lib64" "$WORKER" --help >/tmp/eia_stella_live_worker_help.txt
LD_LIBRARY_PATH="$STELLA_PREFIX/lib:$STELLA_PREFIX/lib64" "$WORKER" --version >/tmp/eia_stella_live_worker_version.txt

REPO_BASE="unknown"
if command -v git >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REPO_BASE="$(git -C "$REPO_ROOT" rev-parse HEAD)"
fi

cat > "$INSTALL_ROOT/EIA_STELLA_LIVE_WORKER_INFO.txt" <<INFO
EIA TECH Phase 8.4B isolated stella_vslam worker
worker_status=PASS
worker_protocol_version=1
worker_executable=$WORKER
stella_install_prefix=$STELLA_PREFIX
stella_vslam_version=$STELLA_VERSION
stella_vslam_commit=$STELLA_COMMIT
stella_FBoW_submodule_commit=$FBOW_COMMIT
source_repo_commit=$REPO_BASE
loop_closure_feature_enabled=false
global_optimization_feature_enabled=false
pixel_transport=posix_shared_memory
control_transport=af_unix_json_lines
INFO

echo "============================================================"
echo " EIA STELLA LIVE WORKER BUILD: PASS"
echo "============================================================"
cat "$INSTALL_ROOT/EIA_STELLA_LIVE_WORKER_INFO.txt"
echo
echo "=== Runtime linkage ==="
LD_LIBRARY_PATH="$STELLA_PREFIX/lib:$STELLA_PREFIX/lib64" ldd "$WORKER" | grep -Ei 'stella|fbow|g2o|opencv|yaml' || true
echo
echo "=== Worker version ==="
cat /tmp/eia_stella_live_worker_version.txt
