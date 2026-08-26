#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

focused=(
  tests/unit/test_i4_2e_1_worker_dependency_isolation.py
  tests/integration/test_i4_2e_1_worker_dependency_isolation_contract.py
)
stress=(
  tests/stress/test_i4_2e_1_worker_dependency_isolation_stress.py
)

case "${1:-all}" in
  focused) python -m pytest -q "${focused[@]}" ;;
  stress) python -m pytest -q "${stress[@]}" ;;
  all) python -m pytest -q "${focused[@]}" "${stress[@]}" ;;
  *) echo "usage: $0 [focused|stress|all]" >&2; exit 2 ;;
esac

PYCOLMAP_PYTHON="${EIA_PYCOLMAP_PYTHON:-/home/eiatech/anaconda3/envs/eia-pycolmap-probe/bin/python3.11}"
WORKER="scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py"
if [[ -x "$PYCOLMAP_PYTHON" ]]; then
  "$PYCOLMAP_PYTHON" - "$WORKER" <<'PYPROBE'
import builtins
from pathlib import Path
import runpy
import sys

worker = Path(sys.argv[1]).resolve()
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "cv2" or name == "core" or name.startswith("core."):
        raise RuntimeError(f"forbidden isolated-worker import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
try:
    runpy.run_path(str(worker), run_name="i4_2e_1_isolated_worker_import_probe")
finally:
    builtins.__import__ = original_import
PYPROBE
  echo "I4.2E.1 isolated-worker import probe: PASS"
else
  echo "I4.2E.1 isolated-worker import probe: SKIP (interpreter not found: $PYCOLMAP_PYTHON)"
fi
