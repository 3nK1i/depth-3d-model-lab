#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

focused=(
  tests/unit/test_i4_2e_2_native_visibility_provenance.py
  tests/integration/test_i4_2e_2_native_visibility_provenance_contract.py
)
stress=(
  tests/stress/test_i4_2e_2_native_visibility_provenance_stress.py
)

case "${1:-all}" in
  focused) python -m pytest -q "${focused[@]}" ;;
  stress) python -m pytest -q "${stress[@]}" ;;
  all) python -m pytest -q "${focused[@]}" "${stress[@]}" ;;
  *) echo "usage: $0 [focused|stress|all]" >&2; exit 2 ;;
esac

PYCOLMAP_PYTHON="${DEPTH3D_RECONSTRUCTION_PYCOLMAP_PYTHON:-$HOME/anaconda3/envs/depth3d-pycolmap-probe/bin/python3.11}"
WORKER="scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py"
if [[ -x "$PYCOLMAP_PYTHON" ]]; then
  "$PYCOLMAP_PYTHON" - "$WORKER" <<'PYPROBE'
import builtins
from pathlib import Path
import runpy
import struct
import sys
import tempfile

worker = Path(sys.argv[1]).resolve()
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in {"cv2", "pycolmap", "core"} or name.startswith("core."):
        raise RuntimeError(f"forbidden I4.2E.2 isolated-worker import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
try:
    namespace = runpy.run_path(str(worker), run_name="i4_2e_2_isolated_worker_probe")
finally:
    builtins.__import__ = original_import

reader = namespace["_read_points_visibility"]
with tempfile.TemporaryDirectory(prefix="i42e2_vis_probe_") as td:
    path = Path(td) / "fused.ply.vis"
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 47))
        for index in range(47):
            handle.write(struct.pack("<II", 1, index))
    offsets, indices = reader(path, expected_points=47, expected_views=47)
    assert offsets.tolist() == list(range(48))
    assert indices.tolist() == list(range(47))
PYPROBE
  echo "I4.2E.2 isolated native-visibility probe: PASS"
else
  echo "I4.2E.2 isolated native-visibility probe: SKIP (interpreter not found: $PYCOLMAP_PYTHON)"
fi
