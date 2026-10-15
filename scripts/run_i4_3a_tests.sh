#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep the complete closed I4.2 regression gate, then add the revised I4.3A v2 foundation.
# On the validated I4.2F baseline: focused=172, stress=92, all=264.
i42_focused=(
  tests/unit/test_i4_2a_temple_ring_preflight.py
  tests/unit/test_i4_2a_1_middlebury_header.py
  tests/integration/test_i4_2a_temple_ring_preflight_contract.py
  tests/integration/test_i4_2a_1_middlebury_header_contract.py
  tests/unit/test_i4_2b_temple_ring_sfm.py
  tests/integration/test_i4_2b_temple_ring_sfm_contract.py
  tests/unit/test_i4_2c_temple_ring_trajectory_validation.py
  tests/integration/test_i4_2c_temple_ring_trajectory_validation_contract.py
  tests/unit/test_i4_2d_temple_ring_mvs.py
  tests/integration/test_i4_2d_temple_ring_mvs_contract.py
  tests/unit/test_i4_2e_temple_ring_detail_qualification.py
  tests/integration/test_i4_2e_temple_ring_detail_qualification_contract.py
  tests/integration/test_i4_2e_temple_ring_detail_preview_contract.py
  tests/unit/test_i4_2e_1_worker_dependency_isolation.py
  tests/integration/test_i4_2e_1_worker_dependency_isolation_contract.py
  tests/unit/test_i4_2e_2_native_visibility_provenance.py
  tests/integration/test_i4_2e_2_native_visibility_provenance_contract.py
  tests/unit/test_i4_2e_2a_native_visibility_coverage_contract.py
  tests/integration/test_i4_2e_2a_native_visibility_coverage_contract.py
  tests/unit/test_i4_2f_temple_ring_closure.py
  tests/integration/test_i4_2f_temple_ring_closure_contract.py
)
i42_stress=(
  tests/stress/test_i4_2a_temple_ring_preflight_stress.py
  tests/stress/test_i4_2a_1_middlebury_header_stress.py
  tests/stress/test_i4_2b_temple_ring_sfm_stress.py
  tests/stress/test_i4_2c_temple_ring_trajectory_validation_stress.py
  tests/stress/test_i4_2d_temple_ring_mvs_stress.py
  tests/stress/test_i4_2e_temple_ring_detail_qualification_stress.py
  tests/stress/test_i4_2e_1_worker_dependency_isolation_stress.py
  tests/stress/test_i4_2e_2_native_visibility_provenance_stress.py
  tests/stress/test_i4_2e_2a_native_visibility_coverage_stress.py
  tests/stress/test_i4_2f_temple_ring_closure_stress.py
)
i43a_focused=(
  tests/unit/test_i4_3a_dense_provider_architecture.py
  tests/integration/test_i4_3a_dense_provider_architecture_contract.py
)
i43a_stress=(
  tests/stress/test_i4_3a_dense_provider_architecture_stress.py
)

case "${1:-all}" in
  focused) python -m pytest -q "${i42_focused[@]}" "${i43a_focused[@]}" ;;
  stress) python -m pytest -q "${i42_stress[@]}" "${i43a_stress[@]}" ;;
  all) python -m pytest -q "${i42_focused[@]}" "${i43a_focused[@]}" "${i42_stress[@]}" "${i43a_stress[@]}" ;;
  *) echo "usage: $0 [focused|stress|all]" >&2; exit 2 ;;
esac

PYCOLMAP_PYTHON="${DEPTH3D_RECONSTRUCTION_PYCOLMAP_PYTHON:-$HOME/anaconda3/envs/eia-pycolmap-probe/bin/python3.11}"
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
        raise RuntimeError(f"forbidden I4.3A backward isolated-worker import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
try:
    namespace = runpy.run_path(str(worker), run_name="i4_3a_backward_isolated_worker_probe")
finally:
    builtins.__import__ = original_import

reader = namespace["_read_points_visibility"]
with tempfile.TemporaryDirectory(prefix="i43a_vis_probe_") as td:
    path = Path(td) / "fused.ply.vis"
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 3))
        handle.write(struct.pack("<III", 2, 2, 5))
        handle.write(struct.pack("<II", 1, 3))
        handle.write(struct.pack("<III", 2, 1, 4))
    offsets, indices = reader(path, expected_points=3, expected_views=6)
    assert offsets.tolist() == [0, 2, 3, 5]
    assert indices.tolist() == [2, 5, 3, 1, 4]
PYPROBE
  echo "I4.3A backward isolated native-visibility probe: PASS"
else
  echo "I4.3A backward isolated native-visibility probe: SKIP (interpreter not found: $PYCOLMAP_PYTHON)"
fi
