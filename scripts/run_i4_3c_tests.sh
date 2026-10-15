#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# I4.3C extends the fully validated I4.3B gate with the canonical DA3 dense
# provider binding while retaining the legacy DA3 contracts as backward sanity.
# I4.3B baseline: focused=193, stress=105, all=298.
# I4.3C adds focused=56 (30 legacy DA3 + 26 new) and stress=14 (2 legacy + 12 new).
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
i43a_stress=(tests/stress/test_i4_3a_dense_provider_architecture_stress.py)
i43b_focused=(
  tests/unit/test_i4_3b_mvs_provider.py
  tests/integration/test_i4_3b_mvs_provider_contract.py
)
i43b_stress=(tests/stress/test_i4_3b_mvs_provider_stress.py)
da3_backward_focused=(
  tests/unit/test_reconstruction_da3.py
  tests/unit/test_reconstruction_da3_provider.py
  tests/integration/test_reconstruction_da3_provider.py
)
da3_backward_stress=(tests/stress/test_reconstruction_da3_provider_stress.py)
i43c_focused=(
  tests/unit/test_i4_3c_da3_dense_provider.py
  tests/integration/test_i4_3c_da3_dense_provider_contract.py
)
i43c_stress=(tests/stress/test_i4_3c_da3_dense_provider_stress.py)

case "${1:-all}" in
  focused) python -m pytest -q "${i42_focused[@]}" "${i43a_focused[@]}" "${i43b_focused[@]}" "${da3_backward_focused[@]}" "${i43c_focused[@]}" ;;
  stress) python -m pytest -q "${i42_stress[@]}" "${i43a_stress[@]}" "${i43b_stress[@]}" "${da3_backward_stress[@]}" "${i43c_stress[@]}" ;;
  all) python -m pytest -q "${i42_focused[@]}" "${i43a_focused[@]}" "${i43b_focused[@]}" "${da3_backward_focused[@]}" "${i43c_focused[@]}" "${i42_stress[@]}" "${i43a_stress[@]}" "${i43b_stress[@]}" "${da3_backward_stress[@]}" "${i43c_stress[@]}" ;;
  *) echo "usage: $0 [focused|stress|all]" >&2; exit 2 ;;
esac

DA3_PYTHON="${DEPTH3D_RECONSTRUCTION_DA3_PYTHON:-$HOME/anaconda3/envs/eia-da3/bin/python3.11}"
DA3_REPO="${DEPTH3D_RECONSTRUCTION_DA3_REPOSITORY:-$HOME/Desktop/eia-da3-backends/depth-anything-3}"
if [[ -x "$DA3_PYTHON" && -d "$DA3_REPO" ]]; then
  "$DA3_PYTHON" - "$DA3_REPO" <<'PYPROBE'
import importlib
import inspect
from pathlib import Path
import subprocess
import sys

repo = Path(sys.argv[1]).resolve()
module = importlib.import_module("depth_anything_3.api")
cls = getattr(module, "DepthAnything3")
api_file = Path(inspect.getfile(cls)).resolve()
source_revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert api_file.is_file()
assert len(source_revision) == 40
print(f"I4.3C isolated DA3 runtime import probe: PASS revision={source_revision}")
PYPROBE
else
  echo "I4.3C isolated DA3 runtime import probe: SKIP (runtime not found: $DA3_PYTHON / $DA3_REPO)"
fi
