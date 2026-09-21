#!/usr/bin/env bash
# I6.7E2B1 single-commit full dense result admission. No live provider execution.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
MODE="${1:-focused}"
if [[ "$MODE" != focused && "$MODE" != closure ]]; then
  echo 'Usage: bash scripts/run_i6_7e2b1_tests.sh focused|closure' >&2
  exit 2
fi
python - <<'PY'
from pathlib import Path
import hashlib
checks = {
 'core/reconstruction/reconstruction_progressive_dense_contract.py': '918c67574ef342909fffdfd495d2503c5909bafd0624fb3856b34c488f2257ee',
 'core/reconstruction/reconstruction_progressive_dense_emission.py': '5c68ec59d1ec0238b15853e95135a56df7e2043824e9c835c66fadaa4f594ff7',
 'core/reconstruction/reconstruction_progressive_dense_transport.py': '0874b96fc0b398a101fa8f432e965e0c178c857a099d99e36377420a12da560c',
 'app/reconstruction_progressive_dense_viewer.py': '453dbbdd8d90852ae0062e01c1370d14d55c849331cc157c551bf9490ab09dd8',
 'app/reconstruction_progressive_sfm_viewer.py': 'cda2e1da490a83a9af0875a95d4ac3672f9637b3dd616a7e6738911702864d76',
 'app/offline_reconstruction_controller.py': 'bd4f9c1cfe1739759d4b95d736cf376f36cffd04288ffb615e133fbb8358aa79',
 'app/reconstruction_controller.py': '57ef77cc5ebad69006b11ad25225e289ef540539e09e1636129b9d9bd4211186',
 'app/widgets/reconstruction_page.py': '2e5b406063f29cd02681fd299136c5836837dfbb60317536161748b62b09a590',
 'app/widgets/viewport/vispy_canvas.py': '2f960d58c2259884408e0a5d1813529eec47d84ec5fc0976282d40e5784d4a25',
 'core/reconstruction/reconstruction_dense_artifact_preflight.py': '7b953e9017d74f48a31b1b05509c3452f7eac7bb6d2591b595d885e4e5aafc79',
 'app/reconstruction_dense_final_handoff.py': 'cf761c3854d28707d56d9924e801d9fe89075f7183f5c04e0fa092baf6eb2490',
 'tests/conftest.py': '923e20444d9e8c942a11f4503401f1e5cd845abae1e61f8b4e32f121be3df810',
}
for name, expected in checks.items():
    f = Path(name)
    if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != expected:
        raise SystemExit('I6_7E2B1_SOURCE_DRIFT=' + name)
if Path('core/reconstruction/reconstruction_dense_final_handoff.py').exists():
    raise SystemExit('I6_7E2B1_ARCHITECTURE_DRIFT=misplaced core module')
print('I6_7E2B1_FROZEN_AND_E2A_SOURCE=PASS')
PY
ANCHOR_RC=$?; echo "I6_7E2B1_ANCHOR_RC=$ANCHOR_RC"; if (( ANCHOR_RC != 0 )); then exit "$ANCHOR_RC"; fi
python -m py_compile app/reconstruction_dense_final_admission.py tests/unit/test_i6_7e2b_final_dense_admission.py tests/integration/test_i6_7e2b_explicit_final_dense_admission.py
COMPILE_RC=$?; echo "I6_7E2B1_COMPILE_RC=$COMPILE_RC"; if (( COMPILE_RC != 0 )); then exit "$COMPILE_RC"; fi
bash -n scripts/run_i6_7e2b1_tests.sh
BASH_RC=$?; echo "I6_7E2B1_BASH_RC=$BASH_RC"; if (( BASH_RC != 0 )); then exit "$BASH_RC"; fi
python -m pytest -q tests/unit/test_i6_7e2b_final_dense_admission.py tests/integration/test_i6_7e2b_explicit_final_dense_admission.py tests/unit/test_phase8_7g4_readiness.py::test_reconstruction_core_remains_ui_independent
FOCUSED_RC=$?; echo "I6_7E2B1_FOCUSED_RC=$FOCUSED_RC"; if (( FOCUSED_RC != 0 )); then exit "$FOCUSED_RC"; fi
if [[ "$MODE" == closure ]]; then
  python -m pytest -q tests/unit/test_i6_7e2a_sparse_dense_handoff_gate.py tests/integration/test_i6_7e2a_completed_provider_candidate.py tests/unit/test_i6_7e1_dense_artifact_preflight.py tests/unit/test_i6_7d_progressive_dense_viewer_projection.py tests/stress/test_i6_7d_progressive_dense_viewer_projection_stress.py tests/unit/test_i6_7c_progressive_dense_transport.py tests/unit/test_i6_7c_progressive_dense_buffer.py tests/unit/test_i6_7b_progressive_dense_emission.py tests/unit/test_i6_7a_progressive_dense_snapshot_contract.py
  PREDECESSOR_RC=$?; echo "I6_7E2B1_PREDECESSOR_RC=$PREDECESSOR_RC"; if (( PREDECESSOR_RC != 0 )); then exit "$PREDECESSOR_RC"; fi
  python - <<'PY'
import shutil
from pathlib import Path
free = shutil.disk_usage(Path.cwd()).free / 2**30
print(f'FREE_DISK_GIB={free:.3f}')
if free < 8.0 + 1/16:
    raise SystemExit('STORAGE_PREFLIGHT=BLOCKED; preserve evidence and recover space')
print('STORAGE_PREFLIGHT=PASS')
PY
  STORAGE_RC=$?; echo "I6_7E2B1_STORAGE_RC=$STORAGE_RC"; if (( STORAGE_RC != 0 )); then exit "$STORAGE_RC"; fi
  python -m pytest -q tests/unit
  FULL_UNIT_RC=$?; echo "I6_7E2B1_FULL_UNIT_RC=$FULL_UNIT_RC"; if (( FULL_UNIT_RC != 0 )); then exit "$FULL_UNIT_RC"; fi
fi
echo 'I6_7E2B1_TEST_RC=0'
