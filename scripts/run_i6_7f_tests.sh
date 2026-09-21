#!/usr/bin/env bash
# I6.7F complete admitted RGB point viewer + explicit dense progress; no SfM/MVS.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
MODE="${1:-focused}"
if [[ "$MODE" != focused && "$MODE" != closure ]]; then
  echo 'Usage: bash scripts/run_i6_7f_tests.sh focused|closure' >&2
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
  'core/reconstruction/reconstruction_dense_artifact_preflight.py': '7b953e9017d74f48a31b1b05509c3452f7eac7bb6d2591b595d885e4e5aafc79',
  'app/reconstruction_dense_final_handoff.py': 'cf761c3854d28707d56d9924e801d9fe89075f7183f5c04e0fa092baf6eb2490',
  'app/reconstruction_dense_final_admission.py': '7168e7b6461dad790ca8513ebc114972440053b3ed05afdda707b2f5155c5987',
  'app/reconstruction_dense_production_worker.py': '7b3db38382300ac30476dc35e33e9c1ce4cfdcafd53c1d759b10fc6e1b4ec66d',
  'app/offline_reconstruction_controller.py': 'bd4f9c1cfe1739759d4b95d736cf376f36cffd04288ffb615e133fbb8358aa79',
}
for name, expected in checks.items():
    path = Path(name)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('I6_7F_FROZEN_SOURCE_DRIFT=' + name)
if Path('core/reconstruction/reconstruction_dense_final_handoff.py').exists():
    raise SystemExit('I6_7F_CORE_UI_BOUNDARY=FAIL')
assert 'I67C_MAX_GEOMETRY_PREVIEW_POINTS' in Path('app/reconstruction_progressive_dense_viewer.py').read_text()
source = Path('app/widgets/viewport/vispy_canvas.py').read_text()
assert 'def show_progressive_dense_reconstruction(self, frame)' in source
assert 'def show_admitted_full_dense(self, frame)' in source
assert 'def clear_admitted_full_dense(' in source
assert 'def show_admitted_full_dense' in Path('tests/conftest.py').read_text()
print('I6_7F_FROZEN_SOURCE_AND_INDEPENDENT_FULL_VIEW=PASS')
print('I6_7F_PRODUCTION_AUTO_ACTIVATED=NO')
print('I6_7F_PROVISIONAL_PREVIEW_CAP_CHANGED=NO')
PY
SCOPE_RC=$?; echo "I6_7F_SCOPE_RC=$SCOPE_RC"; (( SCOPE_RC == 0 )) || exit "$SCOPE_RC"
python -m py_compile \
  app/widgets/reconstruction_page.py app/widgets/viewport/vispy_canvas.py \
  app/reconstruction_full_dense_viewer.py tests/conftest.py \
  tests/unit/test_i6_7f_complete_dense_frame.py \
  tests/ui/test_i6_7f_dense_progress_full_viewer_ui.py \
  scripts/run_i6_7f_full_dense_viewer_demo.py \
  app/reconstruction_dense_production_worker.py \
  tests/unit/test_i6_7e_south_sfm_input_bound.py
COMPILE_RC=$?; echo "I6_7F_COMPILE_RC=$COMPILE_RC"; (( COMPILE_RC == 0 )) || exit "$COMPILE_RC"
bash -n scripts/run_i6_7f_tests.sh
BASH_RC=$?; echo "I6_7F_BASH_RC=$BASH_RC"; (( BASH_RC == 0 )) || exit "$BASH_RC"
python -m pytest -q \
  tests/unit/test_i6_7e_south_sfm_input_bound.py \
  tests/unit/test_i6_7f_complete_dense_frame.py \
  tests/ui/test_i6_7f_dense_progress_full_viewer_ui.py \
  tests/ui/test_i6_7e_explicit_dense_production_gui.py \
  tests/ui/test_i6_7d_incremental_dense_viewer_ui.py \
  tests/unit/test_phase8_7g4_readiness.py::test_reconstruction_core_remains_ui_independent
FOCUSED_RC=$?; echo "I6_7F_FOCUSED_RC=$FOCUSED_RC"; (( FOCUSED_RC == 0 )) || exit "$FOCUSED_RC"
if [[ "$MODE" == closure ]]; then
  python -m pytest -q \
    tests/unit/test_i6_7e_production_dense_completion_verifier.py \
    tests/integration/test_i6_7e_posix_process_tree_cancellation.py \
    tests/integration/test_i6_7e2b2a_isolated_dense_worker.py \
    tests/unit/test_i6_7e2b_final_dense_admission.py \
    tests/integration/test_i6_7e2b_explicit_final_dense_admission.py \
    tests/unit/test_i6_7e2a_sparse_dense_handoff_gate.py \
    tests/integration/test_i6_7e2a_completed_provider_candidate.py \
    tests/unit/test_i6_7e1_dense_artifact_preflight.py \
    tests/unit/test_i6_7d_progressive_dense_viewer_projection.py \
    tests/unit/test_i6_7c_progressive_dense_transport.py \
    tests/unit/test_i6_7c_progressive_dense_buffer.py \
    tests/unit/test_i6_7b_progressive_dense_emission.py \
    tests/unit/test_i6_7a_progressive_dense_snapshot_contract.py
  PREDECESSOR_RC=$?; echo "I6_7F_PREDECESSOR_RC=$PREDECESSOR_RC"
  (( PREDECESSOR_RC == 0 )) || exit "$PREDECESSOR_RC"
  python - <<'PY'
from pathlib import Path
import shutil
free = shutil.disk_usage(Path.cwd()).free / 2**30
print(f'FREE_DISK_GIB={free:.3f}')
if free < 8 + 1/16:
    raise SystemExit('STORAGE_PREFLIGHT=BLOCKED; preserve all prior evidence')
print('STORAGE_PREFLIGHT=PASS')
PY
  STORAGE_RC=$?; echo "I6_7F_STORAGE_RC=$STORAGE_RC"; (( STORAGE_RC == 0 )) || exit "$STORAGE_RC"
  python -m pytest -q tests/unit
  UNIT_RC=$?; echo "I6_7F_FULL_UNIT_RC=$UNIT_RC"; (( UNIT_RC == 0 )) || exit "$UNIT_RC"
fi
echo 'I6_7F_TEST_RC=0'
