#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED_PURE=(
  tests/unit/test_i6_6d_progressive_sfm_viewer_frame.py
  tests/unit/test_i6_6d_incremental_sparse_viewer_static.py
  tests/stress/test_i6_6d_progressive_sfm_viewer_projection_stress.py
)

FOCUSED_QT=(
  tests/ui/test_i6_6d_incremental_sparse_viewer_ui.py
)

I6_6C_REGRESSION=(
  tests/unit/test_i6_6c_progressive_sfm_transport.py
  tests/unit/test_i6_6c_progressive_sfm_buffer.py
  tests/unit/test_i6_6c_progressive_sfm_transport_static.py
  tests/integration/test_i6_6c_progressive_sfm_backend_transport.py
  tests/integration/test_i6_6c_progressive_sfm_controller_transport.py
  tests/stress/test_i6_6c_progressive_sfm_backpressure_stress.py
)

I6_6B_REGRESSION=(
  tests/unit/test_i6_6b_progressive_sfm_worker_emission.py
  tests/unit/test_i6_6b_pycolmap_worker_progressive_static.py
  tests/integration/test_i6_6b_progressive_sfm_worker_emission_contract.py
  tests/stress/test_i6_6b_progressive_sfm_worker_emission_stress.py
)

I6_6A_REGRESSION=(
  tests/unit/test_i6_6a_progressive_sfm_snapshot_contract.py
  tests/integration/test_i6_6a_progressive_sfm_snapshot_transition.py
  tests/stress/test_i6_6a_progressive_sfm_snapshot_stress.py
)

VIEWPORT_PURE_PREDECESSOR=(
  tests/unit/test_phase8_8c2_visualizer_contract.py
  tests/unit/test_phase8_8c3_contract.py
  tests/unit/test_phase8_8d1_contract.py
  tests/unit/test_phase8_8d1_offline_visualization.py
)

VIEWPORT_QT_PREDECESSOR=(
  tests/ui/test_phase8_8d1_offline_reconstruction_visualization.py
  tests/ui/test_viewport_controls.py
)

PAGE_QT_PREDECESSOR=(
  tests/ui/test_phase8_8b3_reconstruction_runtime_controls.py
  tests/ui/test_phase8_8b4_status_progress.py
  tests/ui/test_i6_1c_offline_workspace_ui.py
  tests/ui/test_i6_5d_frame_view_plan_ui.py
)

FOCUSED_PURE_RC=0; FOCUSED_PURE_COUNT=0
FOCUSED_QT_RC=0; FOCUSED_QT_COUNT=0
I66C_RC=0; I66C_COUNT=0
I66B_RC=0; I66B_COUNT=0
I66A_RC=0; I66A_COUNT=0
VIEWPORT_PURE_RC=0; VIEWPORT_PURE_COUNT=0
VIEWPORT_QT_RC=0; VIEWPORT_QT_COUNT=0
PAGE_QT_RC=0; PAGE_QT_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
IMPORT_RC=0; COMPILE_RC=0

run_pytest_count() {
  local rc_var="$1" count_var="$2" label="$3"; shift 3
  local tmp rc count
  tmp="$(mktemp)"
  pytest -q "$@" >"$tmp" 2>&1
  rc=$?
  cat "$tmp"
  count=0
  if [[ $rc -eq 0 ]]; then
    count="$(python - "$tmp" <<'PY'
import re, sys
text=open(sys.argv[1], encoding='utf-8', errors='replace').read()
print(sum(int(v) for v in re.findall(r'(\d+) passed', text)))
PY
)"
  fi
  rm -f "$tmp"
  printf '%s_RC=%s\n' "$label" "$rc"
  printf '%s_COUNT=%s\n' "$label" "$count"
  printf -v "$rc_var" '%s' "$rc"
  printf -v "$count_var" '%s' "$count"
}

probe_import() {
  python - <<'PY'
from app.reconstruction_progressive_sfm_viewer import (
    I66D_VIEWER_PROFILE,
    build_progressive_sfm_viewer_frame,
)
from core.reconstruction.reconstruction_progressive_sfm_contract import ProgressiveSfmStage
from tests.i6_6d_helpers import make_progressive_view_snapshot

frame = build_progressive_sfm_viewer_frame(
    make_progressive_view_snapshot(
        revision=2,
        camera_sample_count=4,
        landmark_sample_count=7,
        registered_camera_total=9,
        landmark_total=222,
        stage=ProgressiveSfmStage.FINALIZING,
    )
)
assert I66D_VIEWER_PROFILE == 'progressive_sparse_replace_v1'
assert frame.revision == 2
assert frame.provisional is True
assert frame.camera_matrices_reconstruction.shape == (4, 4, 4)
assert frame.landmark_points_reconstruction.shape == (7, 3)
assert frame.registered_camera_total == 9
assert frame.landmark_total == 222
assert frame.stage is ProgressiveSfmStage.FINALIZING
print('I6.6D progressive sparse viewer projection import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    app/reconstruction_progressive_sfm_viewer.py \
    app/reconstruction_progressive_sfm_viewer_binding.py \
    app/widgets/reconstruction_page.py \
    app/widgets/viewport/vispy_canvas.py \
    tests/conftest.py \
    tests/i6_6d_helpers.py \
    tests/unit/test_i6_6d_progressive_sfm_viewer_frame.py \
    tests/unit/test_i6_6d_incremental_sparse_viewer_static.py \
    tests/stress/test_i6_6d_progressive_sfm_viewer_projection_stress.py \
    tests/ui/test_i6_6d_incremental_sparse_viewer_ui.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_PURE_RC FOCUSED_PURE_COUNT FOCUSED_PURE "${FOCUSED_PURE[@]}"
  probe_import
  probe_compile
}

run_closure() {
  run_focused
  run_pytest_count FOCUSED_QT_RC FOCUSED_QT_COUNT FOCUSED_QT "${FOCUSED_QT[@]}"
  run_pytest_count I66C_RC I66C_COUNT I6_6C_REGRESSION "${I6_6C_REGRESSION[@]}"
  run_pytest_count I66B_RC I66B_COUNT I6_6B_REGRESSION "${I6_6B_REGRESSION[@]}"
  run_pytest_count I66A_RC I66A_COUNT I6_6A_REGRESSION "${I6_6A_REGRESSION[@]}"
  run_pytest_count VIEWPORT_PURE_RC VIEWPORT_PURE_COUNT VIEWPORT_PURE_PREDECESSOR "${VIEWPORT_PURE_PREDECESSOR[@]}"
  run_pytest_count VIEWPORT_QT_RC VIEWPORT_QT_COUNT VIEWPORT_QT_PREDECESSOR "${VIEWPORT_QT_PREDECESSOR[@]}"
  run_pytest_count PAGE_QT_RC PAGE_QT_COUNT PAGE_QT_PREDECESSOR "${PAGE_QT_PREDECESSOR[@]}"
  run_pytest_count FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused) run_focused ;;
  closure) run_closure ;;
  *) echo "usage: $0 [focused|closure] [receipt.json]" >&2; exit 2 ;;
esac

TEST_RC=0
for rc in \
  "$FOCUSED_PURE_RC" "$IMPORT_RC" "$COMPILE_RC" \
  "$FOCUSED_QT_RC" "$I66C_RC" "$I66B_RC" "$I66A_RC" \
  "$VIEWPORT_PURE_RC" "$VIEWPORT_QT_RC" "$PAGE_QT_RC" "$FULL_UNIT_RC"
do
  [[ "$rc" -eq 0 ]] || TEST_RC=1
done

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_PURE_COUNT" "$FOCUSED_QT_COUNT" "$I66C_COUNT" "$I66B_COUNT" "$I66A_COUNT" \
    "$VIEWPORT_PURE_COUNT" "$VIEWPORT_QT_COUNT" "$PAGE_QT_COUNT" "$FULL_UNIT_COUNT" \
    "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I66C_RC" "$I66B_RC" "$I66A_RC" \
    "$VIEWPORT_PURE_RC" "$VIEWPORT_QT_RC" "$PAGE_QT_RC" "$FULL_UNIT_RC" \
    "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json, sys
(
    path, mode, test_rc,
    focused_pure_count, focused_qt_count, i66c_count, i66b_count, i66a_count,
    viewport_pure_count, viewport_qt_count, page_qt_count, full_count,
    focused_pure_rc, focused_qt_rc, i66c_rc, i66b_rc, i66a_rc,
    viewport_pure_rc, viewport_qt_rc, page_qt_rc, full_rc,
    import_rc, compile_rc,
) = sys.argv[1:]
payload = {
    'schema_version': 1,
    'phase': 'I6.6D_INCREMENTAL_SPARSE_VIEWER_INTEGRATION',
    'mode': mode,
    'status': 'pass' if int(test_rc) == 0 else 'fail',
    'pass': int(test_rc) == 0,
    'manual_viewport_validation_required_before_freeze': True,
    'counts': {
        'focused_pure_tests': int(focused_pure_count),
        'focused_qt_tests': int(focused_qt_count),
        'i6_6c_regression_tests': int(i66c_count),
        'i6_6b_regression_tests': int(i66b_count),
        'i6_6a_regression_tests': int(i66a_count),
        'viewport_pure_predecessor_tests': int(viewport_pure_count),
        'viewport_qt_predecessor_tests': int(viewport_qt_count),
        'page_qt_predecessor_tests': int(page_qt_count),
        'full_unit_tests': int(full_count),
    },
    'return_codes': {
        'focused_pure': int(focused_pure_rc),
        'focused_qt': int(focused_qt_rc),
        'i6_6c_regression': int(i66c_rc),
        'i6_6b_regression': int(i66b_rc),
        'i6_6a_regression': int(i66a_rc),
        'viewport_pure_predecessor': int(viewport_pure_rc),
        'viewport_qt_predecessor': int(viewport_qt_rc),
        'page_qt_predecessor': int(page_qt_rc),
        'full_unit': int(full_rc),
        'import_probe': int(import_rc),
        'compile_probe': int(compile_rc),
    },
    'scope': {
        'controller_transport_modified': False,
        'i6_6b_worker_modified': False,
        'i6_6a_contract_modified': False,
        'progressive_sparse_viewer_bound': True,
        'progressive_snapshot_remains_provisional': True,
        'replace_only_geometry_updates': True,
        'historical_snapshot_geometry_accumulated': False,
        'progressive_camera_path_added': True,
        'viewport_refit_on_every_revision': False,
        'final_result_supersedes_provisional_view': True,
        'admitted_offline_sparse_result_path_modified': False,
        'persistent_canonical_scene_created': False,
        'progressive_dense_visualization_added': False,
        'provider_selection_effect': False,
        'production_auto_activated': False,
        'frame_view_replanning_during_sfm': False,
        'pycolmap_camera_authority_preserved': True,
        'metric_scale_claimed': False,
        'measurement_authoritative': False,
        'certified_metrology_claimed': False,
    },
}
open(path, 'w', encoding='utf-8').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(f'I6.6D receipt={path}')
PY
fi

echo "I6_6D_TEST_RC=$TEST_RC"
exit "$TEST_RC"
