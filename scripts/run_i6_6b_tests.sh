#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED=(
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

WORKER_PREDECESSOR_REGRESSION=(
  tests/unit/test_phase9_6e21_offline_progress_fsm.py
  tests/integration/test_reconstruction_pycolmap_worker_lifecycle.py
)

SFM_PREDECESSOR_REGRESSION=(
  tests/unit/test_reconstruction_sfm_contracts.py
  tests/unit/test_reconstruction_sfm_coordinates.py
  tests/unit/test_reconstruction_sfm_workspace.py
  tests/unit/test_reconstruction_sfm_worker_protocol.py
  tests/unit/test_reconstruction_sfm_worker_hardening.py
  tests/unit/test_reconstruction_sfm_pycolmap.py
  tests/unit/test_reconstruction_sfm_mock.py
  tests/unit/test_reconstruction_sfm_quality.py
  tests/integration/test_reconstruction_sfm_foundation_lifecycle.py
  tests/integration/test_reconstruction_recorded_video_sfm_hardening.py
)

I6_5_PREDECESSOR_REGRESSION=(
  tests/unit/test_i6_5a_reconstruction_frame_view_plan_contract.py
  tests/integration/test_i6_5a_frame_view_plan_contract_integration.py
  tests/stress/test_i6_5a_frame_view_plan_contract_stress.py
  tests/unit/test_i6_5b_deterministic_frame_view_planning.py
  tests/unit/test_i6_5b_frame_view_planning_static.py
  tests/integration/test_i6_5b_deterministic_frame_view_planning_integration.py
  tests/stress/test_i6_5b_deterministic_frame_view_planning_stress.py
  tests/unit/test_i6_5c_reconstruction_frame_view_plan_state.py
  tests/unit/test_i6_5c_controller_frame_view_plan_contract_static.py
  tests/stress/test_i6_5c_frame_view_plan_state_stress.py
  tests/unit/test_i6_5d_frame_view_plan_presentation.py
  tests/unit/test_i6_5d_frame_view_resource_binding.py
  tests/unit/test_i6_5d_reconstruction_page_frame_view_static.py
)

I6_4_BOUNDARY_REGRESSION=(
  tests/unit/test_i6_4a_provider_availability_selection_contract.py
  tests/integration/test_i6_4a_provider_availability_selection_contract.py
  tests/stress/test_i6_4a_provider_availability_selection_contract_stress.py
  tests/unit/test_i6_4c_production_auto_qualification_boundary.py
  tests/unit/test_i6_4c_controller_auto_gate_contract_static.py
  tests/stress/test_i6_4c_production_auto_qualification_boundary_stress.py
)

FOCUSED_RC=0; FOCUSED_COUNT=0
I66A_RC=0; I66A_COUNT=0
WORKER_RC=0; WORKER_COUNT=0
SFM_RC=0; SFM_COUNT=0
I65_RC=0; I65_COUNT=0
I64_RC=0; I64_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
IMPORT_RC=0; COMPILE_RC=0; EXTERNAL_PYCOLMAP_PROBE_RC=0

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
import importlib.util
from pathlib import Path
import sys

from core.reconstruction.reconstruction_progressive_sfm_contract import ProgressiveSfmSnapshot

root = Path.cwd()
helper_path = root / 'scripts/reconstruction_progressive_sfm_emission.py'
worker_path = root / 'scripts/reconstruction_pycolmap_worker.py'

spec = importlib.util.spec_from_file_location('i6_6b_helper_probe', helper_path)
assert spec is not None and spec.loader is not None
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)

spec = importlib.util.spec_from_file_location('i6_6b_worker_probe', worker_path)
assert spec is not None and spec.loader is not None
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)

assert helper.PROGRESSIVE_SFM_EVENT_PREFIX == 'DEPTH3D_SFM_PROGRESSIVE '
assert helper.PROGRESSIVE_MAX_CAMERA_SAMPLES == 64
assert helper.PROGRESSIVE_MAX_LANDMARK_SAMPLES == 4096
assert helper.mapping_snapshot_frame_interval(128) == 11
assert worker.EVENT_PREFIX == 'EIA_SFM_EVENT '
assert hasattr(worker, '_ProgressiveSfmWorkerEmitter')
print('I6.6B progressive worker emission import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    scripts/reconstruction_progressive_sfm_emission.py \
    scripts/reconstruction_pycolmap_worker.py \
    tests/unit/test_i6_6b_progressive_sfm_worker_emission.py \
    tests/unit/test_i6_6b_pycolmap_worker_progressive_static.py \
    tests/integration/test_i6_6b_progressive_sfm_worker_emission_contract.py \
    tests/stress/test_i6_6b_progressive_sfm_worker_emission_stress.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

probe_external_pycolmap() {
  local pycolmap_python
  pycolmap_python="${DEPTH3D_RECONSTRUCTION_PYCOLMAP_PYTHON:-/home/eiatech/anaconda3/envs/eia-pycolmap-probe/bin/python3.11}"
  if [[ ! -x "$pycolmap_python" ]]; then
    echo "I6.6B external PyCOLMAP probe: FAIL (interpreter not found: $pycolmap_python)"
    EXTERNAL_PYCOLMAP_PROBE_RC=2
    echo "EXTERNAL_PYCOLMAP_PROBE_RC=$EXTERNAL_PYCOLMAP_PROBE_RC"
    return
  fi
  "$pycolmap_python" - <<'PY'
import pycolmap

assert str(pycolmap.__version__) == '4.1.1', pycolmap.__version__
options = pycolmap.IncrementalPipelineOptions()
assert hasattr(options, 'snapshot_path')
assert hasattr(options, 'snapshot_frames_freq')
assert int(options.snapshot_frames_freq) == 0
print('I6.6B external PyCOLMAP 4.1.1 native snapshot capability probe: PASS')
PY
  EXTERNAL_PYCOLMAP_PROBE_RC=$?
  echo "EXTERNAL_PYCOLMAP_PROBE_RC=$EXTERNAL_PYCOLMAP_PROBE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_RC FOCUSED_COUNT FOCUSED "${FOCUSED[@]}"
  probe_import
  probe_compile
  probe_external_pycolmap
}

run_predecessors() {
  run_pytest_count I66A_RC I66A_COUNT I6_6A_REGRESSION "${I6_6A_REGRESSION[@]}"
  run_pytest_count WORKER_RC WORKER_COUNT WORKER_PREDECESSOR_REGRESSION "${WORKER_PREDECESSOR_REGRESSION[@]}"
  run_pytest_count SFM_RC SFM_COUNT SFM_PREDECESSOR_REGRESSION "${SFM_PREDECESSOR_REGRESSION[@]}"
  run_pytest_count I65_RC I65_COUNT I6_5_PREDECESSOR_REGRESSION "${I6_5_PREDECESSOR_REGRESSION[@]}"
  run_pytest_count I64_RC I64_COUNT I6_4_BOUNDARY_REGRESSION "${I6_4_BOUNDARY_REGRESSION[@]}"
}

run_full_unit() {
  run_pytest_count FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused) run_focused ;;
  predecessors) run_predecessors ;;
  all) run_focused; run_predecessors ;;
  full-unit) run_full_unit ;;
  closure) run_focused; run_predecessors; run_full_unit ;;
  *) echo "usage: $0 [focused|predecessors|all|full-unit|closure] [receipt.json]" >&2; exit 2 ;;
esac

TEST_RC=0
case "$MODE" in
  focused)
    for rc in "$FOCUSED_RC" "$IMPORT_RC" "$COMPILE_RC" "$EXTERNAL_PYCOLMAP_PROBE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  predecessors)
    for rc in "$I66A_RC" "$WORKER_RC" "$SFM_RC" "$I65_RC" "$I64_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  all)
    for rc in "$FOCUSED_RC" "$I66A_RC" "$WORKER_RC" "$SFM_RC" "$I65_RC" "$I64_RC" "$IMPORT_RC" "$COMPILE_RC" "$EXTERNAL_PYCOLMAP_PROBE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || TEST_RC=1 ;;
  closure)
    for rc in "$FOCUSED_RC" "$I66A_RC" "$WORKER_RC" "$SFM_RC" "$I65_RC" "$I64_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC" "$EXTERNAL_PYCOLMAP_PROBE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_RC" "$FOCUSED_COUNT" \
    "$I66A_RC" "$I66A_COUNT" \
    "$WORKER_RC" "$WORKER_COUNT" \
    "$SFM_RC" "$SFM_COUNT" \
    "$I65_RC" "$I65_COUNT" \
    "$I64_RC" "$I64_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" "$EXTERNAL_PYCOLMAP_PROBE_RC" <<'PY'
import json, sys
(
    path, mode, test_rc,
    focused_rc, focused_count,
    i66a_rc, i66a_count,
    worker_rc, worker_count,
    sfm_rc, sfm_count,
    i65_rc, i65_count,
    i64_rc, i64_count,
    full_rc, full_count,
    import_rc, compile_rc, external_probe_rc,
) = sys.argv[1:]
payload = {
    'schema_version': 1,
    'phase': 'I6.6B_WORKER_PROGRESSIVE_SNAPSHOT_EMISSION_AND_THROTTLING',
    'mode': mode,
    'status': 'pass' if int(test_rc) == 0 else 'fail',
    'pass': int(test_rc) == 0,
    'manual_ui_validation_required': False,
    'real_pycolmap_progressive_emission_validation_required_before_freeze': True,
    'counts': {
        'focused_tests': int(focused_count),
        'i6_6a_regression_tests': int(i66a_count),
        'worker_predecessor_regression_tests': int(worker_count),
        'sfm_predecessor_regression_tests': int(sfm_count),
        'i6_5_predecessor_regression_tests': int(i65_count),
        'i6_4_boundary_regression_tests': int(i64_count),
        'full_unit_tests': int(full_count),
    },
    'return_codes': {
        'focused': int(focused_rc),
        'i6_6a_regression': int(i66a_rc),
        'worker_predecessor_regression': int(worker_rc),
        'sfm_predecessor_regression': int(sfm_rc),
        'i6_5_predecessor_regression': int(i65_rc),
        'i6_4_boundary_regression': int(i64_rc),
        'full_unit': int(full_rc),
        'import_probe': int(import_rc),
        'compile_probe': int(compile_rc),
        'external_pycolmap_probe': int(external_probe_rc),
    },
    'scope': {
        'worker_progressive_emission_added': True,
        'native_pycolmap_snapshot_facility_reused': True,
        'temporary_native_snapshots_removed_after_consumption': True,
        'bounded_camera_preview_limit': 64,
        'bounded_landmark_preview_limit': 4096,
        'mapping_snapshot_target_count': 12,
        'minimum_emit_interval_seconds': 0.75,
        'distinct_progressive_stdout_channel': 'DEPTH3D_SFM_PROGRESSIVE ',
        'existing_worker_event_protocol_modified': False,
        'controller_consumption_added': False,
        'ui_modified': False,
        'viewport_modified': False,
        'worker_request_schema_modified': False,
        'final_result_schema_modified': False,
        'sfm_scientific_threshold_modified': False,
        'feature_extraction_modified': False,
        'matching_modified': False,
        'bundle_adjustment_modified': False,
        'model_selection_modified': False,
        'progressive_failure_blocks_final_sfm': False,
        'snapshot_provisional': True,
        'final_result_admitted_by_progressive_path': False,
        'persistent_canonical_scene_created': False,
        'dense_output_added': False,
        'production_auto_activated': False,
        'frame_view_replanning_during_sfm': False,
        'pycolmap_camera_authority_preserved': True,
        'metric_scale_claimed': False,
        'measurement_authoritative': False,
        'certified_metrology_claimed': False,
    },
}
open(path, 'w', encoding='utf-8').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(f'I6.6B receipt={path}')
PY
fi

echo "I6_6B_TEST_RC=$TEST_RC"
exit "$TEST_RC"
