#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED=(
  tests/unit/test_i6_6a_progressive_sfm_snapshot_contract.py
  tests/integration/test_i6_6a_progressive_sfm_snapshot_transition.py
  tests/stress/test_i6_6a_progressive_sfm_snapshot_stress.py
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
SFM_RC=0; SFM_COUNT=0
I6_5_RC=0; I6_5_COUNT=0
I6_4_RC=0; I6_4_COUNT=0
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
import numpy as np
from core.reconstruction.reconstruction_progressive_sfm_contract import (
    I66A_CAMERA_AUTHORITY_OWNER,
    I66A_PYCOLMAP_BACKEND_ID,
    ProgressiveSfmCameraSample,
    ProgressiveSfmGeometryPreview,
    ProgressiveSfmSnapshot,
    ProgressiveSfmStage,
    ProgressiveSfmStatistics,
    validate_progressive_sfm_snapshot_transition,
)

pose = np.eye(4, dtype=np.float64)
first = ProgressiveSfmSnapshot(
    session_id='11111111-2222-4333-8444-555555555555',
    revision=1,
    timestamp_utc='2026-09-17T12:00:00+00:00',
    elapsed_ns=1,
    stage=ProgressiveSfmStage.MAPPING,
    stage_detail='probe',
    backend_id=I66A_PYCOLMAP_BACKEND_ID,
    backend_version='1',
    request_fingerprint_sha256='e' * 64,
    statistics=ProgressiveSfmStatistics(
        input_views=12,
        registered_cameras=2,
        sparse_landmarks=0,
        observations=0,
    ),
    progress_fraction=0.68,
    geometry=ProgressiveSfmGeometryPreview(
        camera_samples=(ProgressiveSfmCameraSample(
            backend_image_id=1,
            backend_camera_id=0,
            source_image_name='image_0001.jpg',
            T_reconstruction_from_camera=pose,
        ),),
    ),
)
second = ProgressiveSfmSnapshot.from_payload({
    **{k: v for k, v in first.as_dict().items() if k != 'snapshot_sha256'},
    'revision': 3,
    'timestamp_utc': '2026-09-17T12:00:01+00:00',
    'elapsed_ns': 2,
})
validate_progressive_sfm_snapshot_transition(first, second)
assert first.as_dict()['authority']['camera_pose_authority_owner'] == I66A_CAMERA_AUTHORITY_OWNER
assert first.as_dict()['authority']['provisional'] is True
assert first.as_dict()['authority']['final_result_admitted'] is False
assert first.as_dict()['authority']['persistent_canonical_scene'] is False
assert len(first.snapshot_sha256) == 64
print('I6.6A progressive SfM snapshot/authority import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_progressive_sfm_contract.py \
    tests/unit/test_i6_6a_progressive_sfm_snapshot_contract.py \
    tests/integration/test_i6_6a_progressive_sfm_snapshot_transition.py \
    tests/stress/test_i6_6a_progressive_sfm_snapshot_stress.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_RC FOCUSED_COUNT FOCUSED "${FOCUSED[@]}"
  probe_import
  probe_compile
}

run_predecessors() {
  run_pytest_count SFM_RC SFM_COUNT SFM_PREDECESSOR_REGRESSION "${SFM_PREDECESSOR_REGRESSION[@]}"
  run_pytest_count I6_5_RC I6_5_COUNT I6_5_PREDECESSOR_REGRESSION "${I6_5_PREDECESSOR_REGRESSION[@]}"
  run_pytest_count I6_4_RC I6_4_COUNT I6_4_BOUNDARY_REGRESSION "${I6_4_BOUNDARY_REGRESSION[@]}"
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
  focused) for rc in "$FOCUSED_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  predecessors) for rc in "$SFM_RC" "$I6_5_RC" "$I6_4_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  all) for rc in "$FOCUSED_RC" "$SFM_RC" "$I6_5_RC" "$I6_4_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || TEST_RC=1 ;;
  closure) for rc in "$FOCUSED_RC" "$SFM_RC" "$I6_5_RC" "$I6_4_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_RC" "$FOCUSED_COUNT" \
    "$SFM_RC" "$SFM_COUNT" \
    "$I6_5_RC" "$I6_5_COUNT" \
    "$I6_4_RC" "$I6_4_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json, sys
(
    path, mode, test_rc,
    focused_rc, focused_count,
    sfm_rc, sfm_count,
    i65_rc, i65_count,
    i64_rc, i64_count,
    full_rc, full_count,
    import_rc, compile_rc,
) = sys.argv[1:]
payload = {
    'schema_version': 1,
    'phase': 'I6.6A_PROGRESSIVE_SFM_SNAPSHOT_AND_AUTHORITY_CONTRACT',
    'mode': mode,
    'status': 'pass' if int(test_rc) == 0 else 'fail',
    'pass': int(test_rc) == 0,
    'manual_ui_validation_required': False,
    'counts': {
        'focused_tests': int(focused_count),
        'sfm_predecessor_regression_tests': int(sfm_count),
        'i6_5_predecessor_regression_tests': int(i65_count),
        'i6_4_boundary_regression_tests': int(i64_count),
        'full_unit_tests': int(full_count),
    },
    'return_codes': {
        'focused': int(focused_rc),
        'sfm_predecessor_regression': int(sfm_rc),
        'i6_5_predecessor_regression': int(i65_rc),
        'i6_4_boundary_regression': int(i64_rc),
        'full_unit': int(full_rc),
        'import_probe': int(import_rc),
        'compile_probe': int(compile_rc),
    },
    'scope': {
        'progressive_snapshot_contract_added': True,
        'bounded_geometry_preview_contract_added': True,
        'snapshot_digest_added': True,
        'revision_gaps_allowed_for_future_coalescing': True,
        'stale_or_duplicate_revision_rejected': True,
        'cross_session_transition_rejected': True,
        'cross_request_transition_rejected': True,
        'geometry_counts_required_monotonic': False,
        'snapshot_provisional': True,
        'final_result_admitted': False,
        'persistent_canonical_scene_created': False,
        'canonical_geometry_authoritative': False,
        'worker_modified': False,
        'worker_emission_added': False,
        'worker_protocol_modified': False,
        'controller_modified': False,
        'ui_modified': False,
        'viewport_modified': False,
        'dense_output_added': False,
        'provider_selection_effect': False,
        'production_auto_activated': False,
        'frame_view_replanning_during_sfm': False,
        'progressive_dense_visualization_added': False,
        'pycolmap_camera_authority_preserved': True,
        'metric_scale_claimed': False,
        'measurement_authoritative': False,
        'certified_metrology_claimed': False,
    },
}
open(path, 'w', encoding='utf-8').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(f'I6.6A receipt={path}')
PY
fi

echo "I6_6A_TEST_RC=$TEST_RC"
exit "$TEST_RC"
