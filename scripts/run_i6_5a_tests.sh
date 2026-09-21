#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED=(
  tests/unit/test_i6_5a_reconstruction_frame_view_plan_contract.py
  tests/integration/test_i6_5a_frame_view_plan_contract_integration.py
  tests/stress/test_i6_5a_frame_view_plan_contract_stress.py
)

ADAPTIVE_PREDECESSOR_REGRESSION=(
  tests/unit/test_i4_3f1_adaptive_view_selection.py
  tests/integration/test_i4_3f1_adaptive_view_selection_contract.py
  tests/stress/test_i4_3f1_adaptive_view_selection_stress.py
  tests/unit/test_i4_3f2_adaptive_production_planning.py
  tests/unit/test_i4_3f2_production_source_contract.py
  tests/integration/test_i4_3f2_production_wiring_contract.py
  tests/stress/test_i4_3f2_production_wiring_stress.py
  tests/unit/test_i4_3f3_adaptive_selection_closure.py
  tests/integration/test_i4_3f3_production_validation_contract.py
  tests/stress/test_i4_3f3_production_validation_stress.py
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
ADAPTIVE_RC=0; ADAPTIVE_COUNT=0
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
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    I65A_CAMERA_AUTHORITY_OWNER,
    FrameViewPlanMode,
    FrameViewPlanRequest,
    FrameViewPlanResolution,
    FrameViewPlanRealization,
    FrameViewRecord,
    FrameViewRecordDisposition,
    FrameViewSourceProvenance,
)

source = FrameViewSourceProvenance(
    source_id='probe-sequence',
    source_kind='image_sequence',
    source_uri='/probe/images',
    source_fingerprint_sha256='d' * 64,
    estimated_total_views=6,
    expected_frame_rate_hz=30.0,
)
resolution = FrameViewPlanResolution(
    source=source,
    request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
    effective_target_views=4,
    resource_target_views=4,
    reason_codes=('probe_adaptive_plan',),
)
records = tuple(
    FrameViewRecord(
        source_view_id=f'view-{i}',
        sequence_index=i,
        source_locator=f'/probe/images/{i}.jpg',
        source_timestamp_ns=i * 33333333,
        disposition=(FrameViewRecordDisposition.PLANNED if i in {0, 2, 4, 5} else FrameViewRecordDisposition.ELIGIBLE_NOT_PLANNED),
        reason_codes=('selected_by_effective_plan',) if i in {0, 2, 4, 5} else ('not_selected_by_effective_target',),
    )
    for i in range(6)
)
realized = FrameViewPlanRealization(resolution=resolution, records=records)
assert realized.planned_view_count == 4
assert realized.eligible_view_count == 6
assert len(resolution.plan_input_sha256) == 64
assert len(realized.plan_provenance_sha256) == 64
assert realized.as_dict()['authority']['camera_pose_authority_owner'] == I65A_CAMERA_AUTHORITY_OWNER
assert realized.as_dict()['authority']['planned_views_are_registered_cameras'] is False
print('I6.5A frame/view contract and provenance import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_frame_view_plan_contract.py \
    tests/unit/test_i6_5a_reconstruction_frame_view_plan_contract.py \
    tests/integration/test_i6_5a_frame_view_plan_contract_integration.py \
    tests/stress/test_i6_5a_frame_view_plan_contract_stress.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_RC FOCUSED_COUNT FOCUSED "${FOCUSED[@]}"
  probe_import
  probe_compile
}

run_predecessors() {
  run_pytest_count ADAPTIVE_RC ADAPTIVE_COUNT ADAPTIVE_PREDECESSOR_REGRESSION "${ADAPTIVE_PREDECESSOR_REGRESSION[@]}"
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
  predecessors) for rc in "$ADAPTIVE_RC" "$I6_4_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  all) for rc in "$FOCUSED_RC" "$ADAPTIVE_RC" "$I6_4_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || TEST_RC=1 ;;
  closure) for rc in "$FOCUSED_RC" "$ADAPTIVE_RC" "$I6_4_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_RC" "$FOCUSED_COUNT" \
    "$ADAPTIVE_RC" "$ADAPTIVE_COUNT" \
    "$I6_4_RC" "$I6_4_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json, sys
path, mode, test_rc, focused_rc, focused_count, adaptive_rc, adaptive_count, i64_rc, i64_count, full_rc, full_count, import_rc, compile_rc = sys.argv[1:]
payload = {
    'schema_version': 1,
    'phase': 'I6.5A_ADAPTIVE_FRAME_VIEW_CONTRACT_AND_PROVENANCE',
    'mode': mode,
    'status': 'pass' if int(test_rc) == 0 else 'fail',
    'pass': int(test_rc) == 0,
    'manual_ui_validation_required': False,
    'counts': {
        'focused_tests': int(focused_count),
        'adaptive_predecessor_regression_tests': int(adaptive_count),
        'i6_4_boundary_regression_tests': int(i64_count),
        'full_unit_tests': int(full_count),
    },
    'return_codes': {
        'focused': int(focused_rc),
        'adaptive_predecessor_regression': int(adaptive_rc),
        'i6_4_boundary_regression': int(i64_rc),
        'full_unit': int(full_rc),
        'import_probe': int(import_rc),
        'compile_probe': int(compile_rc),
    },
    'scope': {
        'source_provenance_contract_added': True,
        'requested_plan_contract_added': True,
        'effective_plan_contract_added': True,
        'per_view_realization_provenance_added': True,
        'deterministic_plan_digest_added': True,
        'deterministic_realization_digest_added': True,
        'requested_plan_is_effective_plan': False,
        'planned_view_is_registered_camera': False,
        'pycolmap_registration_outcome_claimed': False,
        'permanent_global_view_cap': None,
        'existing_adaptive_planner_modified': False,
        'ingestion_behavior_modified': False,
        'controller_modified': False,
        'ui_modified': False,
        'worker_modified': False,
        'provider_selection_effect': False,
        'provider_activation_authorized': False,
        'provider_execution_authorized': False,
        'production_auto_activated': False,
        'settings_mutation_authorized': False,
        'start_action_authorized': False,
        'scientific_threshold_modified': False,
        'production_default_modified': False,
        'progressive_sfm_visualization_added': False,
        'pycolmap_camera_authority_preserved': True,
        'metric_scale_claimed': False,
        'certified_metrology_claimed': False,
    },
}
open(path, 'w', encoding='utf-8').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(f'I6.5A receipt={path}')
PY
fi

echo "I6_5A_TEST_RC=$TEST_RC"
exit "$TEST_RC"
