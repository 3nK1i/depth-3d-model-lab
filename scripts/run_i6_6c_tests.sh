#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED_PURE=(
  tests/unit/test_i6_6c_progressive_sfm_transport.py
  tests/unit/test_i6_6c_progressive_sfm_buffer.py
  tests/unit/test_i6_6c_progressive_sfm_transport_static.py
  tests/integration/test_i6_6c_progressive_sfm_backend_transport.py
  tests/stress/test_i6_6c_progressive_sfm_backpressure_stress.py
)

FOCUSED_QT=(
  tests/integration/test_i6_6c_progressive_sfm_controller_transport.py
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

FOCUSED_PURE_RC=0; FOCUSED_PURE_COUNT=0
FOCUSED_QT_RC=0; FOCUSED_QT_COUNT=0
I66B_RC=0; I66B_COUNT=0
I66A_RC=0; I66A_COUNT=0
WORKER_RC=0; WORKER_COUNT=0
SFM_RC=0; SFM_COUNT=0
I65_RC=0; I65_COUNT=0
I64_RC=0; I64_COUNT=0
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
from app.reconstruction_progressive_sfm_state import ProgressiveSfmSnapshotBuffer
from core.reconstruction.reconstruction_progressive_sfm_transport import (
    PROGRESSIVE_SFM_EVENT_PREFIX,
    parse_progressive_sfm_worker_line,
    progressive_sfm_transport_line,
)
from tests.i6_6c_helpers import SESSION_ID, make_snapshot

buffer = ProgressiveSfmSnapshotBuffer()
buffer.reset(SESSION_ID)
first = make_snapshot(revision=1)
latest = make_snapshot(revision=4)
assert parse_progressive_sfm_worker_line(progressive_sfm_transport_line(first)) == first
assert buffer.offer(first).value == 'accepted'
assert buffer.offer(latest).value == 'coalesced'
assert buffer.take_pending() == latest
assert buffer.stats.coalesced == 1
assert buffer.stats.delivered == 1
assert PROGRESSIVE_SFM_EVENT_PREFIX == 'DEPTH3D_SFM_PROGRESSIVE '
print('I6.6C progressive SfM transport/backpressure import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_progressive_sfm_transport.py \
    app/reconstruction_progressive_sfm_state.py \
    core/reconstruction/sfm_pycolmap.py \
    scripts/run_reconstruction_offline_job.py \
    app/offline_reconstruction_controller.py \
    app/reconstruction_controller.py \
    scripts/run_i6_6c_progressive_transport_smoke.py \
    tests/i6_6c_helpers.py \
    tests/unit/test_i6_6c_progressive_sfm_transport.py \
    tests/unit/test_i6_6c_progressive_sfm_buffer.py \
    tests/unit/test_i6_6c_progressive_sfm_transport_static.py \
    tests/integration/test_i6_6c_progressive_sfm_backend_transport.py \
    tests/integration/test_i6_6c_progressive_sfm_controller_transport.py \
    tests/stress/test_i6_6c_progressive_sfm_backpressure_stress.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_PURE_RC FOCUSED_PURE_COUNT FOCUSED_PURE "${FOCUSED_PURE[@]}"
  probe_import
  probe_compile
  run_pytest_count FOCUSED_QT_RC FOCUSED_QT_COUNT FOCUSED_QT "${FOCUSED_QT[@]}"
}

run_predecessors() {
  run_pytest_count I66B_RC I66B_COUNT I6_6B_REGRESSION "${I6_6B_REGRESSION[@]}"
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
    for rc in "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  predecessors)
    for rc in "$I66B_RC" "$I66A_RC" "$WORKER_RC" "$SFM_RC" "$I65_RC" "$I64_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  all)
    for rc in "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I66B_RC" "$I66A_RC" "$WORKER_RC" "$SFM_RC" "$I65_RC" "$I64_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || TEST_RC=1 ;;
  closure)
    for rc in "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I66B_RC" "$I66A_RC" "$WORKER_RC" "$SFM_RC" "$I65_RC" "$I64_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done
    ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_PURE_RC" "$FOCUSED_PURE_COUNT" \
    "$FOCUSED_QT_RC" "$FOCUSED_QT_COUNT" \
    "$I66B_RC" "$I66B_COUNT" \
    "$I66A_RC" "$I66A_COUNT" \
    "$WORKER_RC" "$WORKER_COUNT" \
    "$SFM_RC" "$SFM_COUNT" \
    "$I65_RC" "$I65_COUNT" \
    "$I64_RC" "$I64_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json, sys
(
    path, mode, test_rc,
    pure_rc, pure_count, qt_rc, qt_count,
    i66b_rc, i66b_count, i66a_rc, i66a_count,
    worker_rc, worker_count, sfm_rc, sfm_count,
    i65_rc, i65_count, i64_rc, i64_count,
    full_rc, full_count, import_rc, compile_rc,
) = sys.argv[1:]
payload = {
    'schema_version': 1,
    'phase': 'I6.6C_CONTROLLER_TRANSPORT_SESSION_ISOLATION_AND_BACKPRESSURE',
    'mode': mode,
    'status': 'pass' if int(test_rc) == 0 else 'fail',
    'pass': int(test_rc) == 0,
    'manual_ui_validation_required': False,
    'real_progressive_transport_validation_required_before_freeze': True,
    'counts': {
        'focused_pure_tests': int(pure_count),
        'focused_qt_tests': int(qt_count),
        'i6_6b_regression_tests': int(i66b_count),
        'i6_6a_regression_tests': int(i66a_count),
        'worker_predecessor_regression_tests': int(worker_count),
        'sfm_predecessor_regression_tests': int(sfm_count),
        'i6_5_predecessor_regression_tests': int(i65_count),
        'i6_4_boundary_regression_tests': int(i64_count),
        'full_unit_tests': int(full_count),
    },
    'return_codes': {
        'focused_pure': int(pure_rc),
        'focused_qt': int(qt_rc),
        'i6_6b_regression': int(i66b_rc),
        'i6_6a_regression': int(i66a_rc),
        'worker_predecessor_regression': int(worker_rc),
        'sfm_predecessor_regression': int(sfm_rc),
        'i6_5_predecessor_regression': int(i65_rc),
        'i6_4_boundary_regression': int(i64_rc),
        'full_unit': int(full_rc),
        'import_probe': int(import_rc),
        'compile_probe': int(compile_rc),
    },
    'scope': {
        'progressive_channel_consumed_by_application_runtime': True,
        'dedicated_channel_separate_from_offline_status_events': True,
        'i6_6a_contract_remains_authority': True,
        'i6_6b_worker_modified': False,
        'latest_only_pending_buffer': True,
        'bounded_pending_snapshot_count': 1,
        'coalesce_interval_ms': 50,
        'revision_gaps_allowed': True,
        'stale_duplicate_revision_rejected': True,
        'cross_session_snapshot_rejected': True,
        'cross_request_snapshot_rejected': True,
        'malformed_snapshot_fail_soft': True,
        'finalizing_snapshot_flushes_immediately': True,
        'terminal_result_supersedes_provisional_state': True,
        'viewport_modified': False,
        'ui_modified': False,
        'progressive_dense_visualization_added': False,
        'provider_selection_effect': False,
        'production_auto_activated': False,
        'frame_view_replanning_during_sfm': False,
        'final_result_schema_modified': False,
        'pycolmap_camera_authority_preserved': True,
        'metric_scale_claimed': False,
        'measurement_authoritative': False,
        'certified_metrology_claimed': False,
    },
}
open(path, 'w', encoding='utf-8').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(f'I6.6C receipt={path}')
PY
fi

echo "I6_6C_TEST_RC=$TEST_RC"
exit "$TEST_RC"
