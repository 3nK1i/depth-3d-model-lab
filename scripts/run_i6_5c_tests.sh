#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED_PURE=(
  tests/unit/test_i6_5c_reconstruction_frame_view_plan_state.py
  tests/unit/test_i6_5c_controller_frame_view_plan_contract_static.py
  tests/stress/test_i6_5c_frame_view_plan_state_stress.py
)
FOCUSED_QT=(
  tests/integration/test_i6_5c_controller_frame_view_plan.py
)
I6_5B_REGRESSION=(
  tests/unit/test_i6_5b_deterministic_frame_view_planning.py
  tests/unit/test_i6_5b_frame_view_planning_static.py
  tests/integration/test_i6_5b_deterministic_frame_view_planning_integration.py
  tests/stress/test_i6_5b_deterministic_frame_view_planning_stress.py
)
I6_5A_REGRESSION=(
  tests/unit/test_i6_5a_reconstruction_frame_view_plan_contract.py
  tests/integration/test_i6_5a_frame_view_plan_contract_integration.py
  tests/stress/test_i6_5a_frame_view_plan_contract_stress.py
)
I6_CONTROLLER_PREDECESSOR=(
  tests/integration/test_i6_4b_controller_provider_selection.py
  tests/integration/test_i6_4c_controller_production_auto_gate.py
)

FOCUSED_PURE_RC=0; FOCUSED_PURE_COUNT=0
FOCUSED_QT_RC=0; FOCUSED_QT_COUNT=0
I6_5B_RC=0; I6_5B_COUNT=0
I6_5A_RC=0; I6_5A_COUNT=0
I6_CONTROLLER_RC=0; I6_CONTROLLER_COUNT=0
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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tempfile

from app.reconstruction_frame_view_plan_state import (
    FrameViewPlanIntentOrigin,
    FrameViewPlanStateStatus,
    build_reconstruction_frame_view_plan_state,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.reconstruction_frame_view_plan_contract import FrameViewPlanMode, FrameViewPlanRequest

class Workflow(str, Enum):
    IMAGE_SEQUENCE='image_sequence'
@dataclass(frozen=True)
class Selection:
    workflow: Workflow
    ready: bool
    path: Path
    item_count: int

GIB=1024**3
with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    for index in range(8):
        (root/f'{index:03d}.jpg').write_bytes(b'x')
    state=build_reconstruction_frame_view_plan_state(
        source_selection=Selection(Workflow.IMAGE_SEQUENCE, True, root, 8),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        intent_origin=FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE,
        resources=AdaptiveSelectionResourceSnapshot(
            gpu_name='probe', total_vram_bytes=6*GIB, free_vram_bytes=2*GIB,
            available_ram_bytes=8*GIB, cuda_ready=True,
        ),
    )
    assert state.status is FrameViewPlanStateStatus.RESOLVED
    assert state.plan is not None
    assert len(state.planning_provenance_sha256)==64
    assert state.as_dict()['effects']['source_sampling_executed'] is False
    assert state.as_dict()['authority']['planned_views_are_registered_cameras'] is False
print('I6.5C frame/view controller-state import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    app/reconstruction_frame_view_plan_state.py \
    app/reconstruction_controller.py \
    tests/unit/test_i6_5c_reconstruction_frame_view_plan_state.py \
    tests/unit/test_i6_5c_controller_frame_view_plan_contract_static.py \
    tests/integration/test_i6_5c_controller_frame_view_plan.py \
    tests/stress/test_i6_5c_frame_view_plan_state_stress.py
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
  run_pytest_count I6_5B_RC I6_5B_COUNT I6_5B_REGRESSION "${I6_5B_REGRESSION[@]}"
  run_pytest_count I6_5A_RC I6_5A_COUNT I6_5A_REGRESSION "${I6_5A_REGRESSION[@]}"
  run_pytest_count I6_CONTROLLER_RC I6_CONTROLLER_COUNT I6_CONTROLLER_PREDECESSOR "${I6_CONTROLLER_PREDECESSOR[@]}"
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
  focused) for rc in "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  predecessors) for rc in "$I6_5B_RC" "$I6_5A_RC" "$I6_CONTROLLER_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  all) for rc in "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_5B_RC" "$I6_5A_RC" "$I6_CONTROLLER_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || TEST_RC=1 ;;
  closure) for rc in "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_5B_RC" "$I6_5A_RC" "$I6_CONTROLLER_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_PURE_RC" "$FOCUSED_PURE_COUNT" \
    "$FOCUSED_QT_RC" "$FOCUSED_QT_COUNT" \
    "$I6_5B_RC" "$I6_5B_COUNT" \
    "$I6_5A_RC" "$I6_5A_COUNT" \
    "$I6_CONTROLLER_RC" "$I6_CONTROLLER_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json, sys
(
 path, mode, test_rc,
 pure_rc, pure_count, qt_rc, qt_count,
 i65b_rc, i65b_count, i65a_rc, i65a_count,
 controller_rc, controller_count,
 full_rc, full_count, import_rc, compile_rc,
)=sys.argv[1:]
payload={
 'schema_version':1,
 'phase':'I6.5C_CONTROLLER_COMPOSITION_AND_EXPLICIT_USER_OVERRIDE_SEMANTICS',
 'mode':mode,
 'status':'pass' if int(test_rc)==0 else 'fail',
 'pass':int(test_rc)==0,
 'manual_ui_validation_required':False,
 'counts':{
   'focused_pure_tests':int(pure_count), 'focused_qt_tests':int(qt_count),
   'i6_5b_regression_tests':int(i65b_count), 'i6_5a_regression_tests':int(i65a_count),
   'i6_controller_predecessor_tests':int(controller_count), 'full_unit_tests':int(full_count),
 },
 'return_codes':{
   'focused_pure':int(pure_rc), 'focused_qt':int(qt_rc),
   'i6_5b_regression':int(i65b_rc), 'i6_5a_regression':int(i65a_rc),
   'i6_controller_predecessor':int(controller_rc), 'full_unit':int(full_rc),
   'import_probe':int(import_rc), 'compile_probe':int(compile_rc),
 },
 'scope':{
   'controller_owned_request_state':True,
   'default_mode':'adaptive',
   'explicit_user_override_supported':True,
   'all_eligible_preserved':True,
   'manual_limit_is_maximum':True,
   'source_change_recomputes_effective_plan':True,
   'user_request_preserved_across_source_change':True,
   'stale_effective_plan_retained':False,
   'resource_snapshot_explicit_not_probed':True,
   'source_fingerprint_profile':'selection_metadata_manifest_v1',
   'source_fingerprint_is_content_evidence_hash':False,
   'worker_modified':False,
   'ui_modified':False,
   'sampling_execution_modified':False,
   'provider_selection_effect':False,
   'production_auto_activated':False,
   'progressive_sfm_visualization_added':False,
   'planned_view_is_registered_camera':False,
   'pycolmap_camera_authority_preserved':True,
   'metric_scale_claimed':False,
   'certified_metrology_claimed':False,
 }
}
open(path,'w',encoding='utf-8').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(f'I6.5C receipt={path}')
PY
fi

echo "I6_5C_TEST_RC=$TEST_RC"
exit "$TEST_RC"
