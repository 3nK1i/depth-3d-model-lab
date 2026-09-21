#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED=(
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
I6_5A_RC=0; I6_5A_COUNT=0
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
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    FrameViewPlanMode,
    FrameViewPlanRequest,
    FrameViewSourceProvenance,
)
from core.reconstruction.reconstruction_frame_view_planning import (
    I65B_PHASE,
    resolve_deterministic_frame_view_plan,
)

GIB=1024**3
source=FrameViewSourceProvenance(
    source_id='probe',
    source_kind='image_sequence',
    source_uri='/probe/images',
    source_fingerprint_sha256='e'*64,
    estimated_total_views=128,
    expected_frame_rate_hz=30.0,
)
resources=AdaptiveSelectionResourceSnapshot(
    gpu_name='probe-gpu',
    total_vram_bytes=6*GIB,
    free_vram_bytes=2*GIB,
    available_ram_bytes=8*GIB,
    cuda_ready=True,
)
result=resolve_deterministic_frame_view_plan(
    source=source,
    request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
    resources=resources,
)
assert I65B_PHASE == 'I6.5B'
assert result.resolution.effective_target_views == 64
assert result.resolution.resource_target_views == 64
assert result.resource_bound_applied is True
assert len(result.hardware_bounds.hardware_context_sha256) == 64
assert len(result.planning_provenance_sha256) == 64
assert result.as_dict()['policy']['retry_until_pass'] is False
assert result.as_dict()['authority']['planned_views_are_registered_cameras'] is False
print('I6.5B deterministic adaptive planning import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_frame_view_planning.py \
    tests/unit/test_i6_5b_deterministic_frame_view_planning.py \
    tests/unit/test_i6_5b_frame_view_planning_static.py \
    tests/integration/test_i6_5b_deterministic_frame_view_planning_integration.py \
    tests/stress/test_i6_5b_deterministic_frame_view_planning_stress.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_RC FOCUSED_COUNT FOCUSED "${FOCUSED[@]}"
  probe_import
  probe_compile
}

run_predecessors() {
  run_pytest_count I6_5A_RC I6_5A_COUNT I6_5A_REGRESSION "${I6_5A_REGRESSION[@]}"
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
  predecessors) for rc in "$I6_5A_RC" "$ADAPTIVE_RC" "$I6_4_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  all) for rc in "$FOCUSED_RC" "$I6_5A_RC" "$ADAPTIVE_RC" "$I6_4_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || TEST_RC=1 ;;
  closure) for rc in "$FOCUSED_RC" "$I6_5A_RC" "$ADAPTIVE_RC" "$I6_4_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$rc" -eq 0 ]] || TEST_RC=1; done ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_RC" "$FOCUSED_COUNT" \
    "$I6_5A_RC" "$I6_5A_COUNT" \
    "$ADAPTIVE_RC" "$ADAPTIVE_COUNT" \
    "$I6_4_RC" "$I6_4_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json, sys
(
 path, mode, test_rc,
 focused_rc, focused_count,
 i65a_rc, i65a_count,
 adaptive_rc, adaptive_count,
 i64_rc, i64_count,
 full_rc, full_count,
 import_rc, compile_rc,
)=sys.argv[1:]
payload={
 'schema_version':1,
 'phase':'I6.5B_DETERMINISTIC_ADAPTIVE_PLANNING_AND_HARDWARE_BOUNDS',
 'mode':mode,
 'status':'pass' if int(test_rc)==0 else 'fail',
 'pass':int(test_rc)==0,
 'manual_ui_validation_required':False,
 'counts':{
   'focused_tests':int(focused_count),
   'i6_5a_regression_tests':int(i65a_count),
   'adaptive_predecessor_regression_tests':int(adaptive_count),
   'i6_4_boundary_regression_tests':int(i64_count),
   'full_unit_tests':int(full_count),
 },
 'return_codes':{
   'focused':int(focused_rc),
   'i6_5a_regression':int(i65a_rc),
   'adaptive_predecessor_regression':int(adaptive_rc),
   'i6_4_boundary_regression':int(i64_rc),
   'full_unit':int(full_rc),
   'import_probe':int(import_rc),
   'compile_probe':int(compile_rc),
 },
 'scope':{
   'deterministic_product_planner_added':True,
   'legacy_i4_3f2_resource_planner_reused':True,
   'hardware_resource_provenance_added':True,
   'resource_target_is_empirical_runtime_guarantee':False,
   'adaptive_mode_resolved':True,
   'manual_limit_resolved_as_maximum':True,
   'all_eligible_silently_capped':False,
   'request_mode_substituted':False,
   'retry_until_pass':False,
   'silent_replan_after_sfm_failure':False,
   'permanent_global_view_cap':None,
   'existing_i4_3f_planner_modified':False,
   'i6_5a_contract_modified':False,
   'ingestion_behavior_modified':False,
   'controller_modified':False,
   'ui_modified':False,
   'worker_modified':False,
   'provider_selection_effect':False,
   'production_auto_activated':False,
   'scientific_threshold_modified':False,
   'progressive_sfm_visualization_added':False,
   'planned_view_is_registered_camera':False,
   'pycolmap_camera_authority_preserved':True,
   'metric_scale_claimed':False,
   'certified_metrology_claimed':False,
 }
}
open(path,'w',encoding='utf-8').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(f'I6.5B receipt={path}')
PY
fi

echo "I6_5B_TEST_RC=$TEST_RC"
exit "$TEST_RC"
