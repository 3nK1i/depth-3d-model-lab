#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED_PURE=(
  tests/unit/test_i6_5d_frame_view_plan_presentation.py
  tests/unit/test_i6_5d_frame_view_resource_binding.py
  tests/unit/test_i6_5d_reconstruction_page_frame_view_static.py
)
FOCUSED_QT=(
  tests/ui/test_i6_5d_frame_view_plan_ui.py
)
I6_5C_REGRESSION=(
  tests/unit/test_i6_5c_reconstruction_frame_view_plan_state.py
  tests/unit/test_i6_5c_controller_frame_view_plan_contract_static.py
  tests/integration/test_i6_5c_controller_frame_view_plan.py
  tests/stress/test_i6_5c_frame_view_plan_state_stress.py
)
I6_5B_REGRESSION=(
  tests/unit/test_i6_5b_deterministic_frame_view_planning.py
  tests/unit/test_i6_5b_frame_view_planning_static.py
  tests/integration/test_i6_5b_deterministic_frame_view_planning_integration.py
  tests/stress/test_i6_5b_deterministic_frame_view_planning_stress.py
)
I6_4D_REGRESSION=(
  tests/unit/test_i6_4d_provider_auto_presentation.py
  tests/unit/test_i6_4d_reconstruction_page_provider_ui_static.py
  tests/ui/test_i6_4d_provider_auto_ui.py
)
PAGE_QT_PREDECESSOR=(
  tests/ui/test_phase8_8b3_reconstruction_runtime_controls.py
  tests/ui/test_phase8_8b4_status_progress.py
  tests/ui/test_i6_1c_offline_workspace_ui.py
)

FOCUSED_PURE_RC=0; FOCUSED_PURE_COUNT=0
FOCUSED_QT_RC=0; FOCUSED_QT_COUNT=0
I6_5C_RC=0; I6_5C_COUNT=0
I6_5B_RC=0; I6_5B_COUNT=0
I6_4D_RC=0; I6_4D_COUNT=0
PAGE_QT_RC=0; PAGE_QT_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
IMPORT_RC=0; COMPILE_RC=0

run_pytest() {
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
from types import SimpleNamespace
from pathlib import Path
import tempfile

from app.reconstruction_frame_view_plan_presentation import (
    I65D_PHASE,
    present_reconstruction_frame_view_plan,
)
from app.reconstruction_frame_view_plan_state import (
    FrameViewPlanIntentOrigin,
    build_reconstruction_frame_view_plan_state,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.reconstruction_frame_view_plan_contract import FrameViewPlanMode, FrameViewPlanRequest

GIB=1024**3
with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    for index in range(8):
        (root/f'{index:03d}.jpg').write_bytes(b'x')
    state=build_reconstruction_frame_view_plan_state(
        source_selection=SimpleNamespace(ready=True, workflow='image_sequence', path=root),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        intent_origin=FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE,
        resources=AdaptiveSelectionResourceSnapshot(
            gpu_name='probe', total_vram_bytes=6*GIB, free_vram_bytes=2*GIB,
            available_ram_bytes=8*GIB, cuda_ready=True,
        ),
    )
    ui=present_reconstruction_frame_view_plan(state)
    assert I65D_PHASE == 'I6.5D'
    assert ui.status_text == 'Adaptive plan ready'
    assert 'planned' in ui.planned_text
    assert 'no frames sampled' in ui.boundary_text
print('I6.5D frame/view presentation import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    app/reconstruction_frame_view_plan_presentation.py \
    app/reconstruction_frame_view_resources.py \
    app/widgets/reconstruction_page.py \
    app/main_window.py \
    tests/unit/test_i6_5d_frame_view_plan_presentation.py \
    tests/unit/test_i6_5d_frame_view_resource_binding.py \
    tests/unit/test_i6_5d_reconstruction_page_frame_view_static.py \
    tests/ui/test_i6_5d_frame_view_plan_ui.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest FOCUSED_PURE_RC FOCUSED_PURE_COUNT FOCUSED_PURE "${FOCUSED_PURE[@]}"
  probe_import
  probe_compile
}

run_closure() {
  run_focused
  run_pytest FOCUSED_QT_RC FOCUSED_QT_COUNT FOCUSED_QT "${FOCUSED_QT[@]}"
  run_pytest I6_5C_RC I6_5C_COUNT I6_5C_REGRESSION "${I6_5C_REGRESSION[@]}"
  run_pytest I6_5B_RC I6_5B_COUNT I6_5B_REGRESSION "${I6_5B_REGRESSION[@]}"
  run_pytest I6_4D_RC I6_4D_COUNT I6_4D_REGRESSION "${I6_4D_REGRESSION[@]}"
  run_pytest PAGE_QT_RC PAGE_QT_COUNT PAGE_QT_PREDECESSOR "${PAGE_QT_PREDECESSOR[@]}"
  run_pytest FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused) run_focused ;;
  closure) run_closure ;;
  *) echo "usage: $0 [focused|closure] [receipt.json]" >&2; exit 2 ;;
esac

TEST_RC=0
for rc in \
  "$FOCUSED_PURE_RC" "$IMPORT_RC" "$COMPILE_RC" \
  "$FOCUSED_QT_RC" "$I6_5C_RC" "$I6_5B_RC" "$I6_4D_RC" \
  "$PAGE_QT_RC" "$FULL_UNIT_RC"
do
  [[ "$rc" -eq 0 ]] || TEST_RC=1
done

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_PURE_COUNT" "$FOCUSED_QT_COUNT" "$I6_5C_COUNT" "$I6_5B_COUNT" \
    "$I6_4D_COUNT" "$PAGE_QT_COUNT" "$FULL_UNIT_COUNT" \
    "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_5C_RC" "$I6_5B_RC" \
    "$I6_4D_RC" "$PAGE_QT_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json, sys
(
 path, mode, test_rc,
 focused_pure_count, focused_qt_count, i65c_count, i65b_count,
 i64d_count, page_qt_count, full_count,
 focused_pure_rc, focused_qt_rc, i65c_rc, i65b_rc,
 i64d_rc, page_qt_rc, full_rc, import_rc, compile_rc,
)=sys.argv[1:]
payload={
 "schema_version":1,
 "phase":"I6.5D_PROFESSIONAL_FRAME_VIEW_UI_AND_I6_5_CLOSURE",
 "mode":mode,
 "pass":int(test_rc)==0,
 "status":"pass" if int(test_rc)==0 else "fail",
 "i6_5_closure_candidate":int(test_rc)==0,
 "manual_ui_validation_required":True,
 "counts":{
   "focused_pure_tests":int(focused_pure_count),
   "focused_qt_tests":int(focused_qt_count),
   "i6_5c_regression_tests":int(i65c_count),
   "i6_5b_regression_tests":int(i65b_count),
   "i6_4d_regression_tests":int(i64d_count),
   "page_qt_predecessor_tests":int(page_qt_count),
   "full_unit_tests":int(full_count),
 },
 "return_codes":{
   "focused_pure":int(focused_pure_rc),
   "focused_qt":int(focused_qt_rc),
   "i6_5c":int(i65c_rc),
   "i6_5b":int(i65b_rc),
   "i6_4d":int(i64d_rc),
   "page_qt_predecessor":int(page_qt_rc),
   "full_unit":int(full_rc),
   "import_probe":int(import_rc),
   "compile_probe":int(compile_rc),
 },
 "scope":{
   "professional_frame_view_ui_added":True,
   "application_owned_resource_handoff_added":True,
   "controller_probes_hardware":False,
   "default_mode":"adaptive",
   "all_eligible_silently_capped":False,
   "manual_limit_is_maximum":True,
   "source_change_recomputes_effective_plan":True,
   "user_request_preserved_across_source_change":True,
   "source_sampling_executed":False,
   "worker_modified":False,
   "provider_selection_effect":False,
   "production_auto_activated":False,
   "progressive_sfm_visualization_added":False,
   "planned_view_is_registered_camera":False,
   "pycolmap_camera_authority_preserved":True,
   "metric_scale_claimed":False,
   "certified_metrology_claimed":False,
   "ui_modified":True,
 }
}
open(path,'w',encoding='utf-8').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(f'I6.5D receipt={path}')
PY
fi

echo "I6.5D_TEST_RC=$TEST_RC"
exit "$TEST_RC"
