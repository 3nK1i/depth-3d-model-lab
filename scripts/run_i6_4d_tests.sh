#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED_PURE=(
  tests/unit/test_i6_4d_provider_auto_presentation.py
  tests/unit/test_i6_4d_reconstruction_page_provider_ui_static.py
)
FOCUSED_QT=(
  tests/ui/test_i6_4d_provider_auto_ui.py
)
I6_4C_REGRESSION=(
  tests/unit/test_i6_4c_production_auto_qualification_boundary.py
  tests/unit/test_i6_4c_controller_auto_gate_contract_static.py
  tests/integration/test_i6_4c_controller_production_auto_gate.py
  tests/stress/test_i6_4c_production_auto_qualification_boundary_stress.py
)
I6_4B_REGRESSION=(
  tests/unit/test_i6_4b_reconstruction_provider_selection_state.py
  tests/unit/test_i6_4b_controller_provider_selection_contract_static.py
  tests/integration/test_i6_4b_controller_provider_selection.py
  tests/stress/test_i6_4b_provider_selection_state_stress.py
)
I6_3D_REGRESSION=(
  tests/unit/test_i6_3d_reconstruction_advisor_presentation.py
  tests/unit/test_i6_3d_reconstruction_page_advisor_binding_static.py
  tests/ui/test_i6_3d_reconstruction_advisor_ui.py
)
PAGE_QT_PREDECESSOR=(
  tests/ui/test_phase8_8b3_reconstruction_runtime_controls.py
  tests/ui/test_phase8_8b4_status_progress.py
  tests/ui/test_i6_1c_offline_workspace_ui.py
)

FOCUSED_PURE_RC=0; FOCUSED_PURE_COUNT=0
FOCUSED_QT_RC=0; FOCUSED_QT_COUNT=0
I6_4C_RC=0; I6_4C_COUNT=0
I6_4B_RC=0; I6_4B_COUNT=0
I6_3D_RC=0; I6_3D_COUNT=0
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
from app.reconstruction_provider_selection_presentation import (
    I64D_PHASE,
    present_reconstruction_provider_auto,
)
from app.reconstruction_provider_selection_state import build_reconstruction_provider_selection_state
from core.reconstruction.reconstruction_production_auto_gate import build_production_auto_qualification_boundary
from tests.i6_4d_helpers import advisor_state

state = build_reconstruction_provider_selection_state(advisor_state=advisor_state())
gate = build_production_auto_qualification_boundary(state.snapshot)
ui = present_reconstruction_provider_auto(state, gate)
assert I64D_PHASE == 'I6.4D'
assert ui.combo_enabled is True
assert ui.eligibility_text == '1 / 10 selectable'
assert ui.auto_text == 'Not qualified'
assert gate.production_auto_authorized is False
print('I6.4D provider/Auto presentation import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    app/reconstruction_provider_selection_presentation.py \
    app/widgets/reconstruction_page.py \
    tests/i6_4d_helpers.py \
    tests/unit/test_i6_4d_provider_auto_presentation.py \
    tests/unit/test_i6_4d_reconstruction_page_provider_ui_static.py \
    tests/ui/test_i6_4d_provider_auto_ui.py
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
  run_pytest I6_4C_RC I6_4C_COUNT I6_4C_REGRESSION "${I6_4C_REGRESSION[@]}"
  run_pytest I6_4B_RC I6_4B_COUNT I6_4B_REGRESSION "${I6_4B_REGRESSION[@]}"
  run_pytest I6_3D_RC I6_3D_COUNT I6_3D_REGRESSION "${I6_3D_REGRESSION[@]}"
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
  "$FOCUSED_QT_RC" "$I6_4C_RC" "$I6_4B_RC" "$I6_3D_RC" \
  "$PAGE_QT_RC" "$FULL_UNIT_RC"
do
  [[ "$rc" -eq 0 ]] || TEST_RC=1
done

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_PURE_COUNT" "$FOCUSED_QT_COUNT" "$I6_4C_COUNT" "$I6_4B_COUNT" \
    "$I6_3D_COUNT" "$PAGE_QT_COUNT" "$FULL_UNIT_COUNT" \
    "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_4C_RC" "$I6_4B_RC" \
    "$I6_3D_RC" "$PAGE_QT_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json, sys
(
 path, mode, test_rc,
 focused_pure_count, focused_qt_count, i64c_count, i64b_count,
 i63d_count, page_qt_count, full_count,
 focused_pure_rc, focused_qt_rc, i64c_rc, i64b_rc,
 i63d_rc, page_qt_rc, full_rc, import_rc, compile_rc,
)=sys.argv[1:]
payload={
 "schema_version":1,
 "phase":"I6.4D_PROFESSIONAL_PROVIDER_AUTO_UI_AND_I6_4_CLOSURE",
 "mode":mode,
 "pass":int(test_rc)==0,
 "status":"pass" if int(test_rc)==0 else "fail",
 "i6_4_closure_candidate":int(test_rc)==0,
 "manual_ui_validation_required":True,
 "counts":{
   "focused_pure_tests":int(focused_pure_count),
   "focused_qt_tests":int(focused_qt_count),
   "i6_4c_regression_tests":int(i64c_count),
   "i6_4b_regression_tests":int(i64b_count),
   "i6_3d_regression_tests":int(i63d_count),
   "page_qt_predecessor_tests":int(page_qt_count),
   "full_unit_tests":int(full_count),
 },
 "return_codes":{
   "focused_pure":int(focused_pure_rc),
   "focused_qt":int(focused_qt_rc),
   "i6_4c":int(i64c_rc),
   "i6_4b":int(i64b_rc),
   "i6_3d":int(i63d_rc),
   "page_qt_predecessor":int(page_qt_rc),
   "full_unit":int(full_rc),
   "import_probe":int(import_rc),
   "compile_probe":int(compile_rc),
 },
 "scope":{
   "professional_provider_auto_ui_added":True,
   "legacy_persisted_provider_inspector_preserved":True,
   "manual_selection_requires_explicit_eligibility":True,
   "advisor_recommendation_is_selection":False,
   "manual_selection_is_execution":False,
   "provider_activation_authorized":False,
   "provider_execution_authorized":False,
   "settings_mutation_authorized":False,
   "start_action_authorized":False,
   "offline_shadow_auto_qualified":True,
   "production_auto_qualified":False,
   "production_auto_authorized":False,
   "production_auto_selectable":False,
   "live_context_resolver_qualified":False,
   "auto_engine_wired_into_app":False,
   "auto_orchestration_entrypoint_added":False,
   "controller_modified":False,
   "worker_modified":False,
   "global_provider_ranking_created":False,
   "global_scalar_score_created":False,
   "pycolmap_camera_authority_preserved":True,
   "certified_metrology_claimed":False,
   "ui_modified":True,
 }
}
open(path,'w',encoding='utf-8').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(f'I6.4D receipt={path}')
PY
fi

echo "I6.4D_TEST_RC=$TEST_RC"
exit "$TEST_RC"
