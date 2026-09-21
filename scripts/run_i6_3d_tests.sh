#!/usr/bin/env bash
set -u

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

run_pytest() {
  local label="$1"; shift
  local output rc count
  output="$(mktemp)"
  pytest -q "$@" 2>&1 | tee "$output"
  rc=${PIPESTATUS[0]}
  count="$(python - "$output" <<'PY'
import re, sys
text=open(sys.argv[1], encoding='utf-8', errors='replace').read()
values=re.findall(r'(\d+) passed', text)
print(values[-1] if values else '0')
PY
)"
  rm -f "$output"
  printf '%s_RC=%s\n' "$label" "$rc"
  printf '%s_COUNT=%s\n' "$label" "$count"
  eval "${label}_RC=$rc"
  eval "${label}_COUNT=$count"
}

run_pytest FOCUSED_PURE \
  tests/unit/test_i6_3d_reconstruction_advisor_presentation.py \
  tests/unit/test_i6_3d_reconstruction_page_advisor_binding_static.py

python -m py_compile \
  app/reconstruction_advisor_presentation.py \
  app/widgets/reconstruction_page.py \
  tests/unit/test_i6_3d_reconstruction_advisor_presentation.py \
  tests/unit/test_i6_3d_reconstruction_page_advisor_binding_static.py \
  tests/ui/test_i6_3d_reconstruction_advisor_ui.py
COMPILE_RC=$?
echo "COMPILE_RC=$COMPILE_RC"

python - <<'PY'
from app.reconstruction_advisor_presentation import I63D_BOUNDARY_TEXT, present_reconstruction_advisor
assert I63D_BOUNDARY_TEXT == "Advisory only • no settings changed"
assert callable(present_reconstruction_advisor)
print("I6.3D Reconstruction Advisor presentation/import probe: PASS")
PY
IMPORT_RC=$?
echo "IMPORT_RC=$IMPORT_RC"

if [[ "$MODE" == "closure" ]]; then
  run_pytest FOCUSED_QT tests/ui/test_i6_3d_reconstruction_advisor_ui.py
  run_pytest I6_3C_REGRESSION \
    tests/unit/test_i6_3c_reconstruction_advisor_state.py \
    tests/unit/test_i6_3c_controller_advisor_contract_static.py \
    tests/integration/test_i6_3c_controller_advisor_state.py
  run_pytest I6_3B_REGRESSION \
    tests/unit/test_i6_3b_reconstruction_advisor_engine.py \
    tests/integration/test_i6_3b_reconstruction_advisor_engine_contract.py \
    tests/stress/test_i6_3b_reconstruction_advisor_engine_stress.py
  run_pytest I6_3A_REGRESSION \
    tests/unit/test_i6_3a_reconstruction_advisor_contract.py \
    tests/integration/test_i6_3a_reconstruction_advisor_contract.py
  run_pytest I6_2D_QT_PREDECESSOR tests/ui/test_i6_2d_one_start_professional_ux.py
  run_pytest PAGE_QT_PREDECESSOR \
    tests/ui/test_phase8_8b3_reconstruction_runtime_controls.py \
    tests/ui/test_phase8_8b4_status_progress.py \
    tests/ui/test_i6_1c_offline_workspace_ui.py
  run_pytest FULL_UNIT tests/unit
else
  FOCUSED_QT_RC=0; FOCUSED_QT_COUNT=0
  I6_3C_REGRESSION_RC=0; I6_3C_REGRESSION_COUNT=0
  I6_3B_REGRESSION_RC=0; I6_3B_REGRESSION_COUNT=0
  I6_3A_REGRESSION_RC=0; I6_3A_REGRESSION_COUNT=0
  I6_2D_QT_PREDECESSOR_RC=0; I6_2D_QT_PREDECESSOR_COUNT=0
  PAGE_QT_PREDECESSOR_RC=0; PAGE_QT_PREDECESSOR_COUNT=0
  FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
fi

TEST_RC=0
for rc in \
  "$FOCUSED_PURE_RC" "$COMPILE_RC" "$IMPORT_RC" \
  "$FOCUSED_QT_RC" "$I6_3C_REGRESSION_RC" "$I6_3B_REGRESSION_RC" \
  "$I6_3A_REGRESSION_RC" "$I6_2D_QT_PREDECESSOR_RC" \
  "$PAGE_QT_PREDECESSOR_RC" "$FULL_UNIT_RC"
do
  if [[ "$rc" -ne 0 ]]; then TEST_RC=1; fi
done

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_PURE_COUNT" "$FOCUSED_QT_COUNT" "$I6_3C_REGRESSION_COUNT" \
    "$I6_3B_REGRESSION_COUNT" "$I6_3A_REGRESSION_COUNT" \
    "$I6_2D_QT_PREDECESSOR_COUNT" "$PAGE_QT_PREDECESSOR_COUNT" "$FULL_UNIT_COUNT" \
    "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_3C_REGRESSION_RC" \
    "$I6_3B_REGRESSION_RC" "$I6_3A_REGRESSION_RC" \
    "$I6_2D_QT_PREDECESSOR_RC" "$PAGE_QT_PREDECESSOR_RC" "$FULL_UNIT_RC" \
    "$COMPILE_RC" "$IMPORT_RC" <<'PY'
import json, sys
(
 path, mode, test_rc,
 focused_pure_count, focused_qt_count, i63c_count, i63b_count, i63a_count,
 i62d_qt_count, page_qt_count, full_count,
 focused_pure_rc, focused_qt_rc, i63c_rc, i63b_rc, i63a_rc,
 i62d_qt_rc, page_qt_rc, full_rc, compile_rc, import_rc,
)=sys.argv[1:]
payload={
 "schema_version":1,
 "phase":"I6.3D_PROFESSIONAL_ADVISOR_UI_AND_I6_3_CLOSURE",
 "mode":mode,
 "pass":int(test_rc)==0,
 "status":"pass" if int(test_rc)==0 else "fail",
 "manual_ui_validation_required":True,
 "i6_3_closure_candidate":int(test_rc)==0,
 "counts":{
  "focused_pure_tests":int(focused_pure_count),
  "focused_qt_tests":int(focused_qt_count),
  "i6_3c_regression_tests":int(i63c_count),
  "i6_3b_regression_tests":int(i63b_count),
  "i6_3a_regression_tests":int(i63a_count),
  "i6_2d_qt_predecessor_tests":int(i62d_qt_count),
  "page_qt_predecessor_tests":int(page_qt_count),
  "full_unit_tests":int(full_count),
 },
 "return_codes":{
  "focused_pure":int(focused_pure_rc),
  "focused_qt":int(focused_qt_rc),
  "i6_3c":int(i63c_rc),
  "i6_3b":int(i63b_rc),
  "i6_3a":int(i63a_rc),
  "i6_2d_qt_predecessor":int(i62d_qt_rc),
  "page_qt_predecessor":int(page_qt_rc),
  "full_unit":int(full_rc),
  "compile_probe":int(compile_rc),
  "import_probe":int(import_rc),
 },
 "scope":{
  "professional_advisor_ui_added":True,
  "recommendation_and_abstention_presented":True,
  "abstention_is_not_runtime_error":True,
  "advisory_boundary_visible":True,
  "controller_modified":False,
  "advisor_contract_modified":False,
  "advisor_engine_modified":False,
  "advisor_live_state_modified":False,
  "worker_modified":False,
  "provider_selection_modified":False,
  "provider_activation_authorized":False,
  "settings_mutation_authorized":False,
  "runtime_action_semantics_modified":False,
  "start_action_authorized":False,
  "live_context_resolver_qualified":False,
  "production_auto_qualified":False,
  "production_auto_authorized":False,
  "production_auto_activated":False,
  "global_provider_ranking_created":False,
  "global_scalar_score_created":False,
  "pycolmap_camera_authority_preserved":True,
  "certified_metrology_claimed":False,
  "ui_modified":True,
 }
}
open(path,'w',encoding='utf-8').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(f"I6.3D receipt={path}")
PY
fi

echo "I6.3D_TEST_RC=$TEST_RC"
exit "$TEST_RC"
