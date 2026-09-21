#!/usr/bin/env bash
set -u

MODE="${1:-all}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOCUSED_PURE=(
  tests/unit/test_i6_2d_one_start_ux_presentation.py
  tests/unit/test_i6_2d_reconstruction_page_ux_binding_static.py
)
FOCUSED_QT=(
  tests/ui/test_i6_2d_one_start_professional_ux.py
)
I6_2C_PURE=(
  tests/unit/test_i6_2c_offline_result_handoff.py
  tests/unit/test_i6_2c_controller_result_handoff_contract_static.py
)
I6_2C_QT=(
  tests/integration/test_i6_2c_offline_result_handoff_controller.py
  tests/ui/test_i6_2c_offline_result_handoff_ui.py
)
I6_2B_REGRESSION=(
  tests/unit/test_i6_2b_one_start_controller_contract_static.py
  tests/integration/test_i6_2b_one_start_controller_orchestration.py
)
I6_2A_REGRESSION=(
  tests/unit/test_i6_2a_one_start_preflight.py
  tests/integration/test_i6_2a_one_start_preflight_contract.py
)
I6_1_PURE_REGRESSION=(
  tests/unit/test_i6_1a_offline_reconstruction_workspace.py
  tests/integration/test_i6_1a_offline_workspace_contract.py
  tests/unit/test_i6_1b_controller_workspace_contract_static.py
  tests/unit/test_i6_1c_offline_workspace_presentation.py
  tests/unit/test_i6_1c_reconstruction_page_binding_static.py
)
PAGE_QT_PREDECESSORS=(
  tests/ui/test_phase8_8b3_reconstruction_runtime_controls.py
  tests/ui/test_phase8_8b4_status_progress.py
  tests/ui/test_i6_1c_offline_workspace_ui.py
)
STATIC_PREDECESSORS=(
  tests/unit/test_phase8_8b1_reconstruction_page_contract.py
  tests/unit/test_phase8_8b3_runtime_contract.py
  tests/unit/test_phase8_8b4_runtime_contract.py
  tests/unit/test_phase8_8b4_status_presentation.py
)

FOCUSED_PURE_RC=0; FOCUSED_PURE_COUNT=0
FOCUSED_QT_RC=0; FOCUSED_QT_COUNT=0
I6_2C_PURE_RC=0; I6_2C_PURE_COUNT=0
I6_2C_QT_RC=0; I6_2C_QT_COUNT=0
I6_2B_RC=0; I6_2B_COUNT=0
I6_2A_RC=0; I6_2A_COUNT=0
I6_1_RC=0; I6_1_COUNT=0
PAGE_QT_RC=0; PAGE_QT_COUNT=0
STATIC_RC=0; STATIC_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
IMPORT_RC=0
COMPILE_RC=0

run_pytest_count() {
  local rc_var="$1" count_var="$2" label="$3"; shift 3
  local tmp
  tmp="$(mktemp)"
  pytest -q "$@" >"$tmp" 2>&1
  local rc=$?
  cat "$tmp"
  local count=0
  if [[ $rc -eq 0 ]]; then
    count="$(python - "$tmp" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
print(sum(int(value) for value in re.findall(r"(\d+) passed", text)))
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
from app.reconstruction_one_start_ux import (
    OfflineOneStartUxPresentation,
    present_offline_one_start_ux,
)
from app.reconstruction_offline_result_handoff import OfflineReconstructionResultHandoff
assert callable(present_offline_one_start_ux)
assert OfflineOneStartUxPresentation.__name__ == "OfflineOneStartUxPresentation"
assert OfflineReconstructionResultHandoff.__name__ == "OfflineReconstructionResultHandoff"
print("I6.2D One-Start professional UX/import probe: PASS")
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    app/reconstruction_one_start_ux.py \
    app/widgets/reconstruction_page.py \
    tests/unit/test_i6_2d_one_start_ux_presentation.py \
    tests/unit/test_i6_2d_reconstruction_page_ux_binding_static.py \
    tests/ui/test_i6_2d_one_start_professional_ux.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused_pure() {
  run_pytest_count FOCUSED_PURE_RC FOCUSED_PURE_COUNT FOCUSED_PURE "${FOCUSED_PURE[@]}"
  probe_compile
  probe_import
}
run_focused_qt() {
  run_pytest_count FOCUSED_QT_RC FOCUSED_QT_COUNT FOCUSED_QT "${FOCUSED_QT[@]}"
}
run_predecessors_pure() {
  run_pytest_count I6_2C_PURE_RC I6_2C_PURE_COUNT I6_2C_PURE "${I6_2C_PURE[@]}"
  run_pytest_count I6_2A_RC I6_2A_COUNT I6_2A_REGRESSION "${I6_2A_REGRESSION[@]}"
  run_pytest_count I6_1_RC I6_1_COUNT I6_1_PURE_REGRESSION "${I6_1_PURE_REGRESSION[@]}"
  run_pytest_count STATIC_RC STATIC_COUNT STATIC_PREDECESSOR "${STATIC_PREDECESSORS[@]}"
}
run_predecessors_qt() {
  run_pytest_count I6_2C_QT_RC I6_2C_QT_COUNT I6_2C_QT "${I6_2C_QT[@]}"
  run_pytest_count I6_2B_RC I6_2B_COUNT I6_2B_REGRESSION "${I6_2B_REGRESSION[@]}"
  run_pytest_count PAGE_QT_RC PAGE_QT_COUNT PAGE_QT_PREDECESSOR "${PAGE_QT_PREDECESSORS[@]}"
}
run_full_unit() {
  run_pytest_count FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused-pure) run_focused_pure ;;
  focused-qt) run_focused_qt ;;
  predecessors-pure) run_predecessors_pure ;;
  predecessors-qt) run_predecessors_qt ;;
  all) run_focused_pure; run_focused_qt; run_predecessors_pure; run_predecessors_qt ;;
  full-unit) run_full_unit ;;
  closure) run_focused_pure; run_focused_qt; run_predecessors_pure; run_predecessors_qt; run_full_unit ;;
  *) echo "usage: $0 [focused-pure|focused-qt|predecessors-pure|predecessors-qt|all|full-unit|closure] [receipt.json]" >&2; exit 2 ;;
esac

RC=0
case "$MODE" in
  focused-pure) for v in "$FOCUSED_PURE_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$v" -eq 0 ]] || RC=1; done ;;
  focused-qt) [[ "$FOCUSED_QT_RC" -eq 0 ]] || RC=1 ;;
  predecessors-pure) for v in "$I6_2C_PURE_RC" "$I6_2A_RC" "$I6_1_RC" "$STATIC_RC"; do [[ "$v" -eq 0 ]] || RC=1; done ;;
  predecessors-qt) for v in "$I6_2C_QT_RC" "$I6_2B_RC" "$PAGE_QT_RC"; do [[ "$v" -eq 0 ]] || RC=1; done ;;
  all) for v in "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_2C_PURE_RC" "$I6_2C_QT_RC" "$I6_2B_RC" "$I6_2A_RC" "$I6_1_RC" "$PAGE_QT_RC" "$STATIC_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$v" -eq 0 ]] || RC=1; done ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || RC=1 ;;
  closure) for v in "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_2C_PURE_RC" "$I6_2C_QT_RC" "$I6_2B_RC" "$I6_2A_RC" "$I6_1_RC" "$PAGE_QT_RC" "$STATIC_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$v" -eq 0 ]] || RC=1; done ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$RC" \
    "$FOCUSED_PURE_RC" "$FOCUSED_PURE_COUNT" \
    "$FOCUSED_QT_RC" "$FOCUSED_QT_COUNT" \
    "$I6_2C_PURE_RC" "$I6_2C_PURE_COUNT" \
    "$I6_2C_QT_RC" "$I6_2C_QT_COUNT" \
    "$I6_2B_RC" "$I6_2B_COUNT" \
    "$I6_2A_RC" "$I6_2A_COUNT" \
    "$I6_1_RC" "$I6_1_COUNT" \
    "$PAGE_QT_RC" "$PAGE_QT_COUNT" \
    "$STATIC_RC" "$STATIC_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
mode = sys.argv[2]
values = list(map(int, sys.argv[3:]))
(
    rc,
    focused_pure_rc, focused_pure_count,
    focused_qt_rc, focused_qt_count,
    i6_2c_pure_rc, i6_2c_pure_count,
    i6_2c_qt_rc, i6_2c_qt_count,
    i6_2b_rc, i6_2b_count,
    i6_2a_rc, i6_2a_count,
    i6_1_rc, i6_1_count,
    page_qt_rc, page_qt_count,
    static_rc, static_count,
    full_unit_rc, full_unit_count,
    import_rc, compile_rc,
) = values
payload = {
    "schema_version": 1,
    "phase": "I6.2D_PROFESSIONAL_ONE_START_UX_AND_I6_2_CLOSURE",
    "mode": mode,
    "status": "pass" if rc == 0 else "fail",
    "pass": rc == 0,
    "manual_ui_validation_required": True,
    "i6_2_closure_candidate": rc == 0,
    "scope": {
        "professional_one_start_ux_added": True,
        "duplicate_start_prevention_reused": True,
        "offline_pause_semantics_explicit": True,
        "offline_cancel_semantics_explicit": True,
        "completed_partial_result_not_presented_as_dense_final": True,
        "rejected_product_handoff_error_preserved": True,
        "rerun_affordance_is_new_session_only": True,
        "controller_modified": False,
        "runtime_action_semantics_modified": False,
        "worker_modified": False,
        "persistence_format_modified": False,
        "provider_selection_modified": False,
        "reconstruction_algorithm_modified": False,
        "scientific_threshold_modified": False,
        "progressive_sfm_visualization_added": False,
        "dense_execution_authorized": False,
        "persistent_scene_promotion_authorized": False,
        "production_auto_activated": False,
        "production_auto_qualified": False,
        "production_auto_authorized": False,
        "pycolmap_camera_authority_preserved": True,
        "certified_metrology_claimed": False,
        "ui_modified": True,
    },
    "counts": {
        "focused_pure_tests": focused_pure_count,
        "focused_qt_tests": focused_qt_count,
        "i6_2c_pure_regression_tests": i6_2c_pure_count,
        "i6_2c_qt_regression_tests": i6_2c_qt_count,
        "i6_2b_regression_tests": i6_2b_count,
        "i6_2a_regression_tests": i6_2a_count,
        "i6_1_pure_regression_tests": i6_1_count,
        "page_qt_predecessor_tests": page_qt_count,
        "static_predecessor_tests": static_count,
        "full_unit_tests": full_unit_count,
    },
    "return_codes": {
        "focused_pure": focused_pure_rc,
        "focused_qt": focused_qt_rc,
        "i6_2c_pure": i6_2c_pure_rc,
        "i6_2c_qt": i6_2c_qt_rc,
        "i6_2b": i6_2b_rc,
        "i6_2a": i6_2a_rc,
        "i6_1": i6_1_rc,
        "page_qt_predecessor": page_qt_rc,
        "static_predecessor": static_rc,
        "full_unit": full_unit_rc,
        "import_probe": import_rc,
        "compile_probe": compile_rc,
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"I6.2D receipt={path}")
PY
fi

echo "I6.2D_TEST_RC=$RC"
exit "$RC"
