#!/usr/bin/env bash
set -u

MODE="${1:-all}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOCUSED_PURE=(
  tests/unit/test_i6_1c_offline_workspace_presentation.py
  tests/unit/test_i6_1c_reconstruction_page_binding_static.py
)
FOCUSED_UI=(
  tests/ui/test_i6_1c_offline_workspace_ui.py
)
I6_1A_REGRESSION=(
  tests/unit/test_i6_1a_offline_reconstruction_workspace.py
  tests/integration/test_i6_1a_offline_workspace_contract.py
)
I6_1B_REGRESSION=(
  tests/unit/test_i6_1b_controller_workspace_contract_static.py
  tests/integration/test_i6_1b_controller_workspace_plumbing.py
)
PURE_PREDECESSORS=(
  tests/unit/test_phase8_8g1_reconstruction_session_model.py
  tests/unit/test_phase8_8g2_reconstruction_session_interaction.py
  tests/unit/test_phase8_8g3_reconstruction_checkpoint_resume.py
  tests/unit/test_phase8_8g4_reconstruction_failure_recovery.py
  tests/unit/test_phase8_8b1_reconstruction_page_contract.py
)
PAGE_PREDECESSORS=(
  tests/ui/test_phase8_8b2_reconstruction_source_controls.py
  tests/ui/test_phase8_8b4_status_progress.py
  tests/ui/test_phase8_8b5_reconstruction_navigation.py
  tests/ui/test_phase8_8d1_offline_reconstruction_visualization.py
  tests/ui/test_phase9_6e4_workspace_ux.py
)

FOCUSED_PURE_RC=0; FOCUSED_PURE_COUNT=0
FOCUSED_UI_RC=0; FOCUSED_UI_COUNT=0
I6_1A_RC=0; I6_1A_COUNT=0
I6_1B_RC=0; I6_1B_COUNT=0
PURE_RC=0; PURE_COUNT=0
PAGE_RC=0; PAGE_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
PURE_IMPORT_RC=0
QT_IMPORT_RC=0
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

probe_pure_import() {
  python - <<'PY'
from app.reconstruction_offline_workspace import OfflineReconstructionWorkspaceSnapshot
from app.reconstruction_offline_workspace_presentation import (
    OfflineReconstructionWorkspacePresentation,
    present_offline_reconstruction_workspace,
)

assert OfflineReconstructionWorkspaceSnapshot.__dataclass_params__.frozen is True
assert OfflineReconstructionWorkspacePresentation.__dataclass_params__.frozen is True
assert callable(present_offline_reconstruction_workspace)
print("I6.1C workspace presentation/import probe: PASS")
PY
  PURE_IMPORT_RC=$?
  echo "PURE_IMPORT_RC=$PURE_IMPORT_RC"
}

probe_qt_import() {
  python - <<'PY'
from app.widgets.reconstruction_page import ReconstructionPage

assert hasattr(ReconstructionPage, "set_offline_workspace_snapshot")
assert isinstance(ReconstructionPage.offline_workspace_snapshot, property)
print("I6.1C ReconstructionPage Qt binding/import probe: PASS")
PY
  QT_IMPORT_RC=$?
  echo "QT_IMPORT_RC=$QT_IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    app/reconstruction_offline_workspace_presentation.py \
    app/widgets/reconstruction_page.py \
    tests/unit/test_i6_1c_offline_workspace_presentation.py \
    tests/unit/test_i6_1c_reconstruction_page_binding_static.py \
    tests/ui/test_i6_1c_offline_workspace_ui.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused_pure() {
  run_pytest_count FOCUSED_PURE_RC FOCUSED_PURE_COUNT FOCUSED_PURE "${FOCUSED_PURE[@]}"
  probe_pure_import
  probe_compile
}

run_focused_ui() {
  run_pytest_count FOCUSED_UI_RC FOCUSED_UI_COUNT FOCUSED_UI "${FOCUSED_UI[@]}"
  probe_qt_import
}

run_i6_1a() {
  run_pytest_count I6_1A_RC I6_1A_COUNT I6_1A_REGRESSION "${I6_1A_REGRESSION[@]}"
}

run_i6_1b() {
  run_pytest_count I6_1B_RC I6_1B_COUNT I6_1B_REGRESSION "${I6_1B_REGRESSION[@]}"
}

run_pure_predecessors() {
  run_pytest_count PURE_RC PURE_COUNT PURE_PREDECESSOR "${PURE_PREDECESSORS[@]}"
}

run_page_predecessors() {
  run_pytest_count PAGE_RC PAGE_COUNT PAGE_PREDECESSOR "${PAGE_PREDECESSORS[@]}"
}

run_full_unit() {
  run_pytest_count FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused-pure)
    run_focused_pure
    ;;
  focused-ui)
    run_focused_ui
    ;;
  focused)
    run_focused_pure
    run_focused_ui
    ;;
  predecessors-pure)
    run_i6_1a
    run_pure_predecessors
    ;;
  predecessors-ui)
    run_i6_1b
    run_page_predecessors
    ;;
  predecessors)
    run_i6_1a
    run_i6_1b
    run_pure_predecessors
    run_page_predecessors
    ;;
  full-unit)
    run_full_unit
    ;;
  all)
    run_focused_pure
    run_focused_ui
    run_i6_1a
    run_i6_1b
    run_pure_predecessors
    run_page_predecessors
    ;;
  closure)
    run_focused_pure
    run_focused_ui
    run_i6_1a
    run_i6_1b
    run_pure_predecessors
    run_page_predecessors
    run_full_unit
    ;;
  *)
    echo "usage: $0 [focused-pure|focused-ui|focused|predecessors-pure|predecessors-ui|predecessors|full-unit|all|closure] [receipt.json]" >&2
    exit 2
    ;;
esac

RC=0
case "$MODE" in
  focused-pure)
    for value in "$FOCUSED_PURE_RC" "$PURE_IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  focused-ui)
    for value in "$FOCUSED_UI_RC" "$QT_IMPORT_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  focused)
    for value in "$FOCUSED_PURE_RC" "$FOCUSED_UI_RC" "$PURE_IMPORT_RC" "$QT_IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  predecessors-pure)
    for value in "$I6_1A_RC" "$PURE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  predecessors-ui)
    for value in "$I6_1B_RC" "$PAGE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  predecessors)
    for value in "$I6_1A_RC" "$I6_1B_RC" "$PURE_RC" "$PAGE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  full-unit)
    [[ "$FULL_UNIT_RC" -eq 0 ]] || RC=1
    ;;
  all)
    for value in "$FOCUSED_PURE_RC" "$FOCUSED_UI_RC" "$I6_1A_RC" "$I6_1B_RC" "$PURE_RC" "$PAGE_RC" "$PURE_IMPORT_RC" "$QT_IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  closure)
    for value in "$FOCUSED_PURE_RC" "$FOCUSED_UI_RC" "$I6_1A_RC" "$I6_1B_RC" "$PURE_RC" "$PAGE_RC" "$FULL_UNIT_RC" "$PURE_IMPORT_RC" "$QT_IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$RC" \
    "$FOCUSED_PURE_RC" "$FOCUSED_PURE_COUNT" \
    "$FOCUSED_UI_RC" "$FOCUSED_UI_COUNT" \
    "$I6_1A_RC" "$I6_1A_COUNT" \
    "$I6_1B_RC" "$I6_1B_COUNT" \
    "$PURE_RC" "$PURE_COUNT" \
    "$PAGE_RC" "$PAGE_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$PURE_IMPORT_RC" "$QT_IMPORT_RC" "$COMPILE_RC" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
mode = sys.argv[2]
(
    rc,
    focused_pure_rc,
    focused_pure_count,
    focused_ui_rc,
    focused_ui_count,
    i6_1a_rc,
    i6_1a_count,
    i6_1b_rc,
    i6_1b_count,
    pure_rc,
    pure_count,
    page_rc,
    page_count,
    full_unit_rc,
    full_unit_count,
    pure_import_rc,
    qt_import_rc,
    compile_rc,
) = map(int, sys.argv[3:])

payload = {
    "schema_version": 1,
    "phase": "I6.1C_PROFESSIONAL_WORKSPACE_UI_BINDING",
    "mode": mode,
    "status": "pass" if rc == 0 else "fail",
    "pass": rc == 0,
    "scope": {
        "i6_1a_workspace_contract_reused": True,
        "i6_1b_controller_composition_reused": True,
        "workspace_ui_binding_passive": True,
        "workspace_summary_progressively_disclosed": True,
        "identity_conflicts_visible": True,
        "scale_authority_visible": True,
        "production_auto_deferred_visible": True,
        "controller_modified": False,
        "ui_modified": True,
        "runtime_action_semantics_modified": False,
        "persistence_format_modified": False,
        "reconstruction_algorithm_modified": False,
        "provider_selection_modified": False,
        "production_auto_activated": False,
        "production_auto_qualified": False,
        "production_auto_authorized": False,
        "pycolmap_camera_authority_preserved": True,
        "certified_metrology_claimed": False,
        "production_default_modified": False,
        "threshold_modified": False,
    },
    "counts": {
        "focused_pure_tests": focused_pure_count,
        "focused_ui_tests": focused_ui_count,
        "i6_1a_regression_tests": i6_1a_count,
        "i6_1b_regression_tests": i6_1b_count,
        "pure_predecessor_tests": pure_count,
        "page_predecessor_tests": page_count,
        "full_unit_tests": full_unit_count,
        "combined_selected_tests": (
            focused_pure_count
            + focused_ui_count
            + i6_1a_count
            + i6_1b_count
            + pure_count
            + page_count
        ),
    },
    "return_codes": {
        "focused_pure": focused_pure_rc,
        "focused_ui": focused_ui_rc,
        "i6_1a_regression": i6_1a_rc,
        "i6_1b_regression": i6_1b_rc,
        "pure_predecessor": pure_rc,
        "page_predecessor": page_rc,
        "full_unit": full_unit_rc,
        "pure_import_probe": pure_import_rc,
        "qt_import_probe": qt_import_rc,
        "compile_probe": compile_rc,
    },
    "manual_ui_validation_required": True,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"I6.1C receipt={path}")
PY
fi

echo "I6.1C_TEST_RC=$RC"
exit "$RC"
