#!/usr/bin/env bash
set -u

MODE="${1:-all}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOCUSED_PURE=(
  tests/unit/test_i6_2b_one_start_controller_contract_static.py
)
FOCUSED_QT=(
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
START_STATIC_PREDECESSORS=(
  tests/unit/test_phase8_8b1_reconstruction_page_contract.py
  tests/unit/test_phase8_8b3_runtime_contract.py
  tests/unit/test_phase8_8b5_navigation_contract.py
  tests/unit/test_phase8_8h1_contract.py
  tests/unit/test_phase8_8h2_contract.py
)
QT_START_PREDECESSORS=(
  tests/unit/test_phase8_8a5_controller_hardening.py
  tests/unit/test_phase8_8b2_reconstruction_source_selection.py
  tests/unit/test_phase8_8b3_reconstruction_runtime_controls.py
  tests/ui/test_phase8_8a5_controller_readiness.py
  tests/ui/test_phase8_8b2_reconstruction_source_controls.py
  tests/ui/test_phase8_8b3_reconstruction_runtime_controls.py
  tests/ui/test_phase8_8h2_application_resource_coordination.py
  tests/integration/test_phase8_8a3_offline_controller_process.py
  tests/integration/test_i6_1b_controller_workspace_plumbing.py
  tests/ui/test_i6_1c_offline_workspace_ui.py
)

FOCUSED_PURE_RC=0; FOCUSED_PURE_COUNT=0
FOCUSED_QT_RC=0; FOCUSED_QT_COUNT=0
I6_2A_RC=0; I6_2A_COUNT=0
I6_1_PURE_RC=0; I6_1_PURE_COUNT=0
STATIC_RC=0; STATIC_COUNT=0
QT_RC=0; QT_COUNT=0
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
from app.reconstruction_controller import ReconstructionController
from app.reconstruction_one_start_preflight import ONE_START_EXECUTION_SCOPE

assert hasattr(ReconstructionController, "preflight_selected_offline")
assert hasattr(ReconstructionController, "last_offline_one_start_preflight")
assert hasattr(ReconstructionController, "start_selected_offline")
assert ONE_START_EXECUTION_SCOPE == "sparse_sfm_only"
print("I6.2B controller One-Start orchestration/import probe: PASS")
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    app/reconstruction_controller.py \
    app/reconstruction_one_start_preflight.py \
    tests/unit/test_i6_2b_one_start_controller_contract_static.py \
    tests/integration/test_i6_2b_one_start_controller_orchestration.py \
    tests/integration/test_i6_2a_one_start_preflight_contract.py \
    tests/unit/test_phase8_8b3_reconstruction_runtime_controls.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused_pure() {
  run_pytest_count FOCUSED_PURE_RC FOCUSED_PURE_COUNT FOCUSED_PURE "${FOCUSED_PURE[@]}"
  probe_compile
}

run_focused_qt() {
  run_pytest_count FOCUSED_QT_RC FOCUSED_QT_COUNT FOCUSED_QT "${FOCUSED_QT[@]}"
  probe_import
}

run_predecessors_pure() {
  run_pytest_count I6_2A_RC I6_2A_COUNT I6_2A_REGRESSION "${I6_2A_REGRESSION[@]}"
  run_pytest_count I6_1_PURE_RC I6_1_PURE_COUNT I6_1_PURE_REGRESSION "${I6_1_PURE_REGRESSION[@]}"
  run_pytest_count STATIC_RC STATIC_COUNT START_STATIC_PREDECESSOR "${START_STATIC_PREDECESSORS[@]}"
}

run_predecessors_qt() {
  run_pytest_count QT_RC QT_COUNT QT_START_PREDECESSOR "${QT_START_PREDECESSORS[@]}"
}

run_full_unit() {
  run_pytest_count FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused-pure)
    run_focused_pure
    ;;
  focused-qt)
    run_focused_qt
    ;;
  predecessors-pure)
    run_predecessors_pure
    ;;
  predecessors-qt)
    run_predecessors_qt
    ;;
  all)
    run_focused_pure
    run_focused_qt
    run_predecessors_pure
    run_predecessors_qt
    ;;
  full-unit)
    run_full_unit
    ;;
  closure)
    run_focused_pure
    run_focused_qt
    run_predecessors_pure
    run_predecessors_qt
    run_full_unit
    ;;
  *)
    echo "usage: $0 [focused-pure|focused-qt|predecessors-pure|predecessors-qt|all|full-unit|closure] [receipt.json]" >&2
    exit 2
    ;;
esac

RC=0
case "$MODE" in
  focused-pure)
    for value in "$FOCUSED_PURE_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  focused-qt)
    for value in "$FOCUSED_QT_RC" "$IMPORT_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  predecessors-pure)
    for value in "$I6_2A_RC" "$I6_1_PURE_RC" "$STATIC_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  predecessors-qt)
    [[ "$QT_RC" -eq 0 ]] || RC=1
    ;;
  all)
    for value in "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_2A_RC" "$I6_1_PURE_RC" "$STATIC_RC" "$QT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  full-unit)
    [[ "$FULL_UNIT_RC" -eq 0 ]] || RC=1
    ;;
  closure)
    for value in "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_2A_RC" "$I6_1_PURE_RC" "$STATIC_RC" "$QT_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$RC" \
    "$FOCUSED_PURE_RC" "$FOCUSED_PURE_COUNT" \
    "$FOCUSED_QT_RC" "$FOCUSED_QT_COUNT" \
    "$I6_2A_RC" "$I6_2A_COUNT" \
    "$I6_1_PURE_RC" "$I6_1_PURE_COUNT" \
    "$STATIC_RC" "$STATIC_COUNT" \
    "$QT_RC" "$QT_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
mode = sys.argv[2]
(
    rc,
    focused_pure_rc,
    focused_pure_count,
    focused_qt_rc,
    focused_qt_count,
    i6_2a_rc,
    i6_2a_count,
    i6_1_pure_rc,
    i6_1_pure_count,
    static_rc,
    static_count,
    qt_rc,
    qt_count,
    full_unit_rc,
    full_unit_count,
    import_rc,
    compile_rc,
) = map(int, sys.argv[3:])

payload = {
    "schema_version": 1,
    "phase": "I6.2B_START_ORCHESTRATION_SESSION_CREATION",
    "mode": mode,
    "status": "pass" if rc == 0 else "fail",
    "pass": rc == 0,
    "scope": {
        "i6_2a_preflight_wired_to_controller": True,
        "ui_start_delegation_reused": True,
        "ui_modified": False,
        "controller_modified": True,
        "application_resource_guard_evaluated_once_after_passive_preflight": True,
        "commit_time_preflight_revalidation": True,
        "preflight_session_id_equals_worker_session_id": True,
        "preflight_output_dir_equals_worker_output_dir": True,
        "source_locked_by_existing_active_runtime_contract": True,
        "existing_offline_worker_reused": True,
        "execution_scope": "sparse_sfm_only",
        "dense_execution_authorized": False,
        "persistent_scene_promotion_authorized": False,
        "production_auto_activated": False,
        "production_auto_qualified": False,
        "production_auto_authorized": False,
        "pycolmap_camera_authority_preserved": True,
        "certified_metrology_claimed": False,
        "provider_selection_modified": False,
        "persistence_format_modified": False,
        "reconstruction_algorithm_modified": False,
        "production_default_modified": False,
        "scientific_threshold_modified": False,
    },
    "counts": {
        "focused_pure_tests": focused_pure_count,
        "focused_qt_tests": focused_qt_count,
        "i6_2a_regression_tests": i6_2a_count,
        "i6_1_pure_regression_tests": i6_1_pure_count,
        "start_static_predecessor_tests": static_count,
        "qt_start_predecessor_tests": qt_count,
        "full_unit_tests": full_unit_count,
        "combined_selected_tests": focused_pure_count + focused_qt_count + i6_2a_count + i6_1_pure_count + static_count + qt_count,
    },
    "return_codes": {
        "focused_pure": focused_pure_rc,
        "focused_qt": focused_qt_rc,
        "i6_2a_regression": i6_2a_rc,
        "i6_1_pure_regression": i6_1_pure_rc,
        "start_static_predecessor": static_rc,
        "qt_start_predecessor": qt_rc,
        "full_unit": full_unit_rc,
        "import_probe": import_rc,
        "compile_probe": compile_rc,
    },
    "manual_ui_reconstruction_required": True,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"I6.2B receipt={path}")
PY
fi

echo "I6.2B_TEST_RC=$RC"
exit "$RC"
